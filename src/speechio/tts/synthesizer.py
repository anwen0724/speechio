"""合成入口：统一能力校验和引擎资源所有权。"""

from dataclasses import replace
from threading import Condition, Event, Thread

from speechio.adapters.contracts import SynthesisCapabilities
from speechio.audio.results import AudioResult
from speechio.errors import TTSError
from speechio.resources import CloseError, CloseResult
from speechio.tts.options import SynthesisOptions, validate_seconds


class Synthesizer:
    """可复用的合成对象，同一时刻只拥有一个任务。"""

    def __init__(self, config=None, *, engine=None, max_buffer_seconds=10.0):
        validate_seconds(max_buffer_seconds, "max_buffer_seconds", positive=True)
        if (config is None) == (engine is None):
            raise TTSError("INVALID_CONFIGURATION", "configure", "config 和 engine 应且仅提供一个")
        if engine is None:
            from speechio.models.factory import build_synthesis_engine

            engine = build_synthesis_engine(config)
        capabilities = getattr(engine, "capabilities", None)
        methods = ["prepare", "close"]
        if isinstance(capabilities, SynthesisCapabilities):
            methods += [
                name
                for flag, name in (
                    ("synthesis", "synthesize"),
                    ("audio_streaming", "stream"),
                    ("text_streaming", "open_stream"),
                )
                if getattr(capabilities, flag)
            ]
        if not isinstance(capabilities, SynthesisCapabilities) or any(
            not callable(getattr(engine, method, None)) for method in methods
        ):
            raise TTSError("INVALID_CONFIGURATION", "configure", "引擎不符合合成接入契约")
        self._engine = engine
        self._capabilities = capabilities
        self._max_buffer_seconds = max_buffer_seconds
        self._condition = Condition()
        self._prepared = False
        self._busy = False
        self._active = None
        self._streams = []
        self._closing = False
        self._cleanup = None
        self._closed = Event()
        self._close_result = None

    @property
    def capabilities(self):
        return self._capabilities

    def _begin(self):
        """在加载和输入副作用之前占用对象；流任务保持占用至读尽或关闭。"""
        with self._condition:
            if self._closing:
                raise TTSError("RUNTIME_CLOSED", "synthesize", "合成对象已关闭")
            if self._busy or (self._active is not None and not self._active.released):
                raise TTSError("RUNTIME_BUSY", "synthesize", "当前合成任务尚未读尽或关闭")
            self._busy = True

    def _end(self):
        with self._condition:
            self._busy = False
            self._condition.notify_all()

    def _prepare(self):
        if not self._prepared:
            try:
                self._engine.prepare()
            except BaseException as exc:
                # 退出当前调用后再清理；不遗留准备了一半的自有资源。
                self.close(timeout=0)
                if isinstance(exc, (TTSError, KeyboardInterrupt, SystemExit)):
                    raise
                raise TTSError("PREPARATION_FAILED", "prepare", "合成后端准备失败") from None
            self._prepared = True

    def prepare(self):
        self._begin()
        try:
            self._prepare()
        finally:
            self._end()

    def _options(self, options, capability):
        """统一校验能力；未知音色与语言组合仍由后端判断。"""
        if not isinstance(options, SynthesisOptions):
            raise TTSError("INVALID_INPUT", "synthesize", "需要 SynthesisOptions")
        caps = self.capabilities
        if not getattr(caps, capability) or any(
            (
                options.language is not None and not caps.language_selection,
                options.speed != 1.0 and not caps.speed_control,
                options.instruction is not None and not caps.instruction_control,
            )
        ):
            raise TTSError("CAPABILITY_UNSUPPORTED", "synthesize", "后端不支持所请求的合成能力")
        rate = options.sample_rate or caps.default_sample_rate
        if caps.sample_rates is not None and rate not in caps.sample_rates:
            raise TTSError("CAPABILITY_UNSUPPORTED", "synthesize", "后端不支持所请求的采样率")
        return replace(options, sample_rate=rate)

    @staticmethod
    def _text(text):
        if not isinstance(text, str) or not text.strip():
            raise TTSError("INVALID_INPUT", "synthesize", "完整合成文本不能为空")

    def synthesize(self, text, *, options):
        """完整合成只调用一次后端；结果大小随文本增长。"""
        self._text(text)
        selected = self._options(options, "synthesis")
        self._begin()
        try:
            self._prepare()
            result = self._engine.synthesize(text, options=selected)
            if (
                not isinstance(result, AudioResult)
                or result.format.sample_rate != selected.sample_rate
            ):
                raise TTSError("PROTOCOL_ERROR", "synthesize", "后端未返回所请求格式的音频结果")
            return result
        except TTSError:
            raise
        except Exception:
            raise TTSError("SYNTHESIS_FAILED", "synthesize", "后端合成失败") from None
        finally:
            self._end()

    def close(self, *, timeout=5.0):
        """关闭超时保留资源拥有者，执行退出后再释放引擎。"""
        validate_seconds(timeout, "timeout")
        with self._condition:
            if self._cleanup is None:
                self._closing = True
                self._cleanup = Thread(target=self._release, name="speechio-tts-close", daemon=True)
                self._cleanup.start()
        if not self._closed.wait(timeout):
            return CloseResult(status="pending", cleanup_method="deferred")
        return self._close_result

    def stream(self, text, *, options):
        self._text(text)
        return self._start_stream(options, text=text)

    def open_stream(self, *, options):
        return self._start_stream(options)

    def _start_stream(self, options, *, text=None):
        from speechio.tts.stream import SynthesisStream

        selected = self._options(options, "text_streaming" if text is None else "audio_streaming")
        self._begin()
        backend = None
        try:
            self._prepare()
            backend = (
                self._engine.open_stream(options=selected)
                if text is None
                else self._engine.stream(text, options=selected)
            )
            if backend.format.sample_rate != selected.sample_rate:
                raise TTSError("PROTOCOL_ERROR", "open_stream", "后端未采用所请求采样率")
            stream = SynthesisStream(
                backend, incremental=text is None, max_buffer_seconds=self._max_buffer_seconds
            )
            with self._condition:
                self._active = stream
                self._streams = [item for item in self._streams if not item.released]
                self._streams.append(stream)
                if self._closing:
                    stream.cancel()
            return stream
        except BaseException as exc:
            if backend is not None:
                try:
                    backend.cancel()
                    backend.close()
                except Exception:
                    exc.add_note("创建合成流失败后，后端清理未成功")
            if isinstance(exc, (TTSError, KeyboardInterrupt, SystemExit)):
                raise
            raise TTSError("SYNTHESIS_FAILED", "open_stream", "合成流创建失败") from None
        finally:
            self._end()

    def _release(self):
        errors = []
        try:
            with self._condition:
                self._condition.wait_for(lambda: not self._busy)
            for stream in self._streams:
                stream.close(timeout=0)
            for stream in self._streams:
                result = stream.wait_closed()
                if result.status != "ok":
                    errors.extend(
                        result.errors
                        or (CloseError("CLOSE_FAILED", "close_stream", "流资源尚未释放"),)
                    )
            result = self._engine.close()
            while isinstance(result, CloseResult) and result.status == "pending":
                self._closed.wait(0.05)
                result = self._engine.close()
            if isinstance(result, CloseResult):
                self._close_result = CloseResult(
                    status="failed" if errors else result.status,
                    cleanup_method=result.cleanup_method,
                    errors=tuple(errors) + result.errors,
                )
            else:
                self._close_result = CloseResult(
                    status="failed" if errors else "ok", errors=tuple(errors)
                )
        except BaseException:
            errors.append(CloseError("CLOSE_FAILED", "close", "合成引擎释放失败"))
            self._close_result = CloseResult(status="failed", errors=tuple(errors))
        finally:
            self._closed.set()

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, traceback):
        result = self.close()
        if result.status != "ok":
            if exc is not None:
                exc.add_note(f"合成资源关闭状态：{result.status}")
            else:
                raise TTSError("CLOSE_FAILED", "close", f"合成资源关闭状态：{result.status}")
