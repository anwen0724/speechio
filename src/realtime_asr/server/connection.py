"""单连接的归属、控制和有界发送；共享模型仍由 Service 管理。"""

import asyncio
import json
from collections import Counter, OrderedDict, deque
from dataclasses import asdict
from time import monotonic

from anyio import CancelScope
from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from realtime_asr import ASRError
from realtime_asr.protocol import decode_audio_packet
from realtime_asr.server.schemas import AudioMetadata, Command


def preview_key(message):
    """只有可替换预览参与合并；最终、完成及连接错误保持关键出口。"""
    if message["type"] == "state":
        return (message["session_id"], message["generation"]), ("state", None)
    event = message.get("event", {})
    kind = event.get("kind")
    if kind == "realtime":
        return event["unit"], ("realtime", event.get("realtime_role"))
    if kind in ("candidate", "candidate_failed", "correction", "correction_failed"):
        return event["unit"], ("candidate", event["source"])
    return None


class Outbox:
    """保留关键消息，合并等待发送的实时观察；所有操作在所属事件循环。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.wakeup = asyncio.Event()
        self.sending = False

    def put(self, message):
        dropped = Counter()
        key = preview_key(message)
        if key is not None:
            for old in tuple(self.queue):
                if preview_key(old) == key:
                    self.queue.remove(old)
                    dropped[old.get("event", {}).get("kind", old["type"])] += 1
        while len(self.queue) >= self.capacity:
            old = next((item for item in self.queue if preview_key(item) is not None), None)
            if old is not None:
                self.queue.remove(old)
                dropped[old.get("event", {}).get("kind", old["type"])] += 1
            elif key is not None:
                dropped[message.get("event", {}).get("kind", message["type"])] += 1
                return dropped
            else:
                raise ASRError("SLOW_CONSUMER", "send", "关键消息发送缓冲已满")
        self.queue.append(message)
        self.wakeup.set()
        return dropped

    def keep_generation(self, generation):
        self.queue = deque(m for m in self.queue if m["generation"] == generation)


class Connection:
    """一条网络连接只控制自己的 Session，控制应答和事件使用一个发送者。"""

    def __init__(self, service, websocket, session, options):
        self.service = service
        self.websocket = websocket
        self.session = session
        self.options = options
        self.generation = session.generation
        self.audio_sequence = 0
        self.wire_sequence = 0
        self.active = None
        self.last_task = None
        self.active_has_audio = False
        self.pending_units = set()
        self.closed = False
        self.closing = False
        self.close_code = None
        self.close_task = None
        self._last_input_state = None
        self.cache = OrderedDict()
        self.outbox = Outbox(service.config.max_pending_messages)
        self.control_gate = asyncio.Lock()
        self.resume_epochs = {}
        self.candidate_requests = {}

    def message(self, kind, *, generation=None, **fields):
        return {
            "protocol_version": 1,
            "type": kind,
            "session_id": self.session.session_id,
            "generation": self.generation if generation is None else generation,
            **fields,
        }

    def emit(self, message):
        dropped = self.outbox.put(message)
        self.service.counts["realtime_dropped"] += dropped["realtime"]
        self.service.counts["candidate_dropped"] += (
            sum(dropped.values()) - dropped["realtime"] - dropped["state"]
        )
        self.service.counts["state_dropped"] += dropped["state"]

    def _input_status(self):
        if self.options.mode == "auto":
            return {
                "input_state": self.session.input_state,
                "wake_status": self.session.wake_status,
            }
        return {
            "input_state": "recording" if self.active is not None else "listening",
            "wake_status": {"enabled": False, "state": "disabled", "released": True},
        }

    def _emit_input_status(self, *, force=False):
        """音频位置不单独刷屏，只发布当前代次的状态转换。"""
        if not (self.options.status_updates or self.options.wake):
            return
        value = self._input_status()
        wake = value["wake_status"]
        stamp = (
            self.generation,
            value["input_state"],
            wake["state"],
            wake.get("detections"),
            wake.get("timeouts"),
        )
        if force or stamp != self._last_input_state:
            self.emit(self.message("state", **value))
            self._last_input_state = stamp

    def abort(self, error_code, message, *, code=1013):
        """失败清楚结束当前连接；不尝试为残缺输入生成最终。"""
        if self.closed:
            return
        self.closing = True
        self.close_code = code
        self.outbox.queue.clear()
        self.outbox.queue.append(
            self.message("error", code=error_code, operation="connection", message=message)
        )
        self.outbox.wakeup.set()
        self.service.counts["errors"] += 1

    async def run(self):
        self.service.links.add(self)
        self.emit(
            self.message(
                "ready",
                request_id=self.options.request_id,
                mode=self.options.mode,
                format=self.options.format.model_dump(),
                expected_audio_sequence=0,
                candidate=self.service.capabilities()["candidate"],
                wake=self.service.capabilities()["wake"],
                text_options={
                    name: getattr(self.options, name)
                    for name in (
                        "stable_text",
                        "dual_realtime",
                        "word_timestamps",
                        "early_transcription_silence",
                        "dynamic_silence",
                        "punctuation_split",
                        "completion_mode",
                        "retain_for_correction",
                    )
                },
            )
        )
        self._emit_input_status(force=True)
        sender = asyncio.create_task(self._send(), name="ASRWSSend")
        tasks = [
            asyncio.create_task(self._receive(), name="ASRWSReceive"),
            asyncio.create_task(self._events(), name="ASRWSEvents"),
            sender,
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, OSError):
            # 传输已失效，不能把未发送消息改投其他连接。
            self.service.counts["disconnects"] += 1
        except ASRError as exc:
            # 控制拒绝消息也可能达到发送上限，仍经本连接的唯一出口交付。
            self.abort(exc.code, exc.message)
            await sender
        finally:
            self.closed = True
            # ASGI 调用方可以取消连接；清理仍需完成自有线程交接与任务退出。
            with CancelScope(shield=True):
                await self.service.release_session(self.session)
                self.service.links.discard(self)
                if self.close_task is not None:
                    self.close_task.cancel()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(
                    *tasks, *([self.close_task] if self.close_task else []), return_exceptions=True
                )
                self.outbox.queue.clear()

    async def _receive(self):
        while not self.closed:
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive(), self.service.config.idle_timeout
                )
            except TimeoutError:
                self.abort("IDLE_TIMEOUT", "连接超过空闲时限", code=1008)
                await asyncio.Future()  # 让唯一发送者交付错误并关闭连接。
            if message["type"] == "websocket.disconnect":
                return
            try:
                if message.get("bytes") is not None:
                    await self._audio(message["bytes"])
                elif message.get("text") is not None:
                    await self._command(message["text"])
            except (ValueError, UnicodeError, ValidationError):
                if message.get("bytes") is not None:
                    self.abort("INVALID_MESSAGE", "音频包无效", code=1008)
                    await asyncio.Future()
                self.service.counts["errors"] += 1
                self.emit(
                    self.message(
                        "error",
                        code="INVALID_MESSAGE",
                        operation="control",
                        message="控制 JSON 或字段无效",
                        request_id=self._request_id(message),
                    )
                )
            except ASRError as exc:
                if message.get("bytes") is not None or exc.code == "SLOW_CONSUMER":
                    self.abort(
                        exc.code, exc.message, code=1008 if exc.code != "SLOW_CONSUMER" else 1013
                    )
                    await asyncio.Future()
                self.service.counts["errors"] += 1
                self.emit(
                    self.message(
                        "error",
                        code=exc.code,
                        operation=exc.operation,
                        message=exc.message,
                        request_id=self._request_id(message),
                    )
                )

    @staticmethod
    def _request_id(message):
        try:
            request_id = json.loads(message["text"]).get("request_id")
            return (
                request_id if isinstance(request_id, str) and 1 <= len(request_id) <= 128 else None
            )
        except (ValueError, AttributeError):
            return None

    async def _audio(self, packet):
        if self.closing:
            raise ASRError("INPUT_CLOSED", "feed_audio", "连接正在结束输入")
        config = self.service.config
        metadata, audio = decode_audio_packet(
            packet, max_bytes=config.max_packet_bytes, max_metadata=config.max_metadata_bytes
        )
        metadata = AudioMetadata.model_validate(metadata)
        expected_unit = self.active.unit_id if self.active is not None else None
        if (
            metadata.session_id != self.session.session_id
            or metadata.generation != self.generation
            or metadata.unit_id != expected_unit
            or metadata.format != self.options.format
        ):
            raise ASRError("AUDIO_OWNERSHIP_MISMATCH", "feed_audio", "音频不属于当前输入")
        if metadata.audio_sequence != self.audio_sequence:
            raise ASRError("AUDIO_SEQUENCE_MISMATCH", "feed_audio", "音频传输序号不连续")
        expected_bytes = (
            metadata.frame_count
            * metadata.format.channels
            * (2 if metadata.format.sample_type == "int16" else 4)
        )
        if len(audio) != expected_bytes:
            raise ASRError("AUDIO_FRAME_MISMATCH", "feed_audio", "音频字节与源帧数不一致")
        started = monotonic()
        await asyncio.to_thread(self.session.feed_audio, audio)
        self.service.measure("input_accept", monotonic() - started)
        self.audio_sequence += 1
        self.active_has_audio = True
        self.service.counts["audio_packets"] += 1
        self.service.counts["input_bytes"] += len(audio)
        self.service.counts["source_frames"] += metadata.frame_count
        self._emit_input_status()

    async def _command(self, text):
        """应答先进入唯一出口，事件消费者随后领取新代次的观察。"""
        command = Command.model_validate(json.loads(text))
        async with self.control_gate:
            await self._apply_command(command)

    async def _apply_command(self, command):
        fingerprint = command.model_dump()
        cached = self.cache.get(command.request_id)
        if cached is not None:
            if cached[0] != fingerprint:
                raise ASRError("REQUEST_ID_CONFLICT", "control", "请求 ID 已用于不同内容")
            self.emit(dict(cached[1]))
            return
        if command.session_id != self.session.session_id or command.generation != self.generation:
            raise ASRError("SESSION_MISMATCH", "control", "命令不属于本连接当前代次")
        if self.closing and command.type not in ("cancel", "reset", "ping", "close"):
            raise ASRError("INPUT_CLOSED", "control", "连接正在结束输入")
        if command.unit is not None and command.type not in ("cancel", "finish", "correct"):
            raise ASRError("INVALID_MESSAGE", "control", "该命令不接受 unit")
        kind = command.type
        fields = {}
        if kind == "ping" and (self.options.status_updates or self.options.wake):
            fields.update(self._input_status())
        if kind == "start":
            if self.options.mode != "manual":
                raise ASRError("MODE_UNAVAILABLE", "start", "自动输入不单独开始音频单元")
            unit = await asyncio.to_thread(self.session.start)
            self.candidate_requests = {
                key: value
                for key, value in self.candidate_requests.items()
                if key[1] != "correction"
            }
            self.active, self.last_task = unit, None
            self.active_has_audio = False
            self.audio_sequence = 0
            self.pending_units.add(unit.unit_id)
            fields["unit"] = asdict(unit)
        elif kind == "finish":
            target = command.unit.to_ref() if command.unit is not None else None
            if target is not None and target not in (
                self.active,
                self.last_task.unit if self.last_task else None,
            ):
                raise ASRError("INVALID_TARGET", "finish", "结束目标不是当前或最近封口单元")
            task = await asyncio.to_thread(self.session.finish)
            self.last_task, self.active = task, None
            fields["task"] = asdict(task)
        elif kind == "end_input":
            if self.options.mode != "auto":
                raise ASRError("MODE_UNAVAILABLE", "end_input", "仅自动输入使用持续 EOF")
            await asyncio.to_thread(self.session.finish_input)
        elif kind == "candidate":
            request = await asyncio.to_thread(
                self.session.request_candidate, kind=command.candidate_kind, hint=command.hint
            )
            fields["candidate_request"] = asdict(request)
            self.candidate_requests[request.unit.unit_id, request.kind] = request.request_id
        elif kind == "resume":
            resume = await asyncio.to_thread(
                self.session.resume, expected_audio_samples=command.expected_audio_samples
            )
            fields["resume"] = asdict(resume)
            self.resume_epochs[resume.unit.unit_id] = resume.resume_epoch
            self.candidate_requests = {
                key: value
                for key, value in self.candidate_requests.items()
                if key[0] != resume.unit.unit_id
            }
        elif kind == "correct":
            request = await asyncio.to_thread(
                self.session.request_correction, command.unit.to_ref()
            )
            fields["candidate_request"] = asdict(request)
            self.candidate_requests[request.unit.unit_id, request.kind] = request.request_id
        elif kind == "cancel":
            target = command.unit.to_ref() if command.unit is not None else None
            fields["cancelled"] = await asyncio.to_thread(self.session.cancel, target)
            if fields["cancelled"] and (target is None or target == self.active):
                self.active = None
        elif kind == "reset":
            change = await asyncio.to_thread(self.session.reset)
            self.generation = change.generation
            self.audio_sequence = 0
            self.active = self.last_task = None
            self.pending_units.clear()
            self.resume_epochs.clear()
            self.candidate_requests.clear()
            self.outbox.keep_generation(self.generation)
            fields["change"] = asdict(change)
        elif kind == "close":
            self.closing = True
            fields["status"] = "draining"
        response = self.message(
            "pong" if kind == "ping" else "ack",
            request_id=command.request_id,
            command=kind,
            **fields,
        )
        self.cache[command.request_id] = fingerprint, response
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        self.emit(response)
        if kind in ("start", "finish", "cancel", "reset", "end_input"):
            self._emit_input_status(force=True)
        if kind == "close" and self.close_task is None:
            self.close_task = asyncio.create_task(self._drain_close(), name="ASRWSDrain")

    async def _events(self):
        while not self.closed:
            # SDK 可能等待别的会话正在转换的全局协调条件，不能占据网络控制锁。
            event = await asyncio.to_thread(self.session.get_event, timeout=0.05)
            if event is None:
                continue
            async with self.control_gate:
                if event.unit.generation != self.generation:
                    continue
                if event.resume_epoch < self.resume_epochs.get(event.unit.unit_id, 0):
                    continue
                if event.candidate is not None:
                    request = event.candidate.request
                    if (
                        self.candidate_requests.get((event.unit.unit_id, request.kind))
                        != request.request_id
                    ):
                        continue
                self.service.observe(event)
                try:
                    self.emit(self.message("event", event=asdict(event)))
                except ASRError as exc:
                    self.abort(exc.code, exc.message)
                    await asyncio.Future()
                if event.kind in ("completed", "failed", "cancelled"):
                    self.pending_units.discard(event.unit.unit_id)
                    self.resume_epochs.pop(event.unit.unit_id, None)
                    self.candidate_requests = {
                        key: value
                        for key, value in self.candidate_requests.items()
                        if key[0] != event.unit.unit_id or key[1] == "correction"
                    }

    async def _drain_close(self):
        try:
            if self.options.mode == "auto":
                await asyncio.to_thread(self.session.finish_input)
            elif self.active is not None:
                if self.active_has_audio:
                    self.last_task = await asyncio.to_thread(self.session.finish)
                else:
                    await asyncio.to_thread(self.session.cancel)
                self.active = None
            deadline = monotonic() + self.service.config.drain_timeout
            while True:
                pending = (
                    self.session.pending_segments
                    if self.options.mode == "auto"
                    else len(self.pending_units)
                )
                if (
                    not pending
                    and not self.session.pending_candidates
                    and not self.outbox.queue
                    and not self.outbox.sending
                ):
                    break
                if monotonic() >= deadline:
                    raise ASRError("DRAIN_TIMEOUT", "close", "等待本连接结束超时")
                await asyncio.sleep(0.01)
            self.emit(self.message("closed", status="ok"))
            self.close_code = 1000
            self.outbox.wakeup.set()
        except ASRError as exc:
            self.abort(exc.code, exc.message)

    async def _send(self):
        while not self.closed:
            await self.outbox.wakeup.wait()
            while self.outbox.queue:
                message = self.outbox.queue.popleft()
                self.wire_sequence += 1
                message = {**message, "wire_sequence": self.wire_sequence}
                self.outbox.sending = True
                started = monotonic()
                try:
                    await asyncio.wait_for(
                        self.websocket.send_json(message), self.service.config.send_timeout
                    )
                except RuntimeError:
                    self.service.counts["disconnects"] += 1
                    return
                except TimeoutError:
                    self.service.counts["send_timeouts"] += 1
                    await self.websocket.close(code=1013)
                    return
                finally:
                    self.outbox.sending = False
                self.service.measure("send", monotonic() - started)
                self.service.counts["messages_sent"] += 1
            if self.close_code is not None:
                await self.websocket.close(code=self.close_code)
                return
            self.outbox.wakeup.clear()
