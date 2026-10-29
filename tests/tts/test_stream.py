"""从公开合成对象观察顺序交付、结束和资源释放。"""

import unittest
from queue import Queue
from threading import Event, Thread

from speechio import AudioFormat
from speechio.adapters.contracts import TTSAdapter, TTSCapabilities
from speechio.audio.results import AudioChunk
from speechio.errors import TTSError
from speechio.resources import CloseResult
from speechio.tts.options import SynthesisOptions
from speechio.tts.synthesizer import Synthesizer


class ControlledBackend:
    """可控外部模型流；测试决定音频和完成确认的到达时间。"""

    format = AudioFormat(sample_rate=24000)

    def __init__(self):
        self.queue = Queue()
        self.texts = []
        self.finished = 0
        self.closed = Event()

    def send_text(self, text):
        self.texts.append(text)

    def finish_input(self):
        self.finished += 1

    def read(self, *, timeout=None):
        item = self.queue.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def cancel(self):
        self.queue.put(None)

    def close(self, *, timeout=5.0):
        self.closed.set()


class StreamingEngine(TTSAdapter):
    capabilities = TTSCapabilities(
        default_sample_rate=24000, audio_streaming=True, text_streaming=True
    )

    def __init__(self):
        self.backend = ControlledBackend()
        self.closed = Event()

    def open_stream(self, *, options):
        return self.backend

    def stream(self, text, *, options):
        self.backend.send_text(text)
        self.backend.finish_input()
        return self.backend

    def close(self):
        self.closed.set()


class StreamTests(unittest.TestCase):
    def test_failed_start_still_closes_backend_when_cancel_fails(self):
        engine = StreamingEngine()
        engine.backend.format = AudioFormat(sample_rate=16000)

        def fail_cancel():
            raise RuntimeError("cancel failed")

        engine.backend.cancel = fail_cancel
        synth = Synthesizer.from_adapter(engine)
        self.addCleanup(synth.close)
        with self.assertRaises(TTSError) as caught:
            synth.open_stream(options=SynthesisOptions())
        self.assertEqual(caught.exception.code, "PROTOCOL_ERROR")
        result = synth.close(timeout=1)
        self.assertTrue(engine.backend.closed.is_set())
        self.assertTrue(engine.closed.is_set())
        self.assertEqual(result.status, "failed")
        self.assertIn("cancel", [error.operation for error in result.errors])

    def test_failed_start_owns_pending_backend_until_cleanup_finishes(self):
        for final_status in ("ok", "failed"):
            with self.subTest(final_status=final_status):
                engine = StreamingEngine()
                engine.backend.format = AudioFormat(sample_rate=16000)
                allow_cleanup = Event()
                cleanup_attempted = Event()

                def close_backend():
                    cleanup_attempted.set()
                    if not allow_cleanup.is_set():
                        return CloseResult(status="pending")
                    engine.backend.closed.set()
                    return CloseResult(status=final_status)

                engine.backend.close = close_backend
                synth = Synthesizer.from_adapter(engine)
                try:
                    with self.assertRaises(TTSError) as caught:
                        synth.open_stream(options=SynthesisOptions())
                    self.assertEqual(caught.exception.code, "PROTOCOL_ERROR")
                    self.assertTrue(cleanup_attempted.wait(1))
                    self.assertEqual(synth.close(timeout=0).status, "pending")
                    self.assertFalse(engine.closed.is_set())
                    with self.assertRaises(TTSError):
                        synth.open_stream(options=SynthesisOptions())
                finally:
                    allow_cleanup.set()
                    result = synth.close(timeout=1)
                self.assertEqual(result.status, final_status)
                self.assertTrue(engine.backend.closed.is_set())
                self.assertTrue(engine.closed.is_set())
                if final_status == "failed":
                    self.assertTrue(result.errors)

    def test_read_timeout_is_recoverable_and_cancel_wakes_reader(self):
        engine = StreamingEngine()
        synth = Synthesizer.from_adapter(adapter=engine)
        self.addCleanup(synth.close)
        stream = synth.open_stream(options=SynthesisOptions(voice="test"))
        with self.assertRaises(TTSError) as caught:
            stream.read(timeout=0)
        self.assertEqual(caught.exception.code, "READ_TIMEOUT")
        stream.send_text("Hello.")
        engine.backend.queue.put(AudioChunk(b"\x01\x00", stream.format))
        self.assertEqual(stream.read(timeout=1).data, b"\x01\x00")
        outcome = []
        waiting = Event()

        def read():
            waiting.set()
            try:
                stream.read()
            except TTSError as exc:
                outcome.append(exc.code)

        reader = Thread(target=read)
        reader.start()
        self.assertTrue(waiting.wait(1))
        stream.cancel()
        reader.join(1)
        self.assertFalse(reader.is_alive())
        self.assertEqual(outcome, ["CANCELLED"])
        self.assertEqual(stream.close().status, "ok")

    def test_buffer_overflow_fails_without_delivering_queued_audio(self):
        engine = StreamingEngine()
        with Synthesizer.from_adapter(adapter=engine, max_buffer_seconds=0.001) as synth:
            with synth.stream("Hello.", options=SynthesisOptions(voice="test")) as stream:
                engine.backend.queue.put(AudioChunk(b"\x01\x00" * 25, stream.format))
                with self.assertRaises(TTSError) as caught:
                    stream.read(timeout=1)
                self.assertEqual(caught.exception.code, "BUFFER_EXCEEDED")
                self.assertEqual(stream.state, "failed")

    def test_stream_owns_task_until_read_to_end_and_can_then_be_reused(self):
        engine = StreamingEngine()
        with Synthesizer.from_adapter(adapter=engine) as synth:
            stream = synth.stream("One.", options=SynthesisOptions(voice="test"))
            with self.assertRaises(TTSError) as caught:
                synth.open_stream(options=SynthesisOptions(voice="test"))
            self.assertEqual(caught.exception.code, "RUNTIME_BUSY")
            engine.backend.queue.put(None)
            self.assertIsNone(stream.read(timeout=1))
            engine.backend = ControlledBackend()
            with synth.stream("Two.", options=SynthesisOptions(voice="test")) as second:
                engine.backend.queue.put(AudioChunk(b"\x02\x00", second.format))
                engine.backend.queue.put(None)
                self.assertEqual(b"".join(c.data for c in second), b"\x02\x00")

    def test_backend_failure_does_not_become_successful_end(self):
        engine = StreamingEngine()
        with Synthesizer.from_adapter(adapter=engine) as synth:
            with synth.stream("Hello.", options=SynthesisOptions(voice="test")) as stream:
                engine.backend.queue.put(RuntimeError("private backend detail"))
                with self.assertRaises(TTSError) as caught:
                    stream.read(timeout=1)
                self.assertEqual(caught.exception.code, "SYNTHESIS_FAILED")
                self.assertNotIn("private", str(caught.exception))

    def test_worker_exit_exception_is_reported_to_reader(self):
        engine = StreamingEngine()

        def stop_worker(*, timeout=None):
            raise SystemExit("worker exited")

        engine.backend.read = stop_worker
        with Synthesizer.from_adapter(adapter=engine) as synth:
            with synth.stream("Hello.", options=SynthesisOptions(voice="test")) as stream:
                with self.assertRaises(TTSError) as caught:
                    stream.read(timeout=0.1)
                self.assertEqual(caught.exception.code, "SYNTHESIS_FAILED")

    def test_pending_backend_cleanup_keeps_engine_owned_until_release(self):
        engine = StreamingEngine()
        allow_cleanup = Event()
        cleanup_attempted = Event()

        def close_backend(*, timeout=5.0):
            cleanup_attempted.set()
            if not allow_cleanup.is_set():
                return CloseResult(status="pending")
            engine.backend.closed.set()
            return CloseResult()

        engine.backend.close = close_backend
        synth = Synthesizer.from_adapter(adapter=engine)
        stream = synth.open_stream(options=SynthesisOptions(voice="test"))
        stream.cancel()
        self.assertTrue(cleanup_attempted.wait(1))
        try:
            self.assertEqual(synth.close(timeout=0.02).status, "pending")
            self.assertFalse(engine.closed.is_set())
        finally:
            allow_cleanup.set()
        self.assertEqual(synth.close(timeout=1).status, "ok")
        self.assertTrue(stream.closed)

    def test_incremental_text_preserves_spaces_and_waits_for_final_audio(self):
        engine = StreamingEngine()
        with Synthesizer.from_adapter(adapter=engine) as synth:
            with synth.open_stream(options=SynthesisOptions(voice="test")) as stream:
                stream.send_text("Hello")
                stream.send_text(" ")
                stream.send_text("world.")
                stream.send_text("")
                stream.finish_input()
                stream.finish_input()
                self.assertEqual(engine.backend.texts, ["Hello", " ", "world."])
                self.assertEqual(engine.backend.finished, 1)
                self.assertEqual(stream.state, "finishing")
                engine.backend.queue.put(AudioChunk(b"\x01\x00", stream.format))
                self.assertEqual(stream.read(timeout=1).data, b"\x01\x00")
                engine.backend.queue.put(AudioChunk(b"\x02\x00", stream.format))
                engine.backend.queue.put(None)
                self.assertEqual(b"".join(chunk.data for chunk in stream), b"\x02\x00")
                self.assertEqual(stream.state, "completed")
        self.assertTrue(engine.backend.closed.is_set())
        self.assertTrue(engine.closed.is_set())
