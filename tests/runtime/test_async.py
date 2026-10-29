"""从识别公开入口验证取消，保持旧适配器可用。"""

import asyncio
import unittest
from threading import Event

import httpx

from speechio import (
    AudioFormat,
    CancellationToken,
    RemoteModelConfig,
    STTCapabilities,
    STTError,
    STTRuntime,
    Transcriber,
    Transcript,
)
from tests.test_api import RecordingEngine, SpeechDetector
from tests.tts.test_bailian_voice import enrollment_service


class AsyncTranscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_cancellation_stops_polling_and_result_download(self):
        loop = asyncio.get_running_loop()
        polled = asyncio.Event()
        requests = []
        token = CancellationToken()

        def service(request):
            requests.append(request.url.path)
            if request.method == "POST":
                return httpx.Response(200, json={"output": {"task_id": "task-1"}})
            loop.call_soon_threadsafe(polled.set)
            return httpx.Response(
                200, json={"output": {"task_id": "task-1", "task_status": "RUNNING"}}
            )

        transcriber = Transcriber(
            RemoteModelConfig(
                provider="bailian", model="qwen-audio-3.1-asr-flash-filetrans", api_key="test"
            )
        )
        try:
            with enrollment_service(service):
                task = asyncio.create_task(
                    transcriber.transcribe_url_async(
                        "https://example.com/audio.wav", cancellation=token
                    )
                )
                await asyncio.wait_for(polled.wait(), 2)
                token.cancel()
                with self.assertRaises(STTError) as caught:
                    await asyncio.wait_for(task, 1)
                self.assertEqual(caught.exception.code, "CANCELLED")
            self.assertEqual(
                requests, ["/api/v1/services/audio/asr/transcription", "/api/v1/tasks/task-1"]
            )
        finally:
            await transcriber.close_async()

    async def test_runtime_close_is_bounded_while_model_is_preparing(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = Event()

        class Adapter(RecordingEngine):
            def prepare(self):
                loop.call_soon_threadsafe(entered.set)
                release.wait(5)

        runtime = STTRuntime.from_adapter(Adapter())
        preparing = asyncio.create_task(runtime.prepare_async())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            preparing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await preparing
            result = await asyncio.wait_for(runtime.close_async(timeout=0), 1)
            self.assertEqual(result.status, "pending")
        finally:
            release.set()
            self.assertEqual((await runtime.close_async()).status, "ok")

    async def test_auto_input_and_event_wait_do_not_block_loop_or_lose_events(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = Event()

        class Detector(SpeechDetector):
            def detect(self, pcm, start_sample, *, recording=False):
                loop.call_soon_threadsafe(entered.set)
                release.wait(3)
                return super().detect(pcm, start_sample, recording=recording)

        async with STTRuntime.from_adapter(RecordingEngine()) as runtime:
            session = await runtime.create_auto_session_from_detector_async(Detector)
            feeding = asyncio.create_task(session.feed_audio_async(b"\1\0" * 16000))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                waiting = asyncio.Event()

                async def read():
                    waiting.set()
                    return await session.next_event_async()

                reader = asyncio.create_task(read())
                await waiting.wait()
                reader.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await reader
            finally:
                release.set()
                await feeding
            await session.end_input_async()
            final = None
            while True:
                event = await session.next_event_async(timeout=2)
                self.assertIsNotNone(event)
                if event.kind == "final":
                    final = event.result.text
                if event.kind == "completed":
                    break
            self.assertEqual(final, "recognized")

    async def test_cancel_during_inference_discards_result_and_preserves_busy(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = Event()

        class Adapter:
            capabilities = STTCapabilities(engine="test")

            def prepare(self):
                pass

            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                loop.call_soon_threadsafe(entered.set)
                release.wait(5)
                return Transcript(text="迟到结果")

            def close(self):
                pass

        transcriber = Transcriber.from_adapter(Adapter())
        task = asyncio.create_task(
            transcriber.transcribe_async(b"\0\0" * 1600, format=AudioFormat())
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(transcriber.busy)
            self.assertFalse(await transcriber.wait_idle_async(timeout=0))
            self.assertEqual((await transcriber.close_async(timeout=0)).status, "pending")
        finally:
            release.set()
            self.assertEqual((await transcriber.close_async()).status, "ok")

    async def test_cancel_turn_interrupts_its_adapter_and_other_session_completes(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        cancelled = Event()

        class Adapter:
            capabilities = STTCapabilities(engine="test")

            def prepare(self):
                pass

            def transcribe(
                self,
                audio,
                language=None,
                *,
                role="final",
                word_timestamps=False,
                cancellation=None,
            ):
                if audio[0] == 0:
                    loop.call_soon_threadsafe(entered.set)
                    if cancellation is None or not cancellation.wait_cancelled(2):
                        raise AssertionError("取消未到达适配器")
                    cancelled.set()
                    raise STTError("CANCELLED", "transcribe", "已取消")
                return Transcript(text="另一个会话")

            def close(self):
                pass

        runtime = STTRuntime.from_adapter(Adapter())
        runtime.prepare()
        first, second = runtime.create_session(), runtime.create_session()
        try:
            first_turn = first.start()
            first.feed_audio(b"\0\0" * 800)
            first.end_turn()
            await asyncio.wait_for(entered.wait(), 2)
            second.start()
            second.feed_audio(b"\1\0" * 800)
            second.end_turn()
            self.assertTrue(first.cancel_turn(first_turn))
            event = await second.next_event_async(timeout=3)
            self.assertEqual(event.kind, "final")
            self.assertEqual(event.result.text, "另一个会话")
            self.assertTrue(cancelled.is_set())
        finally:
            runtime.close()
