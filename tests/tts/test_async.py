"""异步入口的取消与资源所有权；替身仅位于适配器边界。"""

import asyncio
import json
import unittest
from queue import Queue
from threading import Event

import httpx

import speechio
from speechio import (
    AudioFormat,
    AudioResult,
    CancellationToken,
    RemoteModelConfig,
    SynthesisOptions,
    Synthesizer,
    TTSCapabilities,
    TTSError,
)
from speechio.audio.results import AudioChunk
from speechio.tts.stream import SynthesisStream
from tests.runtime.test_remote import websocket_service
from tests.tts.test_bailian_voice import enrollment_service


class AsyncSynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_cancel_keeps_already_received_cloud_voice_id(self):
        loop = asyncio.get_running_loop()
        querying = asyncio.Event()
        release = Event()

        def service(request):
            if json.loads(request.content)["input"]["action"] == "create_voice":
                return httpx.Response(200, json={"output": {"voice_id": "created-before-cancel"}})
            loop.call_soon_threadsafe(querying.set)
            release.wait(3)
            return httpx.Response(200, json={"output": {"status": "OK"}})

        tts = Synthesizer(
            RemoteModelConfig(provider="bailian", model="qwen-audio-3.1-tts-flash", api_key="test")
        )
        with enrollment_service(service):
            task = asyncio.create_task(
                tts.clone_voice_async(AudioResult(b"\0\0" * 80000, AudioFormat()))
            )
            try:
                await asyncio.wait_for(querying.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError) as caught:
                    await task
                self.assertIn(
                    "created-before-cancel", " ".join(getattr(caught.exception, "__notes__", ()))
                )
            finally:
                release.set()
                await tts.close_async()

    async def test_cancelled_send_keeps_backend_until_sender_exits(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        stop_read = Event()
        release_send = Event()

        class Backend:
            format = AudioFormat(sample_rate=24000)

            def send_text(self, text):
                loop.call_soon_threadsafe(entered.set)
                release_send.wait(3)

            def read(self):
                stop_read.wait(3)
                return None

            def cancel(self):
                stop_read.set()

            def close(self):
                if not release_send.is_set():
                    raise AssertionError("发送尚未退出，不能关闭后端")

        stream = SynthesisStream(Backend(), incremental=True, max_buffer_seconds=1)
        sending = asyncio.create_task(stream.send_text_async("正在发送"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            sending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await sending
            self.assertEqual((await stream.close_async(timeout=0.05)).status, "pending")
        finally:
            release_send.set()
            self.assertEqual((await stream.close_async()).status, "ok")

    async def test_cancel_discards_completed_but_unread_audio(self):
        loop = asyncio.get_running_loop()
        completed = asyncio.Event()

        class Backend:
            format = AudioFormat(sample_rate=24000)

            def __init__(self):
                self.chunks = iter((AudioChunk(b"\1\0", self.format), None))

            def read(self):
                return next(self.chunks)

            def cancel(self):
                pass

            def close(self):
                loop.call_soon_threadsafe(completed.set)

        stream = SynthesisStream(Backend(), incremental=False, max_buffer_seconds=1)
        await asyncio.wait_for(completed.wait(), 2)
        stream.cancel()
        with self.assertRaises(TTSError) as caught:
            await stream.read_async()
        self.assertEqual(caught.exception.code, "CANCELLED")
        self.assertEqual((await stream.close_async()).status, "ok")

    async def test_cancelled_stream_creation_releases_late_backend(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = Event()
        stopped = Event()
        released = Event()

        class Backend:
            format = AudioFormat(sample_rate=24000)

            def read(self):
                stopped.wait(3)
                return None

            def cancel(self):
                stopped.set()

            def close(self):
                released.set()

        class Adapter:
            capabilities = TTSCapabilities(
                default_sample_rate=24000, text_streaming=True, audio_streaming=True
            )

            def prepare(self):
                pass

            def stream(self, text, *, options):
                raise AssertionError("不是整段流")

            def open_stream(self, *, options):
                loop.call_soon_threadsafe(entered.set)
                release.wait(3)
                return Backend()

            def close(self):
                if not released.is_set():
                    raise AssertionError("不能先关闭模型")

        tts = Synthesizer.from_adapter(Adapter())
        task = asyncio.create_task(tts.open_stream_async(options=SynthesisOptions()))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(tts.busy)
        finally:
            release.set()
            self.assertEqual((await tts.close_async()).status, "ok")
        self.assertTrue(stopped.is_set())
        self.assertTrue(released.is_set())

    async def test_precancelled_convenience_call_does_not_prepare_or_connect(self):
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(TTSError) as caught:
            await speechio.synthesize_async(
                "不应发送",
                config=RemoteModelConfig(provider="bailian", model="qwen-audio-3.1-tts-flash"),
                cancellation=token,
            )
        self.assertEqual(caught.exception.code, "CANCELLED")

    async def test_cancelled_native_call_keeps_model_until_execution_exits(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = Event()
        closed = Event()

        class Adapter:
            capabilities = TTSCapabilities(default_sample_rate=24000, synthesis=True)

            def prepare(self):
                pass

            def synthesize(self, text, *, options):
                loop.call_soon_threadsafe(entered.set)
                release.wait(5)
                return AudioResult(b"\0\0", AudioFormat(sample_rate=24000))

            def close(self):
                closed.set()

        tts = Synthesizer.from_adapter(Adapter())
        task = asyncio.create_task(tts.synthesize_async("你好"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(tts.busy)
            self.assertEqual((await tts.close_async(timeout=0)).status, "pending")
            self.assertFalse(closed.is_set())
        finally:
            release.set()
            self.assertEqual((await tts.close_async()).status, "ok")
        self.assertTrue(closed.is_set())

    async def test_cancelling_read_preserves_next_audio(self):
        received = Queue()

        class Backend:
            format = AudioFormat(sample_rate=24000)

            def read(self):
                return received.get(timeout=5)

            def cancel(self):
                received.put(None)

            def close(self):
                pass

        stream = SynthesisStream(Backend(), incremental=False, max_buffer_seconds=1)
        waiting = asyncio.Event()

        async def read():
            waiting.set()
            return await stream.read_async()

        task = asyncio.create_task(read())
        try:
            await waiting.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            received.put(AudioChunk(b"\1\0", Backend.format))
            self.assertEqual((await stream.read_async(timeout=2)).data, b"\1\0")
        finally:
            await stream.close_async()

    async def test_token_cancels_remote_startup_without_waiting_network_timeout(self):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def service(ws):
            json.loads(ws.recv())
            loop.call_soon_threadsafe(started.set)
            try:
                ws.recv(timeout=5)
            except Exception:
                pass

        with websocket_service(service) as url:
            tts = Synthesizer(
                RemoteModelConfig(
                    provider="bailian",
                    model="qwen-audio-3.1-tts-flash",
                    api_key="test",
                    base_url=url,
                    timeout_seconds=10,
                )
            )
            token = CancellationToken()
            task = asyncio.create_task(
                tts.open_stream_async(options=SynthesisOptions(voice="test"), cancellation=token)
            )
            try:
                await asyncio.wait_for(started.wait(), 2)
                token.cancel()
                with self.assertRaises(TTSError) as caught:
                    await asyncio.wait_for(task, 2)
                self.assertEqual(caught.exception.code, "CANCELLED")
            finally:
                await tts.close_async()

    async def test_cancel_after_cloud_voice_creation_preserves_id_and_skips_synthesis(self):
        token = CancellationToken()
        requests = []

        def service(request):
            action = json.loads(request.content)["input"]["action"]
            requests.append(action)
            if action == "create_voice":
                token.cancel()
                return httpx.Response(200, json={"output": {"voice_id": "known-voice"}})
            raise AssertionError("取消后不能查询或合成")

        tts = Synthesizer(
            RemoteModelConfig(provider="bailian", model="qwen-audio-3.1-tts-flash", api_key="test")
        )
        reference = AudioResult(b"\0\0" * 16000 * 5, AudioFormat())
        try:
            with enrollment_service(service):
                with self.assertRaises(TTSError) as caught:
                    await tts.synthesize_async(
                        "不应合成", reference_audio=reference, cancellation=token
                    )
            self.assertEqual(caught.exception.code, "CANCELLED")
            self.assertIn("known-voice", " ".join(caught.exception.__notes__))
            self.assertEqual(requests, ["create_voice"])
        finally:
            await tts.close_async()
