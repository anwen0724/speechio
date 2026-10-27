"""通过公开接口与本机百炼协议服务验证文字、PCM 的对应关系。"""

import json
import unittest
from dataclasses import replace
from queue import Queue
from threading import Barrier, Thread

import speechio
from speechio import AudioChunk, RemoteModelConfig, SynthesisOptions, Synthesizer
from tests.runtime.test_remote import websocket_service
from tests.tts.test_stream import StreamingEngine


def emit(ws, task_id, kind, index, *, text=None, words=None):
    output = {"type": kind, "sentence": {"index": index, "words": words or []}}
    if text is not None:
        output["original_text"] = text
    ws.send(
        json.dumps(
            {
                "header": {"task_id": task_id, "event": "result-generated"},
                "payload": {"output": output},
            }
        )
    )


def start(ws):
    run = json.loads(ws.recv())
    task_id = run["header"]["task_id"]
    ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
    return task_id, run["payload"]["parameters"]


def finish(ws, task_id):
    ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-finished"}}))


def config(url):
    return RemoteModelConfig(
        provider="bailian", model="qwen-audio-3.1-tts-flash", api_key="test", base_url=url
    )


class TextTimingTests(unittest.TestCase):
    def test_malformed_timing_protocol_is_not_reported_as_success(self):
        for failure in ("missing_metadata", "wrong_index", "bad_word_time", "unfinished_sentence"):
            with self.subTest(failure=failure):

                def service(ws):
                    task_id, _ = start(ws)
                    ws.recv()
                    ws.recv()
                    if failure == "missing_metadata":
                        ws.send(b"\x00\x00")
                    else:
                        emit(ws, task_id, "sentence-begin", 0, text="Hi.")
                        if failure == "wrong_index":
                            emit(ws, task_id, "sentence-synthesis", 1)
                        elif failure == "bad_word_time":
                            emit(
                                ws,
                                task_id,
                                "sentence-synthesis",
                                0,
                                words=[{"text": "Hi", "begin_time": 400, "end_time": 20}],
                            )
                        else:
                            finish(ws, task_id)
                    # 等 SDK 的取消消息，避免服务提前断开掩盖协议错误。
                    ws.recv()

                with websocket_service(service) as url, Synthesizer(config(url)) as synth:
                    with synth.stream(
                        "Hi.", options=SynthesisOptions(voice="test", text_timestamps="word")
                    ) as stream:
                        with self.assertRaises(speechio.TTSError) as caught:
                            list(stream.events())
                        self.assertEqual(caught.exception.code, "PROTOCOL_ERROR")

    def test_metadata_without_audio_cannot_grow_queue_without_limit(self):
        engine = StreamingEngine()
        engine.capabilities = replace(engine.capabilities, text_timestamps=("sentence",))
        engine.backend.read_event = engine.backend.read
        with Synthesizer.from_adapter(engine) as synth:
            with synth.open_stream(options=SynthesisOptions(text_timestamps="sentence")) as stream:
                for index in range(1100):
                    engine.backend.queue.put(speechio.TextTiming(index, "Hi.", 0))
                self.assertTrue(engine.backend.closed.wait(1))
                with self.assertRaises(speechio.TTSError) as caught:
                    stream.read_event(timeout=1)
                self.assertEqual(caught.exception.code, "BUFFER_EXCEEDED")

    def test_audio_read_skips_timings_and_disallows_switching_modes(self):
        def service(ws):
            task_id, _ = start(ws)
            ws.recv()
            ws.recv()
            emit(ws, task_id, "sentence-begin", 0, text="Hi.")
            emit(ws, task_id, "sentence-synthesis", 0)
            ws.send(b"\x01\x00")
            emit(ws, task_id, "sentence-end", 0, text="Hi.")
            finish(ws, task_id)

        with websocket_service(service) as url, Synthesizer(config(url)) as synth:
            with synth.stream(
                "Hi.", options=SynthesisOptions(voice="test", text_timestamps="sentence")
            ) as stream:
                self.assertEqual(stream.read(timeout=1).data, b"\x01\x00")
                with self.assertRaises(speechio.TTSError) as caught:
                    stream.read_event(timeout=0)
                self.assertEqual(caught.exception.code, "INVALID_INPUT")
                self.assertIsNone(stream.read(timeout=1))

    def test_capability_checks_reject_unsupported_model_before_prepare(self):
        engine = StreamingEngine()
        with Synthesizer.from_adapter(engine) as synth:
            with self.assertRaises(speechio.TTSError) as caught:
                synth.open_stream(options=SynthesisOptions(text_timestamps="word"))
            self.assertEqual(caught.exception.code, "CAPABILITY_UNSUPPORTED")

    def test_event_cancel_and_buffer_overflow_discard_pending_timings(self):
        for overflow in (False, True):
            with self.subTest(overflow=overflow):
                engine = StreamingEngine()
                engine.capabilities = replace(engine.capabilities, text_timestamps=("sentence",))
                engine.backend.read_event = engine.backend.read
                with Synthesizer.from_adapter(engine, max_buffer_seconds=0.001) as synth:
                    with synth.open_stream(
                        options=SynthesisOptions(text_timestamps="sentence")
                    ) as stream:
                        engine.backend.queue.put(speechio.TextTiming(0, "Hi.", 0))
                        self.assertIsInstance(stream.read_event(timeout=1), speechio.TextTiming)
                        engine.backend.queue.put(speechio.TextTiming(0, "Hi.", 0, 1))
                        if overflow:
                            engine.backend.queue.put(AudioChunk(b"\x00\x00" * 25, stream.format))
                            self.assertTrue(engine.backend.closed.wait(1))
                        else:
                            stream.cancel()
                        with self.assertRaises(speechio.TTSError) as caught:
                            stream.read_event(timeout=1)
                        self.assertEqual(
                            caught.exception.code, "BUFFER_EXCEEDED" if overflow else "CANCELLED"
                        )
                        self.assertEqual(stream.close(timeout=1).status, "ok")

    def test_concurrent_event_readers_are_rejected(self):
        engine = StreamingEngine()
        with Synthesizer.from_adapter(engine) as synth:
            with synth.open_stream(options=SynthesisOptions()) as stream:
                ready = Barrier(2)
                results = Queue()

                def receive():
                    try:
                        ready.wait(timeout=1)
                        stream.read_event(timeout=1)
                    except speechio.TTSError as error:
                        results.put(error.code)

                readers = [Thread(target=receive) for _ in range(2)]
                for reader in readers:
                    reader.start()
                try:
                    self.assertEqual(results.get(timeout=2), "INVALID_INPUT")
                finally:
                    stream.cancel()
                    for reader in readers:
                        reader.join(1)
                self.assertEqual(results.get(timeout=1), "CANCELLED")

    def test_complete_synthesis_keeps_only_final_timings_and_missing_words(self):
        requests = []

        def service(ws):
            task_id, parameters = start(ws)
            requests.append(parameters)
            ws.recv()
            ws.recv()
            emit(ws, task_id, "sentence-begin", 0, text="Hi.")
            emit(ws, task_id, "sentence-synthesis", 0)
            ws.send(b"\x01\x00" * 2000)
            emit(ws, task_id, "sentence-end", 0, text="Hi.")
            finish(ws, task_id)

        with websocket_service(service) as url, Synthesizer(config(url)) as synth:
            result = synth.synthesize(
                "Hi.",
                options=SynthesisOptions(voice="test", sample_rate=8000, text_timestamps="word"),
            )
        self.assertEqual(len(requests), 1)
        self.assertEqual(result.data, b"\x01\x00" * 2000)
        self.assertEqual(result.text_timings, (speechio.TextTiming(0, "Hi.", 0, 0.25),))
        self.assertEqual(speechio.AudioResult(b"", result.format).text_timings, ())

    def test_word_updates_arrive_during_audio_and_keep_task_relative_seconds(self):
        def word(text, begin, end):
            return {"text": text, "begin_time": begin, "end_time": end}

        def service(ws):
            task_id, parameters = start(ws)
            self.assertTrue(parameters["word_timestamp_enabled"])
            ws.recv()
            emit(ws, task_id, "sentence-begin", 0, text="Hello.")
            emit(ws, task_id, "sentence-synthesis", 0, words=[word("Hello", 50, 400)])
            ws.send(b"\x00\x00" * 4000)
            emit(ws, task_id, "sentence-end", 0, text="Hello.", words=[word("Hello", 50, 400)])
            emit(ws, task_id, "sentence-begin", 1, text="Next.", words=[word("Next", 550, 700)])
            emit(ws, task_id, "sentence-synthesis", 1)
            ws.send(b"\x00\x00" * 2000)
            emit(ws, task_id, "sentence-end", 1, text="Next.", words=[word("Next", 550, 700)])
            ws.recv()
            finish(ws, task_id)

        with websocket_service(service) as url, Synthesizer(config(url)) as synth:
            options = SynthesisOptions(voice="test", sample_rate=8000, text_timestamps="word")
            with synth.open_stream(options=options) as stream:
                stream.send_text("Hello. Next.")
                self.assertIsNone(stream.read_event(timeout=1).words)
                update = stream.read_event(timeout=1)
                self.assertEqual(update.words, (speechio.WordTiming("Hello", 0.05, 0.4),))
                self.assertIsInstance(stream.read_event(timeout=1), AudioChunk)
                stream.finish_input()
                timings = [e for e in stream.events() if isinstance(e, speechio.TextTiming)]
        self.assertEqual(timings[-1].words, (speechio.WordTiming("Next", 0.55, 0.7),))
        self.assertEqual((timings[-1].start_seconds, timings[-1].end_seconds), (0.5, 0.75))

    def test_sentence_updates_precede_audio_and_use_pcm_timeline(self):
        def service(ws):
            task_id, parameters = start(ws)
            self.assertNotIn("word_timestamp_enabled", parameters)
            for text in ("Hello ", "world."):
                message = json.loads(ws.recv())
                self.assertEqual(message["header"]["task_id"], task_id)
                self.assertEqual(message["payload"]["input"]["text"], text)
            emit(ws, task_id, "sentence-begin", 0, text="Hello world.")
            emit(ws, task_id, "sentence-synthesis", 0)
            ws.send(b"\x01\x00" * 2000)
            self.assertEqual(json.loads(ws.recv())["header"]["action"], "finish-task")
            emit(ws, task_id, "sentence-synthesis", 0)
            ws.send(b"\x02\x00" * 2000)
            emit(ws, task_id, "sentence-end", 0, text="Hello world.")
            emit(ws, task_id, "sentence-begin", 1, text="Next.")
            emit(ws, task_id, "sentence-synthesis", 1)
            ws.send(b"\x03\x00" * 2000)
            emit(ws, task_id, "sentence-end", 1, text="Next.")
            finish(ws, task_id)

        with websocket_service(service) as url, Synthesizer(config(url)) as synth:
            options = SynthesisOptions(voice="test", sample_rate=8000, text_timestamps="sentence")
            with synth.open_stream(options=options) as stream:
                stream.send_text("Hello ")
                stream.send_text("world.")
                first = stream.read_event(timeout=1)
                self.assertIsInstance(first, speechio.TextTiming)
                self.assertEqual(
                    (first.text, first.start_seconds, first.end_seconds), ("Hello world.", 0, None)
                )
                self.assertEqual(stream.read_event(timeout=1).data, b"\x01\x00" * 2000)
                stream.finish_input()
                events = list(stream.events())
        timings = [event for event in events if isinstance(event, speechio.TextTiming)]
        self.assertEqual(
            [(t.segment_index, t.start_seconds, t.end_seconds) for t in timings],
            [(0, 0, 0.5), (1, 0.5, None), (1, 0.5, 0.75)],
        )
        self.assertTrue(all(t.words is None for t in timings))
        self.assertEqual(sum(len(e.data) for e in events if isinstance(e, AudioChunk)), 8000)
