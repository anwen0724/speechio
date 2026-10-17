"""通过 ASGI WebSocket 验证音频归属和会话控制。"""

import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import soxr
from fastapi.testclient import TestClient

from realtime_asr import RecognizerConfig
from realtime_asr.protocol import encode_audio_packet
from realtime_asr.server import ServerConfig, create_app
from tests.runtime import test_recognizer
from tests.server.test_service import wait_ready
from tests.session import test_automatic


def open_socket(socket, *, mode="manual", **fields):
    socket.send_json(
        {
            "protocol_version": 1,
            "type": "open",
            "request_id": "open",
            "mode": mode,
            "realtime": False,
            **fields,
        }
    )
    return socket.receive_json()


def command(socket, ready, kind, request_id, **fields):
    socket.send_json(
        {
            "protocol_version": 1,
            "type": kind,
            "request_id": request_id,
            "session_id": ready["session_id"],
            "generation": ready["generation"],
            **fields,
        }
    )


class WebSocketTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)
        self.config = ServerConfig(
            recognizer=RecognizerConfig(model=self.path),
            realtime=False,
            idle_timeout=2,
            drain_timeout=2,
        )

    def test_manual_turn_delivers_owned_final_and_replays_finish_ack(self):
        def transcribe(audio, **kwargs):
            return iter([SimpleNamespace(text="正文", start=0, end=0.032)]), (
                SimpleNamespace(language="zh", language_probability=1)
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=transcribe):
            with TestClient(create_app(self.config)) as client:
                wait_ready(client)
                with client.websocket_connect("/api/v1/ws") as socket:
                    ready = open_socket(socket)
                    self.assertEqual(ready["type"], "ready")
                    command(socket, ready, "start", "start")
                    started = socket.receive_json()
                    unit = started["unit"]
                    metadata = {
                        "protocol_version": 1,
                        "session_id": ready["session_id"],
                        "generation": ready["generation"],
                        "unit_id": unit["unit_id"],
                        "audio_sequence": 0,
                        "frame_count": 515,
                        "format": {"sample_rate": 16000, "channels": 1, "sample_type": "int16"},
                    }
                    socket.send_bytes(encode_audio_packet(metadata, b"\x01\x00" * 515))
                    command(socket, ready, "finish", "finish", unit=unit)
                    messages = []
                    while not any(m.get("event", {}).get("kind") == "completed" for m in messages):
                        messages.append(socket.receive_json())
                    final = next(
                        m["event"] for m in messages if m.get("event", {}).get("kind") == "final"
                    )
                    self.assertEqual(final["unit"], unit)
                    self.assertEqual(final["audio_samples"], 515)
                    self.assertEqual(final["result"]["text"], "正文")
                    ack = next(m for m in messages if m.get("request_id") == "finish")
                    command(socket, ready, "finish", "finish", unit=unit)
                    replay = socket.receive_json()
                    self.assertEqual(replay["task"], ack["task"])
                    self.assertGreater(replay["wire_sequence"], ack["wire_sequence"])
                    command(socket, ready, "reset", "finish")
                    conflict = socket.receive_json()
                    self.assertEqual(conflict["code"], "REQUEST_ID_CONFLICT")
                    command(socket, ready, "ping", "after-conflict")
                    self.assertEqual(socket.receive_json()["generation"], ready["generation"])

    def test_invalid_packet_sequence_format_and_length_never_enter_core(self):
        cases = (
            ({"audio_sequence": 1}, b"\x01\x00", "AUDIO_SEQUENCE_MISMATCH"),
            (
                {"format": {"sample_rate": 8000, "channels": 1, "sample_type": "int16"}},
                b"\x01\x00",
                "AUDIO_OWNERSHIP_MISMATCH",
            ),
            ({"frame_count": 2}, b"\x01\x00", "AUDIO_FRAME_MISMATCH"),
            (None, b"\x03\x00\x00\x00{", "INVALID_MESSAGE"),
        )
        with TestClient(create_app(self.config)) as app:
            wait_ready(app)
            for overrides, audio, expected in cases:
                with self.subTest(expected=expected):
                    with app.websocket_connect("/api/v1/ws") as socket:
                        ready = open_socket(socket)
                        command(socket, ready, "start", "start")
                        unit = socket.receive_json()["unit"]
                        metadata = {
                            "protocol_version": 1,
                            "session_id": ready["session_id"],
                            "generation": 0,
                            "unit_id": unit["unit_id"],
                            "audio_sequence": 0,
                            "frame_count": 1,
                            "format": ready["format"],
                        }
                        packet = (
                            encode_audio_packet({**metadata, **overrides}, audio)
                            if overrides is not None
                            else audio
                        )
                        socket.send_bytes(packet)
                        self.assertEqual(socket.receive_json()["code"], expected)
            metrics = app.get("/api/v1/metrics").json()
            self.assertEqual(metrics["counts"].get("audio_packets", 0), 0)
            self.assertEqual(metrics["sessions_open"], 0)

    def test_reset_invalidates_old_packet_and_disconnect_preserves_other_connection(self):
        with TestClient(create_app(self.config)) as client:
            wait_ready(client)
            with client.websocket_connect("/api/v1/ws") as first:
                one = open_socket(first)
                with client.websocket_connect("/api/v1/ws") as second:
                    two = open_socket(second)
                    command(first, one, "start", "s1")
                    unit = first.receive_json()["unit"]
                    command(first, one, "reset", "r1")
                    reset = first.receive_json()
                    self.assertEqual(reset["generation"], one["generation"] + 1)
                    self.assertEqual(reset["change"]["cancelled"][0]["unit"], unit)
                    old = {
                        "protocol_version": 1,
                        "session_id": one["session_id"],
                        "generation": one["generation"],
                        "unit_id": unit["unit_id"],
                        "audio_sequence": 0,
                        "frame_count": 1,
                        "format": {"sample_rate": 16000, "channels": 1, "sample_type": "int16"},
                    }
                    first.send_bytes(encode_audio_packet(old, b"\x01\x00"))
                    self.assertEqual(first.receive_json()["code"], "AUDIO_OWNERSHIP_MISMATCH")
                    command(second, two, "ping", "ping")
                    self.assertEqual(second.receive_json()["type"], "pong")
                    command(second, two, "close", "close")
                    self.assertEqual(second.receive_json()["status"], "draining")
                    self.assertEqual(second.receive_json()["type"], "closed")
            self.assertEqual(client.get("/api/v1/metrics").json()["sessions_open"], 0)

    def test_reset_during_close_sealing_preserves_normal_close_and_new_generation(self):
        entered, release = Event(), Event()
        original = soxr.ResampleStream.resample_chunk

        def resample(stream, audio, last=False):
            if last:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("重采样边界未释放")
            return original(stream, audio, last=last)

        with patch("soxr.ResampleStream.resample_chunk", resample):
            with TestClient(create_app(self.config)) as app:
                wait_ready(app)
                with app.websocket_connect("/api/v1/ws") as socket:
                    ready = open_socket(
                        socket, format={"sample_rate": 48000, "channels": 1, "sample_type": "int16"}
                    )
                    command(socket, ready, "start", "start")
                    unit = socket.receive_json()["unit"]
                    socket.send_bytes(
                        encode_audio_packet(
                            {
                                "protocol_version": 1,
                                "session_id": ready["session_id"],
                                "generation": 0,
                                "unit_id": unit["unit_id"],
                                "audio_sequence": 0,
                                "frame_count": 1545,
                                "format": ready["format"],
                            },
                            bytes(1545 * 2),
                        )
                    )
                    command(socket, ready, "close", "close")
                    self.assertEqual(socket.receive_json()["status"], "draining")
                    try:
                        self.assertTrue(entered.wait(5))
                        command(socket, ready, "reset", "reset")
                        release.set()
                        messages = []
                        while True:
                            message = socket.receive_json()
                            messages.append(message)
                            if message["type"] in ("error", "closed"):
                                break
                    finally:
                        release.set()
                    self.assertEqual(messages[-1]["type"], "closed")
                    self.assertEqual(messages[-1]["generation"], 1)
                    self.assertTrue(any(m.get("command") == "reset" for m in messages))


class AutomaticWebSocketTests(unittest.TestCase):
    def setUp(self):
        test_automatic.AutomaticTests.setUp(self)

    def test_auto_eof_preserves_short_tail_and_repeated_eof_has_no_second_final(self):
        config = ServerConfig(
            recognizer=RecognizerConfig(model=self.path),
            vad_model=self.vad.model_path,
            realtime=False,
        )
        with TestClient(create_app(config)) as app:
            wait_ready(app)
            with app.websocket_connect("/api/v1/ws") as socket:
                ready = open_socket(socket, mode="auto")
                socket.send_bytes(
                    encode_audio_packet(
                        {
                            "protocol_version": 1,
                            "session_id": ready["session_id"],
                            "generation": 0,
                            "unit_id": None,
                            "audio_sequence": 0,
                            "frame_count": 515,
                            "format": ready["format"],
                        },
                        b"\x01\x00" * 515,
                    )
                )
                command(socket, ready, "end_input", "eof")
                command(socket, ready, "end_input", "eof2")
                command(socket, ready, "close", "close")
                messages = []
                while True:
                    message = socket.receive_json()
                    messages.append(message)
                    if message["type"] == "closed":
                        break
                finals = [m["event"] for m in messages if m.get("event", {}).get("kind") == "final"]
                self.assertEqual(len(finals), 1)
                self.assertEqual(finals[0]["boundary"]["end_sample"], 515)
                self.assertEqual(finals[0]["audio_samples"], 515)
                sequences = [m["wire_sequence"] for m in messages]
                self.assertEqual(sequences, sorted(set(sequences)))
