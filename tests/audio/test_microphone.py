"""通过外部 PortAudio 边界验证设备选择与生命周期。"""

import unittest
from threading import Event, Thread
from unittest.mock import patch

from realtime_asr import ASRError
from realtime_asr.core.audio.microphone import MicrophoneConfig, MicrophoneInput, list_input_devices


class PortAudio:
    """外部设备运行库替身，只提供查询和句柄行为。"""

    def __init__(self, *, fail_open=False):
        self.fail_open = fail_open
        self.terminated = False

    def get_default_input_device_info(self):
        return {"index": 0}

    def get_device_info_by_index(self, index):
        if index not in (0, 1):
            raise OSError("device not found")
        return {
            "name": f"device-{index}",
            "hostApi": 0,
            "maxInputChannels": 2,
            "defaultSampleRate": 48000,
        }

    def get_host_api_info_by_index(self, index):
        return {"name": "test-host"}

    def get_device_count(self):
        return 2

    def is_format_supported(self, rate, **kwargs):
        if rate != 48000:
            raise ValueError("unsupported")
        return True

    def open(self, **kwargs):
        if self.fail_open:
            raise OSError("busy device")
        self.opened = kwargs
        return self.stream

    def terminate(self):
        self.terminated = True


class MicrophoneTests(unittest.TestCase):
    def test_explicit_device_reports_actual_fallback_rate_and_releases_resources(self):
        class Stream:
            def stop_stream(self):
                self.stopped = True

            def close(self):
                self.closed = True

        library = PortAudio()
        library.stream = Stream()
        with patch("pyaudio.PyAudio", return_value=library):
            microphone = MicrophoneInput(MicrophoneConfig(device_index=1))
            format = microphone.prepare()
            self.assertEqual((format.sample_rate, format.channels), (48000, 1))
            self.assertEqual(library.opened["input_device_index"], 1)
            self.assertEqual(microphone.prepare(), format)
            first = microphone.close()
            self.assertEqual(microphone.close(), first)
        self.assertTrue(library.terminated and library.stream.closed)

    def test_invalid_explicit_device_is_not_replaced_by_default(self):
        library = PortAudio()
        with patch("pyaudio.PyAudio", return_value=library):
            with self.assertRaises(ASRError) as caught:
                MicrophoneInput(MicrophoneConfig(device_index=9)).prepare()
        self.assertEqual(caught.exception.code, "INPUT_DEVICE_INVALID")
        self.assertTrue(library.terminated)

    def test_enumeration_terminates_its_interface_without_opening_streams(self):
        library = PortAudio()
        with patch("pyaudio.PyAudio", return_value=library):
            devices = list_input_devices()
        self.assertEqual([d.index for d in devices], [0, 1])
        self.assertEqual(devices[0].supported_sample_rates, (48000,))
        self.assertTrue(library.terminated)

    def test_full_queue_keeps_accepted_blocks_then_reports_failure(self):
        cleaned = Event()
        chunks = iter((b"\x01\x00" * 2, b"\x02\x00" * 2, b"\x03\x00" * 2))

        class Stream:
            def start_stream(self):
                pass

            def read(self, frames, **kwargs):
                return next(chunks)

            def stop_stream(self):
                cleaned.set()

            def close(self):
                pass

        library = PortAudio()
        library.stream = Stream()
        with patch("pyaudio.PyAudio", return_value=library):
            microphone = MicrophoneInput(MicrophoneConfig(chunk_frames=2, queue_seconds=0.0001))
            microphone.start()
            try:
                self.assertTrue(cleaned.wait(2))
                self.assertEqual(microphone.received_frames, 4)
                self.assertEqual(microphone.read_audio(), b"\x01\x00" * 2)
                self.assertEqual(microphone.read_audio(), b"\x02\x00" * 2)
                with self.assertRaises(ASRError) as caught:
                    microphone.read_audio()
                self.assertEqual(caught.exception.code, "CAPTURE_CAPACITY_EXCEEDED")
            finally:
                microphone.close()

    def test_cleanup_error_does_not_skip_stream_close_or_terminate(self):
        class Stream:
            def stop_stream(self):
                raise OSError("stop failed")

            def close(self):
                self.closed = True

        library = PortAudio()
        library.stream = Stream()
        with patch("pyaudio.PyAudio", return_value=library):
            microphone = MicrophoneInput()
            microphone.prepare()
            result = microphone.close()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.errors[0].operation, "stop_stream")
        self.assertTrue(library.terminated and library.stream.closed)

    def test_stop_preserves_block_already_being_read_and_releases_after_read_returns(self):
        reading, release = Event(), Event()

        class Stream:
            def start_stream(self):
                pass

            def read(self, frames, **kwargs):
                reading.set()
                if not release.wait(3):
                    raise OSError("读取未释放")
                return b"\x07\x00" * 3

            def stop_stream(self):
                pass

            def close(self):
                pass

        library = PortAudio()
        library.stream = Stream()
        with patch("pyaudio.PyAudio", return_value=library):
            microphone = MicrophoneInput()
            microphone.start()
            try:
                self.assertTrue(reading.wait(2))
                self.assertEqual(microphone.close(timeout=0).status, "pending")
                self.assertFalse(library.terminated)
            finally:
                release.set()
                self.assertEqual(microphone.close(timeout=2).status, "ok")
            self.assertEqual(microphone.read_audio(), b"\x07\x00" * 3)
            self.assertIsNone(microphone.read_audio())
            self.assertEqual(microphone.received_frames, 3)

    def test_device_overflow_is_reported_instead_of_silence(self):
        import pyaudio

        class Stream:
            def start_stream(self):
                pass

            def read(self, frames, **kwargs):
                raise OSError(pyaudio.paInputOverflowed, "overflow")

            def stop_stream(self):
                pass

            def close(self):
                pass

        library = PortAudio()
        library.stream = Stream()
        with patch("pyaudio.PyAudio", return_value=library):
            microphone = MicrophoneInput()
            microphone.start()
            try:
                with self.assertRaises(ASRError) as caught:
                    microphone.read_audio(timeout=1)
                self.assertEqual(caught.exception.code, "MICROPHONE_OVERFLOW")
            finally:
                microphone.close()

    def test_failed_open_releases_acquired_portaudio_instance(self):
        library = PortAudio(fail_open=True)
        with patch("pyaudio.PyAudio", return_value=library):
            microphone = MicrophoneInput(MicrophoneConfig(device_index=1))
            with self.assertRaises(ASRError) as caught:
                microphone.prepare()
        self.assertEqual(caught.exception.code, "MICROPHONE_OPEN_FAILED")
        self.assertTrue(library.terminated, "打开失败后 PortAudio 仍未释放")

    def test_close_timeout_remains_bounded_while_driver_cleanup_is_blocked(self):
        cleaning, release, returned = Event(), Event(), Event()
        outcomes = []

        class Stream:
            def start_stream(self):
                pass

            def read(self, frames, **kwargs):
                return b""

            def stop_stream(self):
                cleaning.set()
                release.wait(5)

            def close(self):
                pass

        library = PortAudio()
        library.stream = Stream()
        with patch("pyaudio.PyAudio", return_value=library):
            microphone = MicrophoneInput()
            microphone.start()

            def close():
                outcomes.append(microphone.close(timeout=0))
                returned.set()

            caller = Thread(target=close)
            try:
                self.assertTrue(cleaning.wait(2))
                caller.start()
                self.assertTrue(returned.wait(1), "关闭等待被驱动清理锁阻塞")
                self.assertEqual(outcomes[0].status, "pending")
            finally:
                release.set()
                if caller.ident is not None:
                    caller.join(2)
                microphone.close(timeout=2)
