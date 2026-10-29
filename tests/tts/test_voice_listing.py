"""通过公开入口验证音色发现；不加载模型，不调用云端真实服务。"""

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from websockets.sync.client import reconnect

from speechio import (
    AudioFormat,
    AudioResult,
    LocalModelConfig,
    RemoteModelConfig,
    SynthesisOptions,
    Synthesizer,
    Transcriber,
    TTSCapabilities,
    TTSError,
    VoiceInfo,
)
from tests.runtime.test_remote import websocket_service
from tests.tts.test_bailian_voice import enrollment_service


class VoiceListingTests(unittest.TestCase):
    def test_bailian_listed_and_new_voice_ids_reach_synthesis_unchanged(self):
        received = []

        def service(ws):
            run = json.loads(ws.recv())
            task_id = run["header"]["task_id"]
            received.append(run["payload"]["parameters"]["voice"])
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
            while json.loads(ws.recv())["header"]["action"] != "finish-task":
                pass
            ws.send(b"\1\0" * 24)
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-finished"}}))

        with websocket_service(service) as url:

            class LocalConnection(reconnect):
                def __init__(self, uri, **kwargs):
                    super().__init__(url.replace("http://", "ws://"), **kwargs)

            with (
                patch("websockets.sync.client.reconnect", LocalConnection),
                Synthesizer(
                    RemoteModelConfig(
                        provider="bailian", model="qwen-audio-3.1-tts-flash", api_key="test-key"
                    )
                ) as tts,
            ):
                selected = next(v.id for v in tts.list_voices() if v.id == "yuxiaoyun_v3.1")
                for voice in (selected, "new-voice-not-in-catalog"):
                    result = tts.synthesize("你好", options=SynthesisOptions(voice=voice))
                    self.assertEqual(result.data, b"\1\0" * 24)
        self.assertEqual(received, ["yuxiaoyun_v3.1", "new-voice-not-in-catalog"])

    def test_custom_adapter_can_use_listed_id_and_old_adapters_still_synthesize(self):
        class OldAdapter:
            capabilities = TTSCapabilities(default_sample_rate=24000, synthesis=True)

            def prepare(self):
                pass

            def close(self):
                pass

            def synthesize(self, text, *, options):
                return AudioResult(b"\1\0", AudioFormat(sample_rate=24000))

        with Synthesizer.from_adapter(OldAdapter()) as tts:
            self.assertIsNone(tts.capabilities.languages)
            self.assertIsNone(tts.capabilities.voice_required)
            with self.assertRaises(TTSError) as caught:
                tts.list_voices()
            self.assertEqual(caught.exception.code, "CAPABILITY_UNSUPPORTED")
            self.assertEqual(tts.synthesize("你好").data, b"\1\0")

        class ListingAdapter(OldAdapter):
            def list_voices(self, *, include_account=False):
                return (VoiceInfo(id="voice-one", name="音色一"),)

            def synthesize(self, text, *, options):
                if options.voice != "voice-one":
                    raise TTSError("VOICE_NOT_FOUND", "synthesize", "音色不存在")
                return super().synthesize(text, options=options)

        tts = Synthesizer.from_adapter(ListingAdapter())
        try:
            voice = tts.list_voices()[0]
            self.assertEqual(
                tts.synthesize("你好", options=SynthesisOptions(voice=voice.id)).data, b"\1\0"
            )
        finally:
            tts.close()
        with self.assertRaises(TTSError) as caught:
            tts.list_voices()
        self.assertEqual(caught.exception.code, "RUNTIME_CLOSED")

    def test_empty_account_is_valid_but_failed_query_is_not_an_empty_result(self):
        tts = Synthesizer(
            RemoteModelConfig(
                provider="bailian", model="qwen-audio-3.1-tts-flash", api_key="test-key"
            )
        )
        try:
            presets = tts.list_voices()
            with enrollment_service(
                lambda request: httpx.Response(200, json={"output": {"voice_list": []}})
            ):
                self.assertEqual(tts.list_voices(include_account=True), presets)
            for status, output, code in (
                (401, {}, "REMOTE_AUTH"),
                (500, {}, "REMOTE_SERVICE"),
                (200, {}, "PROTOCOL_ERROR"),
                (200, {"voice_list": [{"voice_id": "incomplete"}]}, "PROTOCOL_ERROR"),
            ):
                with (
                    self.subTest(status=status, output=output),
                    enrollment_service(
                        lambda request: httpx.Response(status, json={"output": output})
                    ),
                    self.assertRaises(TTSError) as caught,
                ):
                    tts.list_voices(include_account=True)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.operation, "list_voices")
        finally:
            tts.close()

    def test_unknown_region_does_not_return_an_unverified_preset_list(self):
        tts = Synthesizer(
            RemoteModelConfig(
                provider="bailian",
                model="qwen-audio-3.1-tts-flash",
                base_url="https://proxy.example.com",
            )
        )
        try:
            with self.assertRaises(TTSError) as caught:
                tts.list_voices()
            self.assertEqual(caught.exception.code, "CAPABILITY_UNSUPPORTED")
        finally:
            tts.close()

    def test_bailian_declares_and_enforces_speed_and_reference_requirements(self):
        tts = Synthesizer(
            RemoteModelConfig(
                provider="bailian", model="qwen-audio-3.1-tts-flash", api_key="test-key"
            )
        )
        try:
            caps = tts.capabilities
            self.assertEqual(caps.speed_range, (0.5, 2.0))
            requirements = caps.cloning_requirements
            self.assertEqual(requirements.min_audio_seconds, 5)
            self.assertEqual(requirements.max_audio_seconds, 60)
            self.assertEqual(requirements.min_sample_rate, 16000)
            self.assertEqual(requirements.reference_text, "unsupported")
            with (
                patch("httpx.Client", side_effect=AssertionError("无效输入不能联网")),
                patch(
                    "websockets.sync.client.reconnect",
                    side_effect=AssertionError("无效输入不能合成"),
                ),
            ):
                with self.assertRaises(TTSError) as caught:
                    tts.synthesize(
                        "你好", options=SynthesisOptions(voice="yuxiaoyun_v3.1", speed=2.1)
                    )
                self.assertEqual(caught.exception.code, "INVALID_INPUT")
                for duration, rate in ((4, 16000), (61, 16000), (5, 8000)):
                    with (
                        self.subTest(duration=duration, rate=rate),
                        self.assertRaises(TTSError) as caught,
                    ):
                        tts.clone_voice(
                            AudioResult(b"\0\0" * rate * duration, AudioFormat(sample_rate=rate))
                        )
                    self.assertEqual(caught.exception.code, "INVALID_INPUT")
        finally:
            tts.close()

    def test_stt_devices_describe_adapter_support_without_model_loading(self):
        for adapter, expected in (("whisper", ("cpu", "gpu")), ("nemotron", ("cpu",))):
            transcriber = Transcriber(LocalModelConfig(adapter=adapter, model_path="missing-model"))
            try:
                self.assertEqual(transcriber.capabilities.devices, expected)
                self.assertIsNone(transcriber.capabilities.languages)
            finally:
                transcriber.close()
        transcriber = Transcriber(
            RemoteModelConfig(
                provider="bailian", model="qwen-audio-3.1-asr-flash", api_key="test-key"
            )
        )
        try:
            self.assertEqual(transcriber.capabilities.devices, ())
            self.assertIn("zh", transcriber.capabilities.languages)
            self.assertIn("en", transcriber.capabilities.languages)
        finally:
            transcriber.close()

    def test_qwen_describes_selected_model_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text(
                json.dumps(
                    {
                        "talker_config": {
                            "codec_language_id": {"chinese": 1, "english": 2, "japanese": 3}
                        }
                    }
                )
            )
            tts = Synthesizer(LocalModelConfig(adapter="qwen_tts", model_path=folder))
            try:
                self.assertEqual(tts.list_voices(include_account=True), ())
                caps = tts.capabilities
                self.assertEqual(caps.languages, ("zh", "en", "japanese"))
                self.assertEqual(caps.devices, ("cpu", "gpu"))
                self.assertTrue(caps.voice_required)
                self.assertEqual(caps.cloning_requirements.reference_text, "optional")
                self.assertIsNone(caps.cloning_requirements.max_audio_seconds)
            finally:
                tts.close()

    def test_account_query_reads_all_pages_and_filters_model_and_status(self):
        pages = []
        model = "qwen-audio-3.1-tts-flash"

        def service(request):
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertEqual(request.url.host, "workspace.cn-beijing.maas.aliyuncs.com")
            payload = json.loads(request.content)["input"]
            if payload["action"] == "list_voice":
                page = payload["page_index"]
                pages.append(page)
                if page == 0:
                    items = [{"voice_id": f"pending-{i}", "status": "DEPLOYING"} for i in range(8)]
                    items += [
                        {"voice_id": "other-model", "status": "OK"},
                        {"voice_id": "became-unavailable", "status": "OK"},
                    ]
                else:
                    items = [{"voice_id": "my-voice", "status": "OK"}]
                return httpx.Response(200, json={"output": {"voice_list": items}})
            self.assertEqual(payload["action"], "query_voice")
            return httpx.Response(
                200,
                json={
                    "output": {
                        "target_model": "qwen-audio-3.1-tts-plus"
                        if payload["voice_id"] == "other-model"
                        else model,
                        "status": "UNDEPLOYED"
                        if payload["voice_id"] == "became-unavailable"
                        else "OK",
                    }
                },
            )

        tts = Synthesizer(
            RemoteModelConfig(
                provider="bailian",
                model=model,
                api_key="test-key",
                base_url="https://workspace.cn-beijing.maas.aliyuncs.com",
            )
        )
        try:
            presets = tts.list_voices()
            with enrollment_service(service):
                voices = tts.list_voices(include_account=True)
            self.assertEqual(pages, [0, 1])
            self.assertEqual(voices[:-1], presets)
            self.assertEqual(voices[-1].id, "my-voice")
            self.assertEqual(voices[-1].name, "my-voice")
            self.assertIsNone(voices[-1].languages)
            self.assertIsNone(voices[-1].text_timestamps)
        finally:
            tts.close()

    def test_bailian_presets_are_offline_and_filtered_by_region(self):
        results = []
        for base_url in (None, "https://dashscope-intl.aliyuncs.com"):
            tts = Synthesizer(
                RemoteModelConfig(
                    provider="bailian", model="qwen-audio-3.1-tts-flash", base_url=base_url
                )
            )
            try:
                with patch("httpx.Client", side_effect=AssertionError("不应联网")):
                    results.append({v.id: v for v in tts.list_voices()})
                self.assertEqual(tts.capabilities.devices, ())
            finally:
                tts.close()
        beijing, singapore = results
        self.assertIn("yuxiaoyun_v3.1", beijing)
        self.assertNotIn("yuxiaoyun_v3.1", singapore)
        self.assertIn("Emily_v3.1", singapore)
        self.assertEqual(beijing["loongbella_v3.1"].languages, ("zh", "en"))
        self.assertEqual(singapore["Emily_v3.1"].languages, ("en",))
        self.assertEqual(beijing["yuxiaoyun_v3.1"].text_timestamps, ("sentence", "word"))

    def test_kokoro_lists_only_voice_files_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            voices = folder / "voices"
            voices.mkdir()
            for name in ("zf_001.pt", "af_maple.pt", "bf_vale.pt", "notes.txt"):
                (voices / name).write_bytes(b"not a tensor")
            (voices / "directory.pt").mkdir()
            tts = Synthesizer(LocalModelConfig(adapter="kokoro", model_path=folder))
            try:
                original_import = builtins.__import__

                def import_without_models(name, *args, **kwargs):
                    if name.split(".")[0] in ("torch", "kokoro"):
                        raise AssertionError("查询不能导入模型运行库")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=import_without_models):
                    result = tts.list_voices()
                    self.assertEqual(result, tts.list_voices(include_account=True))
                self.assertEqual([voice.id for voice in result], ["af_maple", "bf_vale", "zf_001"])
                self.assertEqual(
                    [voice.languages for voice in result], [("en",), ("en-gb",), ("zh",)]
                )
                self.assertTrue(all(not voice.instruction_control for voice in result))
                self.assertTrue(all(voice.text_timestamps == () for voice in result))
                self.assertEqual(tts.capabilities.devices, ("cpu", "gpu"))
                self.assertTrue(tts.capabilities.voice_required)
                self.assertIsNone(tts.capabilities.default_voice)
            finally:
                tts.close()
