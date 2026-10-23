"""声音克隆结果能够跨实例保存、读取并用于新的合成任务。"""

import tempfile
import unittest
from pathlib import Path

from speechio import AudioFormat, AudioResult, SynthesisOptions, Synthesizer
from speechio.adapters.contracts import TTSAdapter, TTSCapabilities


class CloneAdapter(TTSAdapter):
    capabilities = TTSCapabilities(default_sample_rate=24000, synthesis=True, voice_cloning=True)

    def clone_voice(self, reference_audio, *, reference_text=None):
        from speechio.tts.voices import Voice

        return Voice(adapter="test", model="test-model", data=reference_audio.data)

    def synthesize(self, text, *, options):
        return AudioResult(options.voice.data, AudioFormat(sample_rate=24000))


class VoiceTests(unittest.TestCase):
    def test_saved_clone_survives_closed_instance_and_missing_reference_file(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.wav"
            pcm = b"\x01\x00" * 240
            AudioResult(pcm, AudioFormat(sample_rate=24000)).save(reference)
            with Synthesizer.from_adapter(CloneAdapter()) as tts:
                voice = tts.clone_voice(reference, reference_text="你好")
                saved = voice.save(Path(folder) / "speaker.voice")
            reference.unlink()
            loaded = type(voice).load(saved)
            with Synthesizer.from_adapter(CloneAdapter()) as tts:
                audio = tts.synthesize("再见", options=SynthesisOptions(voice=loaded))
            self.assertEqual(audio.data, pcm)
            self.assertEqual(loaded.model, "test-model")
