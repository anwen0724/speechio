# speechio

[中文](https://github.com/anwen0724/speechio/blob/main/README.zh-CN.md)

speechio is a Python library for speech recognition, speech synthesis, and voice cloning. Use local models or connect to supported cloud services with your API key to add speech capabilities to your application.

## Features

- **Speech recognition**: transcribe audio files, in-memory audio, or microphone input, with live results and utterance segmentation.
- **Speech synthesis**: generate complete audio or receive audio chunks from models that support streaming.
- **Voice cloning**: create a voice from a reference recording, save it for reuse, or provide a reference recording when synthesizing speech.
- **Model extensions**: integrate other local models or cloud services through custom adapters.
- **Example application**: try recognition, synthesis, and cloning in a browser, and inspect parameters and events in the debug console.

Available capabilities depend on the selected model. Your application handles conversation replies, business workflows, and audio playback.

## Install

Requires Python 3.11–3.13:

```bash
pip install speechio
```

The default installation includes dependencies for cloud calls, audio processing, microphone input, voice activity detection, and wake words. Cloud calls require your service credentials. Some wake-word engines also require model files or an AccessKey.

For local recognition or synthesis, install the optional dependency group for your model:

| Local model | Install command |
| --- | --- |
| Whisper | `pip install "speechio[stt-whisper]"` |
| Nemotron | `pip install "speechio[stt-nemotron]"` |
| Qwen3-TTS Base | `pip install "speechio[tts-qwen]"` |
| Kokoro | `pip install "speechio[tts-kokoro]"` |

Installing dependencies does not download recognition or synthesis models. Load an existing complete model directory, or download the model separately first. System dependencies, Python compatibility, and GPU support depend on the selected runtime. See [installation and runtime requirements (Chinese)](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#2-安装与运行环境).

## Quick start

For detailed usage instructions, read the [basic usage guide (Chinese)](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md), which covers model configuration, API parameters, speech recognition, speech synthesis, and voice cloning.

The following examples use Alibaba Cloud Bailian.

### Audio to text

Prepare a `speech.wav` recording no longer than 300 seconds. You can also use a supported format such as MP3 or M4A:

```python
import os

from speechio import RemoteModelConfig, TranscriptionOptions, transcribe_audio

config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-asr-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
result = transcribe_audio(
    "speech.wav",
    config=config,
    options=TranscriptionOptions(language="en"),
)
print(result.text)
```

This short-audio model also checks request size. Use the file transcription model for longer recordings. See [speech recognition (Chinese)](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#4-将音频转换为文字) for microphone, realtime, and long-file examples.

### Text to audio

Select a voice supported by the model, then generate and save a WAV file:

```python
import os

from speechio import RemoteModelConfig, SynthesisOptions, synthesize

config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-tts-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
audio = synthesize(
    "Hello. Welcome to today's practice.",
    config=config,
    options=SynthesisOptions(voice="Emily_v3.1", language="en"),
)
audio.save("hello.wav")
```

The output is `hello.wav`; saving does not overwrite an existing file by default. See [speech synthesis (Chinese)](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#5-将文字转换为音频) for streaming output. To create a voice from your own recording, see [voice cloning and reuse (Chinese)](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#6-克隆声音与复用音色).

For local models, use `LocalModelConfig(adapter=..., model_path=...)` to select an adapter and an existing model directory. See [model configuration (Chinese)](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#3-选择与配置模型) for download and configuration steps.

## Supported models

| Model | Runs on | Main capabilities |
| --- | --- | --- |
| Whisper | Local machine | File transcription; updates live text by repeatedly transcribing accumulated audio |
| Nemotron 3.5 ASR 0.6B | Local machine | File transcription and native streaming; the current adapter supports CPU only |
| Qwen3-TTS 12Hz Base, 0.6B / 1.7B | Local machine | Complete synthesis and voice cloning; the current adapter does not provide streaming synthesis |
| Kokoro 82M v1.1-zh | Local machine | Chinese and English synthesis, with audio output by text segment; no voice cloning |
| `qwen-audio-3.1-asr-flash` | Alibaba Cloud Bailian | Whole-audio transcription for short recordings |
| `qwen-audio-3.1-asr-flash-streaming` | Alibaba Cloud Bailian | Native streaming recognition; also accepts complete audio input |
| `qwen-audio-3.1-asr-flash-filetrans` | Alibaba Cloud Bailian | Long-file transcription from local files or URLs accessible to the service |
| `qwen-audio-3.1-tts-flash` | Alibaba Cloud Bailian | Complete and streaming synthesis, incremental text input, and voice cloning |

Local calls use existing model files. Cloud calls send input to the selected service. See the [usage guide (Chinese)](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md) for input limits, device requirements, and voice compatibility.

## Web example application

The repository includes an example application and a debug console for recording and file transcription, speech synthesis, voice cloning, and voice reuse. The console also exposes parameters and events. Both call the same core package APIs and are not installed by `pip install speechio`.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository, and enter its directory:

```bash
git clone https://github.com/anwen0724/speechio.git
cd speechio
```

On first use, copy `apps/web/config.example.toml` to `apps/web/config.toml`. If you already have a configuration file, edit it directly:

- For cloud models, fill in `apikey` in the model entry, or copy `apps/web/.env.example` to `apps/web/.env` and set your key. Credential precedence is configuration `apikey`, process environment, then `.env`; blank values are skipped.
- For local models, enter an existing model directory and enable the corresponding entry in the configuration example.

To use cloud models, start the application with:

```bash
uv run --group web python -m apps.web --enable-debug
```

For local models, add the corresponding `--extra`. For example, to use Whisper:

```bash
uv run --group web --extra stt-whisper python -m apps.web --enable-debug
```

The other local groups are `stt-nemotron`, `tts-qwen`, and `tts-kokoro`. Repeat `--extra` to select multiple groups. These options install runtime dependencies; they do not download models or guarantee a working GPU environment.

- Example application: <http://127.0.0.1:8767/>
- Debug console: <http://127.0.0.1:8767/debug>

Press `Ctrl+C` in the server terminal to stop it. Omit `--enable-debug` if you do not need the console. The application listens on localhost and runs one task at a time. Local configuration and credential files are covered by Git ignore rules.

## License

speechio is licensed under the [MIT License](https://github.com/anwen0724/speechio/blob/main/LICENSE). Copyright © 2026 anwen.

The bundled Silero detection model includes its license; see the [model asset notes](https://github.com/anwen0724/speechio/blob/main/src/speechio/assets/README.md). Other models, dependencies, and cloud services have their own licenses and terms.
