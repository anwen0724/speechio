# speechio

[中文](https://github.com/anwen0724/speechio/blob/main/README.zh-CN.md)

SpeechIO is a Python speech-to-text library. Transcribe audio files or connect a microphone to display text as you speak and receive a final result for each utterance.

Use local models or cloud services: local recognition reads an existing model directory and processes audio on your machine; cloud recognition sends audio to the selected service using your API key.

## Features

- Transcribe audio files, PCM audio, and sequences of audio chunks.
- Capture microphone audio, detect speech and pauses, and return results by utterance.
- Receive live text through a callback while speaking.
- Use manual segmentation, wake words, and custom recognition engines.
- Try the Web example application and debug console to inspect results and API usage.

Currently supported:

| Model | Runs on | File transcription | Text while speaking |
| --- | --- | --- | --- |
| Whisper | Your machine | Supported | Periodically transcribes accumulated audio to update the preview |
| Nemotron 3.5 ASR 0.6B | Your machine | Supported | Native streaming |
| `qwen-audio-3.1-asr-flash` | Alibaba Cloud Bailian | Short audio | Not supported; can transcribe after each utterance ends |
| `qwen-audio-3.1-asr-flash-streaming` | Alibaba Cloud Bailian | Short audio | Native streaming |

The initial Bailian integration supports short audio and realtime recognition, not long-recording file jobs. Whole-audio calls accept up to 300 seconds; HTTP calls also have an encoded-size limit. Inputs exceeding these limits are rejected.

## Install

Requires 64-bit Python 3.11–3.13.

```bash
python -m pip install speechio
```

Install optional dependencies for the features you need. Each command also installs SpeechIO:

| Use case | Command |
| --- | --- |
| Microphone input | `python -m pip install "speechio[microphone]"` |
| Bailian cloud recognition | `python -m pip install "speechio[remote]"` |
| Bailian with microphone input | `python -m pip install "speechio[remote,microphone]"` |
| Local Nemotron | `python -m pip install "speechio[nemotron]"` |
| OpenWakeWord wake words | `python -m pip install "speechio[wake-openwakeword]"` |
| Porcupine wake words | `python -m pip install "speechio[wake-porcupine]"` |

These commands install the library and its dependencies. They do not download speech recognition models. For local recognition, prepare a model directory as shown below.

On Linux and macOS, prepare PortAudio before installing the microphone dependency. See the [PyAudio installation instructions](https://pypi.org/project/PyAudio/). Wake words also require the corresponding model files; Porcupine requires an AccessKey.

## Quick start

This example uses Whisper base on the CPU to transcribe a local audio file.

Run the download code separately to save the model to a directory you choose. Skip this step if you already have a model directory compatible with faster-whisper.

```python
from speechio.models import download_model

download_model(
    engine="whisper",
    model_id="base",
    destination="models/whisper-base",
)
```

Save your audio as `speech.wav`, then run:

```python
from speechio import LocalModelConfig, transcribe_audio

config = LocalModelConfig(
    engine="whisper",
    model_path="models/whisper-base",
    device="cpu",
    compute_type="int8",
)

result = transcribe_audio("speech.wav", config=config, language="en")
print(result.text)
```

Relative paths are resolved from the directory where you run Python. Recognition only loads existing files from `model_path`; it does not download models automatically. Use `language="en"` for English or `"zh"` for Chinese. Omit it to leave the language unspecified.

## Usage examples

### Cloud transcription

Install the `remote` extra and set the `DASHSCOPE_API_KEY` environment variable to your Bailian API key. The code below reads that variable; the SDK does not automatically load `.env` files.

```python
import os

from speechio import RemoteModelConfig, transcribe_audio

config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-asr-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)

result = transcribe_audio("speech.wav", config=config, language="en")
print(result.text)
```

You can also pass a key obtained by your application directly through `api_key`. Calls send audio to Bailian and are billed according to your account and selected model.

### Live microphone transcription

Install the `microphone` extra and prepare the model directory used in the quick start. Speak into the default microphone and press `Ctrl+C` to exit.

```python
from speechio import LocalModelConfig, StreamTranscriber

config = LocalModelConfig(
    engine="whisper",
    model_path="models/whisper-base",
    device="cpu",
    compute_type="int8",
)

try:
    with StreamTranscriber(
        config=config,
        language="en",
        on_partial=lambda text: print("Live:", text),
    ) as stream:
        for result in stream:
            print("Final:", result.text)
except KeyboardInterrupt:
    pass
```

Text received by `on_partial` may change. The loop yields a final result for each utterance. The bundled Silero model detects speech and pauses; no separate detection-model download is needed. To select a microphone, use `list_input_devices()` to find its index and pass it as `input_device`.

Whisper updates previews while you speak and transcribes the complete audio when each utterance ends. Nemotron and Bailian streaming models use the streaming task's final result by default. Set `completion_mode="final"` to request an additional whole-audio transcription. Cloud retranscription adds requests and may incur additional charges.

For cloud microphone transcription, install `remote,microphone`, replace `config` with a `RemoteModelConfig`, and select `qwen-audio-3.1-asr-flash-streaming`. To stop recording and receive remaining results, call `stream.end_input()` and continue consuming the loop.

See the repository's [examples](https://github.com/anwen0724/speechio/tree/main/examples) for more use cases.

## Web example application

The repository includes two pages: an example application for recording and file transcription, and a debug console for adjusting parameters, inspecting events, and viewing API examples. Both use the same core SDK. The Web application is not included in the installed SDK package.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repository. If you already have a checkout, use its root directory:

```bash
git clone https://github.com/anwen0724/speechio.git
cd speechio
```

On first use, copy `apps/web/config.example.toml` to `apps/web/config.toml`. If you already have a configuration file, edit it directly.

Set local model directories or fill in `apikey` for cloud models. Alternatively, copy `apps/web/.env.example` to `apps/web/.env` and fill in your key. The Web application checks credentials in this order: `apikey` in the configuration file, process environment variables, then `apps/web/.env`. Blank values are skipped.

```bash
uv run --group web python -m apps.web --enable-debug
```

- Example application: <http://127.0.0.1:8767/>
- Debug console: <http://127.0.0.1:8767/debug>

Press `Ctrl+C` in the server terminal to stop it. Omit `--enable-debug` if you do not need the debug console. The application listens only on localhost and runs one task at a time. Local configuration and credential files are covered by Git ignore rules.

## License

SpeechIO is licensed under the [MIT License](https://github.com/anwen0724/speechio/blob/main/LICENSE). Copyright © 2026 anwen.

The bundled Silero model includes its license; see the [model asset notes](https://github.com/anwen0724/speechio/blob/main/src/speechio/assets/README.md). Other models, dependencies, and cloud services are subject to their providers' licenses and terms.
