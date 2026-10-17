# realtime-asr

[中文](README.zh-CN.md)

Local speech-to-text for Python. Transcribe audio files or stream from a microphone, entirely on your machine.

## Features

- File transcription with Whisper, the default engine.
- Live microphone transcription with automatic utterance segmentation.
- Optional Nemotron engine with native streaming.
- Bundled Silero VAD, so live transcription works out of the box.

## Requirements

- Python 3.12, 64-bit.
- A working microphone for live transcription.

## Install

```bash
pip install realtime-asr
```

Live transcription works without a separate model download; the Silero VAD model ships with the package.

On Linux, install PortAudio first:

```bash
sudo apt install portaudio19-dev
pip install realtime-asr
```

### Optional: Nemotron

Nemotron adds native streaming. Install the runtime, then download a model:

```bash
pip install "realtime-asr[nemotron]"
asr-install-sherpa-models --root /models/sherpa-onnx --model 560ms
```

Available model variants are `80ms`, `560ms`, and `1120ms`. Omit `--model` to install all three.

## Quick start

Transcribe a file:

```python
from realtime_asr import transcribe_audio

result = transcribe_audio("speech.wav", model="tiny", language="en")
print(result.text)
```

- `model="tiny"` downloads the Whisper model on first use.
- Omit `language` to detect it automatically.
- `result` is a `Transcript`; `.text` holds the text.

Live microphone input:

```python
from realtime_asr import transcribe_stream

for transcript in transcribe_stream(model="tiny", on_partial=print):
    print(transcript.text)
```

- `on_partial` receives preview text that may change.
- The loop yields one final `Transcript` per utterance.

List input devices and pick one:

```python
from realtime_asr import list_input_devices, transcribe_stream

for device in list_input_devices():
    print(device.index, device.name, "default" if device.is_default else "")

for transcript in transcribe_stream(model="tiny", input_device=1):
    print(transcript.text)
```

## How live transcription works

Whisper uses snapshot mode by default. It re-transcribes the audio received so far about every 0.2 seconds, so previews can change. When an utterance ends, the full audio is transcribed once more to produce the final text.

Nemotron uses native streaming. Text grows as audio arrives, with lower latency.

```python
from realtime_asr import StreamTranscriber

with StreamTranscriber(
    model="/models/sherpa-onnx/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
    engine="nemotron",
    language="en",
) as transcriber:
    for transcript in transcriber:
        print(transcript.text)
```

`StreamTranscriber` also accepts `on_partial` and `input_device`. Call `end_input()` to stop listening and deliver the remaining results.

## Models

- **Whisper** is downloaded from Hugging Face on first use. Set `download_root` to change the cache directory, or `local_files_only=True` to forbid network access.
- A prepared model directory works offline. It must contain `model-manifest.json`, which the library verifies before loading.
- **Silero VAD** is bundled with the package under the MIT license. No download is needed.
- **Nemotron** weights are installed separately with `asr-install-sherpa-models`, then passed as the model path with `engine="nemotron"`.

## API overview

| Entry | Purpose |
| --- | --- |
| `transcribe_audio()` | Transcribe a file, bytes, NumPy array, or iterable of chunks |
| `transcribe_stream()` | Yield final transcripts from the microphone |
| `Transcriber` | Reuse one model across multiple transcriptions |
| `StreamTranscriber` | Control listening, partial results, and shutdown |
| `Session` | Manual utterance boundaries (advanced) |
| `LiveSession` | Automatic utterance segmentation (advanced) |
| `ModelConfig` | Select engine, model, device, and compute type |
| `Transcript` | Transcription result with `.text` and `.words` |
