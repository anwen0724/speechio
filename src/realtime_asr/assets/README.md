# 内置资产

## silero_vad_v6.onnx

- 用途：`transcribe_stream()` 和 `StreamTranscriber` 的默认语音检测模型。
- 来源：`https://raw.githubusercontent.com/SYSTRAN/faster-whisper/v1.2.1/faster_whisper/assets/silero_vad_v6.onnx`
- 上游：Silero VAD，`https://github.com/snakers4/silero-vad`
- 大小：1245151 字节
- SHA-256：`4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2`
- 许可：MIT，见 `silero_vad_v6.LICENSE`

调用方仍可通过 `vad_model=` 传入自己的模型文件，内置文件只作为默认值。
