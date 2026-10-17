"""已接入的固定模型身份；变体的包与文件校验值分开声明。"""

from realtime_asr.core.errors import ASRError

WHISPER_MODEL = {"id": "tiny", "source": "Systran/faster-whisper-tiny"}

DETECTION_MODELS = {
    "silero-vad": {
        "id": "silero-v6",
        "revision": "faster-whisper-v1.2.1",
        "source": "https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1/faster_whisper/assets",
        "files": {
            "silero-v6.onnx": (
                1245151,
                "4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2",
            )
        },
        "urls": {
            "silero-v6.onnx": "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/v1.2.1/faster_whisper/assets/silero_vad_v6.onnx"
        },
    },
    "openwakeword": {
        "id": "openwakeword-alexa-v0.5.1",
        "revision": "v0.5.1",
        "source": "https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1",
        "files": {
            "alexa_v0.1.onnx": (
                854246,
                "6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e",
            ),
            "embedding_model.onnx": (
                1326578,
                "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
            ),
            "melspectrogram.onnx": (
                1087958,
                "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
            ),
        },
    },
}
DETECTION_MODELS["openwakeword"]["urls"] = {
    name: f"https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{name}"
    for name in DETECTION_MODELS["openwakeword"]["files"]
}

NEMOTRON_ID = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
NEMOTRON_URL = (
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{NEMOTRON_ID}.tar.bz2"
)
NEMOTRON_ARCHIVE = (
    475271763,
    "c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a",
)
NEMOTRON_FILES = {
    "encoder.int8.onnx": (
        657601403,
        "012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68",
    ),
    "decoder.int8.onnx": (
        14978075,
        "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0",
    ),
    "joiner.int8.onnx": (
        9504438,
        "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce",
    ),
    "tokens.txt": (131440, "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a"),
}

NEMOTRON_MODELS = {
    "560ms": {"id": NEMOTRON_ID, "archive": NEMOTRON_ARCHIVE, "files": NEMOTRON_FILES},
    "1120ms": {
        "id": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11",
        "archive": (475276334, "adbdd5e9fef87300c37cebfcfc4f1ebe56845c860c8a760af0a1dd65ce9beed3"),
        "files": {
            **NEMOTRON_FILES,
            "encoder.int8.onnx": (
                657601521,
                "2fff2166acaa535bd969fb223c1f0783d71029f143cb298bc54c2afe85abf772",
            ),
        },
    },
    "80ms": {
        "id": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11",
        "archive": (475274007, "fb170128c496db33a1fb9f5f9f823257f42f911224ee218bb429f3c2eaf90a8d"),
        "files": {
            **NEMOTRON_FILES,
            "encoder.int8.onnx": (
                657601516,
                "411e1222810f4a4cf0a3704c7609597a12def5b4ad2c7347a24ccd40d895484d",
            ),
        },
    },
}


def nemotron_model(variant):
    """显式选择固定发布变体，不从目录名称猜测模型身份。"""
    if not isinstance(variant, str) or variant not in NEMOTRON_MODELS:
        raise ASRError(
            "INVALID_CONFIGURATION", "prepare_model", "Nemotron 变体应为 560ms、1120ms 或 80ms"
        )
    return NEMOTRON_MODELS[variant]
