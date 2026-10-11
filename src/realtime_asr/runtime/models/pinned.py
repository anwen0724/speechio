"""P0 固定模型包身份；校验值来自有效 Spec。"""

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
