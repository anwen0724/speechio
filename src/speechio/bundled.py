"""随 wheel 分发的固定资产路径。"""

from pathlib import Path

from speechio.errors import ASRError

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
VAD_MODEL = ASSET_ROOT / "silero_vad_v6.onnx"


def default_vad_model():
    """返回内置 Silero v6 检测模型；文件缺失时给出明确错误。"""
    if not VAD_MODEL.is_file():
        raise ASRError(
            "DEPENDENCY_MISSING",
            "configure",
            f"内置检测模型缺失：{VAD_MODEL}；请重新安装 speechio",
        )
    return VAD_MODEL
