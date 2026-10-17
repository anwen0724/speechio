"""固定权重发布包的不可变描述；安装器据此校验压缩包和展开文件。"""

from dataclasses import dataclass

from realtime_asr.core.runtime.models.pinned import NEMOTRON_MODELS


@dataclass(frozen=True)
class ModelFileManifest:
    """一个展开文件的大小与 SHA-256。"""

    filename: str
    size_bytes: int
    sha256: str
    required: bool = True


@dataclass(frozen=True)
class ModelManifest:
    """一个固定发布包及其展开结果；压缩包哈希是发布边界。"""

    model_id: str
    archive_url: str
    archive_filename: str
    archive_size_bytes: int
    archive_sha256: str
    expected_files: tuple[str, ...]
    license_name: str
    license_url: str
    file_metadata: tuple[ModelFileManifest, ...] = ()


def _nemotron_manifest(variant):
    model = NEMOTRON_MODELS[variant]
    model_id = model["id"]
    archive_size, archive_sha256 = model["archive"]
    files = tuple(
        ModelFileManifest(name, size, digest) for name, (size, digest) in model["files"].items()
    )
    return ModelManifest(
        model_id=model_id,
        archive_url=(
            f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_id}.tar.bz2"
        ),
        archive_filename=f"{model_id}.tar.bz2",
        archive_size_bytes=archive_size,
        archive_sha256=archive_sha256,
        expected_files=tuple(record.filename for record in files),
        license_name="NVIDIA Open Model Data Warehouse License Agreement v1.1",
        license_url="",
        file_metadata=files,
    )


NEMOTRON_560MS = _nemotron_manifest("560ms")
NEMOTRON_1120MS = _nemotron_manifest("1120ms")
NEMOTRON_80MS = _nemotron_manifest("80ms")


def nemotron_manifests():
    """按变体返回固定清单；调用方不用自己拼模型身份。"""
    return {
        "560ms": NEMOTRON_560MS,
        "1120ms": NEMOTRON_1120MS,
        "80ms": NEMOTRON_80MS,
    }
