"""识别结果只包含后端信息，试次归属由验证入口补充。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecognitionResult:
    """保留原文、真实语言来源和可用分段，不伪造未知值。"""

    text: str = ""
    requested_language: str | None = None
    reported_language: str | None = None
    language_source: str = "unavailable"
    language_probability: float | None = None
    segments: list[dict] | None = None
    backend_metadata: dict = field(default_factory=dict)
