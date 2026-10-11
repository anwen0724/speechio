"""将固定 M01—M09 展开为明确的样本和试次，不按效果选样。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixTask:
    """一项声明保存输入路径与试次角色；运行状态由实际记录决定。"""

    matrix_item: str
    backend: str
    sample_id: str
    input_path: str
    input_mode: str
    evaluation_role: str = "diagnostic"
    trial_index: int = 1
    chunk_samples: int | None = None
    auto_language: bool = False
    scenario: str | None = None


def declare_matrix(corpus):
    """预先声明每个应执行项，代表样本只读取固定清单。"""
    categories = {
        name: [item["sample_id"] for item in corpus["samples"] if name in item["categories"]]
        for name in ("C01", "C02", "C04", "C07", "C08", "C09", "T01", "T02")
    }
    bases = [
        item["sample_id"]
        for item in corpus["samples"]
        if not any(c.startswith("T") for c in item["categories"])
    ]
    chinese = corpus["representative_sample_id"]["C01"]
    english = corpus["representative_sample_id"]["C07"]
    snapshots = [chinese, *categories["C02"], *categories["C04"], *categories["T01"]]
    paced = [
        chinese,
        *categories["C02"],
        *categories["C04"],
        english,
        *categories["C08"],
        *categories["C09"],
        *categories["T01"],
    ]
    tasks = []
    for matrix, backend, path, mode in (
        ("M01", "whisper", "whole", "offline"),
        ("M03", "nemotron", "stream", "fast"),
    ):
        for sample in bases:
            tasks.append(
                MatrixTask(
                    matrix,
                    backend,
                    sample,
                    path,
                    mode,
                    "primary",
                    chunk_samples=512 if path == "stream" else None,
                )
            )
        for sample in categories["C01"] + categories["C07"]:
            tasks.append(
                MatrixTask(
                    matrix,
                    backend,
                    sample,
                    path,
                    mode,
                    "repeat",
                    2,
                    512 if path == "stream" else None,
                )
            )
    tasks.extend(MatrixTask("M02", "whisper", sample, "snapshot", "paced") for sample in snapshots)
    tasks.extend(
        MatrixTask("M04", "nemotron", sample, "stream", "paced", chunk_samples=512)
        for sample in paced
    )
    for sample in [chinese, *categories["C02"]]:
        tasks.extend(
            MatrixTask("M05", "nemotron", sample, "stream", "fast", chunk_samples=chunk)
            for chunk in (160, 1600)
        )
    for sample in categories["T01"] + categories["T02"]:
        tasks.extend(
            MatrixTask("M05", "nemotron", sample, "stream", "fast", chunk_samples=chunk)
            for chunk in (160, 512, 1600)
        )
    for scenario in ("reset", "interleave"):
        for sample in (chinese, english):
            tasks.append(
                MatrixTask(
                    "M06",
                    "nemotron",
                    sample,
                    "stream",
                    "fast",
                    chunk_samples=512,
                    auto_language=scenario == "reset",
                    scenario=scenario,
                )
            )
    for backend, path, mode in (("whisper", "whole", "offline"), ("nemotron", "stream", "fast")):
        for sample in (chinese, english):
            tasks.append(
                MatrixTask(
                    "M07",
                    backend,
                    sample,
                    path,
                    mode,
                    chunk_samples=512 if path == "stream" else None,
                    auto_language=True,
                )
            )
    tasks.extend(
        MatrixTask("M08", "nemotron", sample, "whole", "offline") for sample in (chinese, english)
    )
    tasks.extend(
        MatrixTask("M09", "whisper-gpu", sample, "whole", "offline", "primary") for sample in bases
    )
    tasks.extend(
        MatrixTask("M09", "whisper-gpu", sample, "whole", "offline", "repeat", 2)
        for sample in (chinese, english)
    )
    tasks.extend(
        MatrixTask("M09", "whisper-gpu", sample, "snapshot", "paced") for sample in snapshots
    )
    return sorted(tasks, key=lambda item: item.matrix_item)
