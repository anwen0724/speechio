"""阶段报告由必需证据决定，不设未经确认的效果门槛。"""


def phase_status(*, reviewed, core, checks, limits=()):
    """实现后根据 CPU 核心、语料和边界证据形成结论。"""
    if "cancelled" in core or "cancelled" in checks:
        return "cancelled"
    if "failed" in core or "failed" in checks:
        return "failed"
    if not reviewed or not core or not checks:
        return "blocked"
    if any(status != "ok" for status in core) or any(status != "passed" for status in checks):
        return "blocked"
    return "passed_with_limit" if limits else "passed"


def build_report(storage, checks, *, exploratory):
    """列出所有矩阵状态；草稿探索不给出正式 CER／WER。"""
    from collections import Counter

    from evals.p0.acceptance import acceptance_items
    from evals.p0.evaluation.summary import grouped_scores

    trials = [item for item in storage.trials if item["evaluation_role"] != "warmup"]
    core = [item["status"] for item in trials if item["matrix_item"] not in ("M07", "M09")]
    limits = [
        {
            "matrix_item": item["matrix_item"],
            "trial_id": item["trial_id"],
            "status": item["status"],
            "reason": item.get("reason", item.get("error")),
        }
        for item in trials
        if item["matrix_item"] in ("M07", "M09") and item["status"] != "ok"
    ]
    reviewed = all(item["annotation_checked"] for item in storage.samples.values())
    exploratory = exploratory or not reviewed
    for item in trials:
        result = item.get("result") or {}
        if item.get("language_mode") == "auto" and result.get("language_source") == "unavailable":
            limits.append(
                {
                    "matrix_item": item["matrix_item"],
                    "trial_id": item["trial_id"],
                    "reason": "后端未返回检测语言；未推测语言或概率。",
                }
            )
    environment = getattr(storage, "environment", {})
    limits.extend(environment.get("unknown_fields", []))
    acceptance = acceptance_items(storage, checks, reviewed, exploratory)
    by_matrix = {
        f"M{index:02}": dict(
            Counter(
                item["status"]
                for item in trials
                if item["matrix_item"] == f"M{index:02}" and item.get("scenario") != "reload"
            )
        )
        for index in range(1, 10)
    }
    return {
        "schema_version": 1,
        "run_id": storage.writer.run_id,
        "status": phase_status(
            reviewed=reviewed and not exploratory,
            core=core,
            checks=[item["status"] for item in acceptance],
            limits=limits,
        ),
        "exploratory": exploratory,
        "corpus_version": storage.corpus["corpus_version"],
        "annotation_reviewed": reviewed,
        "matrix": by_matrix,
        "boundary_checks": checks,
        "acceptance": acceptance,
        "reload_trial_ids": [
            item["trial_id"] for item in trials if item.get("scenario") == "reload"
        ],
        "resources": getattr(storage, "resource_summary", {}),
        "native_runtime": getattr(storage, "native_runtime", []),
        "limits": limits,
        "scores": [] if exploratory else grouped_scores(trials, storage.corpus["samples"]),
        "score_unavailable_reason": "语料尚未听音核实；探索结果不能作为正式质量基线。"
        if exploratory
        else None,
        "evidence_file": str(storage.writer.path),
    }
