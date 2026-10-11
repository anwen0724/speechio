"""将 P0 验收项连接到实际记录；未知和未执行保持受阻。"""


def acceptance_items(storage, checks, reviewed, exploratory):
    """行为证据与正式语料证据分开，不由一份成功文本推定整阶段完成。"""
    trials = [item for item in storage.trials if item["evaluation_role"] != "warmup"]
    check_map = {item["check_id"]: item for item in checks}
    items = []
    from collections import Counter

    from evals.p0.matrix import declare_matrix

    expected = Counter(task.matrix_item for task in declare_matrix(storage.corpus))

    def add(identifier, matrices=(), boundaries=(), *, additional=True, reason=None):
        selected = [item for item in trials if item["matrix_item"] in matrices]
        proof = [check_map[name] for name in boundaries if name in check_map]
        statuses = [item["status"] for item in selected] + [item["status"] for item in proof]
        complete = (not matrices or bool(selected)) and len(proof) == len(boundaries) and additional
        complete = complete and all(
            sum(
                item["matrix_item"] == matrix and item.get("scenario") != "reload"
                for item in trials
            )
            == expected[matrix]
            for matrix in matrices
        )
        if "failed" in statuses:
            status = "failed"
        elif "cancelled" in statuses:
            status = "cancelled"
        elif complete and all(value in ("ok", "passed") for value in statuses):
            status = "passed"
        else:
            status = "blocked"
        items.append(
            {
                "acceptance_id": identifier,
                "status": status,
                "matrix_items": list(matrices),
                "boundary_check_ids": list(boundaries),
                "trial_ids": [item["trial_id"] for item in selected],
                "reason": reason,
            }
        )

    environment = getattr(storage, "environment", {})
    required = ("python_path", "python_version", "cpu_name", "memory_total_bytes", "dependencies")
    add(
        "A01",
        additional=all(environment.get(name) for name in required) and bool(storage.configurations),
        reason="解释器、CPU、内存、依赖和展开参数均来自实际记录。",
    )
    nonempty = all(
        any(
            item["matrix_item"] == matrix
            and storage.samples[item["sample_id"]]["language"] == language
            and item.get("result")
            and item["result"]["text"].strip()
            for item in trials
        )
        for matrix in ("M01", "M03")
        for language in ("zh", "en")
    )
    add("A02", ("M01", "M03", "M08"), additional=nonempty)
    add("A03", boundaries=("F01", "F02"))
    add(
        "A04",
        boundaries=("F03",),
        additional=reviewed and not exploratory,
        reason=None if reviewed else "公开语料正文、边界和说话人仍待人工听音核实。",
    )
    add("A05", ("M01", "M08"))
    add("A06", ("M02",), ("F10",))
    paced_long = [
        item
        for item in trials
        if item["matrix_item"] == "M04"
        and "C02" in storage.samples[item["sample_id"]]["categories"]
    ]
    early_partial = any(
        item.get("timestamps", {}).get("first_nonempty_partial_ns") is not None
        and item["timestamps"]["first_nonempty_partial_ns"]
        < item["timestamps"].get("finish_called_ns", 0)
        for item in paced_long
    )
    add(
        "A07", ("M04",), additional=early_partial, reason="非空流式中间结果须早于完整输入递交结束。"
    )
    add("A08", ("M03", "M05"), ("F04", "F08"))
    add("A09", boundaries=("F05", "F06", "F07"))
    add("A10", ("M06",), ("F09",))
    add("A11", ("M07",), reason="自动语言的实际文本和元数据独立记录；缺少检测语言或概率不补造。")
    add(
        "A12",
        additional=reviewed and not exploratory,
        reason="评分公式有行为测试；正式语料未核实前不形成正式 CER／WER。",
    )
    measured = all(
        item.get("calls") and item.get("timestamps")
        for item in trials
        if item["matrix_item"] in ("M02", "M03", "M04") and item["status"] == "ok"
    )
    add("A13", ("M02", "M03", "M04"), additional=measured)
    reloads = [item for item in trials if item.get("scenario") == "reload"]
    add(
        "A14",
        boundaries=("F02",),
        additional=len(reloads) == 2,
        reason="重新加载与资源采样有独立记录；不能据引用释放宣称长期宿主回收。",
    )
    gpu = [item for item in trials if item["matrix_item"] == "M09"]
    add(
        "A15",
        additional=bool(gpu) and all(item["status"] != "skipped" for item in gpu),
        reason="GPU 对照按实际结果报告；能力限制由阶段报告另列。",
    )
    add(
        "A16",
        boundaries=("F11", "F12"),
        additional=len(checks) == 12,
        reason="失败、取消、未确认范围、清理错误与全部矩阵声明保留。",
    )
    return items
