"""固定 p0-text-v1 的中文 CER 和英文 WER。"""

import unicodedata


def normalize(text, language):
    """按固定顺序规范化，保留非标点符号，不改简繁体或数字读法。"""
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "zh":
        return "".join(
            c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P")
        )
    if language == "en":
        text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
        return " ".join(text.split())
    return text


def edit_counts(reference, hypothesis):
    """单位代价编辑距离；同代价按匹配、替换、删除、插入择优。"""
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    moves = [[None] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], moves[i][0] = i, "D"
    for j in range(1, columns):
        costs[0][j], moves[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, columns):
            matched = reference[i - 1] == hypothesis[j - 1]
            candidates = [
                (costs[i - 1][j - 1] + (not matched), "M" if matched else "S"),
                (costs[i - 1][j] + 1, "D"),
                (costs[i][j - 1] + 1, "I"),
            ]
            costs[i][j], moves[i][j] = min(candidates, key=lambda item: item[0])
    counts = {"S": 0, "D": 0, "I": 0}
    i, j = rows - 1, columns - 1
    while i or j:
        move = moves[i][j]
        if move != "M":
            counts[move] += 1
        if move in ("M", "S", "D"):
            i -= 1
        if move in ("M", "S", "I"):
            j -= 1
    return counts


def score(reference, hypothesis, language):
    """对结束正文评分；空标注返回不适用，不产生除零分数。"""
    normalized_reference = normalize(reference, language)
    normalized_hypothesis = normalize(hypothesis, language)
    ref = list(normalized_reference) if language == "zh" else normalized_reference.split()
    hyp = list(normalized_hypothesis) if language == "zh" else normalized_hypothesis.split()
    counts = edit_counts(ref, hyp)
    n, errors = len(ref), sum(counts.values())
    return {
        "normalization_id": "p0-text-v1",
        "reference_text": reference,
        "text": hypothesis,
        "normalized_reference": normalized_reference,
        "normalized_text": normalized_hypothesis,
        "N": n,
        "E": errors,
        **counts,
        "error_rate": errors / n if n else None,
    }
