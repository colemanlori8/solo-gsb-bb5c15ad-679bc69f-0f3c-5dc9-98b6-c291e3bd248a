"""敏感短语打码预览核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

把每个**同一说话人连续区间**内所有词的 text 原样拼接（不插空格），在其中
查找每个敏感短语的精确 Unicode 字符命中：命中可横跨多个词，但不允许跨越
说话人边界。所有短语的全部命中汇总后按

    (命中起始全局字符位升序, 短语长度降序, 短语输入顺序升序)

排序，再一趟贪心选出互不重叠的命中——起点靠前者优先，同起点时长命中优先，
完全同键时按短语输入顺序（稳定排序保证）；扫描时维护已选命中的最大终点
busy_end，候选起点 < busy_end 即与已选命中重叠，跳过。全局字符位指字符在
“全部词文本原样拼接”中的 0-based code point 偏移；不同说话人区间互不
相交，故命中是否重叠与区间划分无关。

仅把选中命中覆盖的字符逐字替换为“█”（每字符一个 █），词的时间、说话人与
词序保持不变；无任何命中时返回原词序列与空命中列表。

复杂度：命中枚举 O(Σ|区间文本|·P)（P 为短语数，≤50），选择 O(H log H)
（H 为命中总数），打码 O(总字符数)。
"""

from bisect import bisect_right

from .segmentation import validate_words

REDACT_CHAR = "█"


def redact_preview(words, phrases):
    """选出互不重叠的敏感短语命中并打码，返回 (redacted_words, hits)。

    参数 words 中每个对象需具备属性：start_ms / end_ms / text / speaker
    （沿用同一套词序与时间校验）；phrases 为 1～50 个非空字符串（精确匹配，
    允许重复，重复短语同一位置只会选出一条命中）。

    redacted_words 与输入等长，每项为 dict：start_ms / end_ms / text /
    speaker；仅选中命中覆盖的字符被逐字替换为“█”，其余与输入一致。
    hits 按命中起点升序，每项为 dict：phrase / start_word / end_word /
    char_start / char_end；词序号 0-based、含端点，char_start/char_end 为
    命中首/末字符在首/末词内的 Unicode 字符偏移（首词覆盖
    [char_start, 词末)，末词覆盖 [0, char_end)）。无命中时 hits 为空列表，
    redacted_words 与输入逐字一致。
    """
    validate_words(words)
    n = len(words)

    char_counts = [len(w.text) for w in words]  # Unicode code points
    global_offsets = [0] * n
    for i in range(1, n):
        global_offsets[i] = global_offsets[i - 1] + char_counts[i - 1]

    # ---- 枚举全部命中：(起点, 终点, 短语, 短语输入序号)，起终点为全局字符位 ----
    all_hits = []
    run_start = 0
    for i in range(1, n + 1):
        if i < n and words[i].speaker == words[i - 1].speaker:
            continue
        run_text = "".join(words[k].text for k in range(run_start, i))
        base = global_offsets[run_start]
        for phrase_index, phrase in enumerate(phrases):
            pos = run_text.find(phrase)
            while pos >= 0:
                start = base + pos
                all_hits.append((start, start + len(phrase), phrase, phrase_index))
                pos = run_text.find(phrase, pos + 1)  # 允许重叠命中
        run_start = i

    # ---- 排序 + 一趟贪心选出互不重叠的命中 ----
    # 起点靠前优先；同起点长命中优先；再同则按短语输入顺序（phrase_index
    # 互异，保证全序）。
    all_hits.sort(key=lambda h: (h[0], -(h[1] - h[0]), h[3]))
    selected = []
    busy_end = 0  # 已选命中的最大终点（短语非空且命中互不重叠，故单调不减）
    for start, end, phrase, _ in all_hits:
        if start < busy_end:
            continue  # 与已选命中重叠
        selected.append((start, end, phrase))
        busy_end = end

    # ---- 打码并定位每个命中的首末词与词内偏移 ----
    texts = [list(w.text) for w in words]
    hits = []
    for start, end, phrase in selected:
        first = bisect_right(global_offsets, start) - 1
        last = bisect_right(global_offsets, end - 1) - 1
        for k in range(first, last + 1):
            base = global_offsets[k]
            a = max(start - base, 0)
            b = min(end - base, char_counts[k])
            texts[k][a:b] = REDACT_CHAR * (b - a)
        hits.append(
            {
                "phrase": phrase,
                "start_word": first,
                "end_word": last,
                "char_start": start - global_offsets[first],
                "char_end": end - global_offsets[last],
            }
        )

    redacted = [
        {
            "start_ms": w.start_ms,
            "end_ms": w.end_ms,
            "text": "".join(texts[i]),
            "speaker": w.speaker,
        }
        for i, w in enumerate(words)
    ]
    return redacted, hits
