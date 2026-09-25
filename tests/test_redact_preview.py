"""敏感短语打码预览核心逻辑单元测试 + 独立实现对拍（纯标准库）。

独立实现同样枚举每个同一说话人连续区间内的全部精确命中（允许跨词、允许
重叠），并按 (起点, -长度, 短语输入序号) 排序，但重叠判定改用逐字符占用
集合（而非 redact_preview 的 busy_end 游标），与 redact_preview 互相对拍。
"""

import random
import unittest
from bisect import bisect_right
from types import SimpleNamespace

from app.errors import DomainError
from app.redact_preview import redact_preview


def W(start, end, text, speaker="A"):
    return SimpleNamespace(start_ms=start, end_ms=end, text=text, speaker=speaker)


def brute_force(words, phrases):
    """独立实现：逐字符占用集合判定重叠，返回 (expected_texts, expected_hits)。"""
    n = len(words)
    counts = [len(w.text) for w in words]
    offsets = [0] * n
    for i in range(1, n):
        offsets[i] = offsets[i - 1] + counts[i - 1]

    # 枚举全部命中（全局字符位区间）
    hits = []
    run_start = 0
    for i in range(1, n + 1):
        if i == n or words[i].speaker != words[i - 1].speaker:
            run_text = "".join(words[k].text for k in range(run_start, i))
            for phrase_index, phrase in enumerate(phrases):
                pos = run_text.find(phrase)
                while pos >= 0:
                    s = offsets[run_start] + pos
                    hits.append((s, s + len(phrase), phrase, phrase_index))
                    pos = run_text.find(phrase, pos + 1)
            run_start = i
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0]), h[3]))

    # 逐字符占用集合选互不重叠命中
    occupied = set()
    selected = []
    for s, e, phrase, _ in hits:
        if any(p in occupied for p in range(s, e)):
            continue
        occupied.update(range(s, e))
        selected.append((s, e, phrase))

    full = [c for w in words for c in w.text]
    for s, e, _ in selected:
        for p in range(s, e):
            full[p] = "█"
    texts = []
    idx = 0
    for w in words:
        texts.append("".join(full[idx : idx + len(w.text)]))
        idx += len(w.text)

    expected_hits = []
    for s, e, phrase in selected:
        first = bisect_right(offsets, s) - 1
        last = bisect_right(offsets, e - 1) - 1
        expected_hits.append(
            {
                "phrase": phrase,
                "start_word": first,
                "end_word": last,
                "char_start": s - offsets[first],
                "char_end": e - offsets[last],
            }
        )
    return texts, expected_hits


class BasicRedactionTests(unittest.TestCase):
    def test_single_word_hit(self):
        words = [W(0, 500, "请联系张三", "A")]
        redacted, hits = redact_preview(words, ["张三"])
        self.assertEqual([w["text"] for w in redacted], ["请联系██"])
        self.assertEqual(
            hits,
            [
                {
                    "phrase": "张三",
                    "start_word": 0,
                    "end_word": 0,
                    "char_start": 3,
                    "char_end": 5,
                }
            ],
        )

    def test_cross_word_hit_same_speaker(self):
        words = [W(0, 500, "请联系", "A"), W(500, 1000, "张三", "A")]
        redacted, hits = redact_preview(words, ["系张"])
        self.assertEqual([w["text"] for w in redacted], ["请联█", "█三"])
        self.assertEqual(
            hits,
            [
                {
                    "phrase": "系张",
                    "start_word": 0,
                    "end_word": 1,
                    "char_start": 2,
                    "char_end": 1,
                }
            ],
        )

    def test_speaker_boundary_blocks_hit(self):
        words = [W(0, 500, "张", "A"), W(500, 1000, "三", "B")]
        redacted, hits = redact_preview(words, ["张三"])
        self.assertEqual(hits, [])
        self.assertEqual([w["text"] for w in redacted], ["张", "三"])

    def test_no_hit_returns_original_words(self):
        words = [W(0, 500, "你好", "A"), W(500, 1000, "世界", "B")]
        redacted, hits = redact_preview(words, ["不存在"])
        self.assertEqual(hits, [])
        self.assertEqual(
            redacted,
            [
                {"start_ms": 0, "end_ms": 500, "text": "你好", "speaker": "A"},
                {"start_ms": 500, "end_ms": 1000, "text": "世界", "speaker": "B"},
            ],
        )

    def test_time_speaker_order_preserved(self):
        words = [W(0, 500, "秘密", "A"), W(700, 1200, "公开", "B")]
        redacted, _ = redact_preview(words, ["秘密", "公开"])
        self.assertEqual(
            [(w["start_ms"], w["end_ms"], w["speaker"]) for w in redacted],
            [(0, 500, "A"), (700, 1200, "B")],
        )
        self.assertEqual([w["text"] for w in redacted], ["██", "██"])

    def test_unicode_code_points(self):
        # emoji 与中文字符均按 1 个 code point 计；短语横跨词界
        words = [W(0, 500, "嗨😀秘", "A"), W(500, 1000, "密😀", "A")]
        redacted, hits = redact_preview(words, ["😀秘密"])
        # 命中全局 [1,4)：词 0 偏移 1～2、词 1 偏移 0
        self.assertEqual([w["text"] for w in redacted], ["嗨██", "█😀"])
        self.assertEqual(
            hits,
            [
                {
                    "phrase": "😀秘密",
                    "start_word": 0,
                    "end_word": 1,
                    "char_start": 1,
                    "char_end": 1,
                }
            ],
        )

    def test_hit_covering_whole_run(self):
        words = [W(0, 500, "甲", "A"), W(500, 1000, "乙", "A")]
        redacted, hits = redact_preview(words, ["甲乙"])
        self.assertEqual([w["text"] for w in redacted], ["█", "█"])
        self.assertEqual(
            hits,
            [
                {
                    "phrase": "甲乙",
                    "start_word": 0,
                    "end_word": 1,
                    "char_start": 0,
                    "char_end": 1,
                }
            ],
        )


class OverlapSelectionTests(unittest.TestCase):
    def test_earlier_start_wins(self):
        # "abc"[0,3) 与 "bcd"[1,4) 重叠：起点 0 靠前，选 "abc"
        words = [W(0, 500, "abcd", "A")]
        redacted, hits = redact_preview(words, ["abc", "bcd"])
        self.assertEqual([h["phrase"] for h in hits], ["abc"])
        self.assertEqual([w["text"] for w in redacted], ["███d"])

    def test_same_start_longer_wins(self):
        # 同起点 0："abc" 长 3 优先于 "ab" 长 2
        words = [W(0, 500, "abc", "A")]
        redacted, hits = redact_preview(words, ["ab", "abc"])
        self.assertEqual([h["phrase"] for h in hits], ["abc"])
        self.assertEqual([w["text"] for w in redacted], ["███"])

    def test_same_start_same_length_first_phrase_wins(self):
        # 重复短语：同位置同长，按输入顺序只选出一条
        words = [W(0, 500, "ab", "A")]
        redacted, hits = redact_preview(words, ["ab", "ab"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["phrase"], "ab")
        self.assertEqual([w["text"] for w in redacted], ["██"])

    def test_adjacent_hits_both_selected(self):
        # [0,2) 与 [2,4) 端点相接不算重叠
        words = [W(0, 500, "abcd", "A")]
        redacted, hits = redact_preview(words, ["ab", "cd"])
        self.assertEqual([h["phrase"] for h in hits], ["ab", "cd"])
        self.assertEqual([w["text"] for w in redacted], ["████"])

    def test_overlapping_occurrences_same_phrase(self):
        # "aa" 在 "aaa" 中命中 [0,2) 与 [1,3)：后者与前者重叠被跳过
        words = [W(0, 500, "aaa", "A")]
        redacted, hits = redact_preview(words, ["aa"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["char_start"], 0)
        self.assertEqual([w["text"] for w in redacted], ["██a"])

    def test_blocked_hit_does_not_block_later(self):
        # "ab"[0,2) 选中；"bc"[1,3) 被跳过后，"cd"[2,4) 仍应被选中
        words = [W(0, 500, "abcd", "A")]
        redacted, hits = redact_preview(words, ["ab", "bc", "cd"])
        self.assertEqual([h["phrase"] for h in hits], ["ab", "cd"])
        self.assertEqual([w["text"] for w in redacted], ["████"])

    def test_long_hit_priority_at_same_start_suppresses_inner(self):
        # "abcd" 与 "bc"：同起点不同——构造同起点包含关系：
        # "abx" 中短语 "abx"(idx1) 与 "ab"(idx0) 同起点，长的先选；
        # 末尾 "x" 不被单独请求
        words = [W(0, 500, "abx", "A")]
        redacted, hits = redact_preview(words, ["ab", "abx"])
        self.assertEqual([h["phrase"] for h in hits], ["abx"])
        self.assertEqual([w["text"] for w in redacted], ["███"])

    def test_multiple_speaker_runs_independent(self):
        # 两个说话人区间各自命中（相同短语），互不影响
        words = [W(0, 500, "张三说", "A"), W(500, 1000, "张三答", "B")]
        redacted, hits = redact_preview(words, ["张三"])
        self.assertEqual(len(hits), 2)
        self.assertEqual([h["start_word"] for h in hits], [0, 1])
        self.assertEqual([w["text"] for w in redacted], ["██说", "██答"])

    def test_runs_separated_then_rejoined(self):
        # A-B-A：两个 A 区间内的命中各自独立；跨 A/B 拼接不命中
        words = [
            W(0, 300, "甲", "A"),
            W(300, 600, "乙", "B"),
            W(600, 900, "甲", "A"),
        ]
        redacted, hits = redact_preview(words, ["甲"])
        self.assertEqual(len(hits), 2)
        self.assertEqual([h["start_word"] for h in hits], [0, 2])

    def test_hits_sorted_by_start(self):
        words = [W(0, 500, "丙乙甲", "A")]
        redacted, hits = redact_preview(words, ["甲", "乙", "丙"])
        self.assertEqual([h["phrase"] for h in hits], ["丙", "乙", "甲"])


class ValidationReuseTests(unittest.TestCase):
    def test_time_out_of_order_reused(self):
        with self.assertRaises(DomainError) as ctx:
            redact_preview([W(1000, 2000, "甲"), W(900, 1500, "乙")], ["甲"])
        self.assertEqual(ctx.exception.code, "time_out_of_order")

    def test_overlap_reused(self):
        with self.assertRaises(DomainError) as ctx:
            redact_preview([W(0, 2000, "甲"), W(1500, 3000, "乙")], ["甲"])
        self.assertEqual(ctx.exception.code, "overlapping_time_range")

    def test_empty_words_reused(self):
        with self.assertRaises(DomainError) as ctx:
            redact_preview([], ["甲"])
        self.assertEqual(ctx.exception.code, "empty_words")


class DifferentialTests(unittest.TestCase):
    def test_matches_brute_force(self):
        random.seed(20260925)
        texts = ["甲", "乙", "甲乙", "甲甲", "丙", "a😀", "乙甲", "丁"]
        speakers = ["A", "A", "B", "B"]
        phrase_pool = ["甲", "乙", "甲乙", "甲甲", "丙", "😀", "乙甲", "丁", "甲甲乙"]
        with_hits = 0
        for trial in range(600):
            n = random.randint(1, 10)
            words = []
            t = 0
            for i in range(n):
                t += random.choice([0, 0, 100, 400])
                dur = random.choice([100, 300, 600])
                words.append(W(t, t + dur, random.choice(texts), speakers[i % 4]))
                t += dur
            phrases = [random.choice(phrase_pool) for _ in range(random.randint(1, 6))]

            redacted, hits = redact_preview(words, phrases)
            exp_texts, exp_hits = brute_force(words, phrases)
            self.assertEqual(
                [w["text"] for w in redacted], exp_texts, f"trial {trial} texts mismatch"
            )
            self.assertEqual(hits, exp_hits, f"trial {trial} hits mismatch")

            # 时间、说话人、词序不变
            self.assertEqual(
                [(w["start_ms"], w["end_ms"], w["speaker"]) for w in redacted],
                [(w.start_ms, w.end_ms, w.speaker) for w in words],
            )

            # 自洽校验：命中按全局字符位互不重叠（端点相接允许）
            counts = [len(w.text) for w in words]
            offsets = [0] * n
            for i in range(1, n):
                offsets[i] = offsets[i - 1] + counts[i - 1]
            prev_end = 0
            for h in hits:
                s = offsets[h["start_word"]] + h["char_start"]
                e = offsets[h["end_word"]] + h["char_end"]
                self.assertGreaterEqual(s, prev_end)
                self.assertGreater(e, s)
                self.assertEqual(e - s, len(h["phrase"]))
                prev_end = e

                # 命中区间在打码词中恰为 len(phrase) 个 █
                first, last = h["start_word"], h["end_word"]
                if first == last:
                    covered = redacted[first]["text"][h["char_start"] : h["char_end"]]
                else:
                    covered = redacted[first]["text"][h["char_start"] :]
                    for k in range(first + 1, last):
                        covered += redacted[k]["text"]
                    covered += redacted[last]["text"][: h["char_end"]]
                self.assertEqual(covered, "█" * len(h["phrase"]))

            # █ 的总数等于选中命中长度之和
            self.assertEqual(
                sum(w["text"].count("█") for w in redacted),
                sum(len(h["phrase"]) for h in hits),
            )
            if hits:
                with_hits += 1
        self.assertGreater(with_hits, 300)  # 确保有足够多的有命中样本


if __name__ == "__main__":
    unittest.main()
