#!/usr/bin/env python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Hallucination gates for the distillation pipeline.

Five gates that fire on each teacher response to detect hallucinations
BEFORE the GOLD/SILVER/DROP classification. Designed to integrate with
purify_teacher_outputs.py.

Gate chain (cheapest first):
  Gate 1: Self-consistency    — compare multiple teacher outputs (FREE)
  Gate 2: Semantic drift      — detect off-topic or self-contradicting text (FREE)
  Gate 3: Fact grounding      — token-overlap with reference context (FREE)
  Gate 4: Confidence estimate — perplexity/entropy proxy from text features (FREE)
  Gate 5: Zena cross-exam     — Gemma 4 judges borderline cases (EXPENSIVE)

Each gate returns a GateResult with pass/fail/flag + details.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class GateVerdict(Enum):
    PASS = "pass"      # clean
    FLAG = "flag"      # suspicious, might still be okay
    FAIL = "fail"      # hallucinated, should be demoted or dropped


@dataclass
class GateResult:
    gate: str
    verdict: GateVerdict
    score: float  # 0.0 = definitely clean, 1.0 = definitely hallucinated
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HallucinationReport:
    """Aggregated report across all gates for one teacher response."""
    teacher: str
    prompt_id: str
    gate_results: list[GateResult] = field(default_factory=list)
    final_verdict: GateVerdict = GateVerdict.PASS
    final_score: float = 0.0

    def add(self, result: GateResult):
        self.gate_results.append(result)
        self._recompute()

    def _recompute(self):
        if not self.gate_results:
            return
        # Weighted average score
        weights = {"consistency": 2.0, "drift": 1.0, "grounding": 1.5, "confidence": 1.0, "zena_judge": 2.5}
        total_w = 0.0
        total_s = 0.0
        for r in self.gate_results:
            w = weights.get(r.gate, 1.0)
            total_w += w
            total_s += r.score * w
        self.final_score = round(total_s / total_w, 3) if total_w > 0 else 0.0

        # Any FAIL → FAIL. Multiple FLAGS → FAIL. Single FLAG → FLAG.
        fails = sum(1 for r in self.gate_results if r.verdict == GateVerdict.FAIL)
        flags = sum(1 for r in self.gate_results if r.verdict == GateVerdict.FLAG)
        if fails > 0:
            self.final_verdict = GateVerdict.FAIL
        elif flags >= 2:
            self.final_verdict = GateVerdict.FAIL
        elif flags == 1:
            self.final_verdict = GateVerdict.FLAG
        else:
            self.final_verdict = GateVerdict.PASS


# ---------------------------------------------------------------------------
# Gate 1: Self-Consistency
# ---------------------------------------------------------------------------

def gate_consistency(
    answers: dict[str, str],
    threshold_agree: float = 0.5,
) -> dict[str, GateResult]:
    """Check if teachers give consistent answers.

    Compare each teacher's answer against all others using n-gram similarity.
    A teacher that disagrees with the majority is likely hallucinating.

    Args:
        answers: {teacher_name: answer_text}
        threshold_agree: minimum avg similarity to be considered consistent

    Returns:
        {teacher_name: GateResult}
    """
    names = list(answers.keys())
    if len(names) < 2:
        return {n: GateResult("consistency", GateVerdict.PASS, 0.0) for n in names}

    results = {}
    for name in names:
        sims = []
        for other in names:
            if other == name:
                continue
            sim = _ngram_similarity(answers[name], answers[other])
            sims.append(sim)
        avg_sim = sum(sims) / len(sims)
        # Low similarity with peers → hallucination flag
        halluc_score = max(0.0, 1.0 - avg_sim / threshold_agree) if avg_sim < threshold_agree else 0.0
        halluc_score = min(1.0, halluc_score)

        if avg_sim >= threshold_agree:
            verdict = GateVerdict.PASS
        elif avg_sim >= threshold_agree * 0.6:
            verdict = GateVerdict.FLAG
        else:
            verdict = GateVerdict.FAIL

        results[name] = GateResult(
            gate="consistency",
            verdict=verdict,
            score=round(halluc_score, 3),
            details={"avg_similarity": round(avg_sim, 3), "peer_similarities": {o: round(s, 3) for o, s in zip([n for n in names if n != name], sims)}},
        )
    return results


# ---------------------------------------------------------------------------
# Gate 2: Semantic Drift
# ---------------------------------------------------------------------------

def gate_semantic_drift(
    prompt: str,
    answer: str,
    relevance_threshold: float = 0.15,
    coherence_threshold: float = 0.10,
) -> GateResult:
    """Detect if a response drifts off-topic or contradicts itself.

    Checks:
    1. Prompt relevance: does the answer relate to the question?
    2. Sentence coherence: do adjacent sentences relate to each other?
    3. Contradiction signals: negation patterns that flip claims
    """
    if not answer or len(answer.strip()) < 10:
        return GateResult("drift", GateVerdict.FLAG, 0.5, {"reason": "too_short"})

    sentences = _split_sentences(answer)
    if not sentences:
        return GateResult("drift", GateVerdict.FLAG, 0.5, {"reason": "no_sentences"})

    # 1. Prompt relevance per sentence
    prompt_tokens = set(_tokenize(prompt))
    relevance_scores = []
    for sent in sentences:
        sent_tokens = set(_tokenize(sent))
        if not sent_tokens:
            relevance_scores.append(0.0)
            continue
        overlap = len(prompt_tokens & sent_tokens) / max(len(sent_tokens), 1)
        relevance_scores.append(overlap)
    avg_relevance = sum(relevance_scores) / len(relevance_scores)

    # 2. Adjacent sentence coherence
    coherence_scores = []
    for i in range(len(sentences) - 1):
        t1 = set(_tokenize(sentences[i]))
        t2 = set(_tokenize(sentences[i + 1]))
        if not t1 or not t2:
            coherence_scores.append(0.0)
            continue
        overlap = len(t1 & t2) / max(min(len(t1), len(t2)), 1)
        coherence_scores.append(overlap)
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0

    # 3. Contradiction detection
    contradictions = _detect_contradictions(sentences)

    # Score: combine
    drift_score = 0.0
    flags = []

    if avg_relevance < relevance_threshold:
        drift_score += 0.4
        flags.append("low_relevance")
    if avg_coherence < coherence_threshold:
        drift_score += 0.3
        flags.append("low_coherence")
    if contradictions:
        drift_score += 0.3
        flags.append(f"contradictions: {len(contradictions)}")

    drift_score = min(1.0, drift_score)

    if drift_score >= 0.6:
        verdict = GateVerdict.FAIL
    elif drift_score >= 0.3:
        verdict = GateVerdict.FLAG
    else:
        verdict = GateVerdict.PASS

    return GateResult(
        gate="drift",
        verdict=verdict,
        score=round(drift_score, 3),
        details={
            "avg_relevance": round(avg_relevance, 3),
            "avg_coherence": round(avg_coherence, 3),
            "contradictions": contradictions,
            "flags": flags,
        },
    )


# ---------------------------------------------------------------------------
# Gate 3: Fact Grounding
# ---------------------------------------------------------------------------

def gate_grounding(
    answer: str,
    reference: str,
    threshold: float = 0.3,
) -> GateResult:
    """Check if claims in the answer are grounded in reference material.

    Uses token-overlap faithfulness checking (similar to zen_core_libs).
    """
    if not reference or not reference.strip():
        return GateResult("grounding", GateVerdict.PASS, 0.0, {"reason": "no_reference"})

    answer_tokens = _tokenize(answer)
    ref_tokens = set(_tokenize(reference))

    if not answer_tokens:
        return GateResult("grounding", GateVerdict.FLAG, 0.5, {"reason": "empty_answer"})

    # Sentence-level grounding
    sentences = _split_sentences(answer)
    grounded = 0
    ungrounded_claims = []
    for sent in sentences:
        sent_tokens = set(_tokenize(sent))
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & ref_tokens) / len(sent_tokens)
        if overlap >= threshold:
            grounded += 1
        else:
            ungrounded_claims.append(sent[:100])

    total = max(len(sentences), 1)
    grounding_ratio = grounded / total

    halluc_score = max(0.0, 1.0 - grounding_ratio)

    if grounding_ratio >= 0.7:
        verdict = GateVerdict.PASS
    elif grounding_ratio >= 0.4:
        verdict = GateVerdict.FLAG
    else:
        verdict = GateVerdict.FAIL

    return GateResult(
        gate="grounding",
        verdict=verdict,
        score=round(halluc_score, 3),
        details={
            "grounding_ratio": round(grounding_ratio, 3),
            "ungrounded_claims": ungrounded_claims[:5],
            "total_sentences": total,
        },
    )


# ---------------------------------------------------------------------------
# Gate 4: Confidence Estimate (text-based proxy)
# ---------------------------------------------------------------------------

# Hedge words that signal low confidence
_HEDGE_PATTERNS = [
    r"\bI think\b",
    r"\bprobably\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bnot sure\b",
    r"\bI believe\b",
    r"\bpossibly\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bI'm not certain\b",
    r"\bapproximately\b",
    r"\bsomething like\b",
]

# Overconfidence patterns that often precede hallucination
_OVERCONFIDENT_PATTERNS = [
    r"\bAbsolutely\b",
    r"\bDefinitely\b",
    r"\bWithout a doubt\b",
    r"\bIt is a well-known fact\b",
    r"\bAs everyone knows\b",
    r"\bObviously\b",
    r"\bClearly\b",
]

# Fabrication markers — specific claims that LLMs tend to hallucinate
_FABRICATION_MARKERS = [
    r"\b(in|published|wrote|said|according to)\s+(19|20)\d{2}\b",  # specific years
    r"\b(Dr\.|Prof\.|Professor)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # named people
    r"\bhttps?://\S+\b",  # URLs (often hallucinated)
    r"\bISBN\s*[\d-]+\b",  # ISBNs
    r"\b\d{1,3}\.\d{1,3}%\b",  # precise percentages
]


def gate_confidence(
    answer: str,
    hedge_threshold: int = 3,
    fabrication_threshold: int = 2,
) -> GateResult:
    """Estimate confidence/hallucination risk from text patterns.

    Two signals:
    1. Excessive hedging → uncertain, might be hedging around hallucinated content
    2. Fabrication markers → specific dates, names, URLs that LLMs often fabricate
    """
    if not answer.strip():
        return GateResult("confidence", GateVerdict.FLAG, 0.5, {"reason": "empty"})

    # Count hedges
    hedge_count = sum(
        1 for p in _HEDGE_PATTERNS if re.search(p, answer, re.IGNORECASE)
    )

    # Count overconfidence
    overconf_count = sum(
        1 for p in _OVERCONFIDENT_PATTERNS if re.search(p, answer, re.IGNORECASE)
    )

    # Count fabrication markers
    fab_count = sum(
        len(re.findall(p, answer, re.IGNORECASE)) for p in _FABRICATION_MARKERS
    )

    # Score: hedges and fabrication markers both contribute
    score = 0.0
    flags = []

    if hedge_count >= hedge_threshold:
        score += 0.3
        flags.append(f"hedging ({hedge_count} markers)")

    if overconf_count >= 2:
        score += 0.2
        flags.append(f"overconfident ({overconf_count} markers)")

    if fab_count >= fabrication_threshold:
        score += 0.4
        flags.append(f"fabrication_markers ({fab_count} found)")
    elif fab_count >= 1:
        score += 0.15
        flags.append(f"minor_fabrication ({fab_count} found)")

    # Answer length anomaly: very long answers tend to hallucinate more
    word_count = len(answer.split())
    if word_count > 500:
        score += 0.1
        flags.append(f"verbose ({word_count} words)")

    score = min(1.0, score)

    if score >= 0.6:
        verdict = GateVerdict.FAIL
    elif score >= 0.3:
        verdict = GateVerdict.FLAG
    else:
        verdict = GateVerdict.PASS

    return GateResult(
        gate="confidence",
        verdict=verdict,
        score=round(score, 3),
        details={
            "hedge_count": hedge_count,
            "overconfidence_count": overconf_count,
            "fabrication_markers": fab_count,
            "word_count": word_count,
            "flags": flags,
        },
    )


# ---------------------------------------------------------------------------
# Gate 5: Zena Cross-Exam (requires ZenaDean instance)
# ---------------------------------------------------------------------------

def gate_zena_judge(
    zena_dean,  # ZenaDean instance
    prompt: str,
    answer: str,
) -> GateResult:
    """Use Zena (Gemma 4) as a hallucination judge.

    EXPENSIVE — only call for borderline SILVER cases.
    """
    result = zena_dean.judge_hallucination(prompt, answer)
    h_score = result.get("hallucination_score", 5) / 10.0  # normalize to 0-1

    if h_score <= 0.2:
        verdict = GateVerdict.PASS
    elif h_score <= 0.5:
        verdict = GateVerdict.FLAG
    else:
        verdict = GateVerdict.FAIL

    return GateResult(
        gate="zena_judge",
        verdict=verdict,
        score=round(h_score, 3),
        details=result,
    )


# ---------------------------------------------------------------------------
# Pipeline: run all gates on a sample
# ---------------------------------------------------------------------------

def run_hallucination_pipeline(
    prompt: str,
    teacher_answers: dict[str, str],
    reference: str = "",
    zena_dean=None,
    borderline_only: bool = True,
) -> dict[str, HallucinationReport]:
    """Run the full hallucination gate chain on all teacher answers for one prompt.

    Args:
        prompt: the original prompt
        teacher_answers: {teacher_name: answer_text}
        reference: optional reference text for grounding gate
        zena_dean: optional ZenaDean instance for gate 5
        borderline_only: if True, only run Zena judge on FLAG results (recommended)

    Returns:
        {teacher_name: HallucinationReport}
    """
    reports: dict[str, HallucinationReport] = {}

    # Gate 1: Self-consistency (all teachers at once)
    consistency_results = gate_consistency(teacher_answers)

    for name, answer in teacher_answers.items():
        report = HallucinationReport(teacher=name, prompt_id=prompt[:50])

        # Gate 1: Consistency
        report.add(consistency_results[name])

        # Gate 2: Semantic drift
        report.add(gate_semantic_drift(prompt, answer))

        # Gate 3: Fact grounding (only if reference available)
        if reference:
            report.add(gate_grounding(answer, reference))

        # Gate 4: Confidence estimate
        report.add(gate_confidence(answer))

        # Gate 5: Zena judge (only for borderline, or if explicitly requested)
        if zena_dean is not None:
            if not borderline_only or report.final_verdict == GateVerdict.FLAG:
                report.add(gate_zena_judge(zena_dean, prompt, answer))

        reports[name] = report

    return reports


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple word tokenization, lowercased, no stopwords."""
    # Remove punctuation, lowercase, split
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text.split()
    # Remove common stopwords that don't carry meaning for comparison
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "can", "shall",
                 "of", "in", "to", "for", "with", "on", "at", "by", "from",
                 "as", "into", "through", "during", "before", "after", "and",
                 "but", "or", "not", "no", "if", "than", "that", "this",
                 "it", "its", "i", "you", "he", "she", "we", "they"}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def _ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """N-gram overlap similarity between two texts (0-1)."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    ngrams_a = set()
    for i in range(len(tokens_a) - n + 1):
        ngrams_a.add(tuple(tokens_a[i:i + n]))

    ngrams_b = set()
    for i in range(len(tokens_b) - n + 1):
        ngrams_b.add(tuple(tokens_b[i:i + n]))

    if not ngrams_a or not ngrams_b:
        # Fall back to unigram overlap
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(len(set_a | set_b), 1)

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / max(len(union), 1)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Also split on newlines
    result = []
    for p in parts:
        for sub in p.split("\n"):
            sub = sub.strip()
            if sub and len(sub) > 5:
                result.append(sub)
    return result


def _detect_contradictions(sentences: list[str]) -> list[tuple[int, int, str]]:
    """Detect potential contradictions between sentences.

    Returns list of (sentence_i, sentence_j, pattern_matched).
    """
    # Negation flip patterns
    flip_pairs = [
        (r"\bis\b", r"\bis not\b"),
        (r"\bcan\b", r"\bcannot\b"),
        (r"\bwill\b", r"\bwill not\b"),
        (r"\btrue\b", r"\bfalse\b"),
        (r"\balways\b", r"\bnever\b"),
        (r"\byes\b", r"\bno\b"),
        (r"\bcorrect\b", r"\bincorrect\b"),
        (r"\bpossible\b", r"\bimpossible\b"),
    ]

    contradictions = []
    for i in range(len(sentences)):
        for j in range(i + 1, min(i + 4, len(sentences))):  # check nearby sentences
            si = sentences[i].lower()
            sj = sentences[j].lower()

            # Check if one sentence has the positive and the other the negative
            for pos_pattern, neg_pattern in flip_pairs:
                if re.search(pos_pattern, si) and re.search(neg_pattern, sj):
                    # Also check they share topic tokens
                    shared = set(_tokenize(si)) & set(_tokenize(sj))
                    if len(shared) >= 2:  # share at least 2 content words
                        contradictions.append((i, j, f"{pos_pattern} vs {neg_pattern}"))
                        break
                elif re.search(neg_pattern, si) and re.search(pos_pattern, sj):
                    shared = set(_tokenize(si)) & set(_tokenize(sj))
                    if len(shared) >= 2:
                        contradictions.append((i, j, f"{neg_pattern} vs {pos_pattern}"))
                        break

    return contradictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """Test the gates on sample data."""
    import argparse

    parser = argparse.ArgumentParser(description="Test hallucination gates.")
    parser.add_argument("--input", help="JSONL with teacher_responses (output of multi_teacher_generate).")
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to test.")
    args = parser.parse_args()

    if args.input:

        rows = []
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:  # xray: ignore[PY-005]
                        rows.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        pass  # skip malformed JSON line

        for row in rows[:args.sample]:
            prompt = row.get("prompt", "")
            teachers = row.get("teachers", {})  # xray: ignore[PY-004]
            answers = {name: info.get("answer", info.get("raw", "")) for name, info in teachers.items()}  # xray: ignore[PY-004]

            print(f"\n{'='*60}")  # xray: ignore[PY-004]
            print(f"Prompt: {prompt[:80]}...")  # xray: ignore[PY-004]
            reports = run_hallucination_pipeline(prompt, answers)  # xray: ignore[PY-004]
            for name, report in reports.items():
                icon = {"pass": "✓", "flag": "⚠", "fail": "✗"}[report.final_verdict.value]  # xray: ignore[PY-004]
                print(f"  {icon} {name}: {report.final_verdict.value} (score={report.final_score})")  # xray: ignore[PY-004]
                for gr in report.gate_results:
                    print(f"      Gate {gr.gate}: {gr.verdict.value} ({gr.score:.2f})")  # xray: ignore[PY-004]
    else:
        # Demo with synthetic data
        print("Testing hallucination gates with synthetic data...\n")  # xray: ignore[PY-004]
        answers = {
            "teacher_A": "The capital of France is Paris. It has been the capital since the 10th century.",
            "teacher_B": "The capital of France is Paris, known for the Eiffel Tower built in 1889.",
            "teacher_C": "The capital of France is Lyon, a beautiful city in southern France.",
        }
        prompt = "What is the capital of France?"
        reports = run_hallucination_pipeline(prompt, answers)  # xray: ignore[PY-004]
        for name, report in reports.items():
            icon = {"pass": "✓", "flag": "⚠", "fail": "✗"}[report.final_verdict.value]  # xray: ignore[PY-004]
            print(f"{icon} {name}: {report.final_verdict.value} (score={report.final_score})")  # xray: ignore[PY-004]
            for gr in report.gate_results:
                print(f"    Gate {gr.gate}: {gr.verdict.value} ({gr.score:.2f}) {gr.details.get('flags', '')}")  # xray: ignore[PY-004]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
