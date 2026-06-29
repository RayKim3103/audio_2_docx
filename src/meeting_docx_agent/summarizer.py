from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable, Optional

from .llm import get_llm
from .profiles import RuntimeProfile
from .utils import clean_text_for_xml, humanize_llm_value

PIPELINE_VERSION = "general_meeting_v8"

SYSTEM_PROMPT = """
당신은 보안이 중요한 로컬 환경에서 동작하는 회의록/녹음 정리 전문 AI입니다.
입력 transcript는 Whisper/faster-whisper 같은 ASR 모델이 생성한 결과이므로, 고유명사·약어·숫자·날짜·전문용어·띄어쓰기·문장부호에 오인식이 섞일 수 있습니다.
반드시 transcript에 근거해서만 작성하되, 주변 문맥과 사용자가 제공한 용어/고유명사 힌트가 명확할 때만 자연스럽게 보정하여 표현하세요.
불명확한 이름, 수치, 날짜, 담당자, 결정사항은 추측하지 말고 '확인 필요' 또는 '명시적으로 확인되지 않음'이라고 쓰세요.
ASR 오류처럼 보이는 부분은 내용을 버리지 말고, 의미가 명확한 경우에는 자연어로 정리하고 불확실한 경우에는 확인 필요한 내용으로 분리하세요.
회의가 아닌 강의, 교육, 인터뷰, 발표, 상담, 유튜브/교육 영상도 처리할 수 있도록 일반 도메인으로 정리하세요.
최종 독자는 사람이므로 모든 값은 자연스러운 한국어 문장으로 작성하세요.
출력 언어가 ko인 경우 반드시 자연스러운 한국어 문장으로만 작성하세요. 중국어·일본어·한자식 문장, 번체/간체 한자, 한국어 문장 중간에 끼어드는 중국어 글자를 출력하지 마세요. 단, NVIDIA, HBM, AI, CUDA 같은 고유 약어는 그대로 둘 수 있습니다.
구어체·농담·감탄사는 필요한 경우에만 짧게 맥락으로 남기고, 문서의 중심은 핵심 사실·논점·흐름·근거가 되도록 정리하세요.
회의가 아닌 해설/강의/뉴스/인터뷰/유튜브 녹음이면 회의 참석자나 결정사항을 억지로 만들지 말고, ‘녹음 내용 정리’ 관점에서 주요 내용과 흐름을 정리하세요.
작성 지시문, 예시 문장, placeholder, 마크다운 코드블록을 출력하지 마세요.
Python dict/list 문자열, JSON 문자열, key-value dump를 본문 값으로 넣지 마세요. 예: {'heading': ..., 'bullets': ...} 같은 형식은 금지입니다.
반복 문장을 길게 늘어놓지 말고, 같은 의미는 한 번만 정리하세요.
출력은 요청한 JSON 객체 하나만 반환하세요.
""".strip()

SYSTEM_PROMPT_MARKDOWN = """
당신은 보안이 중요한 로컬 환경에서 동작하는 회의록/녹음 정리 전문 AI입니다.
입력 transcript는 ASR 전사 결과이므로 오인식이 섞일 수 있습니다. 문맥상 명확한 경우에만 자연스럽게 보정하고, 불확실한 내용은 확인 필요로 남기세요.
이 애플리케이션은 general domain 회의/강의/인터뷰/발표/설명 영상 정리를 위한 것입니다. 특정 회사, 산업, 샘플에 편향하지 마세요.
출력 언어가 ko이면 반드시 자연스러운 한국어 문장으로만 작성하세요. 중국어·일본어·한자식 문자를 섞지 마세요. NVIDIA, HBM, AI 같은 통용 약어는 유지할 수 있습니다.
최종 독자는 사람이므로 raw transcript 조각을 그대로 나열하지 말고, 의미 단위로 묶어 문서화하세요.
원문에 없는 사실을 만들지 말고, 숫자·날짜·인물·회사명·결정사항은 특히 보수적으로 다루세요.
Markdown 문서만 반환하세요. 코드블록, JSON, Python dict/list 문자열, 작성 지시문은 출력하지 마세요.
""".strip()

TEMPLATE_PHRASES = [
    "3~7개 bullet로 요약",
    "2~4개",
    "3~5개",
    "주제별로 소제목을 만들고 자세히 정리",
    "원문에 있을 때만 작성",
    "가능한 경우 `[HH:MM:SS]`",
    "가능한 경우 [HH:MM:SS]",
    "중요한 용어가 있으면 설명",
    "불명확하거나 추가 확인이 필요한 부분",
    "실제 transcript",
    "이 구간의 핵심 요약",
]

WEAK_PLACEHOLDERS = {"주제명", "구체적 주제명", "주요 주제", "세부 내용", "설명", "핵심 요약"}

DETAIL_SETTINGS = {
    "brief": {
        "label": "간단 요약",
        "chunk_topics": 3,
        "topic_bullets": 3,
        "summary_bullets": 4,
        "structured_limit": 18,
        "timeline_limit": 18,
        "quotes_limit": 8,
        "token_multiplier": 0.75,
    },
    "standard": {
        "label": "표준 회의록",
        "chunk_topics": 4,
        "topic_bullets": 4,
        "summary_bullets": 5,
        "structured_limit": 35,
        "timeline_limit": 28,
        "quotes_limit": 12,
        "token_multiplier": 0.9,
    },
    "detailed": {
        "label": "상세 회의록",
        "chunk_topics": 6,
        "topic_bullets": 5,
        "summary_bullets": 7,
        "structured_limit": 60,
        "timeline_limit": 45,
        "quotes_limit": 18,
        "token_multiplier": 1.0,
    },
}

KOREAN_STOPWORDS = {
    "그리고", "그래서", "그런데", "하지만", "이렇게", "저렇게", "여러분", "오늘", "지금", "정말", "약간", "조금", "많이",
    "안녕하세요", "안녕", "오늘은", "이것", "저것", "그것", "우리", "제가", "저희", "하는", "하고", "되는", "있는", "없는", "같은", "경우", "부분", "내용", "말씀",
    "생각", "때문", "통해", "대한", "대해", "위해", "해서", "하면", "합니다", "했습니다", "있습니다", "됩니다", "됩니다",
    "입니다", "있고", "있는데", "되는데", "거죠", "거예요", "하죠", "이제", "자", "네", "음", "어", "그", "좀", "더", "수", "것",
    "라고", "그런", "걸로", "미팅도", "만나", "하다", "했다", "된다", "아니", "하나", "요거", "요런", "하하하", "아유", "하튼", "하여튼",
    "선생님", "분들이", "사람들", "언급", "언급되었다", "언급됐다", "나왔고", "나옵니다", "보시면", "중간", "계속", "굉장히", "되게", "완전히", "갑자기",
}


# Small transliteration/cleanup map for accidental Chinese characters that Qwen
# sometimes inserts into Korean names.  This is intentionally conservative and
# only used to make Korean output readable; unknown CJK-heavy text is filtered
# later rather than blindly trusted.
CJK_KO_REPLACEMENTS = {
    "亚": "아",
    "尼克": "닉",
    "尼": "니",
    "克": "크",
    "云": "운",
    "车": "차",
    "电脑": "컴퓨터",
    "成本": "비용",
    "收益": "수익",
    "平台": "플랫폼",
}

COMMON_ASR_TEXT_FIXES = {
    "엔비디亚": "엔비디아",
    "엔비티아": "엔비디아",
    "엔비스티아": "엔비디아",
    "엔비드라": "엔비디아",
    "마이크로서트": "마이크로소프트",
    "네이바": "네이버",
    "네아바": "네이버",
    "구광고": "구광모",
    "구강모": "구광모",
    "구 광모": "구광모",
    "채 태원": "최태원",
    "채태원": "최태원",
    "최태훈": "최태원",
    "최태헌": "최태원",
    "정의성": "정의선",
    "이정성": "정의선",
    "이예진": "이해진",
    "SK아이닉스": "SK하이닉스",
    "SK하이尼克스": "SK하이닉스",
    "SK하이니크스": "SK하이닉스",
    "SKINX": "SK하이닉스",
    "하이尼克스": "하이닉스",
    "하이니크스": "하이닉스",
    "하이니스": "하이닉스",
    "SK하니inicx": "SK하이닉스",
    "HBM 망카페": "HBM 관련 과자",
    "HMB": "HBM",
    "젠스당": "젠슨 황",
    "젠스낭": "젠슨 황",
    "제스당": "젠슨 황",
    "제니스 당": "젠슨 황",
    "젠스튼": "젠슨 황",
    "컴피텍스": "컴퓨텍스",
    "GTC Type A": "GTC Taipei",
    "로봇공 핸드폰 플랫 폴": "로봇공학 플랫폼",
    "로봇공화 폴": "로봇공학 플랫폼",
    "휫지": "휴지",
    "삼겹 살": "삼겹살",
    "삼겹 사": "삼겹살",
    "판도체": "반도체",
}

FILLER_SENTENCE_PREFIXES = (
    "메인 주중에", "요거", "자 ", "네 ", "어 ", "뭐 ", "아 ", "하하", "그리고 ", "그런데 ", "그러니까 ", "하여튼",
)

HALLUCINATION_RISK_WORDS = {
    "기부", "후원", "인수", "합병", "계약", "체결", "확정", "승인", "선정", "투자했다", "결정했다", "합의", "수상", "처벌", "해고",
}


def normalize_language_artifacts(text: Any) -> str:
    s = clean_text_for_xml(str(text or ""))
    for a, b in COMMON_ASR_TEXT_FIXES.items():
        s = s.replace(a, b)
    for a, b in CJK_KO_REPLACEMENTS.items():
        s = s.replace(a, b)
    # If a few stray CJK characters remain inside an otherwise Korean sentence,
    # remove them rather than leaking Chinese glyphs to DOCX.
    if re.search(r"[가-힣]", s):
        s = re.sub(r"[\u3400-\u9fff\u3040-\u30ff]", "", s)
    s = s.replace("  ", " ")
    return s.strip()


def sentence_split(text: str) -> list[str]:
    text = normalize_language_artifacts(text)
    parts = re.split(r"(?<=[.!?。])\s+|\n+|(?<=요)\s+(?=[가-힣A-Z0-9])|(?<=다)\s+(?=[가-힣A-Z0-9])", text)
    out = []
    for p in parts:
        p = clean_item_text(p, 500) if 'clean_item_text' in globals() else re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def content_keywords(text: str) -> set[str]:
    return {w for w in tokenize_keywords(normalize_language_artifacts(text)) if not is_low_value_term(w)}


def is_poor_bullet(text: str) -> bool:
    t = normalize_language_artifacts(text).strip()
    if not t or has_template_leak(t) or is_weak_placeholder(t):
        return True
    if len(t) < 12:
        return True
    if t in {"명시적으로 확인되지 않음", "확인 필요", "원문 기반"}:
        return True
    if any(t.startswith(x) for x in FILLER_SENTENCE_PREFIXES) and len(content_keywords(t)) <= 1:
        return True
    if len(content_keywords(t)) == 0 and len(t) < 40:
        return True
    # Fragments ending in connective / filler often make the DOCX feel like sliced chunks.
    if re.search(r"(그리고|또한|하지만|그런데|하면|해서|라고|걸로|는데|면서)$", t):
        return True
    return False


def transcript_text_index(segments: list[dict]) -> str:
    return normalize_language_artifacts(" ".join(str(s.get("text", "")) for s in segments)).lower()


def is_statement_suspicious(text: str, transcript_index: str) -> bool:
    """Light factual guard.  Avoid rejecting valid paraphrases too aggressively."""
    t = normalize_language_artifacts(text).lower()
    if not t or is_poor_bullet(t):
        return True
    # Any remaining CJK after normalization is too risky for a Korean DOCX.
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff]", t):
        return True
    # Risky verbs should be grounded in the transcript. This catches hallucinations
    # such as '10억 달러를 기부했다' when no donation is discussed.
    if any(w in t for w in HALLUCINATION_RISK_WORDS) and not any(w in transcript_index for w in HALLUCINATION_RISK_WORDS):
        return True
    nums = re.findall(r"\d+(?:[.,]\d+)?", t)
    if nums:
        missing = [n for n in nums if n.replace(",", "") not in transcript_index.replace(",", "")]
        if len(missing) == len(nums) and len(content_keywords(t)) < 2:
            return True
    kws = [k for k in content_keywords(t) if len(k) >= 2]
    if len(kws) >= 3:
        hit = sum(1 for k in kws if k.lower() in transcript_index)
        if hit / max(1, len(kws)) < 0.25:
            return True
    return False


def make_readable_bullet_from_sentences(sentences: list[str], max_len: int = 240) -> str:
    cleaned = []
    for s in sentences:
        s = clean_item_text(s, 300)
        if not is_poor_bullet(s):
            cleaned.append(s)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0][:max_len]
    # Combine adjacent short utterances into a single more readable sentence.
    combined = " ".join(cleaned[:2])
    combined = re.sub(r"\s+", " ", combined).strip()
    return clean_item_text(combined, max_len)


def detail_cfg(detail_level: str) -> dict:
    return DETAIL_SETTINGS.get(detail_level or "standard", DETAIL_SETTINGS["standard"])


def has_template_leak(text: str) -> bool:
    if not text:
        return False
    return any(p in text for p in TEMPLATE_PHRASES)


def collapse_repeated_phrases(text: str, max_repeats: int = 2) -> str:
    """Collapse pathological LLM repetition such as '배당주 투자 결정' repeated dozens of times."""
    if not text:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    # Repeated comma-separated phrases.
    parts = [p.strip() for p in re.split(r"[,，/|]+", s) if p.strip()]
    if len(parts) >= 8:
        counts = Counter(parts)
        dominant, count = counts.most_common(1)[0]
        if count / len(parts) >= 0.35:
            kept: list[str] = []
            seen_count: Counter[str] = Counter()
            for p in parts:
                seen_count[p] += 1
                if seen_count[p] <= max_repeats:
                    kept.append(p)
            s = ", ".join(kept)
    # Repeated sentence fragments.
    sentences = [p.strip() for p in re.split(r"(?<=[.!?。])\s+|\n+", s) if p.strip()]
    if len(sentences) >= 6:
        kept = []
        seen = Counter()
        for sent in sentences:
            key = re.sub(r"\s+", " ", sent)[:120]
            seen[key] += 1
            if seen[key] <= max_repeats:
                kept.append(sent)
        if len(kept) < len(sentences):
            s = " ".join(kept)
    # Repeated short token window, e.g. 'A B C A B C A B C'.
    words = s.split()
    if len(words) > 40:
        for n in range(2, 8):
            chunks = [" ".join(words[i:i+n]) for i in range(0, len(words)-n+1, n)]
            if chunks:
                top, cnt = Counter(chunks).most_common(1)[0]
                if cnt >= 6 and cnt / max(1, len(chunks)) > 0.25:
                    pattern = re.escape(top)
                    s = re.sub(rf"(?:{pattern}\s*){{3,}}", (top + " ") * max_repeats, s).strip()
    return s.strip()


def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|markdown|md)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def extract_json_object(text: str) -> str | None:
    text = strip_code_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None


def parse_json_tolerant(text: str) -> dict | None:
    text = strip_code_fence(text)
    candidates = [text]
    obj = extract_json_object(text)
    if obj and obj != text:
        candidates.append(obj)
    for c in candidates:
        try:
            data = json.loads(c)
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    try:
        from json_repair import repair_json
        repaired = repair_json(obj or text)
        data = json.loads(repaired)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def compact_evidence(evidence: Any, limit: int = 2) -> list[str]:
    items: list[str] = []
    for item in as_list(evidence):
        s = clean_text_for_xml(str(item)).strip()
        if s and s not in items:
            items.append(s[:160])
    return items[:limit]


def clean_item_text(value: Any, max_len: int = 1200) -> str:
    s = humanize_llm_value(value, max_len=max_len).strip()
    s = normalize_language_artifacts(s)
    s = re.sub(r"\s+", " ", s)
    s = collapse_repeated_phrases(s)
    if has_template_leak(s):
        return ""
    if s.startswith("{") and s.endswith("}"):
        s = s.strip("{}").replace("'", "").replace('"', "")
    return s[:max_len].strip()


def is_weak_placeholder(text: str) -> bool:
    return clean_item_text(text).strip() in WEAK_PLACEHOLDERS


def segment_evidence(segments: list[dict], limit: int = 2) -> list[str]:
    ev = []
    for s in segments[:limit]:
        if all(k in s for k in ("id", "start_hms", "end_hms")):
            ev.append(f"[{s['id']} {s['start_hms']}-{s['end_hms']}]")
    return ev


def tokenize_keywords(text: str) -> list[str]:
    raw = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9+.#_-]{1,}", text)
    out = []
    for w in raw:
        w = w.strip("._-+ #").lower()
        if len(w) > 2:
            w = re.sub(r"(으로|에서|에게|까지|부터|처럼|보다|그리고|하고|과|와|은|는|이|가|을|를|에|의)$", "", w)
        if len(w) < 2 or w in KOREAN_STOPWORDS or w.endswith(("합니다", "했습니다", "됩니다", "입니다", "는데요")):
            continue
        if re.fullmatch(r"\d+", w):
            continue
        out.append(w)
    return out


LOW_VALUE_TERMS = {
    "주요", "논의", "년도", "내용", "언급", "언급되었다", "언급됐다", "부분", "그런", "라고", "걸로", "있다", "있는", "없는", "하게", "하면", "해서",
    "미팅도", "만나", "중간", "가게", "형님", "휴지", "먹는", "나왔", "보면", "얘기", "생각", "사람", "문제", "오늘",
}


def is_low_value_term(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t or t in LOW_VALUE_TERMS or t in KOREAN_STOPWORDS:
        return True
    if len(t) < 2:
        return True
    if len(t) <= 3 and re.fullmatch(r"[가-힣]+", t) and t not in {"ai", "pc", "hbm", "esg"}:
        return True
    if re.fullmatch(r"[0-9]+", t):
        return True
    # 조사/어미만 남은 듯한 표현, 발화 습관어 제거
    if t.endswith(("라고", "하죠", "인데", "는데", "거죠", "걸로", "하면", "해서", "하는", "있다", "됐다")):
        return True
    return False


def top_keywords(texts: list[str], limit: int = 5) -> list[str]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize_keywords(t))
    words = []
    for w, _ in counter.most_common(60):
        if is_low_value_term(w):
            continue
        if all(w not in other and other not in w for other in words):
            words.append(w)
        if len(words) >= limit:
            break
    return words


def score_segment_text(text: str, keywords: list[str]) -> float:
    t = text.lower()
    score = min(len(text), 240) / 240.0
    score += sum(0.5 for k in keywords if k and k in t)
    score += 0.3 if re.search(r"\d", text) else 0
    score += 0.25 if any(x in text for x in ["결정", "해야", "필요", "문제", "리스크", "핵심", "중요", "전략", "이유", "결과"] ) else 0
    return score


def pick_representative_segments(segments: list[dict], limit: int = 5) -> list[dict]:
    texts = [s.get("text", "") for s in segments if s.get("text")]
    kws = top_keywords(texts, 8)
    scored = []
    seen = set()
    for s in segments:
        text = clean_item_text(s.get("text", ""), 500)
        if len(text) < 10:
            continue
        key = re.sub(r"\s+", " ", text)[:90]
        if key in seen:
            continue
        seen.add(key)
        scored.append((score_segment_text(text, kws), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in scored[:limit]]
    selected.sort(key=lambda s: float(s.get("start", 0)))
    return selected


def topic_heading_from_segments(segments: list[dict], idx: int) -> str:
    texts = [s.get("text", "") for s in segments if s.get("text")]
    kws = top_keywords(texts, 3)
    if kws:
        # Preserve upper-case acronyms if present in the original text.
        joined = " · ".join(k.upper() if k.isascii() and len(k) <= 5 else k for k in kws)
        return f"주요 논의 {idx}: {joined}"
    if segments:
        text = clean_item_text(segments[0].get("text", ""), 60)
        return f"주요 논의 {idx}: {text[:28]}"
    return f"주요 논의 {idx}"


def sanitize_note(note: dict, source_segments: list[dict] | None = None, detail_level: str = "standard") -> dict:
    cfg = detail_cfg(detail_level)
    if not isinstance(note, dict):
        note = {}
    out = {
        "summary_bullets": [],
        "topics": [],
        "decisions": [],
        "action_items": [],
        "risks_issues": [],
        "open_questions": [],
        "key_quotes": [],
        "timeline": [],
        "terms": [],
    }
    seen_text: set[str] = set()

    for b in as_list(note.get("summary_bullets") or note.get("summary")):
        s = clean_item_text(b, 500)
        key = s[:120]
        if s and not is_poor_bullet(s) and key not in seen_text:
            out["summary_bullets"].append(s)
            seen_text.add(key)
    out["summary_bullets"] = out["summary_bullets"][: cfg["summary_bullets"]]

    topic_seen: set[str] = set()
    for t in as_list(note.get("topics") or note.get("detailed_topics") or note.get("structured_notes")):
        if not isinstance(t, dict):
            continue
        heading = clean_item_text(t.get("heading") or t.get("topic") or t.get("title") or "주요 논점", 120)
        bullets = [clean_item_text(x, 700) for x in as_list(t.get("bullets") or t.get("details") or t.get("items"))]
        cleaned_bullets = []
        local_seen = set()
        for x in bullets:
            k = x[:120]
            if x and not is_poor_bullet(x) and k not in local_seen:
                cleaned_bullets.append(x)
                local_seen.add(k)
        cleaned_bullets = cleaned_bullets[: cfg["topic_bullets"]]
        if cleaned_bullets:
            if is_weak_placeholder(heading):
                heading = cleaned_bullets[0][:40]
            marker = (heading + "|" + "|".join(cleaned_bullets[:2]))[:220]
            if marker not in topic_seen:
                out["topics"].append({"heading": heading or "주요 논점", "bullets": cleaned_bullets, "evidence": compact_evidence(t.get("evidence"), 2)})
                topic_seen.add(marker)
    out["topics"] = out["topics"][: cfg["structured_limit"]]

    for key in ["decisions", "risks_issues", "open_questions"]:
        local_seen = set()
        for item in as_list(note.get(key)):
            if isinstance(item, dict):
                text = item.get("text") or item.get("decision") or item.get("issue") or item.get("question") or item.get("point") or ""
                ev = compact_evidence(item.get("evidence"), 2)
            else:
                text, ev = item, []
            text = clean_item_text(text, 800)
            marker = text[:160]
            if text and not is_poor_bullet(text) and marker not in local_seen:
                out[key].append({"text": text, "evidence": ev})
                local_seen.add(marker)

    for item in as_list(note.get("asr_uncertainties") or note.get("transcription_uncertainties") or note.get("uncertain_terms")):
        if isinstance(item, dict):
            text = item.get("text") or item.get("term") or item.get("uncertainty") or item.get("question") or ""
            ev = compact_evidence(item.get("evidence"), 2)
        else:
            text, ev = item, []
        text = clean_item_text(text, 800)
        if text and not is_poor_bullet(text):
            if not text.startswith("ASR 확인 필요"):
                text = "ASR 확인 필요: " + text
            out["open_questions"].append({"text": text, "evidence": ev})

    action_seen = set()
    for a in as_list(note.get("action_items")):
        if not isinstance(a, dict):
            task = clean_item_text(a, 800)
            if task and task[:120] not in action_seen:
                out["action_items"].append({"task": task, "owner": "확인 필요", "due_date": "확인 필요", "evidence": []})
                action_seen.add(task[:120])
            continue
        task = clean_item_text(a.get("task") or a.get("할 일") or a.get("text"), 800)
        if task and task[:120] not in action_seen:
            out["action_items"].append({
                "task": task,
                "owner": clean_item_text(a.get("owner") or a.get("담당자") or "확인 필요", 80) or "확인 필요",
                "due_date": clean_item_text(a.get("due_date") or a.get("기한") or "확인 필요", 80) or "확인 필요",
                "evidence": compact_evidence(a.get("evidence"), 2),
            })
            action_seen.add(task[:120])

    quote_seen = set()
    for q in as_list(note.get("key_quotes")):
        if isinstance(q, dict):
            quote = clean_item_text(q.get("quote") or q.get("text"), 450)
            ev = compact_evidence(q.get("evidence") or q.get("time"), 2)
        else:
            quote, ev = clean_item_text(q, 450), []
        if quote and not is_poor_bullet(quote) and quote[:120] not in quote_seen:
            out["key_quotes"].append({"quote": quote, "evidence": ev})
            quote_seen.add(quote[:120])
    out["key_quotes"] = out["key_quotes"][: cfg["quotes_limit"]]

    tl_seen = set()
    for tl in as_list(note.get("timeline")):
        if isinstance(tl, dict):
            time = clean_item_text(tl.get("time") or tl.get("시점"), 80)
            event = clean_item_text(tl.get("event") or tl.get("내용") or tl.get("text"), 800)
        else:
            time, event = "", clean_item_text(tl, 800)
        if event and not is_poor_bullet(event) and event[:120] not in tl_seen:
            out["timeline"].append({"time": time, "event": event})
            tl_seen.add(event[:120])
    out["timeline"] = out["timeline"][: cfg["timeline_limit"]]

    term_seen = set()
    for term in as_list(note.get("terms")):
        if isinstance(term, dict):
            name = clean_item_text(term.get("term") or term.get("name"), 120)
            desc = clean_item_text(term.get("description") or term.get("desc"), 500)
        else:
            name, desc = clean_item_text(term, 120), ""
        if name and not is_weak_placeholder(name) and name[:60] not in term_seen:
            out["terms"].append({"term": name, "description": desc})
            term_seen.add(name[:60])

    if not out["summary_bullets"] and out["topics"]:
        for t in out["topics"][: cfg["summary_bullets"]]:
            if t.get("bullets"):
                out["summary_bullets"].append(t["bullets"][0])

    if not out["summary_bullets"] and not out["topics"] and source_segments:
        text = " ".join(s.get("text", "") for s in source_segments[:10]).strip()
        if text:
            out["summary_bullets"] = [text[:400]]
            out["topics"] = [{"heading": "원문 기반 주요 내용", "bullets": [text[:700]], "evidence": segment_evidence(source_segments[:1])}]
    return out


def fallback_note_from_source(segments: list[dict], detail_level: str = "standard") -> dict:
    cfg = detail_cfg(detail_level)
    selected = pick_representative_segments(segments, limit=max(3, cfg["topic_bullets"]))
    bullets = [clean_item_text(s.get("text", ""), 700) for s in selected]
    bullets = [x for x in bullets if x]
    if not bullets:
        lines = [clean_item_text(s.get("text", ""), 400) for s in segments if s.get("text")]
        bullets = [x for x in lines if x][: cfg["topic_bullets"]]
    heading = topic_heading_from_segments(segments, 1)
    note = {
        "summary_bullets": bullets[: cfg["summary_bullets"]],
        "topics": [{"heading": heading, "bullets": bullets[: cfg["topic_bullets"]], "evidence": segment_evidence(selected or segments)}],
        "timeline": [{"time": s.get("start_hms", ""), "event": clean_item_text(s.get("text", ""), 300)} for s in selected[: cfg["timeline_limit"]]],
        "key_quotes": [{"quote": clean_item_text(s.get("text", ""), 300), "evidence": segment_evidence([s], 1)} for s in selected[: cfg["quotes_limit"]]],
    }
    return sanitize_note(note, segments, detail_level)


def loose_note_from_raw(raw: str, segments: list[dict], detail_level: str = "standard") -> dict:
    cfg = detail_cfg(detail_level)
    lines = [clean_item_text(x.strip("-•* 0123456789.\t"), 700) for x in raw.splitlines()]
    bullets = [x for x in lines if len(x) >= 18 and not is_poor_bullet(x)][: max(8, cfg["topic_bullets"] * 2)]
    if not bullets:
        return fallback_note_from_source(segments, detail_level)
    return sanitize_note({
        "summary_bullets": bullets[: cfg["summary_bullets"]],
        "topics": [{"heading": "LLM 출력 복구 내용", "bullets": bullets[: cfg["topic_bullets"]], "evidence": segment_evidence(segments)}],
    }, segments, detail_level)


def chunk_segments(segments: list[dict], max_chars: int, overlap_chars: int = 0) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for s in segments:
        text = s.get("text", "")
        add = len(text) + 45
        if cur and cur_len + add > max_chars:
            chunks.append(cur)
            if overlap_chars > 0:
                overlap: list[dict] = []
                overlap_len = 0
                for prev in reversed(cur):
                    overlap.insert(0, prev)
                    overlap_len += len(prev.get("text", "")) + 45
                    if overlap_len >= overlap_chars:
                        break
                cur = overlap[:]
                cur_len = overlap_len
            else:
                cur = []
                cur_len = 0
        cur.append(s)
        cur_len += add
    if cur:
        chunks.append(cur)
    return chunks


def segments_to_prompt_text(segments: list[dict]) -> str:
    return "\n".join(f"[{s['id']} {s['start_hms']}-{s['end_hms']}] {s['text']}" for s in segments)


def chronological_blocks(
    segments: list[dict],
    block_max_chars: int = 1200,
    max_blocks: int = 18,
    max_total_chars: int = 18000,
) -> list[dict]:
    """Build coherent chronological transcript blocks for fast CPU summarization.

    v5 selected isolated high-scoring segments, which was fast but often lost context.
    v6 keeps short chronological windows so a single CPU LLM call can still understand
    the story flow without running LLM on every chunk.
    """
    raw_blocks: list[dict] = []
    cur: list[dict] = []
    cur_chars = 0
    for seg in segments:
        txt = clean_item_text(seg.get("text", ""), 900)
        if not txt:
            continue
        add = len(txt) + 40
        if cur and cur_chars + add > block_max_chars:
            raw_blocks.append({
                "start": cur[0].get("start_hms", ""),
                "end": cur[-1].get("end_hms", ""),
                "segments": cur,
                "text": " ".join(s.get("text", "") for s in cur),
            })
            cur, cur_chars = [], 0
        cur.append(seg)
        cur_chars += add
    if cur:
        raw_blocks.append({
            "start": cur[0].get("start_hms", ""),
            "end": cur[-1].get("end_hms", ""),
            "segments": cur,
            "text": " ".join(s.get("text", "") for s in cur),
        })
    if len(raw_blocks) <= max_blocks:
        chosen = raw_blocks
    else:
        # Uniformly cover the timeline, with head/tail always included.
        idxs = {0, len(raw_blocks) - 1}
        if max_blocks > 2:
            step = (len(raw_blocks) - 1) / (max_blocks - 1)
            idxs.update(round(i * step) for i in range(max_blocks))
        chosen = [raw_blocks[i] for i in sorted(i for i in idxs if 0 <= i < len(raw_blocks))]
    out = []
    total = 0
    for i, b in enumerate(chosen, start=1):
        text = clean_item_text(b.get("text", ""), block_max_chars)
        if not text:
            continue
        if total + len(text) > max_total_chars and len(out) >= max(6, max_blocks // 2):
            break
        out.append({"id": f"B{i:02d}", "time": f"{b['start']}-{b['end']}", "text": text, "segments": b["segments"]})
        total += len(text)
    return out


def blocks_to_prompt_text(blocks: list[dict]) -> str:
    lines = []
    for b in blocks:
        lines.append(f"[{b['id']} {b['time']}] {b['text']}")
    return "\n\n".join(lines)


def extractive_notes_for_chunks(chunks: list[list[dict]], detail_level: str = "standard") -> list[dict]:
    """Fast, non-LLM notes used as safety net and content reservoir.

    Keep chronological context and avoid junk keyword tables. These notes are not meant
    to be the final prose; they provide factual material if the LLM output is sparse.
    """
    notes = []
    cfg = detail_cfg(detail_level)
    for idx, ch in enumerate(chunks, start=1):
        # Use a mix of first/middle/last + high-value segments, preserving order.
        selected: list[dict] = []
        if ch:
            selected.extend(ch[:2])
            if len(ch) > 4:
                mid = len(ch) // 2
                selected.extend(ch[max(0, mid - 1): mid + 1])
            selected.extend(ch[-2:])
        for s in pick_representative_segments(ch, limit=max(3, cfg["topic_bullets"])):
            if s not in selected:
                selected.append(s)
        selected.sort(key=lambda s: float(s.get("start", 0)))
        seen = set()
        bullets = []
        for s in selected:
            txt = clean_item_text(s.get("text", ""), 500)
            key = txt[:100]
            if txt and key not in seen:
                bullets.append(txt)
                seen.add(key)
        heading = topic_heading_from_segments(ch, idx)
        # Do not create generic term entries from simple word frequency. LLM can add real terms later.
        raw_note = {
            "summary_bullets": bullets[: cfg["summary_bullets"]],
            "topics": [{"heading": heading, "bullets": bullets[: max(3, cfg["topic_bullets"])], "evidence": segment_evidence(selected or ch)}],
            "timeline": [{"time": s.get("start_hms", ""), "event": clean_item_text(s.get("text", ""), 320)} for s in selected[: min(5, cfg["timeline_limit"])]],
            "key_quotes": [{"quote": clean_item_text(s.get("text", ""), 320), "evidence": segment_evidence([s], 1)} for s in selected[: min(3, cfg["quotes_limit"])]],
            "terms": [],
        }
        notes.append(sanitize_note(raw_note, ch, detail_level))
    return notes

def compact_transcript_sample(segments: list[dict], max_chars: int = 9000) -> str:
    if not segments:
        return ""
    selected = pick_representative_segments(segments, limit=80)
    # Ensure beginning and end context are represented.
    head = segments[:12]
    tail = segments[-8:] if len(segments) > 8 else []
    combined = []
    seen_ids = set()
    for s in head + selected + tail:
        sid = s.get("id")
        if sid not in seen_ids:
            combined.append(s)
            seen_ids.add(sid)
    combined.sort(key=lambda s: float(s.get("start", 0)))
    text = segments_to_prompt_text(combined)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[중략: 길이 제한으로 일부 transcript 생략]"
    return text


def make_chunk_prompt(title: str, chunk_idx: int, total_chunks: int, segments: list[dict], language: str, glossary: str = "", detail_level: str = "standard") -> str:
    cfg = detail_cfg(detail_level)
    source = segments_to_prompt_text(segments)
    glossary_text = f"\n사용자 제공 용어/고유명사 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
다음 transcript 구간을 회의록/녹음 정리용 정보로 추출하세요.
문서 제목 후보: {title}
구간: {chunk_idx}/{total_chunks}
출력 언어: {language}
문서 상세도: {cfg['label']}
{glossary_text}
작성 원칙:
- transcript는 ASR 전사 결과이므로 오인식이 있을 수 있습니다. 고유명사·약어·숫자·날짜·전문용어는 문맥과 사용자 제공 힌트가 명확할 때만 자연스럽게 보정하세요.
- 원문에 명시된 사실만 사용하고, 불확실한 내용은 사실처럼 단정하지 말고 open_questions 또는 asr_uncertainties에 넣으세요.
- 일반 도메인으로 작성하고 특정 회사/산업을 가정하지 마세요.
- 이 단계는 '요약'보다 '정보 추출'이 중요합니다. 세부 사실, 숫자, 변화 흐름, 원인/결과, 후속 조치를 보존하세요.
- topics는 {cfg['chunk_topics']}개 이내, 각 topic의 bullets는 {cfg['topic_bullets']}개 이내로 작성하세요.
- topic heading은 transcript의 실제 내용을 반영한 구체적 제목으로 작성하세요. '주요 논의', '주요 주제', '세부 내용' 같은 일반 제목은 쓰지 마세요.
- 같은 문장을 반복하지 마세요.
- evidence는 항목당 최대 1~2개만 쓰고 segment ID를 전부 나열하지 마세요.
- 담당자/기한이 원문에 없으면 '확인 필요'라고 쓰세요.
- 결정사항, 실행 항목, 리스크가 없으면 빈 배열 []을 쓰세요.
- 아래 JSON 구조의 빈 문자열/빈 배열을 실제 내용으로 채우되, placeholder를 그대로 출력하지 마세요.

반환 JSON 구조:
{{
  "summary_bullets": [],
  "topics": [{{"heading": "", "bullets": [], "evidence": []}}],
  "decisions": [{{"text": "", "evidence": []}}],
  "action_items": [{{"task": "", "owner": "", "due_date": "", "evidence": []}}],
  "risks_issues": [{{"text": "", "evidence": []}}],
  "open_questions": [{{"text": "", "evidence": []}}],
  "timeline": [{{"time": "", "event": ""}}],
  "key_quotes": [{{"quote": "", "evidence": []}}],
  "terms": [{{"term": "", "description": ""}}],
  "asr_uncertainties": [{{"text": "", "evidence": []}}]
}}

Transcript:
{source}
""".strip()


def make_fast_final_prompt(title: str, notes: list[dict], segments: list[dict], language: str, glossary: str = "", detail_level: str = "standard") -> str:
    cfg = detail_cfg(detail_level)
    # v7: provide coherent chronological transcript windows instead of isolated keyword snippets.
    # This keeps CPU speed close to v5 fast mode but gives the LLM enough context to write a better document.
    max_blocks = 20 if detail_level == "detailed" else 14
    max_total = 19000 if detail_level == "detailed" else 14500
    blocks = chronological_blocks(segments, block_max_chars=1150, max_blocks=max_blocks, max_total_chars=max_total)
    transcript_digest = blocks_to_prompt_text(blocks)
    compact_notes = json.dumps(notes, ensure_ascii=False)[:9000]
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
다음은 ASR transcript를 시간순으로 압축한 digest입니다. 이 digest를 중심으로 사람이 읽기 좋은 최종 회의록 JSON을 작성하세요.
문서 제목 후보: {title}
출력 언어: {language}
문서 상세도: {cfg['label']}
{glossary_text}
핵심 원칙:
- transcript digest의 실제 내용에 근거해서 작성하세요. 없는 사실을 만들지 마세요.
- 이 녹음이 회의가 아니라 강의/뉴스/유튜브/인터뷰/설명 영상이면, 결정사항·실행항목을 억지로 만들지 말고 주요 내용·흐름·핵심 논점 중심으로 정리하세요.
- ASR 오인식이 있을 수 있습니다. 문맥상 명확한 고유명사/약어만 자연스럽게 보정하고, 애매하면 확인 필요한 내용으로 남기세요.
- 출력은 반드시 자연스러운 한국어 문장으로 작성하세요. 중국어/일본어/한자식 문장을 섞지 마세요.
- one_page_summary는 bullet이 아닌 5~7문장 문단으로 작성하세요.
- overview는 문서 전체를 한눈에 볼 수 있는 짧은 목차형 문구 4~7개로 작성하세요.
- structured_notes는 6~10개 핵심 주제로 나누고, 각 주제마다 3~5개의 구체적 bullet을 작성하세요.
- key_points는 독자가 반드시 기억해야 할 논점 8~12개를 작성하세요.
- timeline은 실제 흐름이 드러나도록 8~15개 작성하세요.
- terms는 실제 의미 있는 고유명사/전문용어만 6~12개 작성하세요. '그런', '라고', '걸로', '있다' 같은 일반 발화어는 절대 용어로 넣지 마세요.
- '명시적으로 확인되지 않음'을 남발하지 마세요. 정말 없는 항목에만 사용하세요.
- Python dict/list 문자열, key-value dump, 작성 지시문을 값으로 넣지 마세요.
- JSON 객체 하나만 반환하세요.

반환 JSON 구조:
{{
  "one_page_summary": "문단형 요약",
  "overview": [],
  "structured_notes": [{{"heading": "", "bullets": [], "evidence": []}}],
  "key_points": [{{"point": "", "evidence": []}}],
  "decisions": [{{"text": "", "evidence": []}}],
  "action_items": [{{"task": "", "owner": "", "due_date": "", "evidence": []}}],
  "risks_issues": [{{"text": "", "evidence": []}}],
  "timeline": [{{"time": "", "event": ""}}],
  "key_quotes": [{{"quote": "", "evidence": []}}],
  "terms": [{{"term": "", "description": ""}}],
  "open_questions": [{{"text": "", "evidence": []}}],
  "asr_uncertainties": [{{"text": "", "evidence": []}}]
}}

시간순 transcript digest:
{transcript_digest}

보조 자동 추출 note(참고용, 부족하거나 중복될 수 있음):
{compact_notes}
""".strip()

def make_final_prompt(title: str, notes: list[dict], segments: list[dict], language: str, glossary: str = "", detail_level: str = "standard") -> str:
    """Final synthesis prompt for GPU/full mode.

    v7 improvement: final merge receives both sanitized chunk notes and a chronological
    transcript digest.  This prevents the final document from over-trusting noisy chunk
    headings and lets the LLM recover the actual flow of a general-domain recording.
    """
    cfg = detail_cfg(detail_level)
    compact_notes = json.dumps(notes, ensure_ascii=False)[:36000]
    max_blocks = 26 if detail_level == "detailed" else 18
    max_total = 28000 if detail_level == "detailed" else 20000
    blocks = chronological_blocks(segments, block_max_chars=1250, max_blocks=max_blocks, max_total_chars=max_total)
    transcript_digest = blocks_to_prompt_text(blocks)
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
다음 자료를 통합하여 사람이 바로 읽을 수 있는 최종 회의록 JSON을 작성하세요.
문서 제목 후보: {title}
출력 언어: {language}
문서 상세도: {cfg['label']}
{glossary_text}
자료 A는 transcript를 시간순으로 압축한 digest이고, 자료 B는 chunk별 LLM 추출 note입니다.
자료 B의 제목이나 bullet이 어색하거나 중복되면, 자료 A의 시간순 맥락을 우선하여 자연스럽게 재구성하세요.

핵심 원칙:
- 이 애플리케이션은 general domain 회의/강의/인터뷰/발표/설명 영상 정리를 위한 것입니다. 특정 산업이나 회사에 편향하지 마세요.
- 원문에 없는 사실을 만들지 마세요. 하지만 ASR 오인식이 명확한 경우에는 문맥과 사용자 힌트를 근거로 자연스럽게 보정하세요.
- 불명확한 고유명사, 숫자, 날짜, 담당자, 결정사항은 단정하지 말고 open_questions 또는 asr_uncertainties에 남기세요.
- 회의가 아닌 뉴스/강의/유튜브/설명 영상이면 결정사항·실행항목을 억지로 만들지 말고 주요 내용·흐름·핵심 논점 중심으로 정리하세요.
- 출력은 반드시 자연스러운 한국어 문장으로 작성하세요. 중국어·일본어·한자식 문장, 중국어 글자, 어색한 직역체를 섞지 마세요.
- one_page_summary는 bullet/list가 아니라 5~8문장 정도의 자연스러운 문단으로 작성하세요.
- overview는 문서 전체를 한눈에 볼 수 있는 4~7개 목차형 문구로 작성하세요.
- structured_notes는 6~12개 핵심 주제로 구성하고, 각 주제에는 실제 내용이 담긴 구체적 bullet 3~6개를 작성하세요.
- key_points는 세부 bullet을 모두 복사하지 말고, 독자가 기억해야 할 상위 논점 8~15개만 작성하세요.
- timeline은 실제 진행 흐름이 보이도록 8~18개 작성하세요.
- key_quotes는 의미 있는 발언만 고르고, 말버릇·농담·잡음성 표현만 단독으로 뽑지 마세요.
- terms는 고유명사, 제품명, 조직명, 기술명, 방법론 등 실제 의미 있는 용어만 작성하세요. 일반 발화어(예: 그런, 라고, 걸로, 있다, 중간, 만나)는 절대 넣지 마세요.
- '명시적으로 확인되지 않음'은 정말 해당 섹션이 원문에 없을 때만 사용하고, 근거 표의 빈칸을 채우기 위해 남발하지 마세요.
- JSON 객체 하나만 반환하세요. Markdown, 코드블록, Python dict/list 문자열, key-value dump를 출력하지 마세요.

반환 JSON 구조:
{{
  "one_page_summary": "사람이 바로 읽을 수 있는 문단형 요약",
  "overview": [],
  "structured_notes": [{{"heading": "", "bullets": [], "evidence": []}}],
  "key_points": [{{"point": "", "evidence": []}}],
  "decisions": [{{"text": "", "evidence": []}}],
  "action_items": [{{"task": "", "owner": "", "due_date": "", "evidence": []}}],
  "risks_issues": [{{"text": "", "evidence": []}}],
  "timeline": [{{"time": "", "event": ""}}],
  "key_quotes": [{{"quote": "", "evidence": []}}],
  "terms": [{{"term": "", "description": ""}}],
  "open_questions": [{{"text": "", "evidence": []}}],
  "asr_uncertainties": [{{"text": "", "evidence": []}}]
}}

자료 A. 시간순 transcript digest:
{transcript_digest}

자료 B. chunk별 추출 note:
{compact_notes}
""".strip()


def has_korean_language_drift(obj: Any) -> bool:
    """Detect Chinese/Japanese/Hanja-like drift in Korean output without being overly strict."""
    text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    # CJK Unified Ideographs.  Korean docs can contain rare Hanja, but generated modern meeting notes should not.
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    # Japanese Kana.
    kana = re.findall(r"[\u3040-\u30ff]", text)
    if kana:
        return True
    # One accidental Chinese char inside Korean transliteration is enough to trigger repair in ko mode.
    if cjk:
        return True
    # Raw literal leakage / key dump often reads badly even if parse succeeded.
    if re.search(r"\{\s*['\"]?(heading|bullets|summary|text)['\"]?\s*:", text):
        return True
    return False


def repair_final_korean(
    llm,
    final_obj: dict,
    notes: list[dict],
    segments: list[dict],
    title: str,
    language: str,
    detail_level: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict | None:
    """One extra GPU-friendly repair pass when final JSON is parseable but linguistically poor."""
    if language != "ko":
        return None
    cfg = detail_cfg(detail_level)
    blocks = chronological_blocks(segments, block_max_chars=1100, max_blocks=18 if detail_level == "detailed" else 12, max_total_chars=20000)
    transcript_digest = blocks_to_prompt_text(blocks)
    current = json.dumps(final_obj, ensure_ascii=False)[:30000]
    support = json.dumps(notes, ensure_ascii=False)[:18000]
    prompt = f"""
아래 기존 최종 JSON은 형식은 맞지만, 일부 표현이 어색하거나 중국어/한자/직역체/반복/빈약한 항목이 섞였을 수 있습니다.
원문 transcript digest와 chunk note를 참고하여 같은 JSON 구조로 다시 작성하세요.

문서 제목 후보: {title}
문서 상세도: {cfg['label']}

수정 원칙:
- 반드시 자연스러운 한국어로 작성하세요. 중국어·일본어·한자식 표현을 제거하세요.
- 내용은 general domain 녹음/회의 정리로 작성하고, 특정 샘플이나 산업에 편향하지 마세요.
- 원문에 없는 사실을 추가하지 마세요.
- 의미 있는 주제, 흐름, 핵심 논점이 보이도록 정리하세요.
- '명시적으로 확인되지 않음'을 남발하지 마세요.
- JSON 객체 하나만 반환하세요.

반환 JSON 구조는 기존 JSON과 동일합니다.

원문 transcript digest:
{transcript_digest}

chunk note 참고:
{support}

기존 최종 JSON:
{current}
""".strip()
    if log_cb:
        log_cb(f"🛠️ 한국어 품질 보정 LLM 생성 / max_new_tokens={max_new_tokens}")
    raw = llm.generate(SYSTEM_PROMPT, prompt, max_new_tokens=max_new_tokens)
    parsed = parse_json_tolerant(raw) if not has_template_leak(raw) else None
    if not parsed:
        return None
    candidate = sanitize_final(parsed, detail_level)
    if candidate and not has_korean_language_drift(candidate):
        return candidate
    # If it still has drift but otherwise has richer structure, return None and keep existing safe path.
    return None

def call_llm_json(
    llm,
    prompt: str,
    segments: list[dict],
    max_new_tokens: int,
    label: str,
    detail_level: str = "standard",
    log_cb: Optional[Callable[[str], None]] = None,
    retries: int = 2,
) -> dict:
    raw_last = ""
    for attempt in range(1, retries + 1):
        if log_cb:
            log_cb(f"🤖 LLM 생성: {label} / attempt {attempt} / max_new_tokens={max_new_tokens}")
        raw = llm.generate(SYSTEM_PROMPT, prompt, max_new_tokens=max_new_tokens)
        raw_last = raw
        parsed = None if has_template_leak(raw) else parse_json_tolerant(raw)
        if parsed:
            note = sanitize_note(parsed, segments, detail_level)
            if note.get("summary_bullets") or note.get("topics") or note.get("timeline"):
                return note
        if log_cb and attempt < retries:
            log_cb("⚠️ JSON 파싱 실패/반복/정보량 부족. 더 짧고 명확한 JSON으로 재시도합니다.")
        prompt = prompt + "\n\n중요: JSON만 반환하세요. evidence는 항목당 1개만 쓰고, topics와 bullets를 줄여서 완성된 JSON으로 답하세요. 같은 문구를 반복하지 마세요."
    if log_cb:
        log_cb("⚠️ 완전한 JSON 생성 실패. LLM 원문 출력에서 유용한 내용을 복구합니다.")
    return loose_note_from_raw(raw_last, segments, detail_level)


def empty_final() -> dict:
    return {
        "one_page_summary": [],
        "overview": [],
        "structured_notes": [],
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "risks_issues": [],
        "timeline": [],
        "key_quotes": [],
        "terms": [],
        "open_questions": [],
    }


def add_unique(out: dict, key: str, item: Any, limit: int, seen: set[str]) -> None:
    text = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
    text = collapse_repeated_phrases(text)
    marker = re.sub(r"\s+", " ", text)[:220]
    if marker in seen or not text.strip() or has_template_leak(text):
        return
    seen.add(marker)
    if len(out[key]) < limit:
        out[key].append(item)


def aggregate_without_final_llm(notes: list[dict], detail_level: str = "standard") -> dict:
    cfg = detail_cfg(detail_level)
    out = empty_final()
    seen: set[str] = set()
    for n in notes:
        for b in n.get("summary_bullets", [])[: cfg["summary_bullets"]]:
            add_unique(out, "one_page_summary", b, cfg["summary_bullets"] * 2, seen)
        for t in n.get("topics", [])[: cfg["chunk_topics"] + 2]:
            add_unique(out, "structured_notes", t, cfg["structured_limit"], seen)
            for b in as_list(t.get("bullets"))[:2]:
                add_unique(out, "key_points", {"point": b, "evidence": t.get("evidence", [])}, cfg["structured_limit"], seen)
        for key, limit in [
            ("decisions", 40), ("action_items", 40), ("risks_issues", 40), ("open_questions", 40),
            ("timeline", cfg["timeline_limit"]), ("key_quotes", cfg["quotes_limit"]), ("terms", 40),
        ]:
            for item in n.get(key, [])[:limit]:
                add_unique(out, key, item, limit, seen)
    if not out["one_page_summary"]:
        out["one_page_summary"] = ["요약을 생성할 수 있는 충분한 LLM 출력이 부족했습니다. transcript를 확인하세요."]
    if not out["overview"] and out["structured_notes"]:
        for t in out["structured_notes"][:5]:
            if isinstance(t, dict) and t.get("heading"):
                out["overview"].append(str(t["heading"]))
    return out


def sanitize_final(obj: dict, detail_level: str = "standard") -> dict:
    out = empty_final()
    limits = {
        "one_page_summary": detail_cfg(detail_level)["summary_bullets"] * 2,
        "overview": 12,
        "structured_notes": detail_cfg(detail_level)["structured_limit"],
        "key_points": 50,
        "decisions": 40,
        "action_items": 40,
        "risks_issues": 40,
        "timeline": detail_cfg(detail_level)["timeline_limit"],
        "key_quotes": detail_cfg(detail_level)["quotes_limit"],
        "terms": 40,
        "open_questions": 40,
    }
    if not isinstance(obj, dict):
        return out
    seen_by_key: dict[str, set[str]] = {k: set() for k in out}
    for k in out:
        for item in as_list(obj.get(k)):
            cleaned_item: Any = None
            if isinstance(item, str):
                s = clean_item_text(item, 1600 if k == "one_page_summary" else 1000)
                if s and not is_poor_bullet(s):
                    cleaned_item = s
            elif isinstance(item, dict) and k in {"one_page_summary", "overview"}:
                s = clean_item_text(item, 1400)
                if s and not is_poor_bullet(s):
                    cleaned_item = s
            elif isinstance(item, dict):
                cleaned = {}
                for kk, vv in item.items():
                    if kk == "evidence":
                        cleaned[kk] = compact_evidence(vv, 2)
                    elif isinstance(vv, list):
                        vals = []
                        local_seen = set()
                        for x in vv:
                            cx = clean_item_text(x, 700)
                            if cx and cx[:120] not in local_seen:
                                vals.append(cx)
                                local_seen.add(cx[:120])
                        cleaned[kk] = vals
                    else:
                        cleaned[kk] = clean_item_text(vv, 700)
                if k == "terms":
                    term_name = clean_item_text(cleaned.get("term") or cleaned.get("name"), 120)
                    term_desc = clean_item_text(cleaned.get("description") or cleaned.get("desc"), 500)
                    if is_low_value_term(term_name) or not term_desc or term_desc in {"녹음 구간에서 반복적으로 언급된 핵심 표현", "명시적으로 확인되지 않음"}:
                        cleaned_item = None
                    else:
                        cleaned["term"] = term_name
                        cleaned["description"] = term_desc
                        cleaned_item = cleaned
                elif not has_template_leak(json.dumps(cleaned, ensure_ascii=False)):
                    if k == "structured_notes":
                        bullets = as_list(cleaned.get("bullets") or cleaned.get("details"))
                        bullets = [clean_item_text(b, 700) for b in bullets]
                        bullets = [b for b in bullets if b and not is_poor_bullet(b) and b != "명시적으로 확인되지 않음"]
                        if bullets:
                            heading = clean_item_text(cleaned.get("heading") or cleaned.get("title"), 120)
                            if is_generic_heading(heading):
                                heading = derive_heading_from_bullets(bullets)
                            cleaned["heading"] = heading
                            cleaned["bullets"] = bullets
                            cleaned_item = cleaned
                        else:
                            cleaned_item = None
                    elif k == "key_points":
                        point = clean_item_text(cleaned.get("point") or cleaned.get("text"), 700)
                        if point and not is_poor_bullet(point) and point != "명시적으로 확인되지 않음" and not is_low_value_term(point):
                            cleaned["point"] = point
                            cleaned_item = cleaned
                        else:
                            cleaned_item = None
                    elif k in {"decisions", "risks_issues", "open_questions"}:
                        text_val = clean_item_text(cleaned.get("text") or cleaned.get("point") or cleaned.get("issue") or cleaned.get("question"), 700)
                        if text_val and not is_poor_bullet(text_val) and text_val != "명시적으로 확인되지 않음":
                            cleaned["text"] = text_val
                            cleaned_item = cleaned
                        else:
                            cleaned_item = None
                    else:
                        cleaned_item = cleaned
            if cleaned_item is not None:
                marker = json.dumps(cleaned_item, ensure_ascii=False, sort_keys=True)[:220] if isinstance(cleaned_item, dict) else str(cleaned_item)[:220]
                if marker not in seen_by_key[k]:
                    out[k].append(cleaned_item)
                    seen_by_key[k].add(marker)
            if len(out[k]) >= limits.get(k, 40):
                break
    for item in as_list(obj.get("asr_uncertainties") or obj.get("transcription_uncertainties") or obj.get("uncertain_terms")):
        if isinstance(item, dict):
            text = clean_item_text(item.get("text") or item.get("term") or item.get("uncertainty") or item.get("question"), 800)
            ev = compact_evidence(item.get("evidence"), 2)
        else:
            text, ev = clean_item_text(item, 800), []
        if text and not is_poor_bullet(text):
            if not text.startswith("ASR 확인 필요"):
                text = "ASR 확인 필요: " + text
            out["open_questions"].append({"text": text, "evidence": ev})
    if not out["one_page_summary"] and obj.get("summary_bullets"):
        out["one_page_summary"] = [clean_item_text(x, 600) for x in as_list(obj.get("summary_bullets"))[:8] if clean_item_text(x, 600)]
    return out


def final_text_corpus(final_obj: dict) -> str:
    return json.dumps(final_obj or {}, ensure_ascii=False)


def looks_repetitive_or_sparse(final_obj: dict) -> bool:
    text = final_text_corpus(final_obj)
    if len(text) < 600:
        return True
    # Too many repeated phrases/sections.
    phrases = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9 ]{3,30}", text)
    if len(phrases) > 20:
        top, cnt = Counter(phrases).most_common(1)[0]
        if cnt >= 8 and cnt / len(phrases) > 0.12:
            return True
    if text.count("명시적으로 확인되지 않음") >= 8 and len(final_obj.get("structured_notes", [])) < 5:
        return True
    # Very long one-page summary with little unique content.
    summary = " ".join(str(x) for x in as_list(final_obj.get("one_page_summary")))
    if len(summary) > 800:
        parts = [p.strip() for p in re.split(r"[,，.。]\s*", summary) if p.strip()]
        if parts and len(set(parts)) / len(parts) < 0.55:
            return True
    return False


def is_generic_heading(text: str) -> bool:
    h = clean_item_text(text, 120).lower()
    if not h:
        return True
    if h in {"주요 논의", "주요 주제", "녹음/회의 개요"}:
        return True
    if re.fullmatch(r"주요 (논의|주제) \d+(:.*)?", h):
        # Allow only if it contains a meaningful named entity/acronym after ':'
        tail = h.split(":", 1)[-1].strip() if ":" in h else ""
        kws = [x.strip() for x in re.split(r"[·, /]+", tail) if x.strip()]
        return not any(not is_low_value_term(x) and len(x) >= 2 for x in kws)
    return False


def derive_heading_from_bullets(bullets: list[Any], fallback: str = "주요 내용") -> str:
    """Create a specific human-readable heading from bullet content.

    Used when the LLM returns generic headings such as '주요 논의 1'.
    This is intentionally general-domain and based only on the supplied bullets.
    """
    texts = [clean_item_text(b, 300) for b in as_list(bullets)]
    texts = [t for t in texts if t]
    kws = top_keywords(texts, 4)
    if kws:
        label = " · ".join(k.upper() if k.isascii() and len(k) <= 6 else k for k in kws[:3])
        return label[:80]
    for t in texts:
        if len(t) >= 12:
            # Prefer a concise noun-like phrase before punctuation.
            cand = re.split(r"[.!?。]| 그리고 | 또한 | 다만 | 때문에 ", t)[0].strip()
            return cand[:70] or fallback
    return fallback

def has_unwanted_cjk(obj: Any) -> bool:
    text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    # Hangul is not in this range. This detects Chinese Han characters and Japanese kana.
    return bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff]", text))

def excessive_unknowns(obj: dict) -> bool:
    text = json.dumps(obj or {}, ensure_ascii=False)
    return text.count("명시적으로 확인되지 않음") >= 8 or text.count("확인 필요") >= 18


def make_one_page_paragraph(final_obj: dict, title: str = "") -> str:
    topics = []
    for t in as_list(final_obj.get("structured_notes"))[:6]:
        if isinstance(t, dict):
            h = clean_item_text(t.get("heading"), 100)
            if h and not is_generic_heading(h):
                topics.append(h)
    facts = []
    for t in as_list(final_obj.get("structured_notes"))[:6]:
        if isinstance(t, dict):
            for b in as_list(t.get("bullets"))[:2]:
                text = clean_item_text(b, 220)
                if text and text[:100] not in {x[:100] for x in facts}:
                    facts.append(text)
    for kp in as_list(final_obj.get("key_points"))[:6]:
        text = clean_item_text(kp.get("point") if isinstance(kp, dict) else kp, 220)
        if text and text[:100] not in {x[:100] for x in facts}:
            facts.append(text)
    topic_part = ", ".join(topics[:4]) if topics else "주요 논의 흐름"
    if facts:
        s1 = f"이 녹음은 {topic_part}을 중심으로 진행됩니다."
        s2 = " ".join(facts[:3])
        s3 = "이후 세부 정리에서는 관련 배경, 핵심 논점, 시간순 흐름과 확인이 필요한 내용을 나누어 정리했습니다."
        return clean_item_text(f"{s1} {s2} {s3}", 1200)
    return f"{title or '이 녹음'}의 주요 내용을 전사 결과에 근거해 정리했습니다. 세부 내용은 아래 주제별 정리와 타임라인을 확인하세요."

def enrich_final_with_chunk_notes(final_obj: dict, notes: list[dict], detail_level: str = "standard", title: str = "") -> dict:
    cfg = detail_cfg(detail_level)
    aggregate = aggregate_without_final_llm(notes, detail_level)
    enriched = final_obj or empty_final()
    min_targets = {
        "structured_notes": min(len(aggregate["structured_notes"]), max(5, cfg["chunk_topics"])),
        "key_points": min(len(aggregate["key_points"]), 10),
        "timeline": min(len(aggregate["timeline"]), 8),
        "key_quotes": min(len(aggregate["key_quotes"]), 5),
    }
    seen: set[str] = set()
    for k, vals in enriched.items():
        for v in as_list(vals):
            seen.add(json.dumps(v, ensure_ascii=False, sort_keys=True)[:220] if isinstance(v, dict) else str(v)[:220])
    for key, target in min_targets.items():
        if len(enriched.get(key, [])) < target:
            for item in aggregate.get(key, []):
                add_unique(enriched, key, item, detail_cfg(detail_level).get("structured_limit", 60), seen)
                if len(enriched[key]) >= target:
                    break
    for key in ["decisions", "action_items", "risks_issues", "open_questions", "terms"]:
        if len(enriched.get(key, [])) < len(aggregate.get(key, [])):
            for item in aggregate.get(key, []):
                add_unique(enriched, key, item, 40, seen)
    # Create a prose one-page summary when LLM produced a repeated or list-like summary.
    summary = " ".join(str(x) for x in as_list(enriched.get("one_page_summary")))
    if not enriched.get("one_page_summary") or looks_repetitive_or_sparse({"one_page_summary": enriched.get("one_page_summary", []), "structured_notes": enriched.get("structured_notes", [])}):
        enriched["one_page_summary"] = [make_one_page_paragraph(enriched, title)]
    if not enriched.get("overview"):
        enriched["overview"] = aggregate.get("overview", [])[:6]
    return enriched


def repair_final_korean_style(
    llm,
    final_obj: dict,
    title: str,
    language: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
    detail_level: str = "standard",
) -> tuple[dict, bool]:
    """Lightweight style/language repair for GPU profiles.

    This does not add new facts. It only rewrites already-produced values into
    natural Korean and removes mixed Chinese/Japanese characters or Python-literal
    artifacts when they appear.
    """
    if not str(language).lower().startswith("ko"):
        return final_obj, False
    if not (has_unwanted_cjk(final_obj) or excessive_unknowns(final_obj) or looks_repetitive_or_sparse(final_obj)):
        return final_obj, False
    compact = json.dumps(final_obj, ensure_ascii=False)[:32000]
    prompt = f"""
다음 JSON은 회의록 초안입니다. 새 사실을 추가하지 말고, 기존 의미를 유지하면서 사람이 읽기 쉬운 자연스러운 한국어로만 다듬어 주세요.
문서 제목 후보: {title}

수정 원칙:
- 중국어·일본어·한자식 문자가 섞인 표현은 한국어 또는 통용 영어 약어로 고치세요.
- ASR 오인식으로 보이는 고유명사는 문맥상 명확할 때만 보정하세요. 불확실하면 확인 필요한 내용으로 유지하세요.
- '명시적으로 확인되지 않음'을 불필요하게 반복하지 말고, 실제 내용이 있는 섹션은 내용 중심으로 정리하세요.
- structured_notes의 제목은 구체적이고 사람이 이해하기 쉬운 제목으로 바꾸세요.
- Python dict/list 문자열, heading/bullets 같은 키 이름을 본문 값으로 쓰지 마세요.
- JSON 구조와 key 이름은 유지하고 JSON 객체 하나만 반환하세요.

초안 JSON:
{compact}
""".strip()
    try:
        if log_cb:
            log_cb("🛠️ 최종 회의록 한국어 문체/언어 혼입 보정 시도")
        raw = llm.generate(SYSTEM_PROMPT, prompt, max_new_tokens=max(1800, min(max_new_tokens, 4500)))
        parsed = parse_json_tolerant(raw) if not has_template_leak(raw) else None
        if parsed:
            repaired = sanitize_final(parsed, detail_level)
            if repaired and len(final_text_corpus(repaired)) > 500:
                return repaired, True
    except Exception as e:
        if log_cb:
            log_cb(f"⚠️ 한국어 문체 보정 실패. 기존 결과를 사용합니다: {e}")
    return final_obj, False


def postprocess_final_quality(final_obj: dict, detail_level: str = "standard", title: str = "") -> dict:
    """General-domain final cleanup before Markdown/DOCX export.

    This keeps the document readable without rejecting too many LLM outputs:
    it removes low-value terms, avoids generic headings, and prevents empty evidence
    placeholders from dominating the report.
    """
    if not isinstance(final_obj, dict):
        return empty_final()
    cfg = detail_cfg(detail_level)
    out = dict(final_obj)

    # Structured notes: drop empty/generic topics and repair headings from their bullets.
    cleaned_topics = []
    seen_topics = set()
    for idx, t in enumerate(as_list(out.get("structured_notes")), start=1):
        if not isinstance(t, dict):
            continue
        bullets = []
        seen_b = set()
        for b in as_list(t.get("bullets")):
            text = clean_item_text(b, 700)
            if not text or is_weak_placeholder(text) or text in {"명시적으로 확인되지 않음", "세부 내용이 명시적으로 확인되지 않음"}:
                continue
            if text[:120] not in seen_b:
                bullets.append(text)
                seen_b.add(text[:120])
        if not bullets:
            continue
        heading = clean_item_text(t.get("heading") or t.get("title") or "", 100)
        if is_generic_heading(heading) or is_low_value_term(heading):
            heading = derive_heading_from_bullets(bullets, fallback=f"주요 내용 {idx}")
        heading = clean_item_text(heading, 100) or f"주요 내용 {idx}"
        marker = (heading + "|" + "|".join(bullets[:2]))[:220]
        if marker in seen_topics:
            continue
        seen_topics.add(marker)
        cleaned_topics.append({"heading": heading, "bullets": bullets[: cfg["topic_bullets"] + 2], "evidence": compact_evidence(t.get("evidence"), 2)})
        if len(cleaned_topics) >= cfg["structured_limit"]:
            break
    out["structured_notes"] = cleaned_topics

    # Overview: prefer human-readable topic headings.
    overview = []
    seen_o = set()
    for item in as_list(out.get("overview")):
        text = clean_item_text(item, 140)
        if text and not is_generic_heading(text) and not is_low_value_term(text) and text[:80] not in seen_o:
            overview.append(text)
            seen_o.add(text[:80])
    if not overview:
        for t in cleaned_topics[:7]:
            h = clean_item_text(t.get("heading"), 120)
            if h and h[:80] not in seen_o:
                overview.append(h)
                seen_o.add(h[:80])
    out["overview"] = overview[:8]

    # Key points: high-level points only; remove low-value / unknown placeholders.
    key_points = []
    seen_kp = set()
    for item in as_list(out.get("key_points")):
        if isinstance(item, dict):
            point = clean_item_text(item.get("point") or item.get("text"), 500)
            ev = compact_evidence(item.get("evidence"), 2)
        else:
            point, ev = clean_item_text(item, 500), []
        if not point or point in {"명시적으로 확인되지 않음", "확인 필요"} or is_weak_placeholder(point):
            continue
        # Avoid tiny filler points.
        if len(point) < 8 or is_low_value_term(point):
            continue
        if point[:140] in seen_kp:
            continue
        key_points.append({"point": point, "evidence": ev})
        seen_kp.add(point[:140])
        if len(key_points) >= 18:
            break
    if len(key_points) < 6:
        for t in cleaned_topics:
            for b in as_list(t.get("bullets"))[:2]:
                point = clean_item_text(b, 500)
                if point and point[:140] not in seen_kp:
                    key_points.append({"point": point, "evidence": compact_evidence(t.get("evidence"), 2)})
                    seen_kp.add(point[:140])
                if len(key_points) >= 12:
                    break
            if len(key_points) >= 12:
                break
    out["key_points"] = key_points

    # Terms: strict cleanup to prevent filler words in the glossary table.
    terms = []
    seen_terms = set()
    for term in as_list(out.get("terms")):
        if isinstance(term, dict):
            name = clean_item_text(term.get("term") or term.get("name"), 120)
            desc = clean_item_text(term.get("description") or term.get("desc"), 500)
        else:
            name, desc = clean_item_text(term, 120), ""
        if is_low_value_term(name) or not name or not desc or desc in {"녹음 구간에서 반복적으로 언급된 핵심 표현", "명시적으로 확인되지 않음"}:
            continue
        if name[:80] in seen_terms:
            continue
        terms.append({"term": name, "description": desc})
        seen_terms.add(name[:80])
        if len(terms) >= 15:
            break
    out["terms"] = terms

    # One-page summary: keep as a single prose paragraph.
    summary_text = " ".join(clean_item_text(x, 900) for x in as_list(out.get("one_page_summary")) if clean_item_text(x, 900))
    if not summary_text or looks_repetitive_or_sparse({"one_page_summary": [summary_text], "structured_notes": cleaned_topics}):
        summary_text = make_one_page_paragraph(out, title)
    out["one_page_summary"] = [clean_item_text(summary_text, 1500)]
    return out



def make_direct_markdown_prompt(
    title: str,
    final_obj: dict,
    notes: list[dict],
    segments: list[dict],
    language: str,
    glossary: str = "",
    detail_level: str = "standard",
    max_digest_chars: int = 22000,
) -> str:
    """Prompt for the final human-facing Markdown writer.

    v8: instead of forcing the last model call to emit another large JSON, ask it
    to write the actual report in Markdown from a grounded draft + time-ordered
    digest. This is more robust and produces less chunk-like prose.
    """
    cfg = detail_cfg(detail_level)
    max_blocks = 26 if detail_level == "detailed" else 18
    blocks = chronological_blocks(
        segments,
        block_max_chars=1150 if detail_level == "detailed" else 1000,
        max_blocks=max_blocks,
        max_total_chars=max_digest_chars,
    )
    transcript_digest = blocks_to_prompt_text(blocks)
    draft = json.dumps(final_obj, ensure_ascii=False)[:24000]
    support = json.dumps(notes, ensure_ascii=False)[:18000]
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
아래 자료를 바탕으로 사람이 읽기 좋은 최종 Markdown 회의록/녹음 정리 문서를 작성하세요.
문서 제목: {title}
출력 언어: {language}
문서 상세도: {cfg['label']}
{glossary_text}
작성 목표:
- 이 문서는 general domain 회의/강의/인터뷰/발표/설명 영상 정리용입니다.
- raw transcript 조각을 그대로 붙이지 말고, 의미 단위로 묶어 사람이 쓴 문서처럼 정리하세요.
- 시간순 digest를 우선 근거로 삼고, draft JSON과 chunk note는 보조 자료로만 사용하세요.
- 원문에 없는 사실을 만들지 마세요. 특히 숫자, 날짜, 돈, 회사명, 인물명, 결정사항은 transcript 근거가 불명확하면 확인 필요로 표시하세요.
- ASR 오인식이 명확한 경우에만 자연스럽게 보정하고, 확신이 없으면 확인 필요로 남기세요.
- 중국어·일본어·한자식 문자를 쓰지 말고 자연스러운 한국어 문장으로 작성하세요.
- '명시적으로 확인되지 않음'은 정말 없는 경우에만 최소한으로 사용하세요.
- 같은 의미를 반복하지 마세요.
- JSON, Python dict/list, 코드블록, 작성 지시문을 출력하지 마세요.

반드시 아래 Markdown 섹션 제목을 그대로 사용하세요:
# {title}

## 1. 한 페이지 요약
- bullet이 아니라 1개 또는 2개의 자연스러운 문단으로 작성하세요.

## 2. 전체 구조화 정리
- 문서 전체 흐름을 5~8개 bullet로 정리하세요.

## 3. 주제별 상세 정리
- 5~10개 주제로 나누어 ### 소제목과 bullet을 작성하세요.
- 각 주제는 원문 흐름을 재구성한 설명이어야 하며, 단순 발화 조각을 그대로 쓰지 마세요.

## 4. 핵심 개념 / 논점
- 표가 아니라 bullet 목록으로 작성하세요. 각 bullet은 '개념/논점: 설명' 형식으로 작성하세요.

## 5. 결정사항 / 결론
- 회의가 아닌 영상/강의라면 억지로 결정사항을 만들지 말고 주요 결론/시사점 중심으로 작성하세요.

## 6. 실행 항목
- 담당자/기한이 명확하지 않으면 '명시적 실행 항목 없음' 또는 '확인 필요'로 작성하세요.

## 7. 리스크 / 이슈
- 실제 리스크, 논란, 불확실성만 작성하세요.

## 8. 타임라인 / 진행 흐름
- 시간순으로 6~12개 bullet을 작성하세요. 가능한 경우 [HH:MM:SS]를 포함하세요.

## 9. 중요 발언 / 근거
- 의미 있는 발언 5~10개를 뽑고, 가능하면 시간 또는 segment 근거를 붙이세요.

## 10. 용어 / 개념
- 실제 고유명사/전문용어만 작성하세요. 일반 발화어는 넣지 마세요.

## 11. 확인 필요한 내용
- ASR 오인식 또는 사실 확인이 필요한 내용만 작성하세요.

시간순 transcript digest:
{transcript_digest}

구조화 draft JSON:
{draft}

chunk note 참고 자료:
{support}
""".strip()


def markdown_has_bad_artifacts(markdown: str) -> bool:
    if not markdown or len(markdown.strip()) < 900:
        return True
    if has_template_leak(markdown):
        return True
    if re.search(r"\{\s*['\"]?(heading|bullets|summary|text)['\"]?\s*:", markdown):
        return True
    if re.search(r"[\u3040-\u30ff]", markdown):
        return True
    if re.search(r"[\u3400-\u9fff]", markdown):
        # Any CJK in the human-facing Markdown is risky for Korean output.
        return True
    required = ["## 1. 한 페이지 요약", "## 3. 주제별 상세 정리", "## 8. 타임라인", "## 11. 확인 필요한 내용"]
    if not all(x in markdown for x in required):
        return True
    # Detect severe repetition.
    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", markdown)
    if len(words) > 100:
        top, cnt = Counter(words).most_common(1)[0]
        if cnt > 35 and cnt / len(words) > 0.08 and top not in {"엔비디아", "회의", "내용", "한국"}:
            return True
    return False


def cleanup_markdown(markdown: str, title: str) -> str:
    md = normalize_language_artifacts(markdown or "")
    md = strip_code_fence(md)
    # Remove accidental preamble before first title.
    if f"# {title}" in md:
        md = md[md.find(f"# {title}"):]
    elif md.lstrip().startswith("#"):
        pass
    else:
        md = f"# {title}\n\n" + md
    # Remove any accidental run_config-like appendix.
    md = re.sub(r"\n+##\s*부록\s*A\..*?(?=\n##\s|\Z)", "", md, flags=re.S)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def generate_direct_markdown(
    llm,
    title: str,
    final_obj: dict,
    notes: list[dict],
    segments: list[dict],
    language: str,
    glossary: str,
    detail_level: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str | None, bool]:
    prompt = make_direct_markdown_prompt(title, final_obj, notes, segments, language, glossary, detail_level)
    try:
        if log_cb:
            log_cb(f"✍️ 최종 사람용 Markdown 작성 LLM 생성 / max_new_tokens={max_new_tokens}")
        raw = llm.generate(SYSTEM_PROMPT_MARKDOWN, prompt, max_new_tokens=max_new_tokens)
        md = cleanup_markdown(raw, title)
        if not markdown_has_bad_artifacts(md):
            return md, True
        if log_cb:
            log_cb("⚠️ Markdown writer 결과에 언어 혼입/반복/형식 문제가 있어 구조화 Markdown으로 대체합니다.")
    except Exception as e:
        if log_cb:
            log_cb(f"⚠️ Markdown writer 실패. 구조화 Markdown으로 대체합니다: {e}")
    return None, False


def ground_final_against_transcript(final_obj: dict, segments: list[dict], detail_level: str = "standard") -> dict:
    """Remove the riskiest hallucinated statements without being overly strict."""
    idx = transcript_text_index(segments)
    out = dict(final_obj or empty_final())
    for key in ["structured_notes"]:
        cleaned_topics = []
        for t in as_list(out.get(key)):
            if not isinstance(t, dict):
                continue
            bullets = []
            for b in as_list(t.get("bullets")):
                text = clean_item_text(b, 700)
                if text and not is_statement_suspicious(text, idx):
                    bullets.append(text)
            if bullets:
                heading = clean_item_text(t.get("heading") or derive_heading_from_bullets(bullets), 120)
                cleaned_topics.append({"heading": heading, "bullets": bullets, "evidence": compact_evidence(t.get("evidence"), 2)})
        out[key] = cleaned_topics
    for key in ["key_points", "decisions", "risks_issues", "open_questions"]:
        vals = []
        for item in as_list(out.get(key)):
            if isinstance(item, dict):
                text = clean_item_text(item.get("point") or item.get("text") or item.get("issue") or item.get("question"), 700)
                if text and not is_statement_suspicious(text, idx):
                    item = dict(item)
                    if key == "key_points":
                        item["point"] = text
                    else:
                        item["text"] = text
                    vals.append(item)
            else:
                text = clean_item_text(item, 700)
                if text and not is_statement_suspicious(text, idx):
                    vals.append(text)
        out[key] = vals
    return out

def effective_strategy(requested: str, profile: RuntimeProfile) -> str:
    requested = (requested or "auto").lower()
    if requested in {"fast", "smart_fast"}:
        return "fast"
    if requested in {"full", "extractive"}:
        return requested
    # CPU transformers generation is slow. Auto therefore uses v8 fast mode:
    # chronological transcript digest + direct Markdown writer + extractive safety net.
    # GPU keeps the richer full chunk-LLM pipeline plus a final Markdown writer.
    if profile.llm_device == "cpu":
        return "fast"
    return "full"


def summarize_segments(
    segments: list[dict],
    title: str,
    profile: RuntimeProfile,
    language: str = "ko",
    glossary: str = "",
    allow_download: bool = True,
    use_final_llm: bool = True,
    detail_level: str = "detailed",
    processing_strategy: str = "auto",
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    cfg = detail_cfg(detail_level)
    strategy = effective_strategy(processing_strategy, profile)
    chunks = chunk_segments(segments, profile.max_chars_per_chunk, profile.chunk_overlap_chars)
    is_cpu = profile.llm_device == "cpu"
    # CPU must stay practical: v8 fast uses one direct Markdown writer call, so
    # token budget is focused on the final human-facing document.
    chunk_tokens = max(700, int(profile.max_new_tokens_chunk * cfg["token_multiplier"]))
    final_tokens = max(1000, int(profile.max_new_tokens_final * cfg["token_multiplier"]))
    if is_cpu:
        chunk_tokens = min(chunk_tokens, 1000)
        # v7 fast mode uses only one LLM synthesis call on CPU, so give that call
        # enough room to produce a useful document while keeping runtime practical.
        if detail_level == "detailed":
            final_tokens = min(max(final_tokens, 3000), 3800)
        elif detail_level == "standard":
            final_tokens = min(max(final_tokens, 2600), 3300)
        else:
            final_tokens = min(max(final_tokens, 2000), 2600)
    notes: list[dict] = []
    final_markdown: str | None = None
    final_markdown_used = False
    fallback_used = False
    final_llm_failed = False
    final_repair_used = False
    llm_calls = 0
    style_repair_used = False
    language_drift_detected = False

    if log_cb:
        log_cb(f"🧩 transcript chunk 수: {len(chunks)} / chunk_chars={profile.max_chars_per_chunk} / overlap={profile.chunk_overlap_chars}")
        log_cb(f"📝 문서 상세도: {detail_level} ({cfg['label']}) / 처리 전략={strategy} / chunk_tokens={chunk_tokens} / final_tokens={final_tokens}")

    if strategy == "extractive":
        notes = extractive_notes_for_chunks(chunks, detail_level)
        final_obj = aggregate_without_final_llm(notes, detail_level)
        fallback_used = True
    elif strategy == "fast":
        # v8 fast mode: build grounded extractive notes, then ask the LLM to write
        # the human-facing Markdown directly from a chronological digest. This keeps
        # CPU runtime much closer to one LLM call while avoiding chunk-like DOCX.
        notes = extractive_notes_for_chunks(chunks, detail_level)
        final_obj = enrich_final_with_chunk_notes(aggregate_without_final_llm(notes, detail_level), notes, detail_level, title)
        if use_final_llm:
            try:
                llm = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
                md, used = generate_direct_markdown(
                    llm, title, final_obj, notes, segments, language, glossary, detail_level,
                    max_new_tokens=max(2200, final_tokens), log_cb=log_cb
                )
                llm_calls += 1
                if used:
                    final_markdown = md
                    final_markdown_used = True
                else:
                    # If the Markdown writer fails, use the older JSON synthesis as a fallback.
                    prompt = make_fast_final_prompt(title, notes, segments, language, glossary, detail_level)
                    if log_cb:
                        log_cb(f"🤖 빠른 JSON 정리 fallback 생성 / max_new_tokens={final_tokens}")
                    raw = llm.generate(SYSTEM_PROMPT, prompt, max_new_tokens=final_tokens)
                    llm_calls += 1
                    parsed = parse_json_tolerant(raw) if not has_template_leak(raw) else None
                    if parsed:
                        candidate = sanitize_final(parsed, detail_level)
                        candidate = enrich_final_with_chunk_notes(candidate, notes, detail_level, title)
                        final_obj = candidate if not looks_repetitive_or_sparse(candidate) else final_obj
                    else:
                        final_llm_failed = True
                        if log_cb:
                            log_cb("⚠️ 빠른 JSON 정리 fallback 파싱 실패. 추출 기반 구조화 결과를 사용합니다.")
            except Exception as e:
                final_llm_failed = True
                if log_cb:
                    log_cb(f"⚠️ 빠른 최종 정리 LLM 실패. 추출 기반으로 진행합니다: {e}")
        else:
            final_obj = enrich_final_with_chunk_notes(final_obj, notes, detail_level, title)
    else:
        # Full mode: LLM extraction per chunk + optional final LLM merge. Recommended for GPU.
        llm = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
        for i, ch in enumerate(chunks, start=1):
            prompt = make_chunk_prompt(title, i, len(chunks), ch, language, glossary, detail_level)
            note = call_llm_json(llm, prompt, ch, chunk_tokens, f"{title}_chunk_{i:02d}", detail_level, log_cb, retries=2)
            llm_calls += 1
            if note.get("topics") and note["topics"][0].get("heading") in {"원문 기반 주요 내용", "LLM 출력 복구 내용"}:
                fallback_used = True
            notes.append(note)
        final_obj = None
        if use_final_llm:
            try:
                final_prompt = make_final_prompt(title, notes, segments, language, glossary, detail_level)
                if log_cb:
                    log_cb(f"🤖 최종 병합 LLM 생성 / max_new_tokens={final_tokens}")
                raw = llm.generate(SYSTEM_PROMPT, final_prompt, max_new_tokens=final_tokens)
                llm_calls += 1
                parsed = parse_json_tolerant(raw) if not has_template_leak(raw) else None
                if parsed:
                    final_obj = sanitize_final(parsed, detail_level)
                else:
                    final_llm_failed = True
                    if log_cb:
                        log_cb("⚠️ 최종 병합 JSON 파싱 실패. chunk note 기반 병합을 보강 사용합니다.")
            except Exception as e:
                final_llm_failed = True
                if log_cb:
                    log_cb(f"⚠️ 최종 병합 LLM 실패. chunk note 기반으로 병합합니다: {e}")
        if not final_obj:
            final_obj = aggregate_without_final_llm(notes, detail_level)
        final_obj = enrich_final_with_chunk_notes(final_obj, notes, detail_level, title)
        if looks_repetitive_or_sparse(final_obj):
            if log_cb:
                log_cb("⚠️ 최종 문서가 반복적/빈약해 보여 chunk note 기반으로 추가 보강합니다.")
            final_obj = enrich_final_with_chunk_notes(final_obj, notes, detail_level, title)
            # If still sparse, fall back to aggregate notes, but do not discard the transcript-aware final result too early.
            if looks_repetitive_or_sparse(final_obj):
                final_obj = enrich_final_with_chunk_notes(aggregate_without_final_llm(notes, detail_level), notes, detail_level, title)
                fallback_used = True

    # v7 GPU quality guard: if the final report still contains CJK drift or looks sparse,
    # run one extra Korean repair pass.  This is intentionally limited to non-CPU full/fast
    # LLM paths so CPU speed remains practical.
    if use_final_llm and profile.llm_device != "cpu" and strategy in {"full", "fast"}:
        needs_repair = (language == "ko" and has_korean_language_drift(final_obj)) or looks_repetitive_or_sparse(final_obj)
        if needs_repair:
            try:
                llm_for_repair = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
                repaired = repair_final_korean(
                    llm_for_repair,
                    final_obj,
                    notes,
                    segments,
                    title,
                    language,
                    detail_level,
                    max_new_tokens=max(2500, min(final_tokens, 5600)),
                    log_cb=log_cb,
                )
                llm_calls += 1
                if repaired:
                    final_obj = enrich_final_with_chunk_notes(repaired, notes, detail_level, title)
                    final_repair_used = True
                    if log_cb:
                        log_cb("✅ 한국어 품질 보정 결과를 최종 문서에 반영했습니다.")
            except Exception as e:
                if log_cb:
                    log_cb(f"⚠️ 한국어 품질 보정 LLM 실패. 기존 보강 결과를 사용합니다: {e}")

    # GPU/full outputs occasionally contain mixed CJK characters or become too sparse/repetitive.
    # Perform one lightweight style repair only on GPU profiles to preserve CPU speed.
    language_drift_detected = has_unwanted_cjk(final_obj)
    if use_final_llm and profile.llm_device == "cuda" and (language_drift_detected or excessive_unknowns(final_obj) or looks_repetitive_or_sparse(final_obj)):
        try:
            llm = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
            repaired, used = repair_final_korean_style(llm, final_obj, title, language, final_tokens, log_cb, detail_level)
            llm_calls += 1
            if used:
                final_obj = enrich_final_with_chunk_notes(repaired, notes, detail_level, title)
                style_repair_used = True
        except Exception as e:
            if log_cb:
                log_cb(f"⚠️ 최종 한국어 문체 보정 단계 오류. 기존 결과를 사용합니다: {e}")

    # v8 factual/language guard before export: remove the riskiest ungrounded claims,
    # then make the structured object readable.
    final_obj = ground_final_against_transcript(final_obj, segments, detail_level)
    final_obj = postprocess_final_quality(final_obj, detail_level, title)

    # GPU profiles get one final writer pass that produces the actual human-facing
    # Markdown.  This avoids a DOCX that reads like stitched chunk notes.
    if use_final_llm and final_markdown is None and profile.llm_device == "cuda":
        try:
            llm_writer = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
            md, used = generate_direct_markdown(
                llm_writer, title, final_obj, notes, segments, language, glossary, detail_level,
                max_new_tokens=max(2800, min(final_tokens, 6200)), log_cb=log_cb
            )
            llm_calls += 1
            if used:
                final_markdown = md
                final_markdown_used = True
        except Exception as e:
            if log_cb:
                log_cb(f"⚠️ 최종 Markdown writer 단계 오류. 구조화 Markdown으로 진행합니다: {e}")

    transcript_chars = sum(len(s.get("text", "")) for s in segments)
    markdown_est_chars = len(json.dumps(final_obj, ensure_ascii=False))
    run_config = {
        "pipeline_version": PIPELINE_VERSION,
        "title": title,
        "detail_level": detail_level,
        "processing_strategy_requested": processing_strategy,
        "processing_strategy_effective": strategy,
        "processing_strategy_note": "v8 fast = chronological transcript digest + direct human Markdown writer + extractive safety net" if strategy == "fast" else "v8 full = chunk LLM extraction + transcript-aware synthesis + final human Markdown writer",
        "profile_name": profile.name,
        "asr_model": profile.asr_model,
        "asr_device": profile.asr_device,
        "asr_compute_type": profile.asr_compute_type,
        "llm_model": profile.llm_model,
        "llm_device": profile.llm_device,
        "max_chars_per_chunk": profile.max_chars_per_chunk,
        "chunk_overlap_chars": profile.chunk_overlap_chars,
        "max_new_tokens_chunk_effective": chunk_tokens,
        "max_new_tokens_final_effective": final_tokens,
        "chunk_count": len(chunks),
        "segment_count": len(segments),
        "transcript_chars": transcript_chars,
        "structured_json_chars": markdown_est_chars,
        "asr_error_aware": True,
        "glossary_provided": bool(glossary.strip()),
        "use_final_llm": use_final_llm,
        "llm_calls": llm_calls,
        "final_llm_failed": final_llm_failed,
        "final_repair_used": final_repair_used,
        "style_repair_used": style_repair_used,
        "final_markdown_used": final_markdown_used,
        "fallback_used": fallback_used,
    }
    return {"chunk_notes": notes, "final": final_obj, "final_markdown": final_markdown, "chunk_count": len(chunks), "run_config": run_config}
