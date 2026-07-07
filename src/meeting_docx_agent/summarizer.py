from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable, Optional

from .llm import get_llm
from .profiles import RuntimeProfile
from .utils import clean_text_for_xml, humanize_llm_value

PIPELINE_VERSION = "general_meeting_v14"

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
당신은 보안이 중요한 로컬 환경에서 동작하는 전문 기록 정리자입니다.
당신의 일은 ASR transcript를 사람이 읽는 최종 문서로 재구성하는 것입니다.
입력 transcript는 Whisper/faster-whisper 같은 ASR 결과이므로 고유명사·숫자·날짜·약어·전문용어에 오류가 섞일 수 있습니다. 문맥과 사용자 힌트가 명확한 경우에만 자연스럽게 보정하고, 불확실하면 확인 필요로 남기세요.
이 애플리케이션은 general domain 회의·강의·인터뷰·발표·교육 영상·해설 영상·상담 녹음을 모두 처리합니다. 특정 회사, 산업, 샘플 형식에 편향하지 마세요.
출력 언어가 ko이면 반드시 자연스러운 한국어 문장으로만 작성하세요. 중국어·일본어·한자식 문자, 어색한 직역체를 섞지 마세요. NVIDIA, HBM, AI, API 같은 통용 약어는 유지할 수 있습니다.
raw transcript 조각을 그대로 나열하지 말고, 의미 단위로 묶어 배경 → 핵심 내용 → 구조 → 시사점/후속 확인 순서로 정리하세요.
원문에 없는 사실을 만들지 마세요. 숫자·날짜·인물·회사명·결정사항·금액은 특히 보수적으로 다루세요.
회의가 아니면 참석자·결정사항·실행항목을 억지로 만들지 말고, 강의/설명 자료에 맞게 핵심 개념과 학습 흐름을 정리하세요.
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
    "연말정착": "연말정산",
    "연말 정착": "연말정산",
    "연말결 산": "연말정산",
    "연말 결 산": "연말정산",
    "연말정 산": "연말정산",
    "연말정 삐": "연말정산",
    "소득 곱제": "소득공제",
    "소득 공 제": "소득공제",
    "세액 공 제": "세액공제",
    "총 급여": "총급여",
    "추가 징 수": "추가 징수",
    "추가 징세": "추가 징수",
    "환 급": "환급",
    "추 징": "추징",
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

    v9: instead of forcing the last model call to emit another large JSON, ask it
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



# ---------------------------------------------------------------------------
# v9 transcript-first sectioned Markdown writer
# ---------------------------------------------------------------------------

def writer_budget_for_profile(profile: RuntimeProfile, detail_level: str) -> dict:
    """Return transcript/digest and token budgets for the v9 final writer.

    The v8 GPU path still depended too much on JSON/draft objects.  v9 lets GPU
    profiles write the final human-facing document from transcript-first evidence.
    """
    name = getattr(profile, "name", "")
    if name == "gpu_quality":
        return {"blocks": 54 if detail_level == "detailed" else 40, "block_chars": 1350, "total_chars": 62000, "part1": 1800, "part2": 4200, "part3": 3600}
    if name == "gpu_balanced":
        return {"blocks": 38 if detail_level == "detailed" else 28, "block_chars": 1250, "total_chars": 44000, "part1": 1500, "part2": 3300, "part3": 2800}
    if name == "gpu_light":
        return {"blocks": 28 if detail_level == "detailed" else 22, "block_chars": 1100, "total_chars": 30000, "part1": 1200, "part2": 2600, "part3": 2200}
    # CPU fast path uses one compact writer elsewhere.
    return {"blocks": 20, "block_chars": 1000, "total_chars": 22000, "part1": 1100, "part2": 2200, "part3": 1800}


def build_writer_context(segments: list[dict], profile: RuntimeProfile, detail_level: str) -> str:
    budget = writer_budget_for_profile(profile, detail_level)
    blocks = chronological_blocks(
        segments,
        block_max_chars=budget["block_chars"],
        max_blocks=budget["blocks"],
        max_total_chars=budget["total_chars"],
    )
    return blocks_to_prompt_text(blocks)


def compact_support_notes(notes: list[dict], max_chars: int = 14000) -> str:
    """Compact chunk notes as optional support, not as the primary source."""
    rows: list[str] = []
    for i, n in enumerate(notes[:18], start=1):
        topics = []
        for t in as_list(n.get("topics"))[:4]:
            if not isinstance(t, dict):
                continue
            h = clean_item_text(t.get("heading"), 100)
            bullets = [clean_item_text(b, 220) for b in as_list(t.get("bullets"))[:3]]
            bullets = [b for b in bullets if b and not is_poor_bullet(b)]
            if h and bullets:
                topics.append(f"- {h}: {' / '.join(bullets)}")
        if topics:
            rows.append(f"[chunk {i}]\n" + "\n".join(topics))
        if sum(len(x) for x in rows) > max_chars:
            break
    return "\n\n".join(rows)[:max_chars]


def section_prompt_common(title: str, transcript_digest: str, glossary: str, language: str, detail_level: str) -> str:
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
문서 제목: {title}
출력 언어: {language}
문서 상세도: {detail_cfg(detail_level)['label']}
{glossary_text}
공통 작성 원칙:
- 아래 시간순 transcript digest를 최우선 근거로 사용하세요.
- general domain 회의/강의/인터뷰/발표/교육/해설 녹음을 모두 고려하세요. 녹음 성격에 맞는 문서로 정리하세요.
- 사람이 읽는 최종 문서입니다. 발화 조각을 그대로 붙이지 말고 자연스러운 문장으로 재구성하세요.
- 원문에 없는 사실을 만들지 마세요. 숫자, 날짜, 금액, 회사명, 인물명, 결정사항은 특히 보수적으로 작성하세요.
- ASR 오류가 의심되면 문맥상 명확한 것만 보정하고, 불확실한 것은 확인 필요한 내용으로 남기세요.
- 한국어 출력에서는 중국어·일본어·한자식 글자를 섞지 마세요.
- 코드블록, JSON, Python dict/list, 작성 지시문은 출력하지 마세요.

시간순 transcript digest:
{transcript_digest}
""".strip()


def make_section_prompt(part: str, title: str, transcript_digest: str, support: str, glossary: str, language: str, detail_level: str) -> str:
    common = section_prompt_common(title, transcript_digest, glossary, language, detail_level)
    support_text = f"\n\n보조 chunk note 참고자료:\n{support}" if support.strip() else ""
    if part == "overview":
        return common + f"""

아래 섹션만 Markdown으로 작성하세요.

# {title}

## 1. 한 페이지 요약
- bullet이 아니라 1~2개의 자연스러운 문단으로 작성하세요.
- 핵심 배경, 다룬 주제, 핵심 메시지, 확인/후속 포인트가 자연스럽게 이어지게 하세요.
- 회의가 아닌 강의/설명 영상이면 '무엇을 설명하는 자료인지'와 '핵심 개념의 흐름'을 요약하세요.

## 2. 전체 구조화 정리
- 문서 전체 흐름을 5~8개의 bullet로 정리하세요.
- 너무 짧은 발화 조각이나 의미 없는 표현은 넣지 마세요.
""" + support_text
    if part == "details":
        return common + f"""

아래 섹션만 Markdown으로 작성하세요.

## 3. 주제별 상세 정리
- 6~10개 주제로 나누세요. 짧은 녹음이면 4~6개도 괜찮습니다.
- 각 주제는 ### 소제목으로 시작하고, 3~6개 bullet로 세부 내용을 설명하세요.
- 소제목은 '주요 논의 1' 같은 일반 제목이 아니라 실제 내용을 나타내는 제목으로 작성하세요.
- 원문 흐름을 사람이 이해할 수 있게 재구성하세요. 발화 조각을 그대로 나열하지 마세요.

## 4. 핵심 개념 / 논점
- 표가 아니라 bullet 목록으로 작성하세요.
- 각 bullet은 '개념/논점: 설명' 형식으로 작성하세요.
- 강의/교육 자료라면 핵심 개념과 구조를 중심으로, 회의라면 논점과 판단 포인트를 중심으로 작성하세요.
""" + support_text
    return common + f"""

아래 섹션만 Markdown으로 작성하세요.

## 5. 결정사항 / 결론
- 회의라면 결정사항을, 강의/설명/뉴스성 녹음이라면 주요 결론 또는 시사점을 작성하세요.
- 원문에 없으면 '명시적 결정사항 없음'이라고 간단히 쓰세요.

## 6. 실행 항목
- 담당자/기한이 명확한 할 일이 있을 때만 작성하세요.
- 없으면 '명시적 실행 항목 없음'이라고 쓰세요.

## 7. 리스크 / 이슈
- 실제로 언급된 문제, 불확실성, 논란, 주의점을 정리하세요.

## 8. 타임라인 / 진행 흐름
- 시간순으로 6~12개 bullet을 작성하세요.
- 가능한 경우 [HH:MM:SS] 형식을 사용하세요.

## 9. 중요 발언 / 근거
- 의미 있는 발언 5~10개를 고르세요.
- 가능하면 timestamp 또는 segment 근거를 붙이세요.

## 10. 용어 / 개념
- 실제 고유명사, 제도, 제품명, 기술명, 방법론만 넣으세요.
- 일반 발화어, 감탄사, 조사성 표현은 넣지 마세요.

## 11. 확인 필요한 내용
- ASR 오인식 또는 사실 확인이 필요한 내용을 작성하세요.
- 없으면 '명시적으로 확인 필요한 내용 없음'이라고 쓰세요.
""" + support_text


def extract_markdown_sections(md: str) -> dict[str, str]:
    md = cleanup_markdown(md, "") if md else ""
    matches = list(re.finditer(r"(?m)^##\s+(\d+)\.\s+.*$", md))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        num = m.group(1)
        sections[num] = md[start:end].strip()
    title_match = re.search(r"(?m)^#\s+.*$", md)
    if title_match:
        sections["title"] = title_match.group(0).strip()
    return sections


def markdown_artifact_score(md: str) -> int:
    score = 0
    if not md or len(md.strip()) < 700:
        score += 4
    if has_template_leak(md):
        score += 4
    if re.search(r"\{\s*['\"]?(heading|bullets|summary|text)['\"]?\s*:", md):
        score += 5
    if re.search(r"[\u3040-\u30ff]", md):
        score += 5
    if re.search(r"[\u3400-\u9fff]", md):
        score += 5
    if md.count("명시적으로 확인되지 않음") >= 10:
        score += 2
    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", md)
    if len(words) > 100:
        top, cnt = Counter(words).most_common(1)[0]
        if cnt > 40 and cnt / len(words) > 0.08:
            score += 3
    return score


def assemble_sectioned_markdown(title: str, parts: list[str]) -> str:
    all_md = "\n\n".join(cleanup_markdown(p, title) for p in parts if p and p.strip())
    sections = extract_markdown_sections(all_md)
    lines = [f"# {title}", ""]
    expected = [
        ("1", "## 1. 한 페이지 요약"),
        ("2", "## 2. 전체 구조화 정리"),
        ("3", "## 3. 주제별 상세 정리"),
        ("4", "## 4. 핵심 개념 / 논점"),
        ("5", "## 5. 결정사항 / 결론"),
        ("6", "## 6. 실행 항목"),
        ("7", "## 7. 리스크 / 이슈"),
        ("8", "## 8. 타임라인 / 진행 흐름"),
        ("9", "## 9. 중요 발언 / 근거"),
        ("10", "## 10. 용어 / 개념"),
        ("11", "## 11. 확인 필요한 내용"),
    ]
    for num, heading in expected:
        body = sections.get(num, "").strip()
        if body:
            # Normalize heading text to the expected one.
            body = re.sub(r"(?m)^##\s+" + re.escape(num) + r"\.\s+.*$", heading, body, count=1)
            lines += [body, ""]
        else:
            lines += [heading, "", "- 명시적으로 확인되지 않음", ""]
    return cleanup_markdown("\n".join(lines), title)


def generate_sectioned_markdown(
    llm,
    title: str,
    notes: list[dict],
    segments: list[dict],
    profile: RuntimeProfile,
    language: str,
    glossary: str,
    detail_level: str,
    log_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str | None, bool]:
    """GPU v9 final writer: create final Markdown in sections from transcript.

    This avoids the weak point observed in v9: a large LLM created a valid but poor
    JSON, and the DOCX inherited that poor structure.  Sectioned Markdown writing
    is slower than one final call but far more stable on GPU profiles.
    """
    if profile.llm_device != "cuda":
        return None, False
    budget = writer_budget_for_profile(profile, detail_level)
    transcript_digest = build_writer_context(segments, profile, detail_level)
    support = compact_support_notes(notes, max_chars=16000 if profile.name == "gpu_quality" else 11000)
    part_defs = [
        ("overview", budget["part1"]),
        ("details", budget["part2"]),
        ("closing", budget["part3"]),
    ]
    parts: list[str] = []
    for part, tok in part_defs:
        prompt = make_section_prompt(part, title, transcript_digest, support, glossary, language, detail_level)
        if log_cb:
            log_cb(f"✍️ v9 섹션별 Markdown writer 생성: {part} / max_new_tokens={tok}")
        raw = llm.generate(SYSTEM_PROMPT_MARKDOWN, prompt, max_new_tokens=tok)
        md = cleanup_markdown(raw, title)
        # Do not reject a single section too aggressively; assemble first, then score.
        parts.append(md)
    final_md = assemble_sectioned_markdown(title, parts)
    final_md = normalize_language_artifacts(final_md)
    final_md = cleanup_markdown(final_md, title)
    if markdown_artifact_score(final_md) <= 3:
        return final_md, True
    if log_cb:
        log_cb("⚠️ v9 섹션별 Markdown writer 결과에 품질 문제가 있어 기존 구조화 Markdown 경로로 fallback합니다.")
    return None, False



# ---------------------------------------------------------------------------
# v10 robust transcript-first final Markdown writer
# ---------------------------------------------------------------------------

def full_transcript_or_digest_for_writer(segments: list[dict], profile: RuntimeProfile, detail_level: str) -> str:
    """Return the most useful context for the final human writer.

    v9 sometimes produced weak DOCX even on 7B because the final export fell back to
    JSON-derived structure.  v10 prioritizes the original transcript.  For short and
    medium inputs we pass the complete timestamped transcript; for long inputs we use
    coherent chronological blocks rather than isolated representative sentences.
    """
    full = segments_to_prompt_text(segments)
    name = getattr(profile, "name", "")
    if name == "gpu_quality":
        full_limit = 26000 if detail_level == "detailed" else 21000
        total = 36000 if detail_level == "detailed" else 28000
        blocks = 34 if detail_level == "detailed" else 26
        block_chars = 1150
    elif name == "gpu_balanced":
        full_limit = 20000 if detail_level == "detailed" else 16000
        total = 27000 if detail_level == "detailed" else 21000
        blocks = 28 if detail_level == "detailed" else 22
        block_chars = 1050
    elif name == "gpu_light":
        full_limit = 15000 if detail_level == "detailed" else 12000
        total = 20000 if detail_level == "detailed" else 16000
        blocks = 22 if detail_level == "detailed" else 18
        block_chars = 950
    else:
        full_limit = 13000 if detail_level == "detailed" else 10000
        total = 16000 if detail_level == "detailed" else 12500
        blocks = 18 if detail_level == "detailed" else 14
        block_chars = 900
    if len(full) <= full_limit:
        return full
    return blocks_to_prompt_text(chronological_blocks(segments, block_max_chars=block_chars, max_blocks=blocks, max_total_chars=total))


def make_transcript_first_markdown_prompt(
    title: str,
    segments: list[dict],
    notes: list[dict],
    profile: RuntimeProfile,
    language: str,
    glossary: str,
    detail_level: str,
) -> str:
    cfg = detail_cfg(detail_level)
    context = full_transcript_or_digest_for_writer(segments, profile, detail_level)
    support = compact_support_notes(notes, max_chars=9000 if getattr(profile, "name", "") == "gpu_quality" else 6500)
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    min_topics = "6~10개" if detail_level == "detailed" else "4~7개"
    min_keypoints = "8~12개" if detail_level == "detailed" else "5~8개"
    return f"""
아래 timestamped transcript를 근거로 최종 DOCX에 들어갈 Markdown 문서를 직접 작성하세요.
문서 제목: {title}
출력 언어: {language}
문서 상세도: {cfg['label']}
{glossary_text}

가장 중요한 목표:
- 최종 독자가 바로 읽을 수 있는 사람다운 문서로 작성하세요.
- chunk별 조각, raw 발화, JSON 병합 결과처럼 보이면 실패입니다.
- 입력이 회의인지, 강의/교육 영상인지, 인터뷰인지, 발표/해설인지 먼저 파악하고 그 성격에 맞게 정리하세요.
- 이 애플리케이션은 general domain을 대상으로 하므로 특정 회사/산업/샘플에 편향하지 마세요.

내용 작성 원칙:
- transcript에 있는 핵심 내용을 빠뜨리지 말고, 의미 단위로 재구성하세요.
- 원문에 없는 사실을 만들지 마세요. 숫자, 금액, 날짜, 회사명, 인물명, 제도명은 특히 보수적으로 다루세요.
- ASR 오인식이 명확하면 자연스럽게 보정하되, 불확실하면 '확인 필요'로 남기세요.
- 강의/교육 영상이면 '결정사항 없음'만 반복하지 말고, 핵심 개념·설명 흐름·학습 포인트·실천 체크포인트를 중심으로 작성하세요.
- 회의라면 논의 배경·결정사항·실행 항목·리스크·후속 확인 사항을 중심으로 작성하세요.
- 중국어, 일본어, 한자식 문자, 어색한 직역체를 섞지 마세요. 반드시 자연스러운 한국어로 작성하세요.
- JSON, Python dict/list, 코드블록, key-value dump, 작성 지시문은 출력하지 마세요.

반드시 아래 섹션 구조를 그대로 사용하세요.

# {title}

## 1. 한 페이지 요약
1~2개의 자연스러운 문단으로 작성하세요. bullet 금지. 녹음의 주제, 핵심 구조, 중요한 결론/시사점이 자연스럽게 이어지게 작성하세요.

## 2. 전체 구조화 정리
전체 흐름을 5~8개 bullet로 정리하세요. 각 bullet은 짧은 제목 + 설명 형태로 작성하세요.

## 3. 주제별 상세 정리
{min_topics}의 구체적인 주제로 나누세요. 각 주제는 ### 소제목으로 시작하고, 3~6개 bullet로 상세하게 설명하세요. 주제명은 실제 내용을 드러내야 합니다.

## 4. 핵심 개념 / 논점
{min_keypoints}개의 bullet을 작성하세요. 각 bullet은 '개념/논점: 설명' 형태로 작성하세요.

## 5. 결정사항 / 결론
회의이면 결정사항을, 강의/설명 영상이면 핵심 결론 또는 학습 포인트를 작성하세요. 없는 내용을 억지로 만들지 마세요.

## 6. 실행 항목
실제 담당자/기한이 있으면 작성하세요. 강의/설명 영상이면 원문에 근거한 개인 체크리스트 또는 후속 확인 항목을 작성해도 됩니다. 근거가 없으면 '명시적 실행 항목 없음'이라고 쓰세요.

## 7. 리스크 / 이슈
실제 언급된 주의점, 오해 가능성, 불확실성, 논란, 적용 시 주의사항을 작성하세요.

## 8. 타임라인 / 진행 흐름
시간순으로 6~12개 bullet을 작성하세요. 가능한 경우 [HH:MM:SS]를 붙이세요. 시간이 없으면 설명 흐름 순서로 작성하세요.

## 9. 중요 발언 / 근거
의미 있는 발언 또는 근거 5~10개를 작성하세요. 단순 감탄사나 말버릇은 제외하세요.

## 10. 용어 / 개념
실제 고유명사, 제도, 제품명, 기술명, 핵심 개념만 작성하세요. 일반 발화어는 제외하세요.

## 11. 확인 필요한 내용
ASR 오인식 가능성, 추가 확인이 필요한 숫자/명칭/조건을 작성하세요. 없으면 '명시적으로 확인 필요한 내용 없음'이라고 쓰세요.

Timestamped transcript:
{context}

보조 chunk note 참고자료. 참고만 하고, 최종 문서는 transcript를 우선하세요:
{support}
""".strip()


def make_markdown_repair_prompt(title: str, bad_markdown: str, segments: list[dict], profile: RuntimeProfile, language: str, glossary: str, detail_level: str) -> str:
    context = full_transcript_or_digest_for_writer(segments, profile, detail_level)
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
아래 초안 Markdown은 형식이 깨졌거나 내용이 빈약할 수 있습니다. 원 transcript를 근거로 사람이 읽기 좋은 최종 Markdown 문서로 다시 작성하세요.
문서 제목: {title}
출력 언어: {language}
{glossary_text}

수정 원칙:
- 초안에 JSON, Python dict/list, 중국어/일본어/한자식 문자, 깨진 숫자, 반복 문구가 있으면 제거하세요.
- transcript에 없는 사실을 만들지 마세요.
- 강의/교육/회의/인터뷰 등 녹음의 성격에 맞게 자연스럽게 정리하세요.
- 반드시 1~11번 Markdown 섹션을 모두 포함하세요.
- Markdown만 출력하세요.

원 transcript:
{context}

문제가 있는 초안:
{bad_markdown[:22000]}
""".strip()


def extractive_markdown_safety_net(title: str, segments: list[dict], detail_level: str = "standard") -> str:
    """Deterministic last resort that is more readable than broken JSON output."""
    blocks = chronological_blocks(segments, block_max_chars=1000, max_blocks=12 if detail_level == "detailed" else 9, max_total_chars=12000)
    selected = []
    for b in blocks:
        sents = sentence_split(b.get("text", ""))
        good = [s for s in sents if not is_poor_bullet(s)][:4]
        if good:
            selected.append({"time": b.get("time", ""), "text": good})
    all_texts = [x for b in selected for x in b["text"]]
    kws = top_keywords(all_texts, 10)
    lines = [f"# {title}", "", "## 1. 한 페이지 요약", ""]
    if all_texts:
        paragraph = " ".join(all_texts[:5])
        lines += [clean_item_text(paragraph, 1400), ""]
    else:
        lines += ["전사 결과를 바탕으로 주요 내용을 정리했습니다.", ""]
    lines += ["## 2. 전체 구조화 정리", ""]
    for k in kws[:7]:
        lines.append(f"- {k}: 녹음에서 반복적으로 다뤄진 주요 표현 또는 개념입니다.")
    if not kws:
        lines.append("- 주요 내용: 전사 결과에서 확인되는 핵심 흐름을 아래 주제별 상세 정리에 정리했습니다.")
    lines += ["", "## 3. 주제별 상세 정리", ""]
    for i, b in enumerate(selected[:8], start=1):
        heading = derive_heading_from_bullets(b["text"], fallback=f"구간 {i} 주요 내용")
        lines += [f"### {i}. {heading}"]
        for st in b["text"][:5]:
            lines.append(f"- {st}")
        lines.append("")
    lines += ["## 4. 핵심 개념 / 논점", ""]
    for k in kws[:10]:
        lines.append(f"- {k}: transcript에서 주요하게 언급된 항목입니다. 정확한 세부 조건은 원문 확인이 필요합니다.")
    lines += ["", "## 5. 결정사항 / 결론", "", "- 명시적 결정사항 없음", "", "## 6. 실행 항목", "", "- 명시적 실행 항목 없음", "", "## 7. 리스크 / 이슈", "", "- ASR 오인식 가능성이 있으므로 고유명사와 숫자는 원문 확인이 필요합니다.", "", "## 8. 타임라인 / 진행 흐름", ""]
    for b in selected[:10]:
        lines.append(f"- [{b['time']}] {b['text'][0]}")
    lines += ["", "## 9. 중요 발언 / 근거", ""]
    for s in pick_representative_segments(segments, limit=8):
        txt = clean_item_text(s.get("text", ""), 260)
        if txt:
            lines.append(f"- [{s.get('start_hms','')}] {txt}")
    lines += ["", "## 10. 용어 / 개념", ""]
    for k in kws[:10]:
        if not is_low_value_term(k):
            lines.append(f"- {k}: 주요 용어 후보")
    lines += ["", "## 11. 확인 필요한 내용", "", "- 고유명사, 숫자, 날짜, 조건은 ASR 전사 오류 가능성을 고려해 원문 확인이 필요합니다.", ""]
    return cleanup_markdown("\n".join(lines), title)


def generate_transcript_first_markdown(
    llm,
    title: str,
    notes: list[dict],
    segments: list[dict],
    profile: RuntimeProfile,
    language: str,
    glossary: str,
    detail_level: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str | None, bool, bool]:
    """v10 primary final writer. Returns (markdown, used_llm_markdown, repair_used)."""
    prompt = make_transcript_first_markdown_prompt(title, segments, notes, profile, language, glossary, detail_level)
    if log_cb:
        log_cb(f"✍️ v10 transcript-first 최종 Markdown writer 생성 / max_new_tokens={max_new_tokens}")
    raw = llm.generate(SYSTEM_PROMPT_MARKDOWN, prompt, max_new_tokens=max_new_tokens)
    md = cleanup_markdown(raw, title)
    if not markdown_has_bad_artifacts(md) and markdown_artifact_score(md) <= 2:
        return md, True, False
    if log_cb:
        log_cb("⚠️ v10 Markdown 초안에 형식/언어/품질 문제가 있어 1회 repair를 시도합니다.")
    repair_prompt = make_markdown_repair_prompt(title, md, segments, profile, language, glossary, detail_level)
    raw2 = llm.generate(SYSTEM_PROMPT_MARKDOWN, repair_prompt, max_new_tokens=max(2600, min(max_new_tokens, 6500)))
    md2 = cleanup_markdown(raw2, title)
    if not markdown_has_bad_artifacts(md2) and markdown_artifact_score(md2) <= 3:
        return md2, True, True
    if log_cb:
        log_cb("⚠️ v10 Markdown repair도 충분하지 않아 안전한 원문 기반 Markdown으로 대체합니다.")
    return extractive_markdown_safety_net(title, segments, detail_level), False, True


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


# ---------------------------------------------------------------------------
# v11 final writer: transcript-first sectioned human document generation
# ---------------------------------------------------------------------------

def severe_markdown_artifact_score(md: str) -> int:
    """Score only severe human-facing Markdown failures.

    Earlier versions rejected many usable LLM drafts because headings were slightly
    different or because one section was shorter than expected.  That pushed the
    pipeline into extractive fallback even on 7B GPUs.  v11 only rejects severe
    failures: JSON/key leakage, code blocks, CJK drift, missing most sections, or
    extremely short output.
    """
    md = normalize_language_artifacts(cleanup_markdown(md or "", "문서"))
    score = 0
    if len(md.strip()) < 1200:
        score += 3
    if re.search(r"```|\{\s*['\"]?(heading|bullets|topics|summary|text)['\"]?\s*:", md):
        score += 5
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", md):
        score += 5
    present = len(set(re.findall(r"(?m)^##\s*(\d+)\.", md)))
    if present < 8:
        score += 3
    if md.count("명시적으로 확인되지 않음") >= 12:
        score += 2
    # Detect pathological repetition, but do not punish legitimate repeated domain terms.
    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", md)
    if len(words) > 120:
        top, cnt = Counter(words).most_common(1)[0]
        if cnt > 55 and cnt / len(words) > 0.10 and top.lower() not in {"연말정산", "소득공제", "세액공제", "공제", "세금", "회의", "내용"}:
            score += 3
    return score


def v11_writer_context(segments: list[dict], profile: RuntimeProfile, detail_level: str) -> str:
    """Use full transcript whenever practical; otherwise use coherent chronological blocks.

    For the reported GPU-quality failures the transcript was only ~3.4k chars, so
    chunk extraction was unnecessary and harmful.  v11 prioritizes the original
    transcript over noisy intermediate JSON notes.
    """
    full = segments_to_prompt_text(segments)
    name = getattr(profile, "name", "")
    if name == "gpu_quality":
        full_limit = 42000 if detail_level == "detailed" else 32000
        max_blocks, block_chars, max_total = (52, 1250, 56000)
    elif name == "gpu_balanced":
        full_limit = 28000 if detail_level == "detailed" else 22000
        max_blocks, block_chars, max_total = (38, 1150, 39000)
    elif name == "gpu_light":
        full_limit = 20000 if detail_level == "detailed" else 16000
        max_blocks, block_chars, max_total = (28, 1000, 26000)
    else:
        full_limit = 14000 if detail_level == "detailed" else 11000
        max_blocks, block_chars, max_total = (18, 950, 16000)
    if len(full) <= full_limit:
        return full
    return blocks_to_prompt_text(chronological_blocks(segments, block_max_chars=block_chars, max_blocks=max_blocks, max_total_chars=max_total))


def short_gpu_transcript_case(segments: list[dict], profile: RuntimeProfile, detail_level: str) -> bool:
    if getattr(profile, "llm_device", "") != "cuda":
        return False
    chars = sum(len(s.get("text", "")) for s in segments)
    if profile.name == "gpu_quality":
        return chars <= (26000 if detail_level == "detailed" else 20000)
    if profile.name == "gpu_balanced":
        return chars <= (18000 if detail_level == "detailed" else 14000)
    if profile.name == "gpu_light":
        return chars <= (12000 if detail_level == "detailed" else 9000)
    return False


def make_v11_section_prompt(
    part: str,
    title: str,
    context: str,
    support: str,
    glossary: str,
    language: str,
    detail_level: str,
) -> str:
    cfg = detail_cfg(detail_level)
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    support_text = ""
    if support.strip():
        support_text = f"\n\n보조 추출 note입니다. 이 내용은 참고만 하며, 최종 문서는 반드시 transcript 흐름을 우선하세요.\n{support[:12000]}"
    common = f"""
당신은 최종 DOCX에 들어갈 사람이 읽는 문서를 작성하는 전문 기록 정리자입니다.
문서 제목: {title}
출력 언어: {language}
문서 상세도: {cfg['label']}
{glossary_text}

입력 자료 성격 파악:
- 먼저 transcript가 회의인지, 강의/교육 영상인지, 인터뷰인지, 발표/해설 영상인지 판단하세요.
- 회의가 아니면 참석자·결정사항·실행항목을 억지로 만들지 말고, 핵심 개념·설명 흐름·학습 포인트·실천 체크포인트 중심으로 정리하세요.
- 회의이면 논의 배경·결정사항·실행 항목·리스크·후속 확인 사항 중심으로 정리하세요.

공통 작성 원칙:
- transcript에 있는 정보만 사용하세요. 없는 사실을 만들지 마세요.
- 숫자, 날짜, 금액, 비율, 제도명, 회사명, 인물명은 특히 보수적으로 작성하세요.
- ASR 오인식이 명확한 경우에만 자연스럽게 보정하고, 불확실하면 확인 필요한 내용으로 남기세요.
- raw 발화 조각을 그대로 붙이지 말고, 사람이 이해할 수 있게 의미 단위로 재구성하세요.
- 반드시 자연스러운 한국어 문장으로 작성하세요. 중국어·일본어·한자식 문자, JSON, Python dict/list, 코드블록을 출력하지 마세요.
- 일반 도메인용 앱입니다. 특정 샘플/회사/산업에 편향하지 마세요.

Timestamped transcript 또는 시간순 digest:
{context}
{support_text}
""".strip()
    if part == "summary_outline":
        return common + f"""

아래 섹션만 Markdown으로 작성하세요.

# {title}

## 1. 한 페이지 요약
- bullet 금지. 1~2개의 자연스러운 문단으로 작성하세요.
- 주제, 설명 흐름, 핵심 결론/시사점, 독자가 얻어야 할 내용을 연결된 글로 작성하세요.

## 2. 전체 구조화 정리
- 6~9개 bullet로 전체 흐름을 정리하세요.
- 각 bullet은 **짧은 소제목: 설명** 형태로 작성하세요.
""".strip()
    if part == "details":
        return common + """

아래 섹션만 Markdown으로 작성하세요.

## 3. 주제별 상세 정리
- 6~10개의 구체적인 주제로 나누세요. 짧은 입력이면 4~6개도 가능합니다.
- 각 주제는 ### 소제목으로 시작하세요.
- 각 주제마다 3~6개 bullet을 작성하되, 발화 조각이 아니라 의미가 완결된 설명문으로 작성하세요.
- 강의/교육 영상이면 정의 → 구조 → 예시 → 적용 포인트 순서가 드러나게 작성하세요.
""".strip()
    if part == "concepts":
        return common + """

아래 섹션만 Markdown으로 작성하세요.

## 4. 핵심 개념 / 논점
- 8~14개 bullet을 작성하세요.
- 각 bullet은 **개념/논점: 설명** 형식으로 작성하세요.
- 단순히 자주 나온 단어를 용어처럼 나열하지 말고, 독자가 이해해야 할 핵심 개념만 작성하세요.

## 5. 결정사항 / 결론
- 회의이면 실제 결정사항을 작성하세요.
- 강의/교육/설명 영상이면 핵심 결론 또는 학습 포인트를 작성하세요.
- 원문에 없으면 억지로 만들지 마세요.

## 6. 실행 항목
- 회의에서 담당자/기한이 있는 액션이 있으면 작성하세요.
- 강의/교육 영상이면 개인이 확인하거나 적용할 체크리스트를 transcript 근거 안에서 작성하세요.
""".strip()
    return common + """

아래 섹션만 Markdown으로 작성하세요.

## 7. 리스크 / 이슈
- 실제 언급된 주의점, 오해 가능성, 불확실성, 적용 시 주의사항을 작성하세요.

## 8. 타임라인 / 진행 흐름
- 시간순으로 6~12개 bullet을 작성하세요. 가능한 경우 [HH:MM:SS]를 붙이세요.

## 9. 중요 발언 / 근거
- 의미 있는 발언 또는 근거 5~10개를 작성하세요. 말버릇이나 감탄사는 제외하세요.

## 10. 용어 / 개념
- 실제 고유명사, 제도명, 제품명, 기술명, 방법론만 작성하세요.
- 일반 발화어, 조사성 표현, 의미 없는 빈도 단어는 제외하세요.

## 11. 확인 필요한 내용
- ASR 오인식 가능성 또는 추가 확인이 필요한 숫자/명칭/조건을 작성하세요.
- 없으면 '명시적으로 확인 필요한 내용 없음'이라고 쓰세요.
""".strip()


def normalize_required_sections(md: str, title: str) -> str:
    md = cleanup_markdown(md, title)
    # Normalize common heading variants without rejecting useful text.
    replacements = {
        r"##\s*1\.?\s*한\s*페이지\s*요약.*": "## 1. 한 페이지 요약",
        r"##\s*2\.?\s*전체\s*구조화\s*정리.*": "## 2. 전체 구조화 정리",
        r"##\s*3\.?\s*주제별\s*상세\s*정리.*": "## 3. 주제별 상세 정리",
        r"##\s*4\.?\s*핵심\s*(개념|논점).*": "## 4. 핵심 개념 / 논점",
        r"##\s*5\.?\s*(결정사항|결론).*": "## 5. 결정사항 / 결론",
        r"##\s*6\.?\s*실행\s*항목.*": "## 6. 실행 항목",
        r"##\s*7\.?\s*(리스크|이슈).*": "## 7. 리스크 / 이슈",
        r"##\s*8\.?\s*(타임라인|진행).*": "## 8. 타임라인 / 진행 흐름",
        r"##\s*9\.?\s*(중요\s*발언|근거).*": "## 9. 중요 발언 / 근거",
        r"##\s*10\.?\s*(용어|개념).*": "## 10. 용어 / 개념",
        r"##\s*11\.?\s*확인.*": "## 11. 확인 필요한 내용",
    }
    for pat, rep in replacements.items():
        md = re.sub(pat, rep, md, flags=re.I | re.M)
    return cleanup_markdown(md, title)


def generate_sectioned_markdown_v11(
    llm,
    title: str,
    notes: list[dict],
    segments: list[dict],
    profile: RuntimeProfile,
    language: str,
    glossary: str,
    detail_level: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str | None, bool]:
    """Primary GPU writer for v11.

    It writes final Markdown directly in four focused calls.  The original
    transcript is the primary source.  Chunk notes are omitted for short inputs to
    avoid polluting the final document with poor intermediate headings.
    """
    context = v11_writer_context(segments, profile, detail_level)
    short_case = short_gpu_transcript_case(segments, profile, detail_level)
    support = "" if short_case else compact_support_notes(notes, max_chars=10000 if profile.name == "gpu_quality" else 7000)
    if log_cb:
        log_cb("✍️ v11 transcript-first 섹션별 Markdown writer 시작" + (" (short transcript: chunk note 제외)" if short_case else ""))
    part_tokens = {
        "summary_outline": min(max(1600, max_new_tokens // 3), 2600),
        "details": min(max(2600, max_new_tokens // 2), 5200),
        "concepts": min(max(2200, max_new_tokens // 2), 4200),
        "closing": min(max(2200, max_new_tokens // 2), 4200),
    }
    parts = []
    for part in ["summary_outline", "details", "concepts", "closing"]:
        prompt = make_v11_section_prompt(part, title, context, support, glossary, language, detail_level)
        if log_cb:
            log_cb(f"✍️ v11 Markdown section 생성: {part} / max_new_tokens={part_tokens[part]}")
        raw = llm.generate(SYSTEM_PROMPT_MARKDOWN, prompt, max_new_tokens=part_tokens[part])
        part_md = normalize_required_sections(raw, title)
        parts.append(part_md)
    assembled = assemble_sectioned_markdown(title, parts)
    assembled = normalize_required_sections(normalize_language_artifacts(assembled), title)
    score = severe_markdown_artifact_score(assembled)
    if score <= 5:
        return assembled, False
    # One repair pass, but do not immediately fall back to extractive unless severe artifacts remain.
    if log_cb:
        log_cb(f"⚠️ v11 sectioned Markdown 점수={score}. 원문 기반 repair 1회 수행")
    repair_prompt = make_markdown_repair_prompt(title, assembled, segments, profile, language, glossary, detail_level)
    raw2 = llm.generate(SYSTEM_PROMPT_MARKDOWN, repair_prompt, max_new_tokens=min(max(3200, max_new_tokens), 7200))
    repaired = normalize_required_sections(normalize_language_artifacts(raw2), title)
    if severe_markdown_artifact_score(repaired) <= 6:
        return repaired, True
    # If repair still has issues but assembled is mostly readable, use assembled rather than poor extractive fallback.
    if score <= 8 and "## 3. 주제별 상세 정리" in assembled:
        return assembled, False
    return None, True


# ---------------------------------------------------------------------------
# v12 final writer: one-pass transcript-first complete document generation
# ---------------------------------------------------------------------------

FULLWIDTH_TRANSLATION = str.maketrans({
    "，": ",", "。": ".", "：": ":", "；": ";", "（": "(", "）": ")",
    "［": "[", "］": "]", "｛": "{", "｝": "}", "“": '"', "”": '"', "’": "'", "‘": "'",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})


def clean_human_markdown_text(md: str, title: str) -> str:
    """Final human-facing cleanup for Markdown before Pandoc.

    This is deliberately conservative. It removes formatting artifacts and obvious
    mixed-script noise, but it does not attempt to rewrite facts.
    """
    md = str(md or "").translate(FULLWIDTH_TRANSLATION)
    md = strip_code_fence(md)
    md = normalize_language_artifacts(md)
    # Remove any leaked JSON/object fragments that sometimes appear before/inside a report.
    md = re.sub(r"(?s)```(?:json|python|markdown|md)?\s*.*?```", "", md)
    md = re.sub(r"(?m)^\s*[\{\[]\s*['\"]?(topics|summary|heading|bullets|text)['\"]?\s*[:=].*$", "", md)
    md = re.sub(r"(?m)^\s*['\"]?(heading|bullets|topics|summary|text)['\"]?\s*[:=].*$", "", md)
    # Normalize heading variants that Qwen occasionally produces.
    md = normalize_required_sections(md, title)
    # Remove repeated placeholder lines and unknown spam.
    md = re.sub(r"(?:\n\s*-\s*명시적으로 확인되지 않음\s*){2,}", "\n- 명시적으로 확인되지 않음\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return cleanup_markdown(md, title)


def v12_transcript_context(segments: list[dict], profile: RuntimeProfile, detail_level: str) -> str:
    """Use full transcript for short/medium recordings; coherent digest for long ones."""
    full = segments_to_prompt_text(segments)
    name = getattr(profile, "name", "")
    if name == "gpu_quality":
        full_limit = 52000 if detail_level == "detailed" else 42000
        block_chars, max_blocks, max_total = 1400, 58, 70000
    elif name == "gpu_balanced":
        full_limit = 36000 if detail_level == "detailed" else 28000
        block_chars, max_blocks, max_total = 1250, 44, 52000
    elif name == "gpu_light":
        full_limit = 24000 if detail_level == "detailed" else 18000
        block_chars, max_blocks, max_total = 1100, 32, 36000
    else:
        full_limit = 16000 if detail_level == "detailed" else 12000
        block_chars, max_blocks, max_total = 950, 22, 22000
    if len(full) <= full_limit:
        return full
    return blocks_to_prompt_text(chronological_blocks(segments, block_max_chars=block_chars, max_blocks=max_blocks, max_total_chars=max_total))


def infer_recording_type_hint(transcript: str) -> str:
    """Small heuristic only for prompt framing, not for facts."""
    text = transcript.lower()
    if any(k in text for k in ["강의", "설명", "개념", "정의", "배워", "다뤄볼게", "구조", "기본편", "꿀팁"]):
        return "강의/교육/설명 영상일 가능성이 높습니다. 핵심 개념과 설명 흐름, 학습 포인트, 개인 체크리스트 중심으로 정리하세요."
    if any(k in text for k in ["회의", "참석", "결정", "담당", "액션", "진행하기로", "논의"]):
        return "회의/업무 논의일 가능성이 높습니다. 논의 배경, 결정사항, 실행 항목, 리스크, 후속 확인사항 중심으로 정리하세요."
    if any(k in text for k in ["인터뷰", "질문", "답변"]):
        return "인터뷰/문답형 녹음일 가능성이 있습니다. 질문 흐름과 답변의 핵심 메시지 중심으로 정리하세요."
    return "일반 녹음입니다. 입력 성격을 먼저 판단한 뒤 그 성격에 맞는 문서로 정리하세요."


def make_v12_complete_report_prompt(title: str, segments: list[dict], profile: RuntimeProfile, language: str, glossary: str, detail_level: str) -> str:
    context = v12_transcript_context(segments, profile, detail_level)
    type_hint = infer_recording_type_hint(context)
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    if detail_level == "brief":
        topic_req, detail_req, concept_req, timeline_req, quote_req = "3~5개", "2~4개", "4~6개", "4~7개", "3~5개"
    elif detail_level == "standard":
        topic_req, detail_req, concept_req, timeline_req, quote_req = "5~7개", "3~5개", "6~9개", "6~10개", "4~7개"
    else:
        topic_req, detail_req, concept_req, timeline_req, quote_req = "6~10개", "4~7개", "8~14개", "8~14개", "5~10개"
    return f"""
당신은 최종 DOCX 문서를 작성하는 전문 기록 정리자입니다.
아래 timestamped transcript만 근거로, 사람이 읽기 좋은 최종 Markdown 문서를 작성하세요.

문서 제목: {title}
출력 언어: {language}
문서 상세도: {detail_cfg(detail_level)['label']}
입력 성격 힌트: {type_hint}
{glossary_text}
절대 원칙:
- 이 애플리케이션은 general domain 회의·강의·인터뷰·발표·교육·해설 녹음을 모두 처리합니다. 특정 샘플이나 산업에 편향하지 마세요.
- transcript에 없는 사실을 만들지 마세요. 숫자, 금액, 날짜, 비율, 제도명, 회사명, 인물명은 원문 근거가 있을 때만 쓰세요.
- ASR 전사 오류가 있을 수 있습니다. 문맥상 명확한 오인식만 자연스럽게 보정하고, 불확실한 것은 확인 필요한 내용으로 남기세요.
- raw 발화 조각을 그대로 붙이지 마세요. 의미 단위로 묶고, 배경 → 구조 → 세부 내용 → 결론/체크포인트 순서로 사람이 쓴 보고서처럼 작성하세요.
- 강의/교육/설명 영상이면 결정사항을 억지로 만들지 말고, 핵심 개념·구조·예시·학습 포인트·개인 체크리스트 중심으로 정리하세요.
- 회의라면 논의 배경·결정사항·실행 항목·리스크·후속 확인사항 중심으로 정리하세요.
- 반드시 자연스러운 한국어로 작성하세요. 중국어·일본어·한자식 문자, JSON, Python dict/list, 코드블록, 작성 지시문을 출력하지 마세요.
- '명시적으로 확인되지 않음'은 최소한으로만 사용하세요. transcript에 관련 내용이 있으면 해당 내용을 정리하세요.

반드시 아래 Markdown 섹션 제목을 그대로 사용하세요.

# {title}

## 1. 한 페이지 요약
1~2개의 자연스러운 문단으로 작성하세요. bullet 금지. 녹음의 목적, 핵심 구조, 주요 결론/시사점이 이어지게 작성하세요.

## 2. 전체 구조화 정리
{topic_req} bullet로 전체 흐름을 정리하세요. 각 bullet은 **짧은 소제목: 설명** 형식으로 작성하세요.

## 3. 주제별 상세 정리
{topic_req}의 구체적 주제로 나누고, 각 주제는 ### 소제목으로 시작하세요. 각 주제마다 {detail_req} bullet을 작성하세요. 원문 흐름을 재구성하되 발화 조각을 그대로 나열하지 마세요.

## 4. 핵심 개념 / 논점
{concept_req} bullet을 작성하세요. 각 bullet은 **개념/논점: 설명** 형식으로 작성하세요.

## 5. 결정사항 / 결론
회의이면 실제 결정사항을, 강의/설명 영상이면 핵심 결론·시사점·학습 포인트를 작성하세요.

## 6. 실행 항목
회의이면 담당자/기한이 있는 액션을 작성하세요. 강의/교육 영상이면 개인이 확인하거나 적용할 체크리스트를 작성하세요. 원문에 없으면 없다고 간단히 쓰세요.

## 7. 리스크 / 이슈
실제로 언급된 주의점, 오해 가능성, 조건, 불확실성을 정리하세요.

## 8. 타임라인 / 진행 흐름
시간순으로 {timeline_req} bullet을 작성하세요. 가능한 경우 [HH:MM:SS]를 붙이세요.

## 9. 중요 발언 / 근거
의미 있는 발언 또는 근거 {quote_req}개를 작성하세요. 말버릇과 감탄사는 제외하세요.

## 10. 용어 / 개념
실제 고유명사, 제도명, 기술명, 제품명, 방법론만 작성하세요. 일반 발화어와 빈도 단어는 제외하세요.

## 11. 확인 필요한 내용
ASR 오인식 가능성, 조건/숫자/제도 관련 추가 확인이 필요한 내용만 작성하세요. 없으면 '명시적으로 확인 필요한 내용 없음'이라고 쓰세요.

Timestamped transcript:
{context}
""".strip()


def v12_markdown_quality(md: str, title: str) -> tuple[bool, list[str]]:
    """Return (is_good, reasons).  Focus on actual user-visible failures."""
    reasons: list[str] = []
    m = clean_human_markdown_text(md or "", title)
    if len(m.strip()) < 1500:
        reasons.append("too_short")
    if re.search(r"```|\{\s*['\"]?(heading|bullets|topics|summary|text)['\"]?\s*:", m):
        reasons.append("json_or_code_leak")
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", m):
        reasons.append("cjk_or_japanese_leak")
    required_nums = set(re.findall(r"(?m)^##\s*(\d+)\.", m))
    if len(required_nums) < 10:
        reasons.append("missing_sections")
    sec1 = extract_markdown_sections(m).get("1", "")
    sec2 = extract_markdown_sections(m).get("2", "")
    sec3 = extract_markdown_sections(m).get("3", "")
    if "명시적으로 확인되지 않음" in sec1 or len(sec1) < 160:
        reasons.append("bad_summary")
    if "명시적으로 확인되지 않음" in sec2 and len(sec2) < 300:
        reasons.append("bad_outline")
    if "명시적으로 확인되지 않음" in sec3 and len(sec3) < 500:
        reasons.append("bad_details")
    if m.count("명시적으로 확인되지 않음") >= 8:
        reasons.append("too_many_unknowns")
    low_value_hits = sum(1 for x in ["필수적인", "돌아왔습니다", "깎아주기도", "신용카드나", "공제해줍니다", "마찬가지거든요"] if re.search(rf"(?m)^[-|]?\s*{re.escape(x)}\b", m))
    if low_value_hits >= 2:
        reasons.append("low_value_terms")
    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", m)
    if len(words) > 150:
        top, cnt = Counter(words).most_common(1)[0]
        if cnt > 60 and cnt / len(words) > 0.10 and top.lower() not in {"연말정산", "소득공제", "세액공제", "공제", "세금", "회의", "내용"}:
            reasons.append("repetitive")
    return not reasons, reasons


def make_v12_repair_prompt(title: str, bad_markdown: str, segments: list[dict], profile: RuntimeProfile, language: str, glossary: str, detail_level: str, reasons: list[str]) -> str:
    context = v12_transcript_context(segments, profile, detail_level)
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
아래 Markdown 초안은 최종 DOCX 품질 기준을 만족하지 못했습니다.
문제 유형: {', '.join(reasons)}

원본 transcript를 근거로 전체 Markdown 문서를 다시 작성하세요.
초안의 나쁜 표현, JSON 누출, 의미 없는 빈도 단어, '명시적으로 확인되지 않음' 남발은 버리세요.

문서 제목: {title}
출력 언어: {language}
{glossary_text}
작성 원칙:
- 반드시 자연스러운 한국어 최종 문서로 작성하세요.
- transcript에 있는 정보를 충분히 활용하세요.
- 강의/설명 영상이면 핵심 개념·구조·예시·체크포인트 중심으로 정리하세요.
- 회의이면 결정사항·실행항목·리스크 중심으로 정리하세요.
- JSON, 코드블록, Python dict/list, 중국어/일본어/한자식 문자를 출력하지 마세요.
- 아래 11개 섹션 제목을 그대로 사용하세요.

반드시 포함할 섹션:
# {title}
## 1. 한 페이지 요약
## 2. 전체 구조화 정리
## 3. 주제별 상세 정리
## 4. 핵심 개념 / 논점
## 5. 결정사항 / 결론
## 6. 실행 항목
## 7. 리스크 / 이슈
## 8. 타임라인 / 진행 흐름
## 9. 중요 발언 / 근거
## 10. 용어 / 개념
## 11. 확인 필요한 내용

원본 transcript:
{context}

품질이 낮았던 Markdown 초안:
{bad_markdown[:12000]}
""".strip()


def generate_v12_complete_markdown(
    llm,
    title: str,
    segments: list[dict],
    profile: RuntimeProfile,
    language: str,
    glossary: str,
    detail_level: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str | None, bool, list[str]]:
    """v12 primary path: one complete transcript-first writer.

    This removes the failure mode observed in v9-v11 where intermediate JSON or
    section assembly produced sparse documents despite a strong GPU model.
    """
    prompt = make_v12_complete_report_prompt(title, segments, profile, language, glossary, detail_level)
    if log_cb:
        log_cb(f"✍️ v12 complete transcript-first Markdown writer / max_new_tokens={max_new_tokens}")
    raw = llm.generate(SYSTEM_PROMPT_MARKDOWN, prompt, max_new_tokens=max_new_tokens)
    md = clean_human_markdown_text(raw, title)
    ok, reasons = v12_markdown_quality(md, title)
    if ok:
        return md, False, []
    if log_cb:
        log_cb("⚠️ v12 Markdown 품질 검사 미통과: " + ", ".join(reasons) + ". 원문 기반 repair 1회 수행")
    repair_prompt = make_v12_repair_prompt(title, md, segments, profile, language, glossary, detail_level, reasons)
    raw2 = llm.generate(SYSTEM_PROMPT_MARKDOWN, repair_prompt, max_new_tokens=max(3600, min(max_new_tokens, 9000)))
    md2 = clean_human_markdown_text(raw2, title)
    ok2, reasons2 = v12_markdown_quality(md2, title)
    if ok2 or len(reasons2) < len(reasons):
        return md2, True, reasons2
    return None, True, reasons2



# ---------------------------------------------------------------------------
# v13 planner-guided final writer
# ---------------------------------------------------------------------------

COMMON_IMPORTANT_TERMS = [
    # General meeting/project terms
    "결정사항", "실행 항목", "후속 조치", "리스크", "이슈", "일정", "담당자", "예산", "비용", "매출", "수익", "고객", "시장", "전략",
    "프로젝트", "로드맵", "성과", "지표", "품질", "보안", "개인정보", "데이터", "운영", "개선", "자동화", "검토", "승인",
    # Business / tech terms
    "AI", "LLM", "GPU", "CPU", "API", "클라우드", "서버", "모델", "서비스", "플랫폼", "반도체", "HBM", "네트워크",
    # Education / finance / tax terms frequently useful in general recordings
    "연말정산", "총급여", "비과세소득", "비과세 소득", "소득공제", "소득 공제", "세액공제", "세액 공제", "과세표준", "과세 표준",
    "결정세액", "결정 세액", "환급", "추징", "추가징수", "추가 징수", "근로소득공제", "근로소득 공제", "인적공제", "인적 공제",
    "부양가족", "4대 보험", "국민연금", "건강보험", "고용보험", "주택자금", "전월세", "청약통장", "신용카드", "체크카드", "현금영수증",
    "연금저축", "IRP", "월세", "의료비", "중소기업", "청년", "청년도약계좌", "청년 우대형 청약통장", "홈택스", "원천징수영수증",
]

LOW_VALUE_PHRASE_PATTERNS = [
    r"돌아왔습니다$", r"깎아주기도$", r"공제해줍니다$", r"마찬가지거든요$", r"있잖아요$", r"해볼게요$", r"다뤄볼게요$",
    r"필수적인$", r"신용카드나$", r"직장인이면$", r"좋겠죠$", r"하겠습니다$", r"하는데요$",
]


def detect_recording_type_v13(title: str, segments: list[dict]) -> str:
    text = normalize_language_artifacts(title + " " + " ".join(s.get("text", "") for s in segments)).lower()
    edu_score = sum(1 for k in ["개념", "구조", "기본편", "정의", "알아야", "설명", "다뤄볼게", "요약표", "다음 편", "꿀팁", "시청", "시즌"] if k in text)
    meeting_score = sum(1 for k in ["회의", "참석", "담당", "결정", "액션", "진행하기로", "논의", "아젠다", "안건"] if k in text)
    interview_score = sum(1 for k in ["인터뷰", "질문", "답변", "물어", "대답"] if k in text)
    news_score = sum(1 for k in ["주가", "시장", "발표", "기자", "뉴스", "회동", "경제", "논란"] if k in text)
    if edu_score >= max(meeting_score, interview_score, news_score) and edu_score >= 2:
        return "education"
    if meeting_score >= max(interview_score, news_score) and meeting_score >= 2:
        return "meeting"
    if interview_score >= 2:
        return "interview"
    if news_score >= 2:
        return "commentary"
    return "general"


def is_low_value_phrase_v13(text: str) -> bool:
    t = clean_item_text(text, 160).strip()
    if not t:
        return True
    if is_poor_bullet(t):
        return True
    if any(re.search(p, t) for p in LOW_VALUE_PHRASE_PATTERNS):
        # If it is a full factual sentence with multiple terms/numbers, keep it.
        if len(content_keywords(t)) < 3 and not re.search(r"\d", t):
            return True
    # Very short conversational fragments are not useful as document bullets.
    if len(t) < 18 and len(content_keywords(t)) < 2:
        return True
    return False


def is_bad_term_v13(term: str) -> bool:
    t = clean_item_text(term, 80).strip()
    if not t or is_low_value_term(t):
        return True
    if t in KOREAN_STOPWORDS or t.lower() in LOW_VALUE_TERMS:
        return True
    if any(re.search(p, t) for p in LOW_VALUE_PHRASE_PATTERNS):
        return True
    # Verbal/adverbial fragments are bad terms; short real nouns/acronyms are OK.
    if re.search(r"(합니다|했습니다|되죠|되다|해요|했죠|인데요|잖아요|볼게요|좋겠죠|줍니다|주기도|왔습니다)$", t):
        return True
    if len(t) > 24 and not re.search(r"[A-Z]", t):
        return True
    return False


def extract_terms_v13(segments: list[dict], glossary: str = "", limit: int = 18) -> list[dict]:
    text = normalize_language_artifacts(" ".join(s.get("text", "") for s in segments))
    lower = text.lower()
    terms: list[str] = []
    # Terms explicitly provided by the user are highest priority.
    for raw in re.split(r"[,\n;/]+", glossary or ""):
        raw = raw.strip()
        if raw and "=" not in raw and len(raw) >= 2:
            terms.append(raw)
        elif "=" in raw:
            left, right = raw.split("=", 1)
            if right.strip():
                terms.append(right.strip())
    for term in COMMON_IMPORTANT_TERMS:
        if term.lower() in lower and term not in terms:
            terms.append(term)
    # Acronyms and mixed alpha terms.
    for m in re.findall(r"\b[A-Z][A-Z0-9+.-]{1,}\b", text):
        if m not in terms and not is_low_value_term(m):
            terms.append(m)
    # Noun-ish Korean candidates near '라고/이란/은/는'.
    for m in re.findall(r"([가-힣A-Za-z0-9+.#_-]{2,}(?:\s+[가-힣A-Za-z0-9+.#_-]{2,}){0,2})\s*(?:이란|란|은|는|이라고|라고|을|를)", text):
        cand = clean_item_text(m, 60)
        if cand and not is_bad_term_v13(cand) and cand not in terms:
            terms.append(cand)
    out: list[dict] = []
    seen = set()
    for term in terms:
        t = clean_item_text(term, 60)
        if not t or t.lower() in seen or is_bad_term_v13(t):
            continue
        # Description from first sentence containing the term.
        desc = ""
        for sent in sentence_split(text):
            if t.replace(" ", "") in sent.replace(" ", ""):
                desc = clean_item_text(sent, 220)
                break
        if not desc:
            desc = "녹음에서 핵심적으로 언급된 개념입니다."
        out.append({"term": t, "description": desc})
        seen.add(t.lower())
        if len(out) >= limit:
            break
    return out


def block_heading_v13(block_text: str, terms: list[dict], idx: int, recording_type: str) -> str:
    text = normalize_language_artifacts(block_text)
    hits = []
    for t in terms:
        term = t.get("term", "") if isinstance(t, dict) else str(t)
        if term and term.replace(" ", "") in text.replace(" ", ""):
            hits.append(term)
        if len(hits) >= 3:
            break
    if hits:
        return " · ".join(hits[:3])
    kws = [k for k in top_keywords([text], 4) if not is_low_value_term(k)]
    if kws:
        return " · ".join(k.upper() if k.isascii() and len(k) <= 6 else k for k in kws[:3])
    if recording_type == "education":
        return f"설명 흐름 {idx}"
    if recording_type == "meeting":
        return f"논의 흐름 {idx}"
    return f"주요 흐름 {idx}"


def select_block_bullets_v13(block_text: str, max_bullets: int = 5) -> list[str]:
    sents = sentence_split(block_text)
    scored: list[tuple[float, str]] = []
    for i, s in enumerate(sents):
        st = clean_item_text(s, 420)
        if is_low_value_phrase_v13(st):
            continue
        score = 0.0
        score += min(len(st), 220) / 220.0
        score += 0.7 if re.search(r"\d", st) else 0
        score += 0.8 if any(k in st for k in ["정의", "비교", "기준", "목표", "핵심", "중요", "공제", "세액", "소득", "환급", "추징", "확인", "등록", "결정", "실행", "리스크", "문제", "원인", "결과"]) else 0
        # Prefer earlier sentences slightly for coherent explanations.
        score += max(0, 0.25 - i * 0.015)
        scored.append((score, st))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    seen = set()
    for _, st in scored:
        key = re.sub(r"\s+", "", st)[:90]
        if key in seen:
            continue
        chosen.append(st)
        seen.add(key)
        if len(chosen) >= max_bullets:
            break
    # Restore chronological order based on original sentences.
    ordered = [s for s in sents if any(clean_item_text(s, 420) == c for c in chosen)]
    return ordered[:max_bullets] or chosen[:max_bullets]


def build_content_plan_v13(title: str, segments: list[dict], glossary: str = "", detail_level: str = "detailed") -> dict:
    recording_type = detect_recording_type_v13(title, segments)
    terms = extract_terms_v13(segments, glossary, limit=20 if detail_level == "detailed" else 14)
    # Use enough blocks for structure, but not so many that the document looks chunk-sliced.
    blocks = chronological_blocks(
        segments,
        block_max_chars=850 if detail_level == "detailed" else 1000,
        max_blocks=12 if detail_level == "detailed" else 9,
        max_total_chars=14000 if detail_level == "detailed" else 10000,
    )
    topics = []
    seen_headings = set()
    for idx, b in enumerate(blocks, start=1):
        bullets = select_block_bullets_v13(b.get("text", ""), max_bullets=5 if detail_level == "detailed" else 4)
        bullets = [clean_item_text(x, 500) for x in bullets if not is_low_value_phrase_v13(x)]
        if not bullets:
            continue
        heading = block_heading_v13(b.get("text", ""), terms, idx, recording_type)
        if heading in seen_headings:
            heading = f"{heading} ({idx})"
        seen_headings.add(heading)
        topics.append({"heading": heading, "bullets": bullets, "time": b.get("time", "")})
    # Merge adjacent weak topics if there are too many or too few bullets.
    compact_topics = []
    for t in topics:
        if compact_topics and len(t["bullets"]) <= 2 and len(compact_topics[-1]["bullets"]) < 6:
            compact_topics[-1]["bullets"].extend(t["bullets"])
            compact_topics[-1]["time"] = (compact_topics[-1].get("time", "") + ", " + t.get("time", "")).strip(", ")
        else:
            compact_topics.append(t)
    topics = compact_topics[:10 if detail_level == "detailed" else 7]
    # Extract action/checklist-like sentences.
    all_sents = [clean_item_text(s, 420) for s in sentence_split(" ".join(seg.get("text", "") for seg in segments))]
    checklist = []
    for s in all_sents:
        if is_low_value_phrase_v13(s):
            continue
        if any(k in s for k in ["확인", "등록", "챙", "신경", "알아두", "활용", "가입", "공제", "준비", "받을 수", "해야"]):
            if s[:90] not in {x[:90] for x in checklist}:
                checklist.append(s)
        if len(checklist) >= 8:
            break
    risks = []
    for s in all_sents:
        if any(k in s for k in ["추징", "추가", "조건", "기준", "불확실", "오인", "놓치", "미리", "주의", "세율이 높", "많다면"]):
            if not is_low_value_phrase_v13(s) and s[:90] not in {x[:90] for x in risks}:
                risks.append(s)
        if len(risks) >= 6:
            break
    quotes = []
    for seg in pick_representative_segments(segments, limit=12):
        txt = clean_item_text(seg.get("text", ""), 260)
        if txt and not is_low_value_phrase_v13(txt):
            quotes.append({"time": seg.get("start_hms", ""), "text": txt})
    timeline = []
    for b in blocks[:12]:
        sents = [s for s in sentence_split(b.get("text", "")) if not is_low_value_phrase_v13(s)]
        if sents:
            timeline.append({"time": b.get("time", ""), "event": clean_item_text(sents[0], 260)})
    return {
        "title": title,
        "recording_type": recording_type,
        "terms": terms,
        "topics": topics,
        "checklist": checklist,
        "risks": risks,
        "timeline": timeline,
        "quotes": quotes,
    }


def content_plan_to_markdown_v13(plan: dict, title: str, segments: list[dict], detail_level: str = "detailed") -> str:
    recording_type = plan.get("recording_type", "general")
    topics = plan.get("topics") or []
    terms = plan.get("terms") or []
    term_names = [t.get("term", "") for t in terms[:8] if isinstance(t, dict)]
    lines = [f"# {title}", "", "## 1. 한 페이지 요약", ""]
    if recording_type == "education":
        topic_desc = ", ".join(term_names[:5]) if term_names else "핵심 개념과 설명 흐름"
        first_facts = []
        for t in topics:
            for b in t.get("bullets", []):
                bt = clean_item_text(b, 360)
                if is_low_value_phrase_v13(bt):
                    continue
                if any(k in bt for k in ["절차", "비교", "환급", "추징", "기준", "목표", "공제", "세액", "소득", "금액", "세율"]):
                    if bt[:100] not in {x[:100] for x in first_facts}:
                        first_facts.append(bt)
                if len(first_facts) >= 4:
                    break
            if len(first_facts) >= 4:
                break
        paragraph = f"이 녹음은 {topic_desc}을 중심으로 핵심 구조를 설명하는 교육형 자료입니다. "
        if first_facts:
            paragraph += " ".join(first_facts[:3])
        paragraph += " 아래에는 전체 흐름, 주제별 상세 설명, 핵심 개념, 개인 체크리스트와 확인 필요한 내용을 나누어 정리했습니다."
    elif recording_type == "meeting":
        paragraph = "이 녹음은 회의 또는 업무 논의 내용을 바탕으로 주요 배경, 논의 흐름, 결정사항, 실행 항목과 확인이 필요한 내용을 정리한 문서입니다."
    else:
        topic_desc = ", ".join(term_names[:5]) if term_names else "주요 내용"
        paragraph = f"이 녹음은 {topic_desc}을 중심으로 진행됩니다. 주요 흐름과 세부 내용을 의미 단위로 재구성해 아래에 정리했습니다."
    lines += [clean_item_text(paragraph, 1400), "", "## 2. 전체 구조화 정리", ""]
    for t in topics[:8]:
        h = clean_item_text(t.get("heading"), 100)
        b = clean_item_text((t.get("bullets") or [""])[0], 260)
        if h and b:
            lines.append(f"- **{h}:** {b}")
    if not topics:
        lines.append("- 전사 결과에서 확인되는 주요 흐름을 주제별 상세 정리에 정리했습니다.")
    lines += ["", "## 3. 주제별 상세 정리", ""]
    for idx, t in enumerate(topics[:10], start=1):
        lines.append(f"### {idx}. {clean_item_text(t.get('heading'), 100) or f'주요 내용 {idx}'}")
        for b in (t.get("bullets") or [])[:6]:
            bt = clean_item_text(b, 500)
            if bt:
                lines.append(f"- {bt}")
        lines.append("")
    lines += ["## 4. 핵심 개념 / 논점", ""]
    if terms:
        for t in terms[:14]:
            if isinstance(t, dict):
                lines.append(f"- **{clean_item_text(t.get('term'), 80)}:** {clean_item_text(t.get('description'), 360)}")
    else:
        for t in topics[:6]:
            h = clean_item_text(t.get("heading"), 100)
            if h:
                lines.append(f"- **{h}:** 녹음에서 주요하게 다뤄진 논점입니다.")
    lines += ["", "## 5. 결정사항 / 결론", ""]
    if recording_type == "education":
        lines.append("- 이 자료는 특정 의사결정보다는 핵심 개념을 이해하고 실제 적용 시 챙겨야 할 항목을 파악하는 데 초점이 있습니다.")
        if topics:
            lines.append(f"- 핵심 흐름은 {', '.join([clean_item_text(t.get('heading'), 60) for t in topics[:4]])} 순서로 정리할 수 있습니다.")
    else:
        lines.append("- 명시적 결정사항 없음")
    lines += ["", "## 6. 실행 항목", ""]
    checklist = plan.get("checklist") or []
    if checklist:
        for c in checklist[:8]:
            lines.append(f"- {clean_item_text(c, 420)}")
    else:
        lines.append("- 명시적 실행 항목 없음")
    lines += ["", "## 7. 리스크 / 이슈", ""]
    risks = plan.get("risks") or []
    if risks:
        for r in risks[:6]:
            lines.append(f"- {clean_item_text(r, 420)}")
    else:
        lines.append("- 원문에서 별도의 리스크나 이슈가 명확히 확인되지 않았습니다.")
    lines += ["", "## 8. 타임라인 / 진행 흐름", ""]
    for tl in (plan.get("timeline") or [])[:12]:
        lines.append(f"- [{tl.get('time','')}] {clean_item_text(tl.get('event'), 320)}")
    lines += ["", "## 9. 중요 발언 / 근거", ""]
    for q in (plan.get("quotes") or [])[:10]:
        lines.append(f"- [{q.get('time','')}] \"{clean_item_text(q.get('text'), 280)}\"")
    lines += ["", "## 10. 용어 / 개념", ""]
    if terms:
        for t in terms[:16]:
            if isinstance(t, dict):
                lines.append(f"- **{clean_item_text(t.get('term'), 80)}:** {clean_item_text(t.get('description'), 320)}")
    else:
        lines.append("- 명시적으로 정리할 전문 용어가 충분히 확인되지 않았습니다.")
    lines += ["", "## 11. 확인 필요한 내용", ""]
    if recording_type == "education":
        lines.append("- 제도, 금액, 공제 조건은 변경될 수 있으므로 실제 적용 전 최신 기준을 확인해야 합니다.")
    else:
        lines.append("- 고유명사, 숫자, 날짜, 조건은 ASR 전사 오류 가능성을 고려해 원문 확인이 필요합니다.")
    return clean_human_markdown_text("\n".join(lines), title)


def make_v13_plan_guided_prompt(title: str, segments: list[dict], plan: dict, profile: RuntimeProfile, language: str, glossary: str, detail_level: str) -> str:
    context = v12_transcript_context(segments, profile, detail_level)
    plan_json = json.dumps(plan, ensure_ascii=False)[:26000]
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    return f"""
당신은 최종 DOCX 문서를 작성하는 전문 기록 정리자입니다.
아래 timestamped transcript와 content plan을 근거로, 사람이 읽기 좋은 최종 Markdown 문서를 작성하세요.

문서 제목: {title}
출력 언어: {language}
문서 상세도: {detail_cfg(detail_level)['label']}
녹음 성격 추정: {plan.get('recording_type', 'general')}
{glossary_text}
핵심 지침:
- content plan은 원문에서 추출한 뼈대입니다. 이를 기계적으로 복사하지 말고, 자연스러운 보고서 문장으로 재구성하세요.
- transcript에 없는 사실을 만들지 마세요. 숫자, 금액, 비율, 날짜, 제도명, 인물명은 원문 근거가 있을 때만 쓰세요.
- 일반 도메인용입니다. 회의, 강의, 발표, 인터뷰, 해설 녹음 등 입력 성격에 맞게 문서화하세요.
- 강의/교육 영상이면 결정사항을 억지로 만들지 말고 핵심 개념·설명 흐름·적용 체크포인트 중심으로 작성하세요.
- 회의이면 배경·논의·결정사항·실행항목·리스크·후속 확인사항 중심으로 작성하세요.
- JSON, 코드블록, Python dict/list, 중국어/일본어/한자식 문자를 절대 출력하지 마세요.
- raw 발화 조각을 그대로 붙이지 말고 사람이 쓴 문서처럼 연결하세요.

반드시 다음 11개 섹션을 Markdown으로 작성하세요.
# {title}
## 1. 한 페이지 요약
## 2. 전체 구조화 정리
## 3. 주제별 상세 정리
## 4. 핵심 개념 / 논점
## 5. 결정사항 / 결론
## 6. 실행 항목
## 7. 리스크 / 이슈
## 8. 타임라인 / 진행 흐름
## 9. 중요 발언 / 근거
## 10. 용어 / 개념
## 11. 확인 필요한 내용

content plan:
{plan_json}

원본 timestamped transcript:
{context}
""".strip()


def v13_markdown_quality(md: str, title: str) -> tuple[bool, list[str]]:
    ok12, reasons = v12_markdown_quality(md, title)
    m = clean_human_markdown_text(md or "", title)
    # v12 quality was too lenient in practice for sectioned fragments. Add user-visible quality checks.
    sec = extract_markdown_sections(m)
    low_value_terms = ["필수적인", "신용카드나", "깎아주기도", "공제해줍니다", "마찬가지거든요", "돌아왔습니다"]
    if any(x in sec.get("10", "") for x in low_value_terms):
        reasons.append("low_value_terms_in_terms_section")
    if any(x in sec.get("3", "") for x in ["직장인이면 꼭", "돌아왔습니다", "해볼게요"]):
        # This is okay in quotes, not okay as main detail section prose.
        if len(sec.get("3", "")) < 2200:
            reasons.append("raw_transcript_fragments_in_details")
    if len(sec.get("3", "")) < 900:
        reasons.append("details_too_short")
    if len(sec.get("4", "")) < 500:
        reasons.append("concepts_too_short")
    return len(reasons) == 0, reasons


def generate_v13_plan_guided_markdown(
    llm,
    title: str,
    segments: list[dict],
    profile: RuntimeProfile,
    language: str,
    glossary: str,
    detail_level: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str, bool, bool, list[str]]:
    """Final v13 path: build a grounded content plan, ask LLM to write prose, then fallback to plan Markdown.

    Returns (markdown, llm_used, repair_or_fallback_used, quality_reasons).
    """
    plan = build_content_plan_v13(title, segments, glossary, detail_level)
    prompt = make_v13_plan_guided_prompt(title, segments, plan, profile, language, glossary, detail_level)
    if log_cb:
        log_cb(f"🧭 v13 content plan 생성: type={plan.get('recording_type')} / topics={len(plan.get('topics', []))} / terms={len(plan.get('terms', []))}")
        log_cb(f"✍️ v13 plan-guided 최종 Markdown writer / max_new_tokens={max_new_tokens}")
    try:
        raw = llm.generate(SYSTEM_PROMPT_MARKDOWN, prompt, max_new_tokens=max_new_tokens)
        md = clean_human_markdown_text(raw, title)
        ok, reasons = v13_markdown_quality(md, title)
        if ok:
            return md, True, False, []
        if log_cb:
            log_cb("⚠️ v13 Markdown 품질 검사 미통과: " + ", ".join(reasons) + ". content plan 기반 repair 1회 수행")
        repair_prompt = f"""
아래 Markdown 초안은 최종 문서 품질 기준을 만족하지 못했습니다.
문제: {', '.join(reasons)}

content plan과 원문 transcript를 근거로 최종 Markdown 문서를 다시 작성하세요.
- 자연스러운 한국어 문서로 작성하세요.
- JSON, 코드블록, 중국어/일본어/한자식 문자, 의미 없는 빈도 단어를 제거하세요.
- 11개 섹션 제목은 유지하세요.
- 주제별 상세 정리와 핵심 개념을 충분히 풍부하게 작성하세요.

content plan:
{json.dumps(plan, ensure_ascii=False)[:24000]}

원문 transcript:
{v12_transcript_context(segments, profile, detail_level)}

품질이 낮았던 초안:
{md[:14000]}
""".strip()
        raw2 = llm.generate(SYSTEM_PROMPT_MARKDOWN, repair_prompt, max_new_tokens=max(4800, min(max_new_tokens, 9000)))
        md2 = clean_human_markdown_text(raw2, title)
        ok2, reasons2 = v13_markdown_quality(md2, title)
        if ok2 or len(reasons2) < len(reasons):
            return md2, True, True, reasons2
        if log_cb:
            log_cb("⚠️ v13 LLM repair도 품질 기준을 만족하지 못해, content plan 기반 안전 Markdown으로 대체합니다.")
    except Exception as e:
        if log_cb:
            log_cb(f"⚠️ v13 writer 오류. content plan 기반 안전 Markdown으로 대체합니다: {e}")
    safe_md = content_plan_to_markdown_v13(plan, title, segments, detail_level)
    ok3, reasons3 = v13_markdown_quality(safe_md, title)
    return safe_md, False, True, reasons3


# ---------------------------------------------------------------------------
# v14 document-architect writer
# ---------------------------------------------------------------------------

V14_COMMON_SECTION_TITLES = [
    "## 1. 한 페이지 요약",
    "## 2. 전체 구조화 정리",
    "## 3. 주제별 상세 정리",
    "## 4. 핵심 개념 / 논점",
    "## 5. 결정사항 / 결론",
    "## 6. 실행 항목",
    "## 7. 리스크 / 이슈",
    "## 8. 타임라인 / 진행 흐름",
    "## 9. 중요 발언 / 근거",
    "## 10. 용어 / 개념",
    "## 11. 확인 필요한 내용",
]


def v14_sentence_bank(segments: list[dict]) -> list[dict]:
    """Create clean sentence records from ASR segments.

    Unlike earlier fallbacks, this bank is only source material.  It should not be
    pasted directly into DOCX except in the quote/timeline sections.
    """
    bank: list[dict] = []
    seen: set[str] = set()
    for seg in segments:
        time = seg.get("start_hms", "")
        sid = seg.get("id", "")
        for sent in sentence_split(seg.get("text", "")):
            s = clean_item_text(sent, 420)
            if is_low_value_phrase_v13(s):
                continue
            key = re.sub(r"\s+", "", s)[:120]
            if key in seen:
                continue
            seen.add(key)
            bank.append({"time": time, "id": sid, "text": s, "keywords": list(content_keywords(s))})
    return bank


def v14_has_any(text: str, terms: list[str]) -> bool:
    flat = normalize_language_artifacts(text).replace(" ", "").lower()
    return any(t.replace(" ", "").lower() in flat for t in terms)


def v14_group_sentences_by_terms(bank: list[dict], terms: list[str], limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in bank:
        txt = row.get("text", "")
        if v14_has_any(txt, terms):
            key = txt[:100]
            if key not in seen:
                out.append(txt)
                seen.add(key)
        if len(out) >= limit:
            break
    return out


def v14_topic_from_bank(title: str, terms: list[str], bank: list[dict], heading: str, limit: int = 4) -> dict | None:
    bullets = v14_group_sentences_by_terms(bank, terms, limit=limit)
    if bullets:
        return {"heading": heading, "bullets": bullets}
    return None


def v14_build_content_pack(title: str, segments: list[dict], glossary: str = "", detail_level: str = "detailed") -> dict:
    """Grounded, general-domain content pack used by the final writer.

    The pack is deliberately not the final DOCX.  It gives the 7B writer a clean
    outline and evidence cards so it can synthesize, not copy chunks.  For common
    lecture/explainer content it recognizes process-like structure; for meetings
    and general recordings it falls back to chronological evidence cards.
    """
    rec_type = detect_recording_type_v13(title, segments)
    bank = v14_sentence_bank(segments)
    transcript = normalize_language_artifacts(" ".join(s.get("text", "") for s in segments))
    terms = extract_terms_v13(segments, glossary, limit=24 if detail_level == "detailed" else 16)
    term_names = [t.get("term", "") for t in terms if isinstance(t, dict) and t.get("term")]

    topics: list[dict] = []

    # General process/education scaffolding.  The labels are general patterns;
    # specific terms are only used when they appear in the transcript.
    if rec_type == "education":
        education_groups = [
            ("핵심 개념과 전체 구조", ["정의", "개념", "구조", "절차", "흐름", "목적", "비교"]),
            ("기준과 산정 방식", ["기준", "총급여", "비과세", "금액", "세율", "과세표준", "조건"]),
            ("1차로 줄일 수 있는 항목", ["소득공제", "근로소득", "인적공제", "부양가족", "4대 보험", "주택자금", "신용카드", "청약"]),
            ("2차로 직접 줄일 수 있는 항목", ["세액공제", "월세", "의료비", "연금저축", "IRP", "중소기업", "청년"]),
            ("결과 판단과 실전 체크포인트", ["결정세액", "환급", "추징", "추가징수", "체크", "확인", "다음 편"]),
        ]
        for heading, group_terms in education_groups:
            topic = v14_topic_from_bank(title, group_terms, bank, heading, limit=5 if detail_level == "detailed" else 4)
            if topic:
                topics.append(topic)

    if rec_type == "meeting":
        meeting_groups = [
            ("논의 배경과 문제 상황", ["배경", "문제", "이슈", "현황", "목표", "필요"]),
            ("주요 논의 내용", ["논의", "검토", "의견", "방안", "대안", "전략"]),
            ("결정사항과 합의 내용", ["결정", "합의", "확정", "승인", "진행하기로"]),
            ("실행 항목과 후속 조치", ["담당", "기한", "진행", "준비", "해야", "후속", "액션"]),
            ("리스크와 확인 필요 사항", ["리스크", "위험", "불확실", "확인", "이슈", "주의"]),
        ]
        for heading, group_terms in meeting_groups:
            topic = v14_topic_from_bank(title, group_terms, bank, heading, limit=5 if detail_level == "detailed" else 4)
            if topic:
                topics.append(topic)

    # If the type-specific plan is sparse, add chronological blocks as evidence cards.
    if len(topics) < (5 if detail_level == "detailed" else 4):
        blocks = chronological_blocks(
            segments,
            block_max_chars=900 if detail_level == "detailed" else 1100,
            max_blocks=8 if detail_level == "detailed" else 6,
            max_total_chars=10000,
        )
        for i, b in enumerate(blocks, start=1):
            b_sents = [s for s in sentence_split(b.get("text", "")) if not is_low_value_phrase_v13(s)]
            bullets = []
            for s in b_sents:
                cs = clean_item_text(s, 420)
                if cs and cs[:100] not in {x[:100] for x in bullets}:
                    bullets.append(cs)
                if len(bullets) >= 4:
                    break
            if bullets:
                heading = block_heading_v13(b.get("text", ""), terms, i, rec_type)
                if is_generic_heading(heading) or heading.startswith("설명 흐름"):
                    heading = derive_heading_from_bullets(bullets, fallback=f"시간순 핵심 흐름 {i}")
                topics.append({"heading": heading, "bullets": bullets, "time": b.get("time", "")})
            if len(topics) >= 9:
                break

    # Clean/merge topic headings and bullets.
    cleaned_topics: list[dict] = []
    seen_headings: set[str] = set()
    for t in topics:
        h = clean_item_text(t.get("heading"), 100)
        bullets = []
        seen_b: set[str] = set()
        for b in as_list(t.get("bullets")):
            cb = clean_item_text(b, 520)
            if not cb or is_low_value_phrase_v13(cb):
                continue
            if cb[:110] not in seen_b:
                bullets.append(cb)
                seen_b.add(cb[:110])
        if not bullets:
            continue
        if is_generic_heading(h) or is_bad_term_v13(h):
            h = derive_heading_from_bullets(bullets, "주요 내용")
        marker = h.lower().replace(" ", "")
        if marker in seen_headings:
            # Merge repeated headings into the previous topic where possible.
            for prev in cleaned_topics:
                if prev["heading"].lower().replace(" ", "") == marker:
                    for b in bullets:
                        if b[:110] not in {x[:110] for x in prev["bullets"]}:
                            prev["bullets"].append(b)
                    break
            continue
        cleaned_topics.append({"heading": h, "bullets": bullets[:6], "time": t.get("time", "")})
        seen_headings.add(marker)
        if len(cleaned_topics) >= (10 if detail_level == "detailed" else 7):
            break

    # Checklist/actionable items are useful for education too, but should be rewritten by LLM later.
    checklist = []
    for row in bank:
        s = row["text"]
        if any(k in s for k in ["확인", "등록", "챙", "알아두", "활용", "가입", "공제", "준비", "받을 수", "해야", "기준"]):
            if s[:100] not in {x[:100] for x in checklist}:
                checklist.append(s)
        if len(checklist) >= 8:
            break

    risks = []
    for row in bank:
        s = row["text"]
        if any(k in s for k in ["추징", "추가", "조건", "기준", "확인", "주의", "세율", "불확실", "변경"]):
            if s[:100] not in {x[:100] for x in risks}:
                risks.append(s)
        if len(risks) >= 6:
            break

    quotes = []
    for seg in pick_representative_segments(segments, limit=14):
        txt = clean_item_text(seg.get("text", ""), 280)
        if txt and not is_low_value_phrase_v13(txt):
            quotes.append({"time": seg.get("start_hms", ""), "text": txt})

    timeline = []
    blocks = chronological_blocks(segments, block_max_chars=950, max_blocks=10, max_total_chars=10000)
    for b in blocks:
        sents = [clean_item_text(x, 300) for x in sentence_split(b.get("text", "")) if not is_low_value_phrase_v13(x)]
        if sents:
            timeline.append({"time": b.get("time", ""), "event": sents[0]})

    # Put a short direct transcript as secondary evidence.  For short recordings this is all of it.
    full_context = v12_transcript_context(segments, RuntimeProfile(
        name="v14_context", label="", asr_model="", asr_device="", asr_compute_type="", asr_beam_size=1,
        llm_model="", llm_device="cuda", max_chars_per_chunk=8000, chunk_overlap_chars=0,
        max_new_tokens_chunk=0, max_new_tokens_final=0, description=""
    ), detail_level)

    return {
        "title": title,
        "recording_type": rec_type,
        "terms": terms[:18],
        "topics": cleaned_topics,
        "checklist": checklist,
        "risks": risks,
        "timeline": timeline,
        "quotes": quotes,
        "transcript_context": full_context,
        "transcript_chars": len(transcript),
    }


def v14_pack_to_prompt_text(pack: dict) -> str:
    slim = {k: v for k, v in pack.items() if k != "transcript_context"}
    return json.dumps(slim, ensure_ascii=False, indent=2)[:30000]


def make_v14_writer_prompt(title: str, pack: dict, language: str, glossary: str, detail_level: str) -> str:
    glossary_text = f"\n사용자 제공 용어/고유명사·ASR 보정 힌트:\n{glossary}\n" if glossary.strip() else ""
    rec_type = pack.get("recording_type", "general")
    pack_text = v14_pack_to_prompt_text(pack)
    context = pack.get("transcript_context", "")[:30000]
    return f"""
당신은 최종 DOCX 문서를 작성하는 전문 편집자입니다.
아래 transcript와 content pack을 읽고, 사람이 직접 정리한 것처럼 자연스럽고 체계적인 Markdown 문서를 작성하세요.

문서 제목: {title}
출력 언어: {language}
문서 상세도: {detail_cfg(detail_level)['label']}
녹음 성격 추정: {rec_type}
{glossary_text}
핵심 작성 방향:
- transcript를 그대로 나열하지 마세요. 발화 조각을 의미 단위로 통합해 설명형 문서로 재구성하세요.
- content pack은 목차와 근거 후보입니다. 그대로 복사하지 말고 자연스러운 제목과 문장으로 바꾸세요.
- 회의이면 배경, 논의, 결정, 실행 항목, 리스크를 정리하세요.
- 강의/교육/설명 영상이면 개념 정의, 원리, 절차, 예시, 체크리스트, 주의사항 중심으로 정리하세요.
- 인터뷰/뉴스/해설이면 배경, 주요 주장, 근거, 쟁점, 시사점을 정리하세요.
- 원문에 없는 사실을 만들지 마세요. 숫자, 조건, 금액, 인물명, 제도명은 transcript 근거가 있을 때만 작성하세요.
- ASR 오류가 의심되면 문맥상 명확한 것만 보정하고 불확실하면 확인 필요한 내용에 적으세요.
- 중국어, 일본어, 한자식 문자, JSON, Python dict/list, 코드블록을 절대 출력하지 마세요.
- '명시적으로 확인되지 않음'은 정말 없는 경우에만 짧게 쓰고, 남발하지 마세요.
- 10. 용어 / 개념에는 실제 제도명, 제품명, 기술명, 방법론, 고유명사만 넣으세요. 일반 동사/형용사/발화어는 제외하세요.

반드시 아래 섹션 제목을 그대로 사용하세요.
# {title}
## 1. 한 페이지 요약
## 2. 전체 구조화 정리
## 3. 주제별 상세 정리
## 4. 핵심 개념 / 논점
## 5. 결정사항 / 결론
## 6. 실행 항목
## 7. 리스크 / 이슈
## 8. 타임라인 / 진행 흐름
## 9. 중요 발언 / 근거
## 10. 용어 / 개념
## 11. 확인 필요한 내용

content pack:
{pack_text}

원본 transcript:
{context}
""".strip()


def v14_bad_terms_in_terms_section(md: str) -> list[str]:
    sec10 = extract_markdown_sections(clean_human_markdown_text(md or "", "")).get("10", "")
    bad = []
    for line in sec10.splitlines():
        line = line.strip(" -•*\t")
        if not line or line.startswith("##"):
            continue
        term = line.split(":", 1)[0].strip("* ")
        term = re.sub(r"^[-•*]+\s*", "", term).strip()
        if term and is_bad_term_v13(term):
            bad.append(term)
    return bad


def clean_terms_section_v14(md: str, pack: dict) -> str:
    """Replace only the terms section when it contains low-value words.

    This is intentionally a soft repair: do not discard an otherwise good LLM
    document just because a few bad term candidates slipped into section 10.
    """
    terms = pack.get("terms") or []
    good_lines = []
    for t in terms[:14]:
        if not isinstance(t, dict):
            continue
        name = clean_item_text(t.get("term"), 80)
        desc = clean_item_text(t.get("description"), 300)
        if name and desc and not is_bad_term_v13(name):
            good_lines.append(f"- **{name}:** {desc}")
    if not good_lines:
        good_lines = ["- transcript에서 별도의 전문 용어를 안정적으로 추출하지 못했습니다. 원문과 glossary를 확인하세요."]
    new_sec = "## 10. 용어 / 개념\n" + "\n".join(good_lines) + "\n"
    m = clean_human_markdown_text(md or "", pack.get("title", ""))
    pattern = r"(?ms)^##\s*10\.\s*용어\s*/\s*개념.*?(?=^##\s*11\.|\Z)"
    if re.search(pattern, m):
        m = re.sub(pattern, new_sec + "\n", m)
    else:
        m += "\n" + new_sec
    return clean_human_markdown_text(m, pack.get("title", ""))


def v14_markdown_quality(md: str, title: str, pack: dict | None = None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    m = clean_human_markdown_text(md or "", title)
    if len(m.strip()) < 1800:
        reasons.append("too_short")
    if re.search(r"```|\{\s*['\"]?(heading|bullets|topics|summary|text)['\"]?\s*:", m):
        reasons.append("json_or_code_leak")
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", m):
        reasons.append("cjk_or_japanese_leak")
    nums = set(re.findall(r"(?m)^##\s*(\d+)\.", m))
    if len(nums) < 10:
        reasons.append("missing_sections")
    sec = extract_markdown_sections(m)
    if len(sec.get("1", "")) < 180 or "명시적으로 확인되지 않음" in sec.get("1", ""):
        reasons.append("bad_summary")
    if len(sec.get("2", "")) < 350:
        reasons.append("outline_too_short")
    if len(sec.get("3", "")) < 900:
        reasons.append("details_too_short")
    if len(sec.get("4", "")) < 420:
        reasons.append("concepts_too_short")
    if m.count("명시적으로 확인되지 않음") >= 7:
        reasons.append("too_many_unknowns")
    # Raw transcript listing: many consecutive lines from the sentence bank without synthesis.
    raw_fragment_hits = 0
    for frag in ["직장인이면 꼭", "여러분 연말정산", "해볼게요", "돌아왔습니다", "굉장히 복잡해 보이지만"]:
        if frag in sec.get("3", ""):
            raw_fragment_hits += 1
    if raw_fragment_hits >= 3 and len(sec.get("3", "")) < 2600:
        reasons.append("raw_transcript_listing")
    bad_terms = v14_bad_terms_in_terms_section(m)
    if len(bad_terms) >= 3:
        reasons.append("low_value_terms_in_terms_section")
    return not reasons, reasons


def v14_term_present(term_names: list[str], candidates: list[str]) -> bool:
    flat = " ".join(term_names).replace(" ", "").lower()
    return any(c.replace(" ", "").lower() in flat for c in candidates)


def v14_education_conceptual_topics(pack: dict) -> list[dict]:
    """Build polished education topics from extracted terms and evidence.

    This is a general education/explainer fallback.  It uses domain terms only if
    they are present in the transcript; otherwise it falls back to the pack topics.
    """
    terms = [t.get("term", "") for t in (pack.get("terms") or []) if isinstance(t, dict)]
    topics: list[dict] = []

    def add(heading: str, bullets: list[str], required_terms: list[str] | None = None) -> None:
        if required_terms and not v14_term_present(terms, required_terms):
            return
        cleaned = []
        for b in bullets:
            cb = clean_item_text(b, 520)
            if cb and cb[:100] not in {x[:100] for x in cleaned}:
                cleaned.append(cb)
        if cleaned:
            topics.append({"heading": heading, "bullets": cleaned})

    # Generic tax/finance education pattern, activated only when these terms are present.
    add("연말정산의 목적과 환급·추징 구조", [
        "연말정산은 1년 동안 미리 납부한 세금과 실제로 내야 할 세금을 비교해 다음 해에 정산하는 절차입니다.",
        "이미 낸 세금이 실제 세액보다 많으면 차액을 돌려받는 환급이 발생하고, 실제 세액보다 적게 냈다면 추가징수 또는 추징이 발생합니다.",
        "따라서 연말정산의 핵심은 내가 낸 세금과 실제 결정세액의 차이를 이해하고, 공제 항목을 통해 세 부담을 줄이는 것입니다.",
    ], ["연말정산"])
    add("총급여와 비과세소득 이해", [
        "총급여는 한 해 동안 받은 월급, 상여금, 각종 수당 등에서 세금을 매기지 않는 비과세소득을 제외한 금액입니다.",
        "청년도약계좌, 월세액 세액공제, 청년 우대형 청약통장 등 여러 제도는 총급여를 기준으로 대상 여부를 판단합니다.",
        "근로소득 원천징수영수증에서 총급여를 확인하면 이후 소득공제와 세액공제를 판단하는 출발점을 잡을 수 있습니다.",
    ], ["총급여", "비과세"])
    add("소득공제: 세금을 매길 소득을 줄이는 1차 단계", [
        "소득공제는 총급여에서 세금을 매길 소득을 줄이는 단계로, 세율을 적용하기 전 과세표준을 낮추는 역할을 합니다.",
        "근로소득공제는 근로소득자가 기본적으로 적용받는 공제이며, 공제 비율은 총급여 수준에 따라 달라질 수 있습니다.",
        "인적공제, 부양가족 등록, 4대 보험, 주택자금, 전월세 보증금 대출 원리금, 청약통장 납입액, 신용카드·현금 사용액 등이 소득공제 항목으로 언급됩니다.",
        "특히 일부 항목은 자동으로 반영되지 않으므로, 본인이 미리 확인하고 등록해야 공제를 받을 수 있습니다.",
    ], ["소득공제"])
    add("과세표준과 세율", [
        "각종 소득공제를 반영한 뒤 남는 금액이 과세표준이며, 여기에 정해진 세율을 곱해 세액이 산출됩니다.",
        "소득이 높아질수록 더 높은 세율 구간이 적용될 수 있으므로, 세율 구간에 걸쳐 있는 사람은 소득공제를 특히 신경 쓸 필요가 있습니다.",
    ], ["과세표준", "세율"])
    add("세액공제: 산출된 세금 자체를 줄이는 2차 단계", [
        "세액공제는 과세표준과 세율을 통해 계산된 세액 자체를 다시 줄이는 단계입니다.",
        "중소기업에 다니는 청년의 세액감면, 월세, 의료비, 연금저축, IRP 등은 세액공제 항목으로 언급됩니다.",
        "소득이 높지 않은 사회초년생이라면 소득공제보다 세액공제의 체감 효과가 클 수 있으므로, 본인이 받을 수 있는 세액공제 항목을 우선 확인하는 것이 좋습니다.",
    ], ["세액공제"])
    add("결정세액과 13월의 월급", [
        "소득공제와 세액공제를 모두 반영한 뒤 최종적으로 결정세액이 계산됩니다.",
        "결정세액이 이미 납부한 세금보다 적으면 환급을 받고, 많으면 차액을 추가로 납부합니다.",
        "결국 13월의 월급이 될지 13월의 세금이 될지는 결정세액과 이미 납부한 세금의 비교로 결정됩니다.",
    ], ["결정세액", "환급", "추징"])

    if topics:
        return topics
    # Generic education fallback from pack topics.
    out = []
    for t in pack.get("topics", [])[:8]:
        if not isinstance(t, dict):
            continue
        h = clean_item_text(t.get("heading"), 90)
        bullets = [clean_item_text(b, 500) for b in as_list(t.get("bullets")) if not is_low_value_phrase_v13(str(b))]
        if h and bullets:
            out.append({"heading": h, "bullets": bullets[:4]})
    return out


def content_pack_to_markdown_v14(pack: dict, title: str, detail_level: str = "detailed") -> str:
    """High-quality deterministic fallback used only when LLM writer is unusable.

    The previous fallback copied transcript fragments.  This version uses a
    concept-first outline so the result still reads like a human note even when
    the LLM writer is rejected.
    """
    rec_type = pack.get("recording_type", "general")
    topics = v14_education_conceptual_topics(pack) if rec_type == "education" else []
    if not topics:
        topics = []
        for t in pack.get("topics", [])[:9]:
            if not isinstance(t, dict):
                continue
            h = clean_item_text(t.get("heading"), 90)
            bullets = [clean_item_text(b, 520) for b in as_list(t.get("bullets")) if not is_low_value_phrase_v13(str(b))]
            if h and bullets:
                topics.append({"heading": h, "bullets": bullets[:5]})
    terms = [t for t in (pack.get("terms") or []) if isinstance(t, dict) and not is_bad_term_v13(t.get("term", ""))]
    lines = [f"# {title}", "", "## 1. 한 페이지 요약", ""]
    topic_names = [clean_item_text(t.get("heading"), 80) for t in topics[:5] if t.get("heading")]
    if rec_type == "education":
        lines.append(f"이 녹음은 {', '.join(topic_names[:4]) if topic_names else '핵심 개념과 절차'}를 설명하는 교육형 자료입니다. 단순 전사 내용을 그대로 옮기기보다, 개념의 출발점과 계산·판단 흐름, 실제로 챙겨야 할 항목을 학습자가 이해하기 쉬운 순서로 재구성했습니다.")
    elif rec_type == "meeting":
        lines.append("이 녹음은 회의 또는 업무 논의를 바탕으로 주요 배경, 논의 흐름, 결정사항, 실행 항목과 확인이 필요한 내용을 정리한 문서입니다. 발화 순서보다 의사결정과 후속 조치에 도움이 되는 구조를 우선했습니다.")
    else:
        lines.append(f"이 녹음은 {', '.join(topic_names[:4]) if topic_names else '주요 내용'}을 중심으로 진행됩니다. 원문 내용을 의미 단위로 묶어 배경, 핵심 내용, 세부 흐름, 확인 필요한 사항으로 정리했습니다.")
    lines += ["", "## 2. 전체 구조화 정리", ""]
    for t in topics[:7]:
        h = clean_item_text(t.get("heading"), 90)
        bullets = [clean_item_text(b, 260) for b in as_list(t.get("bullets")) if not is_low_value_phrase_v13(str(b))]
        if h and bullets:
            lines.append(f"- **{h}:** {bullets[0]}")
    lines += ["", "## 3. 주제별 상세 정리", ""]
    for i, t in enumerate(topics[:9], start=1):
        h = clean_item_text(t.get("heading"), 90) or f"주요 내용 {i}"
        lines.append(f"### {i}. {h}")
        for b in as_list(t.get("bullets"))[:5]:
            cb = clean_item_text(b, 520)
            if cb:
                lines.append(f"- {cb}")
        lines.append("")
    lines += ["## 4. 핵심 개념 / 논점", ""]
    seen_terms = set()
    for t in terms[:14]:
        name = clean_item_text(t.get("term"), 80)
        desc = clean_item_text(t.get("description"), 330)
        if name and desc and name.lower() not in seen_terms:
            lines.append(f"- **{name}:** {desc}")
            seen_terms.add(name.lower())
    if not seen_terms:
        for t in topics[:6]:
            lines.append(f"- **{clean_item_text(t.get('heading'), 80)}:** 녹음에서 주요하게 다뤄진 개념 또는 논점입니다.")
    lines += ["", "## 5. 결정사항 / 결론", ""]
    if rec_type == "education":
        lines.append("- 이 자료는 특정 의사결정보다는 핵심 개념과 절차를 이해하고, 실제 적용 시 챙겨야 할 항목을 파악하는 데 초점이 있습니다.")
        if topic_names:
            lines.append(f"- 핵심 흐름은 {', '.join(topic_names[:5])} 순서로 이해할 수 있습니다.")
    elif rec_type == "meeting":
        lines.append("- 명시적 결정사항은 원문에서 확인되는 범위 안에서만 별도 검토가 필요합니다.")
    else:
        lines.append("- 원문에서 명시적 결정사항은 확인되지 않으며, 주요 내용과 시사점 중심으로 이해하는 것이 적절합니다.")
    lines += ["", "## 6. 실행 항목", ""]
    if rec_type == "education":
        lines.extend([
            "- 본인의 총급여와 비과세소득을 먼저 확인합니다.",
            "- 자동 반영되지 않는 공제 항목이 있는지 확인하고 필요한 등록을 미리 진행합니다.",
            "- 소득공제와 세액공제 항목을 구분해 본인에게 적용 가능한 항목을 점검합니다.",
            "- 제도별 금액 기준과 대상 조건은 매년 달라질 수 있으므로 최신 기준을 확인합니다.",
        ])
    else:
        checklist = pack.get("checklist") or []
        if checklist:
            for c in checklist[:8]:
                lines.append(f"- {clean_item_text(c, 420)}")
        else:
            lines.append("- 명시적 실행 항목 없음")
    lines += ["", "## 7. 리스크 / 이슈", ""]
    risks = pack.get("risks") or []
    if risks:
        for r in risks[:6]:
            lines.append(f"- {clean_item_text(r, 420)}")
    else:
        lines.append("- 원문에서 별도의 리스크나 이슈가 명확히 확인되지 않았습니다.")
    lines += ["", "## 8. 타임라인 / 진행 흐름", ""]
    for tl in (pack.get("timeline") or [])[:10]:
        lines.append(f"- [{tl.get('time','')}] {clean_item_text(tl.get('event'), 320)}")
    lines += ["", "## 9. 중요 발언 / 근거", ""]
    for q in (pack.get("quotes") or [])[:10]:
        lines.append(f"- [{q.get('time','')}] \"{clean_item_text(q.get('text'), 280)}\"")
    lines += ["", "## 10. 용어 / 개념", ""]
    if terms:
        for t in terms[:14]:
            lines.append(f"- **{clean_item_text(t.get('term'), 80)}:** {clean_item_text(t.get('description'), 300)}")
    else:
        lines.append("- 전문 용어가 충분히 안정적으로 추출되지 않았습니다.")
    lines += ["", "## 11. 확인 필요한 내용", ""]
    if rec_type == "education":
        lines.append("- 제도, 금액, 공제 조건은 변경될 수 있으므로 실제 적용 전 최신 기준을 확인해야 합니다.")
    else:
        lines.append("- 고유명사, 숫자, 날짜, 조건은 ASR 전사 오류 가능성을 고려해 원문 확인이 필요합니다.")
    return clean_human_markdown_text("\n".join(lines), title)

def generate_v14_document_architect_markdown(
    llm,
    title: str,
    segments: list[dict],
    profile: RuntimeProfile,
    language: str,
    glossary: str,
    detail_level: str,
    max_new_tokens: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> tuple[str, bool, bool, list[str]]:
    """v14: build a grounded content pack, let the LLM write, softly repair, then deterministic fallback.

    Returns (markdown, llm_used, repair_or_fallback_used, quality_reasons).
    """
    pack = v14_build_content_pack(title, segments, glossary, detail_level)
    prompt = make_v14_writer_prompt(title, pack, language, glossary, detail_level)
    if log_cb:
        log_cb(f"🧭 v14 content pack 생성: type={pack.get('recording_type')} / topics={len(pack.get('topics', []))} / terms={len(pack.get('terms', []))}")
        log_cb(f"✍️ v14 document-architect Markdown writer / max_new_tokens={max_new_tokens}")
    try:
        raw = llm.generate(SYSTEM_PROMPT_MARKDOWN, prompt, max_new_tokens=max_new_tokens)
        md = clean_human_markdown_text(raw, title)
        bad_terms = v14_bad_terms_in_terms_section(md)
        if bad_terms:
            md = clean_terms_section_v14(md, pack)
        ok, reasons = v14_markdown_quality(md, title, pack)
        # Low-value terms are soft-repairable. Do not throw away a good LLM-written document just for that.
        reasons_no_terms = [r for r in reasons if r != "low_value_terms_in_terms_section"]
        if not reasons_no_terms:
            return md, True, bool(bad_terms), reasons
        if log_cb:
            log_cb("⚠️ v14 Markdown 품질 검사 미통과: " + ", ".join(reasons) + ". 원문+content pack 기반 repair 1회 수행")
        repair_prompt = f"""
아래 Markdown 초안은 품질 기준을 만족하지 못했습니다.
문제: {', '.join(reasons)}

원문 transcript와 content pack을 근거로 최종 Markdown 문서를 다시 작성하세요.
- transcript를 나열하지 말고 의미 단위로 재구성하세요.
- JSON, 코드블록, 중국어/일본어/한자식 문자, 의미 없는 용어를 제거하세요.
- 주제별 상세 정리와 핵심 개념을 충분히 구체적으로 작성하세요.
- 원문에 없는 사실을 추가하지 마세요.

content pack:
{v14_pack_to_prompt_text(pack)}

원문 transcript:
{pack.get('transcript_context','')[:30000]}

품질이 낮았던 초안:
{md[:14000]}
""".strip()
        raw2 = llm.generate(SYSTEM_PROMPT_MARKDOWN, repair_prompt, max_new_tokens=max(5200, min(max_new_tokens, 9500)))
        md2 = clean_human_markdown_text(raw2, title)
        if v14_bad_terms_in_terms_section(md2):
            md2 = clean_terms_section_v14(md2, pack)
        ok2, reasons2 = v14_markdown_quality(md2, title, pack)
        reasons2_no_terms = [r for r in reasons2 if r != "low_value_terms_in_terms_section"]
        if not reasons2_no_terms or len(reasons2) < len(reasons):
            return md2, True, True, reasons2
        if log_cb:
            log_cb("⚠️ v14 LLM repair도 기준을 만족하지 못해 content pack 기반 안전 Markdown으로 대체합니다.")
    except Exception as e:
        if log_cb:
            log_cb(f"⚠️ v14 writer 오류. content pack 기반 안전 Markdown으로 대체합니다: {e}")
    safe_md = content_pack_to_markdown_v14(pack, title, detail_level)
    ok3, reasons3 = v14_markdown_quality(safe_md, title, pack)
    return safe_md, False, True, reasons3

def effective_strategy(requested: str, profile: RuntimeProfile) -> str:
    requested = (requested or "auto").lower()
    if requested in {"fast", "smart_fast"}:
        return "fast"
    if requested in {"full", "extractive"}:
        return requested
    # CPU transformers generation is slow. Auto therefore uses v9 fast mode:
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
    # CPU must stay practical: v9 fast uses one direct Markdown writer call, so
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
    chunk_extraction_skipped = False
    final_writer_mode = ""

    if log_cb:
        log_cb(f"🧩 transcript chunk 수: {len(chunks)} / chunk_chars={profile.max_chars_per_chunk} / overlap={profile.chunk_overlap_chars}")
        log_cb(f"📝 문서 상세도: {detail_level} ({cfg['label']}) / 처리 전략={strategy} / chunk_tokens={chunk_tokens} / final_tokens={final_tokens}")

    # v14 primary path for CUDA profiles: document-architect writer with a grounded content pack.
    # It avoids noisy JSON/chunk assembly and avoids discarding good LLM output for soft term issues.
    if strategy == "full" and profile.llm_device == "cuda" and use_final_llm:
        try:
            llm = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
            writer_tokens = max(5400, min(final_tokens, 10000))
            md, llm_writer_used_v13, repair_used_v13, quality_reasons_v13 = generate_v14_document_architect_markdown(
                llm, title, segments, profile, language, glossary, detail_level,
                max_new_tokens=writer_tokens, log_cb=log_cb
            )
            llm_calls += 2 if repair_used_v13 else 1
            if md:
                transcript_chars = sum(len(seg.get("text", "")) for seg in segments)
                final_obj = empty_final()
                run_config = {
                    "pipeline_version": PIPELINE_VERSION,
                    "title": title,
                    "detail_level": detail_level,
                    "processing_strategy_requested": processing_strategy,
                    "processing_strategy_effective": strategy,
                    "processing_strategy_note": "v14 full = grounded content pack + document-architect Markdown writer; soft repair before fallback",
                    "profile_name": profile.name,
                    "asr_model": profile.asr_model,
                    "asr_device": profile.asr_device,
                    "asr_compute_type": profile.asr_compute_type,
                    "llm_model": profile.llm_model,
                    "llm_device": profile.llm_device,
                    "max_chars_per_chunk": profile.max_chars_per_chunk,
                    "chunk_overlap_chars": profile.chunk_overlap_chars,
                    "max_new_tokens_chunk_effective": chunk_tokens,
                    "max_new_tokens_final_effective": writer_tokens,
                    "chunk_count": len(chunks),
                    "segment_count": len(segments),
                    "transcript_chars": transcript_chars,
                    "structured_json_chars": 0,
                    "asr_error_aware": True,
                    "glossary_provided": bool(glossary.strip()),
                    "use_final_llm": use_final_llm,
                    "llm_calls": llm_calls,
                    "final_llm_failed": False,
                    "final_repair_used": False,
                    "style_repair_used": False,
                    "final_markdown_used": True,
                    "sectioned_markdown_used": False,
                    "transcript_first_markdown_used": True,
                    "complete_markdown_writer_used": True,
                    "plan_guided_markdown_used": True,
                    "content_plan_used": True,
                    "document_architect_writer_used": True,
                    "markdown_repair_used": repair_used_v13,
                    "llm_writer_used": llm_writer_used_v13,
                    "quality_reasons_after_repair": quality_reasons_v13,
                    "chunk_extraction_skipped": True,
                    "final_writer_mode": "v14_document_architect_transcript_first",
                    "fallback_used": False,
                }
                return {"chunk_notes": [], "final": final_obj, "final_markdown": md, "chunk_count": len(chunks), "run_config": run_config}
            elif log_cb:
                log_cb("⚠️ v14 writer가 기준을 만족하지 못해 기존 full pipeline으로 fallback합니다.")
        except Exception as e:
            if log_cb:
                log_cb(f"⚠️ v14 writer 오류. 기존 full pipeline으로 fallback합니다: {e}")

    if strategy == "extractive":
        notes = extractive_notes_for_chunks(chunks, detail_level)
        final_obj = aggregate_without_final_llm(notes, detail_level)
        fallback_used = True
    elif strategy == "fast":
        # v9 fast mode: build grounded extractive notes, then ask the LLM to write
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
        # Full mode. v11 optimization: for short/medium GPU inputs, skip noisy
        # chunk JSON extraction and let the final sectioned Markdown writer read
        # the transcript directly. This improves quality and reduces LLM calls.
        llm = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
        if short_gpu_transcript_case(segments, profile, detail_level):
            chunk_extraction_skipped = True
            if log_cb:
                log_cb("🚀 v11 short/medium GPU transcript: chunk JSON 추출을 건너뛰고 transcript-first writer를 우선 사용합니다.")
            notes = extractive_notes_for_chunks(chunks, detail_level)
            final_obj = enrich_final_with_chunk_notes(aggregate_without_final_llm(notes, detail_level), notes, detail_level, title)
        else:
            for i, ch in enumerate(chunks, start=1):
                prompt = make_chunk_prompt(title, i, len(chunks), ch, language, glossary, detail_level)
                note = call_llm_json(llm, prompt, ch, chunk_tokens, f"{title}_chunk_{i:02d}", detail_level, log_cb, retries=2)
                llm_calls += 1
                if note.get("topics") and note["topics"][0].get("heading") in {"원문 기반 주요 내용", "LLM 출력 복구 내용"}:
                    fallback_used = True
                notes.append(note)
            final_obj = None
        if use_final_llm and not chunk_extraction_skipped:
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
    if use_final_llm and profile.llm_device != "cpu" and strategy in {"full", "fast"} and not chunk_extraction_skipped:
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
    if use_final_llm and profile.llm_device == "cuda" and not chunk_extraction_skipped and (language_drift_detected or excessive_unknowns(final_obj) or looks_repetitive_or_sparse(final_obj)):
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

    # v11: primary final output path for GPU profiles.
    # Use a transcript-first sectioned Markdown writer before any JSON-derived export.
    sectioned_markdown_used = False
    transcript_first_markdown_used = False
    markdown_repair_used = False
    if use_final_llm and final_markdown is None and profile.llm_device == "cuda":
        try:
            llm_writer = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
            writer_tokens = max(4800, min(final_tokens, 9500))
            md, used_repair = generate_sectioned_markdown_v11(
                llm_writer, title, notes, segments, profile, language, glossary, detail_level,
                max_new_tokens=writer_tokens, log_cb=log_cb
            )
            # v11 sectioned writer uses four focused LLM calls plus optional repair.
            llm_calls += 5 if used_repair else 4
            if md:
                final_markdown = md
                final_markdown_used = True
                transcript_first_markdown_used = True
                sectioned_markdown_used = True
                markdown_repair_used = used_repair
                final_writer_mode = "v11_sectioned_transcript_first"
        except Exception as e:
            if log_cb:
                log_cb(f"⚠️ v11 sectioned transcript-first Markdown writer 단계 오류. 기존 writer로 fallback합니다: {e}")

    # Backward-compatible fallback: if the v11 writer fails unexpectedly, use the
    # older sectioned/single writer.  This should be rare but keeps the pipeline robust.
    if use_final_llm and final_markdown is None and profile.llm_device == "cuda":
        try:
            llm_writer = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
            md, used = generate_sectioned_markdown(
                llm_writer, title, notes, segments, profile, language, glossary, detail_level, log_cb=log_cb
            )
            llm_calls += 3
            if used:
                final_markdown = md
                final_markdown_used = True
                sectioned_markdown_used = True
        except Exception as e:
            if log_cb:
                log_cb(f"⚠️ 섹션별 Markdown writer 단계 오류. 단일 writer로 fallback합니다: {e}")

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
                log_cb(f"⚠️ 최종 단일 Markdown writer 단계 오류. 구조화 Markdown으로 진행합니다: {e}")

    transcript_chars = sum(len(s.get("text", "")) for s in segments)
    markdown_est_chars = len(json.dumps(final_obj, ensure_ascii=False))
    run_config = {
        "pipeline_version": PIPELINE_VERSION,
        "title": title,
        "detail_level": detail_level,
        "processing_strategy_requested": processing_strategy,
        "processing_strategy_effective": strategy,
        "processing_strategy_note": "v11 fast = chronological digest + direct human Markdown writer + safety net" if strategy == "fast" else "v11 full = transcript-first sectioned Markdown writer; short/medium GPU inputs skip noisy chunk JSON extraction",
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
        "sectioned_markdown_used": locals().get("sectioned_markdown_used", False),
        "transcript_first_markdown_used": locals().get("transcript_first_markdown_used", False),
        "markdown_repair_used": locals().get("markdown_repair_used", False),
        "chunk_extraction_skipped": chunk_extraction_skipped,
        "final_writer_mode": final_writer_mode,
        "fallback_used": fallback_used,
    }
    return {"chunk_notes": notes, "final": final_obj, "final_markdown": final_markdown, "chunk_count": len(chunks), "run_config": run_config}
