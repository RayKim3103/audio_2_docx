from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from .llm import get_llm
from .profiles import RuntimeProfile
from .utils import clean_text_for_xml, humanize_llm_value

PIPELINE_VERSION = "general_meeting_v3"

SYSTEM_PROMPT = """
당신은 보안이 중요한 로컬 환경에서 동작하는 회의록/녹음 정리 전문 AI입니다.
반드시 입력 transcript에 근거해서만 작성하세요.
회의가 아닌 강의, 교육, 인터뷰, 발표 녹음도 처리할 수 있도록 일반 도메인으로 정리하세요.
최종 독자는 사람이므로 모든 값은 자연스러운 한국어 문장으로 작성하세요.
원문에 없는 참석자, 담당자, 기한, 결정사항은 만들지 말고 '확인 필요' 또는 '명시적으로 확인되지 않음'이라고 쓰세요.
작성 지시문, 예시 문장, placeholder, 마크다운 코드블록을 출력하지 마세요.
Python dict/list 문자열, JSON 문자열, key-value dump를 본문 값으로 넣지 마세요. 예: {'heading': ..., 'bullets': ...} 같은 형식은 금지입니다.
출력은 요청한 JSON 객체 하나만 반환하세요.
""".strip()

# Strong template leaks: these are prompt instructions/placeholders that must never
# become final content. We intentionally keep this list focused to avoid rejecting
# legitimate Korean words such as '근거' or '설명'.
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
        "structured_limit": 25,
        "timeline_limit": 25,
        "quotes_limit": 12,
        "token_multiplier": 0.85,
    },
    "standard": {
        "label": "표준 회의록",
        "chunk_topics": 5,
        "topic_bullets": 5,
        "summary_bullets": 6,
        "structured_limit": 50,
        "timeline_limit": 40,
        "quotes_limit": 18,
        "token_multiplier": 1.0,
    },
    "detailed": {
        "label": "상세 회의록",
        "chunk_topics": 7,
        "topic_bullets": 7,
        "summary_bullets": 8,
        "structured_limit": 90,
        "timeline_limit": 70,
        "quotes_limit": 30,
        "token_multiplier": 1.15,
    },
}


def detail_cfg(detail_level: str) -> dict:
    return DETAIL_SETTINGS.get(detail_level or "standard", DETAIL_SETTINGS["standard"])


def has_template_leak(text: str) -> bool:
    if not text:
        return False
    return any(p in text for p in TEMPLATE_PHRASES)


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
    s = re.sub(r"\s+", " ", s)
    if has_template_leak(s):
        return ""
    # If a literal-looking fragment survived parsing, do not invalidate the whole result;
    # just strip the most visible braces and quotes so the DOCX remains human-readable.
    if s.startswith("{") and s.endswith("}"):
        s = s.strip("{}").replace("'", "").replace('\"', "")
    return s[:max_len]


def is_weak_placeholder(text: str) -> bool:
    return clean_item_text(text).strip() in WEAK_PLACEHOLDERS


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
    for b in as_list(note.get("summary_bullets") or note.get("summary")):
        s = clean_item_text(b, 500)
        if s and not is_weak_placeholder(s):
            out["summary_bullets"].append(s)
    out["summary_bullets"] = out["summary_bullets"][: cfg["summary_bullets"]]

    for t in as_list(note.get("topics") or note.get("detailed_topics") or note.get("structured_notes")):
        if not isinstance(t, dict):
            continue
        heading = clean_item_text(t.get("heading") or t.get("topic") or t.get("title") or "주요 논점", 120)
        bullets = [clean_item_text(x, 700) for x in as_list(t.get("bullets") or t.get("details") or t.get("items"))]
        bullets = [x for x in bullets if x and not is_weak_placeholder(x)][: cfg["topic_bullets"]]
        if bullets:
            if is_weak_placeholder(heading):
                heading = bullets[0][:40]
            out["topics"].append({"heading": heading or "주요 논점", "bullets": bullets, "evidence": compact_evidence(t.get("evidence"), 2)})
    out["topics"] = out["topics"][: cfg["structured_limit"]]

    for key in ["decisions", "risks_issues", "open_questions"]:
        for item in as_list(note.get(key)):
            if isinstance(item, dict):
                text = item.get("text") or item.get("decision") or item.get("issue") or item.get("question") or item.get("point") or ""
                ev = compact_evidence(item.get("evidence"), 2)
            else:
                text, ev = item, []
            text = clean_item_text(text, 800)
            if text and not is_weak_placeholder(text):
                out[key].append({"text": text, "evidence": ev})

    for a in as_list(note.get("action_items")):
        if not isinstance(a, dict):
            task = clean_item_text(a, 800)
            if task:
                out["action_items"].append({"task": task, "owner": "확인 필요", "due_date": "확인 필요", "evidence": []})
            continue
        task = clean_item_text(a.get("task") or a.get("할 일") or a.get("text"), 800)
        if task:
            out["action_items"].append({
                "task": task,
                "owner": clean_item_text(a.get("owner") or a.get("담당자") or "확인 필요", 80) or "확인 필요",
                "due_date": clean_item_text(a.get("due_date") or a.get("기한") or "확인 필요", 80) or "확인 필요",
                "evidence": compact_evidence(a.get("evidence"), 2),
            })

    for q in as_list(note.get("key_quotes")):
        if isinstance(q, dict):
            quote = clean_item_text(q.get("quote") or q.get("text"), 450)
            ev = compact_evidence(q.get("evidence") or q.get("time"), 2)
        else:
            quote, ev = clean_item_text(q, 450), []
        if quote and not is_weak_placeholder(quote):
            out["key_quotes"].append({"quote": quote, "evidence": ev})
    out["key_quotes"] = out["key_quotes"][: cfg["quotes_limit"]]

    for tl in as_list(note.get("timeline")):
        if isinstance(tl, dict):
            time = clean_item_text(tl.get("time") or tl.get("시점"), 80)
            event = clean_item_text(tl.get("event") or tl.get("내용") or tl.get("text"), 800)
        else:
            time, event = "", clean_item_text(tl, 800)
        if event and not is_weak_placeholder(event):
            out["timeline"].append({"time": time, "event": event})
    out["timeline"] = out["timeline"][: cfg["timeline_limit"]]

    for term in as_list(note.get("terms")):
        if isinstance(term, dict):
            name = clean_item_text(term.get("term") or term.get("name"), 120)
            desc = clean_item_text(term.get("description") or term.get("desc"), 500)
        else:
            name, desc = clean_item_text(term, 120), ""
        if name and not is_weak_placeholder(name):
            out["terms"].append({"term": name, "description": desc})

    # If the LLM created some topics but no summary, derive a short summary from the topics.
    if not out["summary_bullets"] and out["topics"]:
        for t in out["topics"][: cfg["summary_bullets"]]:
            if t.get("bullets"):
                out["summary_bullets"].append(t["bullets"][0])

    # Source-based fallback only when the model output is almost empty.
    if not out["summary_bullets"] and not out["topics"] and source_segments:
        text = " ".join(s.get("text", "") for s in source_segments[:10]).strip()
        if text:
            out["summary_bullets"] = [text[:400]]
            out["topics"] = [{"heading": "원문 기반 주요 내용", "bullets": [text[:700]], "evidence": segment_evidence(source_segments[:1])}]
    return out


def segment_evidence(segments: list[dict], limit: int = 2) -> list[str]:
    ev = []
    for s in segments[:limit]:
        if all(k in s for k in ("id", "start_hms", "end_hms")):
            ev.append(f"[{s['id']} {s['start_hms']}-{s['end_hms']}]")
    return ev


def fallback_note_from_source(segments: list[dict], detail_level: str = "standard") -> dict:
    cfg = detail_cfg(detail_level)
    lines = [clean_item_text(s.get("text", ""), 400) for s in segments if s.get("text")]
    lines = [x for x in lines if x]
    if not lines:
        return sanitize_note({}, segments, detail_level)
    bullets = [" ".join(lines[i : i + 3])[:700] for i in range(0, min(len(lines), cfg["topic_bullets"] * 3), 3)]
    note = {
        "summary_bullets": bullets[: cfg["summary_bullets"]],
        "topics": [{"heading": "원문 기반 주요 내용", "bullets": bullets, "evidence": segment_evidence(segments)}],
        "timeline": [{"time": s.get("start_hms", ""), "event": clean_item_text(s.get("text", ""), 300)} for s in segments[: cfg["timeline_limit"]]],
    }
    return sanitize_note(note, segments, detail_level)


def loose_note_from_raw(raw: str, segments: list[dict], detail_level: str = "standard") -> dict:
    cfg = detail_cfg(detail_level)
    lines = [clean_item_text(x.strip("-•* 0123456789.\t"), 700) for x in raw.splitlines()]
    bullets = [x for x in lines if len(x) >= 18 and not is_weak_placeholder(x)][: max(8, cfg["topic_bullets"] * 2)]
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
            # Add a small overlap to preserve context across boundaries.
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
- 원문에 명시된 사실만 사용하세요.
- 일반 도메인으로 작성하고 특정 회사/산업을 가정하지 마세요.
- 이 단계는 '요약'보다 '정보 추출'이 중요합니다. 세부 사실, 숫자, 변화 흐름, 원인/결과, 후속 조치를 최대한 보존하세요.
- topics는 가능한 한 {cfg['chunk_topics']}개 이내, 각 topic의 bullets는 {cfg['topic_bullets']}개 이내로 작성하세요.
- evidence는 항목당 최대 2개만 쓰고 segment ID를 전부 나열하지 마세요.
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
  "terms": [{{"term": "", "description": ""}}]
}}

Transcript:
{source}
""".strip()


def make_final_prompt(title: str, notes: list[dict], language: str, detail_level: str = "standard") -> str:
    cfg = detail_cfg(detail_level)
    compact = json.dumps(notes, ensure_ascii=False)[:45000]
    return f"""
다음은 transcript 구간별로 추출한 JSON note입니다. 이를 통합하여 최종 회의록 JSON을 작성하세요.
문서 제목 후보: {title}
출력 언어: {language}
문서 상세도: {cfg['label']}

원칙:
- note에 없는 사실은 추가하지 마세요.
- 중복은 줄이되, 주제별 상세 내용·결정사항·실행 항목·이슈·확인 필요 사항·타임라인은 누락하지 마세요.
- 최종 문서는 빈약하면 안 됩니다. structured_notes에는 핵심 주제뿐 아니라 관련 세부 내용을 충분히 포함하세요.
- evidence는 항목당 최대 2개만 유지하세요.
- one_page_summary는 bullet/list가 아니라 사람이 읽기 쉬운 문단형 글로 작성하세요.
- 값 안에 Python dict/list, JSON 객체 문자열, 키 이름(heading/bullets 등)을 그대로 넣지 마세요.
- 아래 JSON 구조의 빈 문자열/빈 배열을 실제 내용으로 채우되, placeholder를 그대로 출력하지 마세요.

반환 JSON 구조:
{{
  "one_page_summary": "사람이 바로 읽을 수 있는 5~8문장 내외의 자연스러운 문단형 요약",
  "overview": [],
  "structured_notes": [{{"heading": "", "bullets": [], "evidence": []}}],
  "key_points": [{{"point": "", "evidence": []}}],
  "decisions": [{{"text": "", "evidence": []}}],
  "action_items": [{{"task": "", "owner": "", "due_date": "", "evidence": []}}],
  "risks_issues": [{{"text": "", "evidence": []}}],
  "timeline": [{{"time": "", "event": ""}}],
  "key_quotes": [{{"quote": "", "evidence": []}}],
  "terms": [{{"term": "", "description": ""}}],
  "open_questions": [{{"text": "", "evidence": []}}]
}}

Chunk notes:
{compact}
""".strip()


def call_llm_json(
    llm,
    prompt: str,
    segments: list[dict],
    max_new_tokens: int,
    label: str,
    detail_level: str = "standard",
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    raw_last = ""
    for attempt in range(1, 4):
        if log_cb:
            log_cb(f"🤖 LLM 생성: {label} / attempt {attempt} / max_new_tokens={max_new_tokens}")
        raw = llm.generate(SYSTEM_PROMPT, prompt, max_new_tokens=max_new_tokens)
        raw_last = raw
        parsed = None if has_template_leak(raw) else parse_json_tolerant(raw)
        if parsed:
            note = sanitize_note(parsed, segments, detail_level)
            # Accept if it has some real content. Do not require perfect JSON richness.
            if note.get("summary_bullets") or note.get("topics") or note.get("timeline"):
                return note
        if log_cb:
            log_cb("⚠️ JSON 파싱 실패/템플릿 문구/정보량 부족. 더 짧고 명확한 JSON으로 재시도합니다.")
        prompt = prompt + "\n\n중요: JSON만 반환하세요. evidence는 항목당 1개만 쓰고, topics와 bullets를 줄여서 완성된 JSON으로 답하세요."
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
            ("decisions", 60), ("action_items", 60), ("risks_issues", 60), ("open_questions", 60),
            ("timeline", cfg["timeline_limit"]), ("key_quotes", cfg["quotes_limit"]), ("terms", 50),
        ]:
            for item in n.get(key, [])[:limit]:
                add_unique(out, key, item, limit, seen)
    if not out["one_page_summary"]:
        out["one_page_summary"] = ["요약을 생성할 수 있는 충분한 LLM 출력이 부족했습니다. transcript를 확인하세요."]
    if not out["overview"] and out["structured_notes"]:
        for t in out["structured_notes"][:3]:
            if isinstance(t, dict) and t.get("heading"):
                out["overview"].append(str(t["heading"]))
    return out


def sanitize_final(obj: dict, detail_level: str = "standard") -> dict:
    out = empty_final()
    limits = {
        "one_page_summary": detail_cfg(detail_level)["summary_bullets"] * 2,
        "overview": 12,
        "structured_notes": detail_cfg(detail_level)["structured_limit"],
        "key_points": 60,
        "decisions": 60,
        "action_items": 60,
        "risks_issues": 60,
        "timeline": detail_cfg(detail_level)["timeline_limit"],
        "key_quotes": detail_cfg(detail_level)["quotes_limit"],
        "terms": 50,
        "open_questions": 60,
    }
    if not isinstance(obj, dict):
        return out
    for k in out:
        for item in as_list(obj.get(k)):
            if isinstance(item, str):
                s = clean_item_text(item, 1600 if k == "one_page_summary" else 1200)
                if s and not is_weak_placeholder(s):
                    out[k].append(s)
            elif isinstance(item, dict) and k in {"one_page_summary", "overview"}:
                s = clean_item_text(item, 1800)
                if s and not is_weak_placeholder(s):
                    out[k].append(s)
            elif isinstance(item, dict):
                cleaned = {}
                for kk, vv in item.items():
                    if kk == "evidence":
                        cleaned[kk] = compact_evidence(vv, 2)
                    elif isinstance(vv, list):
                        cleaned[kk] = [clean_item_text(x, 800) for x in vv if clean_item_text(x, 800)]
                    else:
                        cleaned[kk] = clean_item_text(vv, 800)
                if not has_template_leak(json.dumps(cleaned, ensure_ascii=False)):
                    out[k].append(cleaned)
            if len(out[k]) >= limits.get(k, 60):
                break
    if not out["one_page_summary"] and obj.get("summary_bullets"):
        out["one_page_summary"] = [clean_item_text(x, 600) for x in as_list(obj.get("summary_bullets"))[:8] if clean_item_text(x, 600)]
    return out


def enrich_final_with_chunk_notes(final_obj: dict, notes: list[dict], detail_level: str = "standard") -> dict:
    """Preserve detail from chunk notes even if the final LLM over-compresses."""
    cfg = detail_cfg(detail_level)
    aggregate = aggregate_without_final_llm(notes, detail_level)
    enriched = final_obj or empty_final()
    # Minimum target counts. If final LLM is sparse, backfill from chunk extraction.
    min_targets = {
        "structured_notes": min(len(aggregate["structured_notes"]), max(10, cfg["chunk_topics"] * 2)),
        "key_points": min(len(aggregate["key_points"]), 12),
        "timeline": min(len(aggregate["timeline"]), 12),
        "key_quotes": min(len(aggregate["key_quotes"]), 6),
    }
    seen: set[str] = set()
    for k, vals in enriched.items():
        for v in as_list(vals):
            seen.add(json.dumps(v, ensure_ascii=False, sort_keys=True)[:220] if isinstance(v, dict) else str(v)[:220])
    for key, target in min_targets.items():
        if len(enriched.get(key, [])) < target:
            for item in aggregate.get(key, []):
                add_unique(enriched, key, item, detail_cfg(detail_level).get("structured_limit", 80), seen)
                if len(enriched[key]) >= target:
                    break
    # Always preserve explicit decisions/actions/issues/questions if final LLM dropped them.
    for key in ["decisions", "action_items", "risks_issues", "open_questions", "terms"]:
        if len(enriched.get(key, [])) < len(aggregate.get(key, [])):
            for item in aggregate.get(key, []):
                add_unique(enriched, key, item, 60, seen)
    if not enriched.get("one_page_summary"):
        enriched["one_page_summary"] = aggregate.get("one_page_summary", [])[: cfg["summary_bullets"]]
    if not enriched.get("overview"):
        enriched["overview"] = aggregate.get("overview", [])[:8]
    return enriched


def summarize_segments(
    segments: list[dict],
    title: str,
    profile: RuntimeProfile,
    language: str = "ko",
    glossary: str = "",
    allow_download: bool = True,
    use_final_llm: bool = True,
    detail_level: str = "detailed",
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    cfg = detail_cfg(detail_level)
    chunks = chunk_segments(segments, profile.max_chars_per_chunk, profile.chunk_overlap_chars)
    chunk_tokens = max(900, int(profile.max_new_tokens_chunk * cfg["token_multiplier"]))
    final_tokens = max(1200, int(profile.max_new_tokens_final * cfg["token_multiplier"]))
    llm = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
    notes = []
    fallback_used = False
    if log_cb:
        log_cb(f"🧩 transcript chunk 수: {len(chunks)} / chunk_chars={profile.max_chars_per_chunk} / overlap={profile.chunk_overlap_chars}")
        log_cb(f"📝 문서 상세도: {detail_level} ({cfg['label']}) / chunk_tokens={chunk_tokens} / final_tokens={final_tokens}")
    for i, ch in enumerate(chunks, start=1):
        prompt = make_chunk_prompt(title, i, len(chunks), ch, language, glossary, detail_level)
        note = call_llm_json(llm, prompt, ch, chunk_tokens, f"{title}_chunk_{i:02d}", detail_level, log_cb)
        if note.get("topics") and note["topics"][0].get("heading") in {"원문 기반 주요 내용", "LLM 출력 복구 내용"}:
            fallback_used = True
        notes.append(note)
    final_obj = None
    final_llm_failed = False
    if use_final_llm:
        try:
            final_prompt = make_final_prompt(title, notes, language, detail_level)
            if log_cb:
                log_cb(f"🤖 최종 병합 LLM 생성 / max_new_tokens={final_tokens}")
            raw = llm.generate(SYSTEM_PROMPT, final_prompt, max_new_tokens=final_tokens)
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
    final_obj = enrich_final_with_chunk_notes(final_obj, notes, detail_level)
    transcript_chars = sum(len(s.get("text", "")) for s in segments)
    markdown_est_chars = len(json.dumps(final_obj, ensure_ascii=False))
    run_config = {
        "pipeline_version": PIPELINE_VERSION,
        "title": title,
        "detail_level": detail_level,
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
        "use_final_llm": use_final_llm,
        "final_llm_failed": final_llm_failed,
        "fallback_used": fallback_used,
    }
    return {"chunk_notes": notes, "final": final_obj, "chunk_count": len(chunks), "run_config": run_config}
