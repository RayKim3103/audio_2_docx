from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .llm import get_llm
from .profiles import RuntimeProfile
from .utils import clean_text_for_xml, seconds_to_hms

SYSTEM_PROMPT = """
당신은 보안이 중요한 로컬 환경에서 동작하는 회의록/녹음 정리 전문 AI입니다.
반드시 입력 transcript에 근거해서만 작성하세요.
회의가 아닌 강의/교육/인터뷰/발표 녹음일 수도 있으므로, 원문 성격에 맞게 일반 도메인으로 정리하세요.
원문에 없는 참석자, 담당자, 기한, 결정사항은 만들지 말고 '확인 필요' 또는 '명시적으로 확인되지 않음'이라고 쓰세요.
출력은 요청한 JSON 형식만 반환하고, 설명 문장이나 마크다운 코드블록은 붙이지 마세요.
""".strip()

TEMPLATE_PHRASES = [
    "3~7개 bullet로 요약",
    "주제별로 소제목을 만들고 자세히 정리",
    "원문에 있을 때만 작성",
    "가능한 경우 `[HH:MM:SS]`",
    "가능한 경우 [HH:MM:SS]",
    "중요한 용어가 있으면 설명",
    "불명확하거나 추가 확인이 필요한 부분",
]


def has_template_leak(text: str) -> bool:
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
    if isinstance(value, list):
        return value
    return [value]


def compact_evidence(evidence: Any, limit: int = 2) -> list[str]:
    items = []
    for item in as_list(evidence):
        s = str(item).strip()
        if not s:
            continue
        items.append(s[:120])
    return items[:limit]


def sanitize_note(note: dict, source_segments: list[dict] | None = None) -> dict:
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
    for b in as_list(note.get("summary_bullets")):
        s = clean_text_for_xml(str(b)).strip()
        if s and not has_template_leak(s):
            out["summary_bullets"].append(s)
    for t in as_list(note.get("topics") or note.get("detailed_topics")):
        if not isinstance(t, dict):
            continue
        heading = clean_text_for_xml(str(t.get("heading") or t.get("topic") or "주요 논점")).strip()
        bullets = [clean_text_for_xml(str(x)).strip() for x in as_list(t.get("bullets") or t.get("details"))]
        bullets = [x for x in bullets if x and not has_template_leak(x)][:5]
        if bullets:
            if heading in {"주제명", "세부 내용", "설명"}:
                heading = bullets[0][:30]
            out["topics"].append({"heading": heading, "bullets": bullets, "evidence": compact_evidence(t.get("evidence"))})
    for key in ["decisions", "risks_issues", "open_questions"]:
        for item in as_list(note.get(key)):
            if isinstance(item, dict):
                text = item.get("text") or item.get("decision") or item.get("issue") or item.get("question") or ""
                ev = compact_evidence(item.get("evidence"))
            else:
                text = item
                ev = []
            text = clean_text_for_xml(str(text)).strip()
            if text and not has_template_leak(text):
                out[key].append({"text": text, "evidence": ev})
    for a in as_list(note.get("action_items")):
        if isinstance(a, dict):
            task = clean_text_for_xml(str(a.get("task") or a.get("할 일") or "")).strip()
            if task and not has_template_leak(task):
                out["action_items"].append({
                    "task": task,
                    "owner": clean_text_for_xml(str(a.get("owner") or a.get("담당자") or "확인 필요")).strip() or "확인 필요",
                    "due_date": clean_text_for_xml(str(a.get("due_date") or a.get("기한") or "확인 필요")).strip() or "확인 필요",
                    "evidence": compact_evidence(a.get("evidence")),
                })
    for q in as_list(note.get("key_quotes")):
        if isinstance(q, dict):
            quote = clean_text_for_xml(str(q.get("quote") or q.get("text") or "")).strip()
            ev = compact_evidence(q.get("evidence") or q.get("time"))
        else:
            quote = clean_text_for_xml(str(q)).strip()
            ev = []
        if quote and not has_template_leak(quote):
            out["key_quotes"].append({"quote": quote[:300], "evidence": ev})
    for tl in as_list(note.get("timeline")):
        if isinstance(tl, dict):
            time = str(tl.get("time") or tl.get("시점") or "").strip()
            event = str(tl.get("event") or tl.get("내용") or tl.get("text") or "").strip()
        else:
            time, event = "", str(tl).strip()
        if event and not has_template_leak(event):
            out["timeline"].append({"time": clean_text_for_xml(time), "event": clean_text_for_xml(event)})
    for term in as_list(note.get("terms")):
        if isinstance(term, dict):
            name = clean_text_for_xml(str(term.get("term") or term.get("name") or "")).strip()
            desc = clean_text_for_xml(str(term.get("description") or term.get("desc") or "")).strip()
        else:
            name, desc = clean_text_for_xml(str(term)).strip(), ""
        if name and not has_template_leak(name + desc):
            out["terms"].append({"term": name, "description": desc})
    if not out["summary_bullets"] and source_segments:
        text = " ".join(s.get("text", "") for s in source_segments[:8]).strip()
        if text:
            out["summary_bullets"] = [text[:250]]
    return out


def fallback_note_from_source(segments: list[dict]) -> dict:
    # This is a safety fallback; the document still indicates that it is source-based fallback.
    lines = [s.get("text", "") for s in segments if s.get("text")]
    first = " ".join(lines[:8]).strip()
    topics = []
    if first:
        topics.append({
            "heading": "원문 기반 주요 내용",
            "bullets": [" ".join(lines[i : i + 3])[:400] for i in range(0, min(len(lines), 12), 3)],
            "evidence": [f"[{segments[0]['id']} {segments[0]['start_hms']}-{segments[0]['end_hms']}]"] if segments else [],
        })
    return sanitize_note({"summary_bullets": [first[:300]] if first else [], "topics": topics}, segments)


def loose_note_from_raw(raw: str, segments: list[dict]) -> dict:
    lines = [clean_text_for_xml(x.strip("-•* 0123456789.\t")) for x in raw.splitlines()]
    bullets = [x for x in lines if len(x) >= 20 and not has_template_leak(x)][:6]
    if not bullets:
        return fallback_note_from_source(segments)
    return sanitize_note({
        "summary_bullets": bullets[:3],
        "topics": [{"heading": "LLM 출력 복구 내용", "bullets": bullets[:5], "evidence": []}],
    }, segments)


def chunk_segments(segments: list[dict], max_chars: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for s in segments:
        text = s.get("text", "")
        add = len(text) + 40
        if cur and cur_len + add > max_chars:
            chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(s)
        cur_len += add
    if cur:
        chunks.append(cur)
    return chunks


def segments_to_prompt_text(segments: list[dict]) -> str:
    return "\n".join(f"[{s['id']} {s['start_hms']}-{s['end_hms']}] {s['text']}" for s in segments)


def make_chunk_prompt(title: str, chunk_idx: int, total_chunks: int, segments: list[dict], language: str, glossary: str = "") -> str:
    source = segments_to_prompt_text(segments)
    glossary_text = f"\n사용자 제공 용어/고유명사 힌트: {glossary}\n" if glossary.strip() else ""
    return f"""
다음은 하나의 회의/녹음 transcript 중 {chunk_idx}/{total_chunks}번째 구간입니다.
문서 제목 후보: {title}
출력 언어: {language}
{glossary_text}
요구사항:
- 일반 도메인 회의록/녹음 정리입니다. 특정 회사/산업을 가정하지 마세요.
- 원문에 명시된 사실만 사용하세요.
- evidence는 항목당 최대 2개만 쓰세요. segment ID를 전부 나열하지 마세요.
- 담당자/기한이 원문에 없으면 '확인 필요'라고 쓰세요.
- decision/action item이 없으면 빈 배열 []을 쓰세요.
- JSON 객체 하나만 반환하세요.

JSON 스키마:
{{
  "summary_bullets": ["이 구간의 핵심 요약 2~4개"],
  "topics": [{{"heading": "구체적 주제명", "bullets": ["세부 내용"], "evidence": ["[S0001 00:00:00-00:00:05]"]}}],
  "decisions": [{{"text": "결정사항", "evidence": ["[S0001 ...]"]}}],
  "action_items": [{{"task": "할 일", "owner": "담당자 또는 확인 필요", "due_date": "기한 또는 확인 필요", "evidence": ["[S0001 ...]"]}}],
  "risks_issues": [{{"text": "이슈/리스크", "evidence": ["[S0001 ...]"]}}],
  "open_questions": [{{"text": "추가 확인 질문", "evidence": ["[S0001 ...]"]}}],
  "timeline": [{{"time": "00:00:00", "event": "진행 흐름"}}],
  "key_quotes": [{{"quote": "중요 발언", "evidence": ["[S0001 ...]"]}}],
  "terms": [{{"term": "용어", "description": "원문 기반 설명"}}]
}}

Transcript:
{source}
""".strip()


def make_final_prompt(title: str, notes: list[dict], language: str) -> str:
    compact = json.dumps(notes, ensure_ascii=False)[:35000]
    return f"""
다음은 여러 transcript 구간을 정리한 JSON note입니다. 이를 통합해 최종 회의록 JSON을 작성하세요.
문서 제목 후보: {title}
출력 언어: {language}

원칙:
- note에 없는 사실은 추가하지 마세요.
- 일반 도메인 회의록입니다. 특정 기업/주제를 가정하지 마세요.
- 중복을 줄이되 중요한 결정사항, 실행항목, 이슈, 확인 필요사항은 누락하지 마세요.
- evidence는 항목당 최대 2개만 유지하세요.
- JSON 객체 하나만 반환하세요.

최종 JSON 스키마:
{{
  "one_page_summary": ["핵심 요약"],
  "overview": ["녹음/회의의 목적, 배경, 맥락"],
  "structured_notes": [{{"heading": "주제", "bullets": ["세부 내용"], "evidence": ["근거"]}}],
  "key_points": [{{"point": "핵심 논점", "evidence": ["근거"]}}],
  "decisions": [{{"text": "결정사항", "evidence": ["근거"]}}],
  "action_items": [{{"task": "할 일", "owner": "담당자 또는 확인 필요", "due_date": "기한 또는 확인 필요", "evidence": ["근거"]}}],
  "risks_issues": [{{"text": "리스크/이슈", "evidence": ["근거"]}}],
  "timeline": [{{"time": "시점", "event": "진행 흐름"}}],
  "key_quotes": [{{"quote": "중요 발언", "evidence": ["근거"]}}],
  "terms": [{{"term": "용어", "description": "설명"}}],
  "open_questions": [{{"text": "확인 필요 사항", "evidence": ["근거"]}}]
}}

Chunk notes:
{compact}
""".strip()


def call_llm_json(llm, prompt: str, segments: list[dict], max_new_tokens: int, label: str, log_cb: Optional[Callable[[str], None]] = None) -> dict:
    raw_last = ""
    for attempt in range(1, 4):
        if log_cb:
            log_cb(f"🤖 LLM 생성: {label} / attempt {attempt}")
        raw = llm.generate(SYSTEM_PROMPT, prompt, max_new_tokens=max_new_tokens)
        raw_last = raw
        if has_template_leak(raw):
            if log_cb:
                log_cb("⚠️ 템플릿 문구가 감지되어 재시도합니다.")
            continue
        parsed = parse_json_tolerant(raw)
        if parsed:
            note = sanitize_note(parsed, segments)
            if note.get("summary_bullets") or note.get("topics"):
                return note
        if log_cb:
            log_cb("⚠️ JSON 파싱 실패 또는 정보량 부족. 더 짧은 형식으로 재시도합니다.")
        prompt = prompt + "\n\n중요: evidence는 항목당 1개만 쓰고, 더 짧은 JSON으로 답하세요."
    if log_cb:
        log_cb("⚠️ 완전한 JSON 생성 실패. LLM 원문 출력에서 유용한 내용을 복구합니다.")
    return loose_note_from_raw(raw_last, segments)


def aggregate_without_final_llm(notes: list[dict]) -> dict:
    out = {
        "one_page_summary": [], "overview": [], "structured_notes": [], "key_points": [],
        "decisions": [], "action_items": [], "risks_issues": [], "timeline": [],
        "key_quotes": [], "terms": [], "open_questions": [],
    }
    seen = set()
    def add_unique(key, item, item_key=None, limit=80):
        text = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
        marker = item_key or text[:180]
        if marker in seen or not text.strip():
            return
        seen.add(marker)
        if len(out[key]) < limit:
            out[key].append(item)
    for n in notes:
        for b in n.get("summary_bullets", [])[:3]:
            add_unique("one_page_summary", b, b[:120], 12)
        for t in n.get("topics", [])[:6]:
            add_unique("structured_notes", t, (t.get("heading", "") + str(t.get("bullets", [])[:1]))[:160], 80)
            if t.get("bullets"):
                add_unique("key_points", {"point": t["bullets"][0], "evidence": t.get("evidence", [])}, t["bullets"][0][:140], 30)
        for key in ["decisions", "action_items", "risks_issues", "open_questions", "timeline", "key_quotes", "terms"]:
            for item in n.get(key, [])[:10]:
                add_unique(key, item, None, 80)
    if not out["one_page_summary"]:
        out["one_page_summary"] = ["요약을 생성할 수 있는 충분한 LLM 출력이 부족했습니다. 전체 transcript를 확인하세요."]
    return out


def summarize_segments(
    segments: list[dict],
    title: str,
    profile: RuntimeProfile,
    language: str = "ko",
    glossary: str = "",
    allow_download: bool = True,
    use_final_llm: bool = True,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    chunks = chunk_segments(segments, profile.max_chars_per_chunk)
    llm = get_llm(profile.llm_model, profile.llm_device, allow_download=allow_download)
    notes = []
    for i, ch in enumerate(chunks, start=1):
        prompt = make_chunk_prompt(title, i, len(chunks), ch, language, glossary)
        note = call_llm_json(llm, prompt, ch, profile.max_new_tokens_chunk, f"{title}_chunk_{i:02d}", log_cb)
        notes.append(note)
    final_obj = None
    if use_final_llm:
        try:
            final_prompt = make_final_prompt(title, notes, language)
            raw = llm.generate(SYSTEM_PROMPT, final_prompt, max_new_tokens=profile.max_new_tokens_final)
            parsed = parse_json_tolerant(raw)
            if parsed and not has_template_leak(raw):
                final_obj = sanitize_final(parsed)
        except Exception as e:
            if log_cb:
                log_cb(f"⚠️ 최종 병합 LLM 실패. chunk note 기반으로 병합합니다: {e}")
    if not final_obj:
        final_obj = aggregate_without_final_llm(notes)
    return {"chunk_notes": notes, "final": final_obj, "chunk_count": len(chunks)}


def sanitize_final(obj: dict) -> dict:
    out = aggregate_without_final_llm([])
    out.update({k: [] for k in out})
    for k in out:
        for item in as_list(obj.get(k)):
            if isinstance(item, str):
                s = clean_text_for_xml(item).strip()
                if s and not has_template_leak(s):
                    out[k].append(s)
            elif isinstance(item, dict):
                # Reuse generic dict after compacting evidence.
                cleaned = {}
                for kk, vv in item.items():
                    if kk == "evidence":
                        cleaned[kk] = compact_evidence(vv)
                    else:
                        cleaned[kk] = clean_text_for_xml(str(vv)).strip() if not isinstance(vv, list) else [clean_text_for_xml(str(x)).strip() for x in vv]
                if not has_template_leak(json.dumps(cleaned, ensure_ascii=False)):
                    out[k].append(cleaned)
    if not out["one_page_summary"] and obj.get("summary_bullets"):
        out["one_page_summary"] = [str(x) for x in as_list(obj.get("summary_bullets"))[:8]]
    return out
