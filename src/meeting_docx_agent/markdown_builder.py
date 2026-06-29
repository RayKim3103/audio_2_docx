from __future__ import annotations

from typing import Any

from .asr import segments_to_timestamped
from .utils import clean_text_for_xml, humanize_llm_value


def md_escape_cell(text: Any) -> str:
    s = clean_text_for_xml(str(text or "")).replace("|", "\\|").replace("\n", "<br>")
    return s.strip()


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _evidence_text(item: dict, limit: int = 2) -> str:
    ev = item.get("evidence") if isinstance(item, dict) else None
    if not ev:
        return ""
    return ", ".join(str(x) for x in as_list(ev)[:limit] if str(x).strip())



def item_to_text(item: Any) -> str:
    """Return human-readable text from strings/dicts/lists without leaking raw literals."""
    if isinstance(item, dict):
        text = item.get("text") or item.get("point") or item.get("event") or item.get("quote") or item.get("term")
        if text:
            return clean_text_for_xml(humanize_llm_value(text)).strip()
        return clean_text_for_xml(humanize_llm_value(item)).strip()
    return clean_text_for_xml(humanize_llm_value(item)).strip()


def paragraph_section(items, empty="명시적으로 확인되지 않음") -> str:
    parts: list[str] = []
    for item in as_list(items):
        text = item_to_text(item)
        if text:
            # Remove leading bullet markers if the LLM returned bullet-like sentences.
            text = text.strip("-•* ").strip()
            parts.append(text)
    if not parts:
        return empty
    paragraph = " ".join(parts)
    paragraph = paragraph.replace("..", ".")
    return paragraph


def bullet_section(items, empty="명시적으로 확인되지 않음") -> str:
    lines = []
    for item in as_list(items):
        text = item_to_text(item)
        ev = _evidence_text(item) if isinstance(item, dict) else ""
        if ev and text:
            text = f"{text} ({ev})"
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if lines else f"- {empty}"


def build_topic_blocks(structured: list, max_topics: int = 80, max_bullets: int = 10) -> list[str]:
    lines: list[str] = []
    if not structured:
        return ["- 구조화할 주요 내용이 명시적으로 확인되지 않음"]
    for idx, t in enumerate(structured[:max_topics], start=1):
        if isinstance(t, dict):
            heading = item_to_text(t.get("heading") or f"주요 주제 {idx}")
            lines += [f"### {idx}. {heading}"]
            bullets = [item_to_text(b) for b in as_list(t.get("bullets")) if item_to_text(b)]
            if bullets:
                for b in bullets[:max_bullets]:
                    lines.append(f"- {b}")
            else:
                lines.append("- 세부 내용이 명시적으로 확인되지 않음")
            ev = as_list(t.get("evidence"))[:2]
            if ev:
                lines.append(f"  - 근거: {', '.join(str(x) for x in ev)}")
            lines.append("")
        else:
            text = clean_text_for_xml(str(t)).strip()
            if text:
                lines.append(f"- {text}")
    return lines or ["- 구조화할 주요 내용이 명시적으로 확인되지 않음"]


def build_markdown(
    title: str,
    final: dict,
    segments: list[dict],
    include_transcript_appendix: bool = False,
    detail_level: str = "detailed",
    run_config: dict | None = None,
) -> str:
    title = clean_text_for_xml(title)
    lines = [f"# {title}", ""]

    # 1. One-page summary
    # Keep this section as prose paragraphs, not bullet lists.
    lines += ["## 1. 한 페이지 요약", ""]
    lines.append(paragraph_section(final.get("one_page_summary")))
    overview_text = paragraph_section(final.get("overview"), empty="")
    if overview_text:
        lines += ["", "### 녹음/회의 개요", "", overview_text]

    # 2. Overall structured outline
    lines += ["", "## 2. 전체 구조화 정리", ""]
    overview = as_list(final.get("overview"))
    if overview:
        for item in overview[:12]:
            lines.append(f"- {item_to_text(item)}")
    else:
        structured = as_list(final.get("structured_notes"))
        if structured:
            for t in structured[:8]:
                if isinstance(t, dict):
                    lines.append(f"- {item_to_text(t.get('heading') or '주요 주제')}")
        else:
            lines.append("- 전체 구조화 정보가 명시적으로 확인되지 않음")

    # 3. Detailed topic notes - this section was added to avoid sparse DOCX outputs.
    lines += ["", "## 3. 주제별 상세 정리", ""]
    lines += build_topic_blocks(as_list(final.get("structured_notes")), max_topics=90 if detail_level == "detailed" else 50, max_bullets=10 if detail_level == "detailed" else 7)

    # 4. Key points
    lines += ["", "## 4. 핵심 개념 / 논점", ""]
    key_points = as_list(final.get("key_points"))
    if key_points:
        lines += ["| 항목 | 설명/논점 | 근거 |", "|---|---|---|"]
        for i, item in enumerate(key_points[:60], start=1):
            if isinstance(item, dict):
                point = item.get("point") or item.get("text") or ""
                ev = _evidence_text(item)
            else:
                point, ev = str(item), ""
            lines.append(f"| {i} | {md_escape_cell(point)} | {md_escape_cell(ev or '원문 기반')} |")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    # 5. Decisions
    lines += ["", "## 5. 결정사항 / 결론", ""]
    lines.append(bullet_section(final.get("decisions"), empty="명시적 결정사항 없음"))

    # 6. Action items
    lines += ["", "## 6. 실행 항목", ""]
    actions = as_list(final.get("action_items"))
    if actions:
        lines += ["| 할 일 | 담당자 | 기한 | 근거 |", "|---|---|---|---|"]
        for a in actions[:60]:
            if isinstance(a, dict):
                task = a.get("task") or a.get("text") or ""
                owner = a.get("owner") or "확인 필요"
                due = a.get("due_date") or "확인 필요"
                ev = _evidence_text(a)
            else:
                task, owner, due, ev = str(a), "확인 필요", "확인 필요", ""
            lines.append(f"| {md_escape_cell(task)} | {md_escape_cell(owner)} | {md_escape_cell(due)} | {md_escape_cell(ev or '원문 기반')} |")
    else:
        lines.append("- 명시적 실행 항목 없음")

    # 7. Risks/issues
    lines += ["", "## 7. 리스크 / 이슈", ""]
    lines.append(bullet_section(final.get("risks_issues")))

    # 8. Timeline
    lines += ["", "## 8. 타임라인 / 진행 흐름", ""]
    timeline = as_list(final.get("timeline"))
    if timeline:
        for t in timeline[:80]:
            if isinstance(t, dict):
                time = clean_text_for_xml(t.get("time") or "").strip()
                event = clean_text_for_xml(t.get("event") or t.get("text") or "").strip()
                if event:
                    lines.append(f"- {time + ': ' if time else ''}{event}")
            else:
                text = item_to_text(t)
                if text:
                    lines.append(f"- {text}")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    # 9. Quotes/evidence
    lines += ["", "## 9. 중요 발언 / 근거", ""]
    quotes = as_list(final.get("key_quotes"))
    if quotes:
        for q in quotes[:40]:
            if isinstance(q, dict):
                quote = q.get("quote") or q.get("text") or ""
                ev = _evidence_text(q)
                quote = clean_text_for_xml(str(quote)).strip()
                if quote:
                    lines.append(f"- {ev + ' ' if ev else ''}\"{quote}\"")
            else:
                text = clean_text_for_xml(str(q)).strip()
                if text:
                    lines.append(f"- {text}")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    # 10. Terms
    lines += ["", "## 10. 용어 / 개념", ""]
    terms = as_list(final.get("terms"))
    if terms:
        lines += ["| 용어 | 설명 |", "|---|---|"]
        for term in terms[:60]:
            if isinstance(term, dict):
                lines.append(f"| {md_escape_cell(term.get('term'))} | {md_escape_cell(term.get('description'))} |")
            else:
                lines.append(f"| {md_escape_cell(term)} |  |")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    # 11. Open questions
    lines += ["", "## 11. 확인 필요한 내용", ""]
    lines.append(bullet_section(final.get("open_questions")))

    # run_config is saved as JSON and included in the ZIP, but intentionally not written into the human-facing DOCX.

    if include_transcript_appendix:
        lines += ["", "---", "", "## 부록 B. 전체 Timestamped Transcript", "", "```text", segments_to_timestamped(segments), "```"]

    return "\n".join(lines).strip() + "\n"
