from __future__ import annotations

from typing import Any

from .asr import segments_to_timestamped
from .utils import clean_text_for_xml


def md_escape_cell(text: Any) -> str:
    s = clean_text_for_xml(str(text or "")).replace("|", "\\|").replace("\n", "<br>")
    return s.strip()


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def bullet_section(items, empty="명시적으로 확인되지 않음") -> str:
    lines = []
    for item in as_list(items):
        if isinstance(item, dict):
            text = item.get("text") or item.get("point") or item.get("event") or item.get("quote") or item.get("term") or ""
            ev = item.get("evidence")
            if ev:
                text = f"{text} ({', '.join(as_list(ev)[:2])})"
        else:
            text = str(item)
        text = clean_text_for_xml(text).strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if lines else f"- {empty}"


def build_markdown(title: str, final: dict, segments: list[dict], include_transcript_appendix: bool = False) -> str:
    title = clean_text_for_xml(title)
    lines = [f"# {title}", ""]

    lines += ["## 1. 한 페이지 요약", "", "### 핵심 요약"]
    lines.append(bullet_section(final.get("one_page_summary")))
    lines += ["", "### 녹음/회의 개요"]
    lines.append(bullet_section(final.get("overview")))

    lines += ["", "## 2. 전체 구조화 정리", ""]
    structured = as_list(final.get("structured_notes"))
    if structured:
        for t in structured[:30]:
            if isinstance(t, dict):
                heading = clean_text_for_xml(t.get("heading") or "주요 주제").strip()
                lines += [f"### {heading}"]
                for b in as_list(t.get("bullets"))[:8]:
                    b = clean_text_for_xml(str(b)).strip()
                    if b:
                        lines.append(f"- {b}")
                ev = as_list(t.get("evidence"))[:2]
                if ev:
                    lines.append(f"  - 근거: {', '.join(str(x) for x in ev)}")
                lines.append("")
            else:
                lines.append(f"- {clean_text_for_xml(str(t))}")
    else:
        lines.append("- 구조화할 주요 내용이 명시적으로 확인되지 않음")

    lines += ["", "## 3. 핵심 논점", ""]
    key_points = as_list(final.get("key_points"))
    if key_points:
        lines += ["| 항목 | 설명/논점 | 근거 |", "|---|---|---|"]
        for i, item in enumerate(key_points[:30], start=1):
            if isinstance(item, dict):
                point = item.get("point") or item.get("text") or ""
                ev = ", ".join(as_list(item.get("evidence"))[:2])
            else:
                point, ev = str(item), ""
            lines.append(f"| {i} | {md_escape_cell(point)} | {md_escape_cell(ev or '명시적으로 확인되지 않음')} |")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    lines += ["", "## 4. 결정사항 / 결론", ""]
    lines.append(bullet_section(final.get("decisions")))

    lines += ["", "## 5. 실행 항목", ""]
    actions = as_list(final.get("action_items"))
    if actions:
        lines += ["| 할 일 | 담당자 | 기한 | 근거 |", "|---|---|---|---|"]
        for a in actions[:30]:
            if isinstance(a, dict):
                task = a.get("task") or a.get("text") or ""
                owner = a.get("owner") or "확인 필요"
                due = a.get("due_date") or "확인 필요"
                ev = ", ".join(as_list(a.get("evidence"))[:2])
            else:
                task, owner, due, ev = str(a), "확인 필요", "확인 필요", ""
            lines.append(f"| {md_escape_cell(task)} | {md_escape_cell(owner)} | {md_escape_cell(due)} | {md_escape_cell(ev or '명시적으로 확인되지 않음')} |")
    else:
        lines.append("- 명시적 실행 항목 없음")

    lines += ["", "## 6. 리스크 / 이슈", ""]
    lines.append(bullet_section(final.get("risks_issues")))

    lines += ["", "## 7. 타임라인 / 진행 흐름", ""]
    timeline = as_list(final.get("timeline"))
    if timeline:
        for t in timeline[:40]:
            if isinstance(t, dict):
                time = clean_text_for_xml(t.get("time") or "").strip()
                event = clean_text_for_xml(t.get("event") or t.get("text") or "").strip()
                lines.append(f"- {time + ': ' if time else ''}{event}")
            else:
                lines.append(f"- {clean_text_for_xml(str(t))}")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    lines += ["", "## 8. 중요 발언 / 근거", ""]
    quotes = as_list(final.get("key_quotes"))
    if quotes:
        for q in quotes[:20]:
            if isinstance(q, dict):
                quote = q.get("quote") or q.get("text") or ""
                ev = ", ".join(as_list(q.get("evidence"))[:2])
                lines.append(f"- {ev + ' ' if ev else ''}\"{clean_text_for_xml(str(quote)).strip()}\"")
            else:
                lines.append(f"- {clean_text_for_xml(str(q))}")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    lines += ["", "## 9. 용어 / 개념", ""]
    terms = as_list(final.get("terms"))
    if terms:
        lines += ["| 용어 | 설명 |", "|---|---|"]
        for term in terms[:30]:
            if isinstance(term, dict):
                lines.append(f"| {md_escape_cell(term.get('term'))} | {md_escape_cell(term.get('description'))} |")
            else:
                lines.append(f"| {md_escape_cell(term)} |  |")
    else:
        lines.append("- 명시적으로 확인되지 않음")

    lines += ["", "## 10. 확인 필요한 내용", ""]
    lines.append(bullet_section(final.get("open_questions")))

    if include_transcript_appendix:
        lines += ["", "---", "", "## 부록. 전체 Timestamped Transcript", "", "```text", segments_to_timestamped(segments), "```"]

    return "\n".join(lines).strip() + "\n"
