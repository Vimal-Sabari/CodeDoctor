"""Stack Exchange XML data-dump parser for frontend build-error Q&A."""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.collect.common import (
    DEFAULT_COLLECT_CONFIG,
    RawLogRecord,
    append_raw_records,
    load_collect_config,
    resolve_project_path,
)

if TYPE_CHECKING:
    import typer

BUILD_ERROR_REGEX = re.compile(
    r"(Error:|Module not found|Cannot find module|error TS\d{4}:|"
    r"ERR_[A-Z0-9_]+|SyntaxError:|FATAL ERROR|Failed to compile|ERESOLVE)",
    re.IGNORECASE,
)
TAG_REGEX = re.compile(r"<([^>]+)>")
HTML_TAG_REGEX = re.compile(r"<[^>]+>")


class StackExchangeConfig(BaseModel):
    """Configuration for Stack Exchange dump parsing."""

    tags: list[str]
    min_score: int = 5
    require_accepted_answer: bool = True
    license: str = "CC BY-SA 4.0"
    site_base_url: str = "https://stackoverflow.com/questions"
    output_jsonl: Path


def _build_config(raw_cfg: dict[str, Any]) -> StackExchangeConfig:
    se = raw_cfg["stackexchange"]
    return StackExchangeConfig(
        tags=se["tags"],
        min_score=se.get("min_score", 5),
        require_accepted_answer=se.get("require_accepted_answer", True),
        license=se.get("license", "CC BY-SA 4.0"),
        site_base_url=se.get("site_base_url", "https://stackoverflow.com/questions").rstrip("/"),
        output_jsonl=resolve_project_path(se["output_jsonl"]),
    )


def _parse_tags(raw_tags: str) -> list[str]:
    if not raw_tags:
        return []
    if "<" in raw_tags:
        return [t.lower() for t in TAG_REGEX.findall(raw_tags)]
    return [t.strip("|").lower() for t in raw_tags.split("|") if t.strip("|")]


def _clean_html_text(body: str) -> str:
    unescaped = html.unescape(body)
    text = HTML_TAG_REGEX.sub(" ", unescaped)
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_stackexchange_dump(
    posts_xml: Path,
    cfg: StackExchangeConfig,
    limit: int | None = None,
) -> dict[str, Any]:
    """Stream-parse `Posts.xml`, extract build-error Q&A with accepted answers, and write JSONL."""
    target_tags = {t.lower() for t in cfg.tags}
    matched_questions: dict[str, dict[str, Any]] = {}

    # Pass 1: find matching questions and map AcceptedAnswerId -> question
    for _, elem in ET.iterparse(posts_xml, events=("end",)):
        if elem.tag != "row":
            elem.clear()
            continue
        if elem.attrib.get("PostTypeId") != "1":
            elem.clear()
            continue

        accepted_id = elem.attrib.get("AcceptedAnswerId")
        if cfg.require_accepted_answer and not accepted_id:
            elem.clear()
            continue

        score = int(elem.attrib.get("Score", "0"))
        if score < cfg.min_score:
            elem.clear()
            continue

        post_tags = _parse_tags(elem.attrib.get("Tags", ""))
        if not target_tags.intersection(post_tags):
            elem.clear()
            continue

        body_text = _clean_html_text(elem.attrib.get("Body", ""))
        title = html.unescape(elem.attrib.get("Title", ""))
        if not BUILD_ERROR_REGEX.search(f"{title}\n{body_text}"):
            elem.clear()
            continue

        if accepted_id:
            matched_questions[accepted_id] = {
                "question_id": elem.attrib["Id"],
                "title": title,
                "body": body_text,
                "tags": post_tags,
                "score": score,
                "author": elem.attrib.get("OwnerDisplayName")
                or f"user:{elem.attrib.get('OwnerUserId', 'unknown')}",
            }
        elem.clear()
        if limit is not None and len(matched_questions) >= limit:
            break

    if not matched_questions:
        return {"questions_matched": 0, "records_written": 0, "output_jsonl": str(cfg.output_jsonl)}

    # Pass 2: stream answers to retrieve accepted answer bodies & authors
    records: list[RawLogRecord] = []
    for _, elem in ET.iterparse(posts_xml, events=("end",)):
        if elem.tag != "row":
            elem.clear()
            continue
        if elem.attrib.get("PostTypeId") != "2":
            elem.clear()
            continue

        ans_id = elem.attrib.get("Id")
        if ans_id and ans_id in matched_questions:
            q = matched_questions[ans_id]
            ans_body = _clean_html_text(elem.attrib.get("Body", ""))
            ans_author = elem.attrib.get("OwnerDisplayName") or (
                f"user:{elem.attrib.get('OwnerUserId', 'unknown')}"
            )
            q_id = q["question_id"]
            records.append(
                RawLogRecord(
                    id=f"se_{q_id}",
                    source="stackexchange",
                    raw_log=q["body"],
                    title=q["title"],
                    answer_text=ans_body,
                    metadata={
                        "post_id": q_id,
                        "accepted_answer_id": ans_id,
                        "post_url": f"{cfg.site_base_url}/{q_id}",
                        "author": q["author"],
                        "answer_author": ans_author,
                        "tags": q["tags"],
                        "score": q["score"],
                        "license": cfg.license,
                    },
                )
            )
        elem.clear()

    written = append_raw_records(cfg.output_jsonl, records)
    return {
        "questions_matched": len(matched_questions),
        "records_written": written,
        "output_jsonl": str(cfg.output_jsonl),
    }


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="stackexchange-collect",
        help="Parse a Stack Exchange Posts.xml dump for frontend build error Q&A.",
        add_completion=False,
    )

    @cli.command("parse-dump")
    def parse_dump_cmd(
        posts_xml: Path = typer.Argument(..., help="Path to Stack Exchange Posts.xml."),
        config: Path = typer.Option(DEFAULT_COLLECT_CONFIG, "--config", "-c"),
        limit: int | None = typer.Option(None, "--limit", "-n"),
    ) -> None:
        """Parse Posts.xml and append unified raw records to data/raw/stackexchange.jsonl."""
        cfg = _build_config(load_collect_config(config))
        summary = parse_stackexchange_dump(posts_xml, cfg, limit=limit)
        typer.echo(json.dumps(summary, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
