#!/usr/bin/env python3
"""Static and rendered-artifact audit for the ECG reconstruction Quarto book.

The audit is deliberately evidence-only: it does not execute cells or infer that
a prose claim is true. It inventories chapters, Python blocks, labels, local
links, render freshness, and deployed resource closure so reviewers can focus on
scientific gaps without overlooking mechanical release defects.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html.parser
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LocalRefs(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.refs: list[str] = []; self.external_refs: list[str] = []; self.external_runtime: list[str] = []; self.generator: str | None = None

    def handle_starttag(self, _tag, attrs):
        attrs_dict=dict(attrs)
        if _tag == "meta" and attrs_dict.get("name") == "generator":
            self.generator=attrs_dict.get("content")
        for key, value in attrs:
            if key not in {"src", "href"} or not value:
                continue
            if value.startswith(("http:", "https:")):
                self.external_refs.append(value)
                if _tag in {"script", "iframe", "img"} or (_tag == "link" and key == "href"):
                    self.external_runtime.append(value)
                continue
            if value.startswith(("mailto:", "data:", "javascript:", "#")):
                continue
            self.refs.append(value.split("#", 1)[0].split("?", 1)[0])


def configured_chapters(config: Path) -> list[str]:
    text = config.read_text()
    return re.findall(r"^\s{4,}-\s+([\w.-]+\.qmd)\s*$", text, flags=re.M)


def code_blocks(text: str) -> list[dict]:
    blocks=[]
    for match in re.finditer(r"^```\{([^}]+)\}\s*\n(.*?)^```\s*$", text, flags=re.M|re.S):
        body=match.group(2);line=text.count("\n",0,match.start())+1
        label=re.search(r"^#\|\s*label:\s*(\S+)\s*$",body,flags=re.M)
        eval_opt=re.search(r"^#\|\s*eval:\s*(\S+)\s*$",body,flags=re.M)
        blocks.append({"engine":match.group(1),"line":line,"label":label.group(1) if label else None,
                       "eval":eval_opt.group(1) if eval_opt else None,"lines":body.count("\n")+1})
    return blocks


def audit_chapter(book: Path, name: str) -> dict:
    source=book/name;text=source.read_text();blocks=code_blocks(text);html=book/"_book"/name.replace(".qmd",".html")
    refs=[];missing=[];complete=False;generator=None;external=[];runtime=[];output_blocks=0;identifier_hits=[]
    if html.exists():
        rendered=html.read_text(errors="ignore");parser=LocalRefs();parser.feed(rendered);refs=sorted(set(parser.refs))
        generator=parser.generator;external=sorted(set(parser.external_refs));runtime=sorted(set(parser.external_runtime));output_blocks=len(re.findall(r'class="[^"]*\bcell-output\b',rendered))
        identifier_hits=sorted(set(re.findall(r'\bMUSE_[A-Za-z0-9_-]+\b',rendered)))[:20]
        missing=[r for r in refs if not (html.parent/r).resolve().exists()]
        complete=html.read_bytes().rstrip().endswith(b"</html>")
    headings=[m.group(0).lstrip("#").strip() for m in re.finditer(r"^#{1,6}\s+.+$",text,flags=re.M)]
    return {"chapter":name,"source_bytes":source.stat().st_size,"source_mtime":source.stat().st_mtime,
            "headings":len(headings),"python_blocks":sum(b["engine"]=="python" for b in blocks),
            "all_blocks":len(blocks),"unlabeled_blocks":[b["line"] for b in blocks if not b["label"]],
            "labels":[b["label"] for b in blocks if b["label"]],"html_exists":html.exists(),
            "html_complete":complete,"html_bytes":html.stat().st_size if html.exists() else None,
            "html_stale":bool(html.exists() and html.stat().st_mtime<source.stat().st_mtime),
            "local_resource_count":len(refs),"missing_local_resources":missing,
            "generator":generator,"external_resources":external,"external_runtime":runtime,"output_blocks":output_blocks,
            "identifier_hits":identifier_hits,
            "expected_executable_blocks":sum(b["engine"]=="python" and str(b["eval"]).lower()!="false" for b in blocks)}


def markdown(report: dict) -> str:
    lines=["# Quarto Book Mechanical Audit","",f"Generated: `{report['generated_at']}`","",
           "> This audit verifies structure and deployed artifacts, not scientific truth.","",
           "## Release gates","",
           f"- Configured chapters: **{report['summary']['configured_chapters']}**",
           f"- Missing source chapters: **{len(report['summary']['missing_sources'])}**",
           f"- Missing rendered chapters: **{report['summary']['missing_html']}**",
           f"- Incomplete HTML files: **{report['summary']['incomplete_html']}**",
           f"- Source-newer-than-HTML chapters: **{report['summary']['stale_html']}**",
           f"- Missing deployed local resources: **{report['summary']['missing_resources']}**",
           f"- Unlabeled executable blocks: **{report['summary']['unlabeled_blocks']}**",
           f"- Duplicate executable labels: **{len(report['summary']['duplicate_labels'])}**","",
           f"- Wrong renderer identity: **{report['summary']['wrong_renderer']}**",
           f"- Executable chapters with zero outputs: **{report['summary']['zero_output_chapters']}**",
           f"- Distinct external runtime resources: **{report['summary']['external_resources']}**","",
           f"- Unvendored external runtime dependencies: **{report['summary']['external_runtime']}**","",
           f"- Public acquisition-identifier hits: **{report['summary']['identifier_hits']}**","",
           "## Chapter inventory","",
           "| Chapter | Headings | Python blocks | Outputs | Renderer | HTML | Stale | Missing resources |",
           "|---|---:|---:|---:|---|---|---|---:|"]
    for row in report["chapters"]:
        lines.append(f"| `{row['chapter']}` | {row['headings']} | {row['python_blocks']} | {row['output_blocks']} | {row['generator']} | "
                     f"{'complete' if row['html_complete'] else 'missing/incomplete'} | {row['html_stale']} | {len(row['missing_local_resources'])} |")
    lines += ["","## Exact defects","", "```json",json.dumps({
        "missing_sources":report["summary"]["missing_sources"],
        "duplicate_labels":report["summary"]["duplicate_labels"],
        "chapters_with_unlabeled_blocks":{r["chapter"]:r["unlabeled_blocks"] for r in report["chapters"] if r["unlabeled_blocks"]},
        "chapters_with_missing_resources":{r["chapter"]:r["missing_local_resources"] for r in report["chapters"] if r["missing_local_resources"]},
        "wrong_renderer":{r["chapter"]:r["generator"] for r in report["chapters"] if r["generator"] != "quarto-1.10.18"},
        "zero_output_chapters":[r["chapter"] for r in report["chapters"] if r["expected_executable_blocks"] and not r["output_blocks"]],
        "external_resources":sorted({x for r in report["chapters"] for x in r["external_resources"]}),
        "external_runtime":sorted({x for r in report["chapters"] for x in r["external_runtime"]}),
        "identifier_hits":{r["chapter"]:r["identifier_hits"] for r in report["chapters"] if r["identifier_hits"]},
    },indent=2),"```",""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument("--book",type=Path,default=ROOT/"book");p.add_argument("--json",type=Path);p.add_argument("--markdown",type=Path);a=p.parse_args()
    chapters=configured_chapters(a.book/"_quarto.yml");missing=[x for x in chapters if not (a.book/x).exists()]
    rows=[audit_chapter(a.book,x) for x in chapters if (a.book/x).exists()]
    labels=[label for row in rows for label in row["labels"]];dupes=sorted(k for k,v in Counter(labels).items() if v>1)
    report={"generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),"book":str(a.book.resolve()),"chapters":rows,
            "summary":{"configured_chapters":len(chapters),"missing_sources":missing,
                       "missing_html":sum(not r["html_exists"] for r in rows),"incomplete_html":sum(not r["html_complete"] for r in rows),
                       "stale_html":sum(r["html_stale"] for r in rows),"missing_resources":sum(len(r["missing_local_resources"]) for r in rows),
                       "unlabeled_blocks":sum(len(r["unlabeled_blocks"]) for r in rows),"duplicate_labels":dupes,
                       "wrong_renderer":sum(r["generator"] != "quarto-1.10.18" for r in rows),
                       "zero_output_chapters":sum(bool(r["expected_executable_blocks"] and not r["output_blocks"]) for r in rows),
                       "external_resources":len({x for r in rows for x in r["external_resources"]}),
                       "external_runtime":len({x for r in rows for x in r["external_runtime"]}),
                       "identifier_hits":sum(len(r["identifier_hits"]) for r in rows)}}
    if a.json:a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(report,indent=2)+"\n")
    if a.markdown:a.markdown.parent.mkdir(parents=True,exist_ok=True);a.markdown.write_text(markdown(report))
    print(json.dumps(report["summary"],indent=2))
    return int(bool(missing or report["summary"]["missing_html"] or report["summary"]["incomplete_html"] or report["summary"]["stale_html"] or
                    report["summary"]["missing_resources"] or dupes or report["summary"]["wrong_renderer"] or
                    report["summary"]["zero_output_chapters"] or report["summary"]["identifier_hits"] or
                    report["summary"]["external_runtime"]))


if __name__=="__main__":raise SystemExit(main())
