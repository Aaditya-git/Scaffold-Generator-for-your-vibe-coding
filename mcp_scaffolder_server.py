#!/usr/bin/env python3
"""
MCP Server: Project Scaffolder (Python)

Expose a single MCP Tool `scaffold_project` that takes a natural-language or
ASCII tree prompt describing a folder layout and creates it on the local
filesystem (within an allowed base path).

Transports: stdio (works with Claude Desktop, Cursor, etc.)

Install deps:
  pip install fastmcp pyyaml

Run locally for inspection:
  python mcp_scaffolder_server.py
  # or via MCP CLI:
  mcp dev mcp_scaffolder_server.py

Claude Desktop config example (~/.claude/servers/scaffolder.json):
{
  "name": "Scaffolder",
  "command": "python",
  "args": ["/ABSOLUTE/PATH/mcp_scaffolder_server.py"],
  "env": {"MCP_SCAFFOLDER_BASE": "/Users/me/dev"}
}

SECURITY
- All writes are restricted under the directory specified by env var
  `MCP_SCAFFOLDER_BASE`. Requests targeting paths outside this base are denied.
- No stdout prints (stdio transport requirement). Use returned JSON for results.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Union, Literal

import os
import re
import json
import pathlib
import textwrap

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Use FastMCP from the fastmcp package (not mcp.server.*)
from fastmcp import FastMCP

# -----------------------------
# Utility & parsing
# -----------------------------
BOX_CHARS = "│├└─"

class SpecError(Exception):
    pass


def _leading_units(line: str) -> int:
    tmp = line
    for ch in BOX_CHARS:
        tmp = tmp.replace(ch, " ")
    m = re.search(r"(├─ |└─ )", line)
    if m:
        prefix = tmp[: m.start()]
    else:
        prefix = re.match(r"^\s*", tmp).group(0) or ""  # type: ignore
    return max(0, len(prefix) // 2)


def parse_ascii_tree(tree_text: str) -> List[Tuple[int, str, bool]]:
    lines = [ln.rstrip() for ln in tree_text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    out: List[Tuple[int, str, bool]] = []
    start_idx = 1 if lines[0].strip().endswith('/') else 0
    for raw in lines[start_idx:]:
        name = re.sub(r"^.*?(├─ |└─ )", "", raw).strip()
        if name == raw:
            name = raw.lstrip(" \t" + BOX_CHARS).lstrip("- ")
        is_dir = name.endswith('/')
        if is_dir:
            name = name[:-1]
        depth = _leading_units(raw)
        out.append((depth, name, is_dir))
    return out


def collapse_tree(items: List[Tuple[int, str, bool]]) -> List[Tuple[str, bool]]:
    stack: List[str] = []
    paths: List[Tuple[str, bool]] = []
    for depth, name, is_dir in items:
        while len(stack) > depth:
            stack.pop()
        if is_dir:
            if len(stack) == depth:
                stack.append(name)
            else:
                if stack:
                    stack[-1] = name
                else:
                    stack.append(name)
        base = "/".join([s for s in stack])
        rel = f"{base + ('/' if base else '')}{name}" if base else name
        paths.append((rel, is_dir))
    return paths

Entry = Dict[str, Union[str, Dict]]

def parse_yaml_or_json(data: Union[dict, list]) -> List[Tuple[str, bool, Optional[str]]]:
    results: List[Tuple[str, bool, Optional[str]]] = []

    def add(path: str, is_dir: bool, content: Optional[str] = None):
        results.append((path.rstrip('/'), is_dir, content))

    def walk(prefix: str, mapping: dict):
        for k, v in mapping.items():
            if isinstance(v, dict):
                p = f"{prefix}/{k}" if prefix else k
                add(p, True)
                walk(p, v)
            else:
                p = f"{prefix}/{k}" if prefix else k
                add(p, False, None if v is None else str(v))

    if isinstance(data, dict) and ("entries" in data or "root" in data):
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise SpecError("YAML/JSON flat shape requires 'entries' list.")
        for it in entries:
            if not isinstance(it, dict) or "path" not in it:
                raise SpecError("Each entry must include 'path'.")
            path = str(it["path"])  # type: ignore
            typ = str(it.get("type", "file")).lower()
            is_dir = typ in {"dir","folder","directory"} or path.endswith('/')
            content = it.get("content")
            add(path, is_dir, None if content is None else str(content))
        return results

    mapping = data.get("tree") if isinstance(data, dict) and "tree" in data else data
    if not isinstance(mapping, dict):
        raise SpecError("YAML/JSON nested shape must be object or under 'tree'.")
    walk("", mapping)
    return results

DEFAULT_SNIPPETS: Dict[str, str] = {
    "README.md": "# Project\n\nDescribe your project here.\n",
    ".env.example": "# Copy to .env and fill values\nAPI_KEY=\n",
    "requirements.txt": "# Add Python dependencies here\n",
}

# -----------------------------
# Filesystem
# -----------------------------

def safe_join(base: pathlib.Path, target: str) -> pathlib.Path:
    """Resolve target within base; raise if escaping the base."""
    base_res = base.resolve()
    p = (base_res / target).resolve()
    if not str(p).startswith(str(base_res)):
        raise SpecError(f"Target escapes base directory: {p} not under {base_res}")
    return p


# -----------------------------
# MCP server + tool
# -----------------------------

mcp = FastMCP("Scaffolder")

@mcp.tool(description="Create folders/files from a prompt (ASCII tree, YAML/JSON, or line-based). Writes only under MCP_SCAFFOLDER_BASE.")
def scaffold_project(
    prompt: str,
    target_path: str,
    mode: Literal["auto","tree","yaml","json"] = "auto",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """Generate the requested structure under the allowed base directory.

    Returns a JSON payload with created/skipped/errors arrays.
    """
    base = os.environ.get("MCP_SCAFFOLDER_BASE")
    if not base:
        raise SpecError("Set MCP_SCAFFOLDER_BASE to an allowed write root.")

    base_path = pathlib.Path(base)
    if not base_path.exists():
        raise SpecError(f"MCP_SCAFFOLDER_BASE does not exist: {base_path}")

    created: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    # Determine parse mode
    selected = mode
    if mode == "auto":
        text = prompt.strip()
        first = text.splitlines()[0] if text else ""
        if text.startswith("{") or text.startswith("["):
            selected = "json"
        elif re.search(r"^\s*\w+\s*:\s*", text) or "entries:" in text or "tree:" in text:
            selected = "yaml"
        elif ("├" in text or "└" in text or "/" in first or first.endswith('/')):
            selected = "tree"
        else:
            selected = "tree"  # treat NL lines as a basic tree

    paths: List[Tuple[str, bool]] = []
    contents: Dict[str, Optional[str]] = {}

    try:
        if selected == "tree":
            items = parse_ascii_tree(prompt)
            if not items:
                # NL fallback: each line => dir if endswith '/', else file at root
                lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
                for ln in lines:
                    isd = ln.endswith('/')
                    name = ln[:-1] if isd else ln
                    items.append((0, name, isd))
            for rel, isd in collapse_tree(items):
                paths.append((rel, isd))
        elif selected in ("yaml","json"):
            if selected == "yaml":
                if yaml is None:
                    raise SpecError("YAML requested but PyYAML is not installed.")
                data = yaml.safe_load(prompt)
            else:
                data = json.loads(prompt)
            for p, isd, content in parse_yaml_or_json(data):
                paths.append((p, isd))
                if not isd:
                    contents[p] = content
        else:
            raise SpecError(f"Unsupported mode: {selected}")
    except Exception as e:  # parse errors
        raise SpecError(f"Failed to parse prompt as {selected}: {e}")

    # Create filesystem entries
    base_target = safe_join(base_path, target_path.strip().lstrip('/'))
    if dry_run:
        return {
            "selected_mode": selected,
            "base": str(base_path),
            "target": str(base_target),
            "plan": [
                {"path": str(base_target / rel), "type": "dir" if isd else "file"}
                for rel, isd in paths
            ],
            "created": created,
            "skipped": skipped,
            "errors": errors,
            "dry_run": True,
        }

    for rel, isd in paths:
        abs_path = safe_join(base_target, rel)
        try:
            if isd:
                abs_path.mkdir(parents=True, exist_ok=True)
                created.append(str(abs_path))
            else:
                if abs_path.exists() and not overwrite:
                    skipped.append(str(abs_path))
                    continue
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                content = contents.get(rel)
                if content is None:
                    content = DEFAULT_SNIPPETS.get(os.path.basename(rel), "")
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(content)
                created.append(str(abs_path))
        except Exception as e:
            errors.append(f"{abs_path}: {e}")

    return {
        "selected_mode": selected,
        "base": str(base_path),
        "target": str(base_target),
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "dry_run": False,
    }


# Optional: helper prompt template
@mcp.prompt()
def scaffold_prompt_template() -> str:
    return textwrap.dedent(
        """
        Provide either:
        1) ASCII tree (box-drawing ok):
           my-app/
           ├─ README.md
           └─ src/
              └─ main.py
        2) YAML nested mapping:
           tree:
             README.md: "# Title\n"
             src:
               main.py: "print('hi')\n"
        3) YAML entries list:
           entries:
             - { path: README.md, type: file, content: "# Title\n" }
             - { path: src/, type: dir }
             - { path: src/main.py, type: file }
        """
    ).strip()


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # STDIO is default; avoids manual stdio_server wiring.
    mcp.run()
