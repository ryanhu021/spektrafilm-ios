#!/usr/bin/env python3
"""Check that the documentation still describes the repository it ships with.

Docs drift quietly. A count in prose stays plausible long after it stops being true, and a relative
link keeps rendering as a link after the file moves. Every check here verifies a claim a reader
would act on; none of them need the parity oracle, so this runs on a bare checkout.

Usage:  Tools/check_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".build", ".git", "oracle", "DerivedData"}

problems: list[str] = []
notes: list[tuple[str, str]] = []


def problem(message: str) -> None:
    problems.append(message)


def note(label: str, value: str) -> None:
    notes.append((label, value))


def markdown_files() -> list[Path]:
    return sorted(
        p
        for p in REPO.rglob("*.md")
        if not any(part in SKIP_DIRS for part in p.relative_to(REPO).parts)
    )


LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def check_links() -> None:
    """Every relative markdown link resolves to a file that exists."""
    checked = 0
    for doc in markdown_files():
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = (doc.parent / target.split("#", 1)[0]).resolve()
            checked += 1
            if not path.exists():
                problem(f"{doc.relative_to(REPO)} links to missing {target}")
    note("relative links checked", str(checked))


def check_profiles() -> None:
    """Prose that states a profile count agrees with the bundled JSON."""
    profiles = list((REPO / "Sources/SpektraFilm/Resources/profiles").glob("*.json"))
    note("bundled profiles", str(len(profiles)))
    stated = re.compile(r"\b(\d+)\s+(?:upstream\s+)?(?:measured\s+)?(?:film|profiles)\b")
    for doc in markdown_files():
        for match in stated.finditer(doc.read_text(encoding="utf-8")):
            if int(match.group(1)) != len(profiles):
                problem(
                    f"{doc.relative_to(REPO)} says {match.group(0)!r}"
                    f" but {len(profiles)} profiles are bundled"
                )


def check_goldens() -> None:
    """The committed manifest and the .spkg files on disk describe the same set."""
    goldens = REPO / "Tests/SpektraFilmTests/Goldens"
    manifest_path = goldens / "manifest.json"
    if not manifest_path.exists():
        problem("Tests/SpektraFilmTests/Goldens/manifest.json is missing; run `make goldens`")
        return
    manifest = json.loads(manifest_path.read_text())
    on_disk = {p.stem for p in goldens.glob("*.spkg")}
    listed = set(manifest)
    for name in sorted(listed - on_disk):
        problem(f"manifest lists golden {name!r} with no .spkg file; run `make goldens`")
    for name in sorted(on_disk - listed):
        problem(f"golden {name}.spkg is not in the manifest; run `make goldens`")
    note("goldens", f"{len(on_disk)} fixtures")


def check_oracle_pin() -> None:
    """The oracle is pinned to a full SHA; a short one can become ambiguous."""
    pin = json.loads((REPO / "Tools/parity/upstream_pin.json").read_text())
    commit = pin["upstream"]["oracle_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        problem(f"oracle_commit {commit!r} is not a full 40-character SHA")
    else:
        note("oracle pin", commit[:12] + "…")


def check_ci_references() -> None:
    """Targets and scripts that CI invokes exist, so a green local run means something."""
    makefile = (REPO / "Makefile").read_text()
    targets = {m.group(1) for m in re.finditer(r"^([a-z][a-z-]*):", makefile, re.M)}
    workflows = sorted((REPO / ".github/workflows").glob("*.yml"))
    for workflow in workflows:
        text = workflow.read_text()
        for target in re.findall(r"\bmake ([a-z-]+)", text):
            if target not in targets:
                problem(
                    f"{workflow.name} runs `make {target}` but the Makefile has no such target"
                )
        for script in re.findall(r"\bTools/[A-Za-z0-9_/.-]+\.(?:sh|py)", text):
            path = REPO / script
            if not path.exists():
                problem(f"{workflow.name} runs {script} which does not exist")
            elif not path.stat().st_mode & 0o111:
                problem(f"{script} is invoked by {workflow.name} but is not executable")
    note("workflows checked", str(len(workflows)))


def check_swift_paths() -> None:
    """Paths the lint and build steps name actually exist.

    A lint step pointed at a directory that was renamed passes by checking nothing.
    """
    lint_paths = set()
    for workflow in (REPO / ".github/workflows").glob("*.yml"):
        lines = workflow.read_text().splitlines()
        for i, line in enumerate(lines):
            if "swift format lint" not in line:
                continue
            # Collect the command's own arguments, following backslash continuations. Scanning the
            # whole file instead would pick up `-project App/Spektrafilm.xcodeproj` from the build
            # job, which is generated and deliberately untracked.
            block = [line]
            while block[-1].rstrip().endswith("\\") and i + len(block) < len(lines):
                block.append(lines[i + len(block)])
            for token in " ".join(block).replace("\\", " ").split():
                if token.startswith(("Sources", "Tests", "App")) or token == "Package.swift":
                    lint_paths.add(token)
    for path in sorted(lint_paths):
        if not (REPO / path).exists():
            problem(f"swift-format lint names {path!r}, which does not exist")
    note("lint paths", ", ".join(sorted(lint_paths)) or "none")


def main() -> int:
    for check in (
        check_links,
        check_profiles,
        check_goldens,
        check_oracle_pin,
        check_ci_references,
        check_swift_paths,
    ):
        check()

    for label, value in notes:
        print(f"  {label:<28} {value}")
    if problems:
        print()
        for message in problems:
            print(f"  FAIL  {message}")
        print(f"\n{len(problems)} documentation problem(s)")
        return 1
    print("\ndocumentation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
