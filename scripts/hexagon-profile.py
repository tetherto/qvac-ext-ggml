#!/usr/bin/env python3
"""Summarize GGML_HEXAGON_PROFILE logs without double-counting OPBATCH.

Each ``=== label ===`` starts a separate section, including repeated labels.
Unmarked input is reported as ``unlabelled``; no warmup exclusion is inferred.
Only standard-library modules are required.
"""

import argparse
import json
from pathlib import Path
import re


MARKER = re.compile(r"^\s*===\s*(.*?)\s*===\s*$")
RECORD = re.compile(r"\b(HTP\d+) profile-op (.*)")
TIMING = re.compile(r"^usec (\d+) cycles (\d+) start (\d+) mhz \S+(?: pmu \[.*\])?$")


def summarize(lines):
    """Return per-section/session totals and groups sorted by leaf cycles."""
    sections = []
    label = "unlabelled"
    sessions = {}

    def finish():
        if not sessions:
            return
        result = {"section": label, "sessions": []}
        for name, session in sessions.items():
            groups = sorted(session.pop("groups").values(), key=lambda g: g["cycles"], reverse=True)
            leaf_cycles = session["leaf_cycles"]
            batch_cycles = session["batch_cycles_inclusive"]
            for group in groups:
                group["mean_cycles"] = group["cycles"] / group["count"]
                group["leaf_cycle_percent"] = 100 * group["cycles"] / leaf_cycles if leaf_cycles else 0
                group["batch_cycle_percent"] = 100 * group["cycles"] / batch_cycles if batch_cycles else None
            result["sessions"].append({"backend": name, **session, "groups": groups})
        sections.append(result)

    for lineno, line in enumerate(lines, 1):
        marker = MARKER.match(line)
        if marker:
            finish()
            label = marker[1]
            sessions = {}
            continue
        record = RECORD.search(line)
        if not record:
            continue
        fields = record[2].strip().split("|")
        timing = TIMING.match(fields[-1])
        if len(fields) != 7 or not timing:
            raise ValueError(f"line {lineno}: malformed profile-op record")
        op, _names, shape, types, _strides, params, _timing = fields
        usec, cycles = int(timing[1]), int(timing[2])
        session = sessions.setdefault(record[1], {
            "leaf_count": 0, "leaf_cycles": 0, "leaf_usec": 0,
            "batch_count": 0, "batch_cycles_inclusive": 0, "batch_usec_inclusive": 0,
            "groups": {},
        })
        if op == "OPBATCH":
            session["batch_count"] += 1
            session["batch_cycles_inclusive"] += cycles
            session["batch_usec_inclusive"] += usec
            continue
        session["leaf_count"] += 1
        session["leaf_cycles"] += cycles
        session["leaf_usec"] += usec
        kernel = params.split(" vtcm ", 1)[0]
        group = session["groups"].setdefault((op, shape, types, kernel), {
            "op": op, "shape": shape, "types": types, "kernel": kernel,
            "count": 0, "cycles": 0, "usec": 0,
        })
        group["count"] += 1
        group["cycles"] += cycles
        group["usec"] += usec
    finish()
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--section", help="keep only sections with this exact label")
    parser.add_argument("--json", action="store_true", help="emit all groups as JSON")
    parser.add_argument("--top", type=int, default=20, help="groups per section/session in text output")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        with args.log.open() as stream:
            sections = summarize(stream)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.section is not None:
        sections = [section for section in sections if section["section"] == args.section]
    if not sections:
        parser.error("no matching profile-op records")
    if args.json:
        print(json.dumps({"source": str(args.log), "sections": sections}, indent=2))
        return
    print("OPBATCH includes leaf operations: never add batch and leaf totals.")
    print("Percentages describe profiled DSP cycles, not end-to-end inference time.")
    for section in sections:
        for session in section["sessions"]:
            print(f"\n{section['section']} / {session['backend']}")
            print(f"Leaf ops: {session['leaf_count']}, {session['leaf_cycles']:,} cycles")
            print(f"Batches: {session['batch_count']}, {session['batch_cycles_inclusive']:,} cycles (inclusive)")
            for group in session["groups"][:args.top]:
                print(f"{group['leaf_cycle_percent']:6.2f}% leaf | {group['count']:6d} calls | "
                      f"{group['cycles']:12,d} cycles | {group['op']} | {group['kernel']} | "
                      f"{group['types']} | {group['shape']}")


if __name__ == "__main__":
    main()
