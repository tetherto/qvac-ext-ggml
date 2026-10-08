#!/usr/bin/env python3
"""Regression tests for inclusive batch timing and run boundaries."""

import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "hexagon_profile", Path(__file__).resolve().parents[1] / "scripts" / "hexagon-profile.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def record(op, cycles, kernel="hmx-tiled", backend="HTP0", shape="96:96 x 96:33 -> 96:33"):
    return (f"ggml-hex: {backend} profile-op {op}|weights x input -> out|{shape}|f16 x f32 -> f32|"
            f"2:192 x 4:384 -> 4:384|{kernel} vtcm 4096|usec 1 cycles {cycles} start 4294967290 mhz 1.0\n")


class TestProfile(unittest.TestCase):
    def test_batch_is_inclusive(self):
        result = PROFILE.summarize([record("OPBATCH", 100), record("MUL_MAT+ADD", 40), record("ADD", 10)])
        session = result[0]["sessions"][0]
        self.assertEqual(session["leaf_cycles"], 50)
        self.assertEqual(session["batch_cycles_inclusive"], 100)
        self.assertEqual(session["groups"][0]["leaf_cycle_percent"], 80)
        self.assertEqual(session["groups"][0]["batch_cycle_percent"], 40)

    def test_sections_and_repeated_labels_stay_separate(self):
        result = PROFILE.summarize([
            "=== profile warmup (not captured) ===\n", record("MUL_MAT+ADD", 40),
            "=== profile run (captured) ===\n", record("MUL_MAT+ADD", 30),
            "=== profile run (captured) ===\n", record("MUL_MAT+ADD", 20),
        ])
        self.assertEqual(len(result), 3)
        self.assertEqual([r["sessions"][0]["leaf_cycles"] for r in result], [40, 30, 20])

    def test_group_by_shape_type_kernel_and_session(self):
        result = PROFILE.summarize([
            record("MUL_MAT+ADD", 30), record("MUL_MAT+ADD", 50),
            record("MUL_MAT+ADD", 10, kernel="hvx-flat"),
            record("MUL_MAT+ADD", 20, shape="96:96 x 96:1 -> 96:1"),
            record("MUL_MAT+ADD", 60).replace("f16 x f32", "f32 x f32"),
            record("MUL_MAT+ADD", 70, backend="HTP1"),
        ])
        self.assertEqual(result[0]["section"], "unlabelled")
        sessions = result[0]["sessions"]
        self.assertEqual(len(sessions), 2)
        self.assertEqual(len(sessions[0]["groups"]), 4)
        self.assertEqual(sessions[0]["groups"][0]["count"], 2)
        self.assertEqual(sessions[0]["groups"][0]["mean_cycles"], 40)
        self.assertIsNone(sessions[0]["groups"][0]["batch_cycle_percent"])

    def test_pmu_suffix_and_zero_duration(self):
        line = record("MUL_MAT+ADD", 0).replace("usec 1", "usec 0").rstrip() + " pmu [1,2,3,4,5,6,7,8]\n"
        result = PROFILE.summarize(["unrelated application output\n", line])
        self.assertEqual(result[0]["sessions"][0]["groups"][0]["leaf_cycle_percent"], 0)

    def test_malformed_record_fails_instead_of_silently_omitting_work(self):
        with self.assertRaisesRegex(ValueError, "line 1"):
            PROFILE.summarize([record("MUL_MAT+ADD", 42).replace("cycles 42", "cycles ???")])


if __name__ == "__main__":
    unittest.main()
