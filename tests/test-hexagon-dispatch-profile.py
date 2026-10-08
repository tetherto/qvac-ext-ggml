#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('dispatch_profile',
    Path(__file__).resolve().parents[1] / 'scripts' / 'hexagon-dispatch-profile.py')
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def batch(n=1):
    return f'ggml-hex: HTP0 profile-op OPBATCH|----|n-ops {n}|----|----|----|usec 10 cycles 100 start 0 mhz 10'


def state(capacity=100, mask=0):
    return f'ggml-hex: HTP0 trace-state capacity {capacity} saturated-mask {mask}'


def event(stage, edge, cycle, info=0, thread=0):
    return f'ggml-hex: HTP0 trace-evt {stage}: thread {thread} info {info} {edge} {cycle}'


def complete():
    return [batch(), state()] + [event(s, e, c) for s in ('BUFF', 'TENSOR_PREP', 'WORKER_WAKE', 'WORKER_SUSPEND',
                                                                  'L2FLUSH', 'L2FLUSH', 'OP_SETUP', 'OP_EXECUTE', 'OP_RETIRE')
                                 for e, c in (('start', 100), ('stop', 150))]


class TestDispatchProfile(unittest.TestCase):
    def read(self, lines):
        return PROFILE.summarize(lines)['sessions']['HTP0']

    def test_complete_and_slot_reuse(self):
        out = self.read(complete() + complete())
        self.assertTrue(out['complete_trace'])
        self.assertEqual(out['stages']['OP_SETUP'], dict(count=2, cycles=100))

    def test_saturation_is_reported_even_with_matched_pairs(self):
        lines = complete()
        lines[1] = state(mask=2)  # worker saturation also invalidates coverage
        out = self.read(lines)
        self.assertFalse(out['complete_trace'])
        self.assertEqual(out['saturated_batches'], 1)

    def test_missing_stop_never_pairs_across_batch(self):
        out = self.read([batch(), state(), event('OP_SETUP', 'start', 1), batch(), state(),
                         event('OP_SETUP', 'stop', 10)])
        self.assertEqual(out['unmatched_starts'], 1)
        self.assertEqual(out['unmatched_stops'], 1)
        self.assertNotIn('OP_SETUP', out['stages'])
        self.assertFalse(out['complete_trace'])

    def test_missing_entire_stage_or_capacity_is_incomplete(self):
        for lines in ([batch(), state()], [x for x in complete() if 'trace-state' not in x]):
            self.assertFalse(self.read(lines)['complete_trace'])

    def test_wrapping_cycles_and_nested_spans(self):
        lines = [batch(), state(), event('OP_SETUP', 'start', 0xfffffff0),
                 event('L2FLUSH', 'start', 0xfffffff8), event('L2FLUSH', 'stop', 2),
                 event('OP_SETUP', 'stop', 16),
                 event('OP_EXECUTE', 'start', 20, thread=1),
                 event('OP_EXECUTE', 'stop', 900, thread=1)]
        out = self.read(lines)
        self.assertEqual(out['stages']['OP_SETUP']['cycles'], 32)
        self.assertEqual(out['stages']['L2FLUSH']['cycles'], 10)
        self.assertNotIn('OP_EXECUTE', out['stages'])

    def test_duplicate_indices_cannot_mask_missing_operations(self):
        lines = complete()
        lines[0] = batch(2)
        lines += [event(s, e, c) for s in ('OP_SETUP', 'OP_EXECUTE', 'OP_RETIRE')
                  for e, c in (('start', 200), ('stop', 250))]
        out = self.read(lines)
        self.assertFalse(out['complete_trace'])
        self.assertEqual(out['duplicate_dispatch_pairs'], 3)
        self.assertEqual(out['missing_dispatch_pairs'], 3)

    def test_out_of_range_indices_and_missing_batch_spans(self):
        out = self.read(complete() + [event('OP_SETUP', 'start', 1, info=5),
                                     event('OP_SETUP', 'stop', 2, info=5)])
        self.assertFalse(out['complete_trace'])
        self.assertEqual(out['invalid_dispatch_indices'], 1)
        out = self.read([x for x in complete() if 'WORKER_WAKE:' not in x])
        self.assertFalse(out['complete_trace'])
        self.assertEqual(out['missing_batch_spans'], 1)

    def test_host_lifetimes_are_kept_separate(self):
        out = self.read(complete() + ['ggml-hex: HTP0 profile-host batches 2 ops 2 wait-us 90 lifetime-sum-us 180'])
        self.assertEqual(out['host'][0]['lifetime-sum-us'], 180)
        self.assertEqual(out['stages']['OP_EXECUTE']['cycles'], 50)

    def test_malformed_records_fail(self):
        for bad in ('trace-state capacity ??? saturated-mask 0', 'trace-evt OP_SETUP: garbage',
                    'profile-host ops many'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.read([batch(), 'ggml-hex: HTP0 ' + bad])


if __name__ == '__main__':
    unittest.main()
