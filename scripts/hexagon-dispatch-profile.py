#!/usr/bin/env python3
"""Summarize PROFILE=3 dispatcher spans without treating nested spans as additive.

Only matched start/stop pairs contribute cycles. Trace saturation or missing
pairs marks coverage incomplete. Cycle deltas use the trace's 32-bit counter:
a single span must be shorter than one counter period. Host lifetime sums
include concurrent outstanding batches; pop includes profiling output.
"""
import argparse
import collections
import json
import re
from pathlib import Path

DISPATCH = frozenset(('OP_SETUP', 'OP_EXECUTE', 'OP_RETIRE'))
STAGES = frozenset(('TENSOR_PREP', 'WORKER_WAKE', 'WORKER_SUSPEND',
                    'OP_SETUP', 'OP_EXECUTE', 'OP_RETIRE', 'BUFF', 'L2FLUSH'))
PREFIX = re.compile(r'ggml-hex: (\S+) (.*)')
EVENT = re.compile(r'trace-evt (\w+): thread (\d+) info (\d+) (start|stop) (\d+)$')
STATE = re.compile(r'trace-state capacity (\d+) saturated-mask (\d+)$')


def summarize(lines):
    sessions = {}
    pending = {}

    def session(name):
        return sessions.setdefault(name, dict(batches=0, trace_batches=0, saturated_batches=0,
            unknown_capacity_batches=0, missing_dispatch_pairs=0, duplicate_dispatch_pairs=0,
            invalid_dispatch_indices=0, missing_batch_spans=0, unmatched_starts=0, unmatched_stops=0,
            stages={}, host=[]))

    def finish(name):
        batch = pending.pop(name, None)
        if batch is None or not batch['seen']:
            return
        out = session(name)
        out['trace_batches'] += 1
        out['saturated_batches'] += bool(batch['saturated'])
        out['unknown_capacity_batches'] += batch['capacity'] is None
        out['unmatched_starts'] += sum(len(v) for v in batch['starts'].values())
        expected = set(range(batch['n_ops']))
        for stage in DISPATCH:
            out['missing_dispatch_pairs'] += len(expected - batch['matched_ops'][stage])
        for stage, count in (('BUFF', 1), ('TENSOR_PREP', 1), ('WORKER_WAKE', 1),
                             ('WORKER_SUSPEND', 1), ('L2FLUSH', 2)):
            out['missing_batch_spans'] += max(0, count - batch['matched'][stage])

    for lineno, line in enumerate(lines, 1):
        prefix = PREFIX.search(line.strip())
        if not prefix:
            continue
        name, body = prefix.groups()
        if not body.startswith(('profile-op OPBATCH|', 'trace-state ', 'trace-evt ', 'profile-host ')):
            continue
        out = session(name)
        if body.startswith('profile-op OPBATCH|'):
            finish(name)
            out['batches'] += 1
            n_ops = re.search(r'\|n-ops (\d+)\|', body)
            if not n_ops:
                raise ValueError(f'line {lineno}: missing batch operation count')
            pending[name] = dict(seen=False, saturated=False, capacity=None, n_ops=int(n_ops[1]),
                                 starts=collections.defaultdict(list), matched=collections.Counter(),
                                 matched_ops=collections.defaultdict(set))
        elif body.startswith('trace-state ') or body.startswith('trace-evt '):
            batch = pending.get(name)
            if batch is None:
                raise ValueError(f'line {lineno}: trace without preceding OPBATCH')
            batch['seen'] = True
            if body.startswith('trace-state '):
                m = STATE.fullmatch(body)
                if not m:
                    raise ValueError(f'line {lineno}: malformed trace state')
                batch['capacity'], mask = map(int, m.groups())
                batch['saturated'] = bool(mask) or batch['capacity'] == 0
                continue
            m = EVENT.fullmatch(body)
            if not m:
                raise ValueError(f'line {lineno}: malformed trace event')
            stage, thread, info, edge, cycles = m.groups()
            # Main-thread stages contain worker activity. Keep worker spans out
            # of these totals, and do not add nested L2FLUSH to OP_SETUP/EXECUTE.
            if stage not in STAGES or int(thread) != 0:
                continue
            key = (stage, int(info))
            stack = batch['starts'][key]
            if edge == 'start':
                stack.append(int(cycles))
            elif not stack:
                out['unmatched_stops'] += 1
            else:
                delta = (int(cycles) - stack.pop()) & 0xffffffff
                total = out['stages'].setdefault(stage, dict(count=0, cycles=0))
                batch['matched'][stage] += 1
                if stage in DISPATCH:
                    index = int(info)
                    if index >= batch['n_ops']:
                        out['invalid_dispatch_indices'] += 1
                    if index in batch['matched_ops'][stage]:
                        out['duplicate_dispatch_pairs'] += 1
                    batch['matched_ops'][stage].add(index)
                total['count'] += 1
                total['cycles'] += delta
        elif body.startswith('profile-host '):
            fields = body.split()[1:]
            if len(fields) % 2 or any(not v.isdecimal() for v in fields[1::2]):
                raise ValueError(f'line {lineno}: malformed host totals')
            out['host'].append(dict(zip(fields[::2], map(int, fields[1::2]))))
    for name in list(pending):
        finish(name)
    for out in sessions.values():
        out['complete_trace'] = (out['batches'] > 0 and out['trace_batches'] == out['batches'] and
            not any(out[k] for k in ('saturated_batches', 'unknown_capacity_batches',
                                    'unmatched_starts', 'unmatched_stops', 'missing_dispatch_pairs',
                                    'duplicate_dispatch_pairs', 'invalid_dispatch_indices', 'missing_batch_spans')))
    return dict(notes=[
        'Stage cycles are inclusive; nested L2FLUSH/worker/kernel timings are not additive.',
        'Only matched spans contribute; incomplete/saturated captures provide partial totals.',
        'Each span must be shorter than one 32-bit cycle-counter period.',
        'Host lifetime sums overlap queued batches; host pop includes profile logging.',
    ], sessions=sessions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log.read_text().splitlines()), indent=2))


if __name__ == '__main__':
    main()
