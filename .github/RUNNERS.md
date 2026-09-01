# Runner label catalog

CI runner labels live in one place: [`.github/runners.yaml`](./runners.yaml).
This lets a runner-image migration (OS bump, GPU pool rename, hosted-image
retirement) update a single file instead of grepping every workflow.

## How it works

`runners.yaml` is the source of truth. A generated reusable workflow,
[`.github/workflows/reusable-runner-names.yml`](./workflows/reusable-runner-names.yml),
exports each catalog entry as a job output. Callers pull the label from that
output instead of hardcoding it, because `runs-on:` is evaluated before any step
runs, so a composite action cannot supply the label — a reusable workflow's
outputs can.

A catalog value is one of:

- a **scalar** label (`ubuntu-22.04`) — consumed as
  `runs-on: ${{ needs.runner_names.outputs.<key> }}`
- a **composite label set** (`[self-hosted, Linux, X64, NVIDIA]`) — exported as a
  JSON array string and consumed as
  `runs-on: ${{ fromJSON(needs.runner_names.outputs.<key>) }}`

Rolling `-latest` aliases (`ubuntu-latest`, `macos-latest`, `windows-latest`)
are intentionally left hardcoded — they are GitHub aliases, not fleet labels.

## Wiring a job

```yaml
jobs:
  runner_names:
    permissions:
      contents: read
    uses: ./.github/workflows/reusable-runner-names.yml

  my-gpu-job:
    needs: runner_names
    runs-on: ${{ fromJSON(needs.runner_names.outputs.selfhosted_linux_x64_nvidia) }}
    steps: ...
```

## Changing a label

1. Edit `.github/runners.yaml`.
2. Regenerate: `node .github/scripts/sync-runner-names.mjs`
3. Test: `node --test .github/scripts/test/runner-names.test.mjs`

CI enforces both invariants (catalog in sync with the generated workflow, and no
workflow hardcodes a catalog target) via the `lint-runner-names` job in
[`ci.yml`](./workflows/ci.yml), which runs
`node .github/scripts/validate-runner-names.mjs`.
