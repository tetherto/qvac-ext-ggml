import test from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { join } from 'node:path'
import {
  REUSABLE_WORKFLOW,
  RUNNERS_YAML,
  assertReusableMatchesCatalog,
  findHardcodedLabelViolations,
  findMissingRunnerNamesNeeds,
  listAddonWorkflows,
  loadRunners,
  outputValue,
  parseRunnersYaml,
  readRepoFile,
  renderReusableWorkflow,
  repoRoot,
  runsOnExpression,
} from '../lib/runner-names.mjs'

test('runners.yaml parses scalar + array entries with unique keys/targets', () => {
  const runners = loadRunners()
  assert.ok(runners.length >= 5)
  assert.ok(runners.some((e) => e.kind === 'scalar' && e.label === 'ubuntu-22.04'))
  const nvidia = runners.find((e) => e.key === 'selfhosted_linux_x64_nvidia')
  assert.equal(nvidia.kind, 'array')
  assert.deepEqual(nvidia.labels, ['self-hosted', 'Linux', 'X64', 'NVIDIA'])
  assert.equal(new Set(runners.map((e) => e.key)).size, runners.length)
})

test('array vs scalar output value + runs-on expression', () => {
  const [scalar] = parseRunnersYaml('ubuntu_2204: ubuntu-22.04\n')
  assert.equal(outputValue(scalar), 'ubuntu-22.04')
  assert.equal(runsOnExpression(scalar), '${{ needs.runner_names.outputs.ubuntu_2204 }}')

  const [arr] = parseRunnersYaml('gpu: [self-hosted, Linux, X64, NVIDIA]\n')
  assert.equal(outputValue(arr), '["self-hosted","Linux","X64","NVIDIA"]')
  assert.equal(runsOnExpression(arr), '${{ fromJSON(needs.runner_names.outputs.gpu) }}')
})

test('parseRunnersYaml rejects quoted values and junk', () => {
  assert.throws(() => parseRunnersYaml('k: "ubuntu-22.04"\n'), /invalid/)
  assert.throws(() => parseRunnersYaml('k: [self-hosted, "Linux"]\n'), /bare/)
  assert.throws(() => parseRunnersYaml('k: macos\nk: other\n'), /duplicate runner key/)
  assert.throws(() => parseRunnersYaml('not yaml at all\n'), /invalid/)
})

test('reusable-runner-names.yml matches the catalog', () => {
  const runners = loadRunners()
  assert.doesNotThrow(() => assertReusableMatchesCatalog(runners, readRepoFile(REUSABLE_WORKFLOW)))
  const rendered = renderReusableWorkflow(runners)
  assert.match(rendered, /AUTO-GENERATED/)
  assert.match(rendered, /runs-on: ubuntu-latest/)
  assert.doesNotMatch(rendered, /actions\/checkout/)
  // Array entry exported as a single-quoted JSON string.
  assert.match(rendered, /selfhosted_linux_x64_nvidia=\["self-hosted","Linux","X64","NVIDIA"\]/)
})

test('CI workflows do not hardcode catalog runner targets', () => {
  const runners = loadRunners()
  const findings = []
  for (const file of listAddonWorkflows()) {
    const source = readRepoFile(file)
    findings.push(...findHardcodedLabelViolations(file, source, runners))
    findings.push(...findMissingRunnerNamesNeeds(file, source))
  }
  assert.deepEqual(findings, [], JSON.stringify(findings, null, 2))
})

test('detector catches scalar and composite-array hardcoded runs-on', () => {
  const runners = parseRunnersYaml(
    'ubuntu_2204: ubuntu-22.04\ngpu: [self-hosted, Linux, X64, NVIDIA]\nmac: [self-hosted, macOS, ARM64]\n',
  )
  const source = [
    'jobs:',
    '  a:',
    '    runs-on: ubuntu-22.04',
    '  b:',
    '    runs-on: [self-hosted, Linux, X64, NVIDIA]',
    '  c:',
    '    runs-on: ${{ fromJSON(needs.runner_names.outputs.mac) }}',
    '  d:',
    '    runs-on: ubuntu-latest',
    '  e:',
    '    os: ubuntu-22.04',
    '',
  ].join('\n')
  const findings = findHardcodedLabelViolations('x.yml', source, runners)
  assert.deepEqual(
    findings.map((f) => [f.line, f.target]).sort((a, b) => a[0] - b[0]),
    [
      [3, 'ubuntu-22.04'],
      [5, '[self-hosted,Linux,X64,NVIDIA]'],
    ],
  )
})

test('detector matches composite arrays regardless of token order', () => {
  const runners = parseRunnersYaml('gpu: [self-hosted, Linux, X64, NVIDIA]\n')
  const source = ['jobs:', '  a:', '    runs-on: [self-hosted, NVIDIA, X64, Linux]', ''].join('\n')
  assert.equal(findHardcodedLabelViolations('x.yml', source, runners).length, 1)
})

test('validate-runner-names.mjs exits 0', () => {
  const result = spawnSync(process.execPath, [join(repoRoot, '.github/scripts/validate-runner-names.mjs')], {
    encoding: 'utf8',
    cwd: repoRoot,
  })
  assert.equal(result.status, 0, `stdout=${result.stdout}\nstderr=${result.stderr}`)
})

test('catalog path constant points at a tracked file', () => {
  assert.equal(RUNNERS_YAML, '.github/runners.yaml')
  assert.match(readRepoFile(RUNNERS_YAML), /selfhosted_linux_x64_nvidia/)
})
