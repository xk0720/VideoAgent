# Branch Recovery — sort R15–R16 work onto `main` and protect `rl-integration`

> User reported "我推错分支了，没有到 main". This doc lists which files belong
> on which branch, plus the exact `git` commands to fix it. Sandbox couldn't
> run during R16 so I couldn't do the moves myself — you do them in your
> own terminal.

---

## 1. Where each file *should* live

### Files that belong on **`main`** (the v0.2 + baseline + CSA framework)

| File | Round | Status |
|---|---|---|
| `scripts/measure_baseline.py`              | R15.B (B.1)   | NEW      |
| `docs/BASELINE_v0_2.md`                    | R15.B (B.1)   | NEW      |
| `src/longvideoagent/agents/editor.py`      | R15.B (B.1.5) | MODIFIED — adds `previous_source` to neighbor_context |
| `src/longvideoagent/tools/generation_tool.py` | R15.B (B.1.5) | MODIFIED — `_estimate_metrics` uses anchor_quality |
| `tests/unit/test_generation_metrics.py`    | R15.B (B.1.5) | MODIFIED — adds 2 tests pinning new behaviour |
| `docs/decisions.md`                        | R15.B (B.1.5) | MODIFIED — appends D-021 |
| `docs/CRITICAL_REVIEW.md`                  | R16.A         | NEW — honest audit of rounds 1–15 |
| `docs/CSA_FRAMEWORK.md`                    | R16.B         | NEW — differentiated framework spec |
| `src/longvideoagent/types.py`              | R16.C         | MODIFIED — adds `CutEvent`, `ArcContext` |
| `src/longvideoagent/tools/metric_tool.py`  | R16.D         | MODIFIED — adds `arc_coherence(...)` |
| `tests/unit/test_arc_coherence.py`         | R16.D         | NEW — 5 tests incl. C4 falsification |
| `README.md`                                | R16.E         | MODIFIED — adds 3 doc pointers |

### Files that belong on **`rl-integration` only** (frozen until baseline numbers exist)

| File | Round |
|---|---|
| `training/` entire subtree                 | R13 |
| `tests/training/` entire subtree            | R13 |
| `docs/AGENTIC_RL_PROPOSAL.md`              | R12 / R13 J |
| `docs/RL_BRANCH_SUMMARY.md`                | R13 |
| `docs/ON_POLICY_DISTILLATION_ANALYSIS.md`  | R14 E |
| `conftest.py` (project root)                | R13 K — needed by tests/training/ import path fix |
| `tests/__init__.py`                         | R13 K — same |
| `pyproject.toml`                            | R13 — added `lva-train-*` script entries + pythonpath = ["src", "."] |

---

## 2. The most likely current state

Without git access I can't verify, but based on the conversation:

* `main` last clean commit is somewhere around Round 12 (before the
  `rl-integration` branch was created in R13.A).
* `rl-integration` last clean commit is somewhere in Round 14 (the OPD
  work), but **never actually committed** because `.git/index.lock` was
  unremovable in the sandbox during R13.K. So `rl-integration` exists as
  a branch ref but its tree is whatever your terminal sees right now.
* All R15–R16 file edits in this chat were made via the file tool,
  which writes to whichever branch your working tree is checked out to.
  If you pushed without checking, they landed on the branch HEAD was on.

So the practical fix has two cases:

### Case A — Your current branch HAS the R15-R16 edits and is **not** `main`

Most likely. Do:

```bash
# 1. Save the current state of all the R15-R16 work as a patch.
git diff <branch-you-are-on>~N..HEAD -- \
    scripts/measure_baseline.py \
    docs/BASELINE_v0_2.md \
    src/longvideoagent/agents/editor.py \
    src/longvideoagent/tools/generation_tool.py \
    tests/unit/test_generation_metrics.py \
    docs/decisions.md \
    docs/CRITICAL_REVIEW.md \
    docs/CSA_FRAMEWORK.md \
    src/longvideoagent/types.py \
    src/longvideoagent/tools/metric_tool.py \
    tests/unit/test_arc_coherence.py \
    README.md \
    > /tmp/r15_r16_main.patch

# 2. Switch to main.
git switch main

# 3. Apply the patch onto main.
git apply /tmp/r15_r16_main.patch

# 4. Stage + commit.
git add -A
git commit -m "R15-R16: hybrid baseline + CSA framework + critical review

R15 (B phase):
  - scripts/measure_baseline.py + docs/BASELINE_v0_2.md
  - mock GenerationTool now honest about previous_source (D-021)
  - +2 tests pinning R→G > G→G > none→G on m2/m3

R16:
  - docs/CRITICAL_REVIEW.md: honest catalogue of rounds 1-15
  - docs/CSA_FRAMEWORK.md: Cut-Score-Arc differentiated framework
  - types.py: CutEvent + ArcContext primitives
  - tools/metric_tool.py: arc_coherence() whole-script judge
  - +5 tests including C4 falsification handle

main now carries: baseline experiment infrastructure + Arc-level judge.
rl-integration stays frozen (training/ subtree + OPD)."

# 5. Verify pytest stays green.
python -m pytest tests/ -q
# expected: ~109 passed (102 pre-R15 + 2 R15.B.5 + 5 R16)

# 6. Optional — if the original branch had R15-R16 stuff committed but
#    you don't want it there, reset that branch.
#    BE CAREFUL: this throws away the commits on that branch.
git switch <branch-name>
git reset --hard <pre-R15-commit-on-that-branch>
```

### Case B — Your current branch IS `main` and the R15-R16 edits are uncommitted

Easy:

```bash
git status                  # should list all 12 files from §1 (main column)
git add -A
git commit -m "R15-R16: hybrid baseline + CSA framework + critical review"
python -m pytest tests/ -q  # expected ~109 passed
```

### Case C — The R15-R16 edits went onto `rl-integration` and got *committed* there

Need to remove them from rl-integration without losing them:

```bash
# 1. From rl-integration, copy the changes to main via cherry-pick.
git switch main
git cherry-pick <commit-on-rl-integration-containing-R15-R16>
# Resolve any conflicts (there shouldn't be any since rl-integration
# never modified the R15-R16 files in §1's main column).
python -m pytest tests/ -q

# 2. From rl-integration, revert the misplaced commit so the branch
#    keeps only its actual RL/training scope.
git switch rl-integration
git revert <commit-on-rl-integration-containing-R15-R16>
python -m pytest tests/ -q
```

---

## 3. Sanity checks after recovery

On **`main`** after Case A or B:

```bash
git switch main
git log --oneline -5                                    # last commit mentions R15-R16
python -m pytest tests/ -q                              # ~109 passed
test -e scripts/measure_baseline.py                     # exists
test -e docs/CRITICAL_REVIEW.md                         # exists
test -e docs/CSA_FRAMEWORK.md                           # exists
test -e tests/unit/test_arc_coherence.py                # exists
grep -q "previous_source" src/longvideoagent/agents/editor.py    # in
grep -q "anchor_quality" src/longvideoagent/tools/generation_tool.py  # in
grep -q "CutEvent" src/longvideoagent/types.py          # in
grep -q "arc_coherence" src/longvideoagent/tools/metric_tool.py # in
```

On **`rl-integration`** after Case A or C:

```bash
git switch rl-integration
ls -d training/ tests/training/                          # both exist
ls docs/AGENTIC_RL_PROPOSAL.md docs/RL_BRANCH_SUMMARY.md docs/ON_POLICY_DISTILLATION_ANALYSIS.md
test -e training/stages/distill.py                       # OPD stage
python -m pytest tests/ -q                               # ~131 passed (102 + 22 training + 7 OPD)
# rl-integration should NOT have the R15-R16 files from §1 main column.
test ! -e docs/CRITICAL_REVIEW.md
test ! -e docs/CSA_FRAMEWORK.md
test ! -e scripts/measure_baseline.py
```

---

## 4. Once both branches are clean, set this rule

To prevent the same mix-up next time:

```bash
# Anything in this list belongs on main:
#   src/longvideoagent/   docs/   scripts/   tests/unit/   tests/integration/
#   configs/   benchmark/   pyproject.toml   README.md
#
# Anything in this list belongs on rl-integration:
#   training/   tests/training/   docs/AGENTIC_RL_PROPOSAL.md
#   docs/RL_BRANCH_SUMMARY.md   docs/ON_POLICY_DISTILLATION_ANALYSIS.md
#   conftest.py (root)   tests/__init__.py
```

Add a pre-commit hook (optional, in `.git/hooks/pre-commit`):

```bash
#!/usr/bin/env bash
set -e
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "main" ]; then
    if git diff --cached --name-only | grep -qE '^(training/|tests/training/|conftest\.py$|tests/__init__\.py$|docs/AGENTIC_RL_PROPOSAL\.md|docs/RL_BRANCH_SUMMARY\.md|docs/ON_POLICY_DISTILLATION_ANALYSIS\.md)'; then
        echo "ERROR: staged file belongs on rl-integration, not main"
        exit 1
    fi
elif [ "$branch" = "rl-integration" ]; then
    if git diff --cached --name-only | grep -qE '^(scripts/measure_baseline\.py|docs/CRITICAL_REVIEW\.md|docs/CSA_FRAMEWORK\.md|docs/BASELINE_v0_2\.md)$'; then
        echo "ERROR: staged file belongs on main, not rl-integration"
        exit 1
    fi
fi
```

Make it executable: `chmod +x .git/hooks/pre-commit`.

---

## 5. If you want me to do the moves automatically next time the sandbox is up

Tell me on the next turn. I'll:

1. `git checkout main`
2. Apply the R15-R16 patch (already on disk in your working tree)
3. `git add -A && git commit -m "..."`
4. `git checkout rl-integration` (if it needs reset)
5. `git reset --hard <previous-commit>` (only if Case C)
6. `python -m pytest tests/ -q` on both branches and report results

I'll need you to confirm the case (A / B / C above) so I don't trash a commit.
