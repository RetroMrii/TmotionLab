# Git checkpoint review

Scope: the completed local TmotionLab application and research source checkpoint. This is not a deployment plan. All commands below are proposals for the supervisor; no staging, commits, pushes, resets, untracking, deletion or history rewriting were performed.

## Changes made for this checkpoint

| File | Change |
|---|---|
| `README.md` | Project presentation, completed Reading Studio features, model/study selection, correctly defined held-out metrics, exact local commands, model boundary and scientific limitations. No invented public deployment/download or internal engineering-report dependency. |
| `frontend/README.md` | Correct Compare entry, research generation order, full-motion description and test prerequisites. |
| `.gitignore` | Explicit scientific report/provenance allowlists; ignore internal reports, generated research assets, dependencies/caches, data rows, synthetic outputs and weights. Existing tracked aggregate diagnostics remain eligible. |
| `.gitattributes` (new) | Disable line-ending conversion for exactly the 38 hash-pinned originals. Preserve their approved bytes across platforms. |
| `frontend/package.json` | `research:generate`; generation before normal build/check and before dev. No dependency/version changes. |
| `frontend/tests/fixtures/ribbon-geometry.json` (new) | Small 61-path reference count/checksum copied from the existing internal baseline. |
| `frontend/tests/refinement.spec.ts`, `frontend/tests/waves.spec.ts` | Read that fixture instead of depending on an ignored internal engineering report. No behavior/assertion changes. |
| `docs/GIT-CHECKPOINT-REVIEW.md` | This review, staging proposal and clean-checkout limitations. |

The pre-existing completed work remains unstaged and is included in the proposed checkpoint: GoEmotions API integration, Reading Studio, research pages/exporter, motion and emotion colors, the controlled research scripts/configs, and their regression tests. No application design, model logic, weights, thresholds, source hashes or scientific findings changed in this task.

## Intended source inventory

Paths in this document are relative to the repository root. Use the grouped staging commands below rather than `git add .` or a blanket force-add.

| Group | New/modified material intended for commit |
|---|---|
| Root | `.gitattributes`, `.gitignore`, `.env.example`, `README.md`, `requirements.txt`; retain existing `pyproject.toml`. |
| Backend | Existing modifications in `backend/app/config.py`, `main.py`, historical routes/schemas/artifact store and smoke script; new `routes/goemotions.py`, `schemas/goemotions.py`, `services/goemotions.py`, `scripts/verify_goemotions_serving.py`. |
| Frontend | `frontend/package.json`, existing dependency lock changes from completed work, index/config/README/ignore rules, `src/`, `public/favicon.svg`, exporter and research lock, Playwright config/tests/fixture. The already-deleted `src/App.css` is a pre-existing redesign deletion; the proposal records it rather than restoring it. |
| Research implementation | New/modified files under `ml/src/`, `ml/scripts/`, `ml/configs/` and `tests/ml/`; includes four-strategy sampling, synthetic-generation/audit methodology and multiseed analysis. Adding scripts does not execute training, external generation, evaluation or cleanup. |
| Backend tests | Existing changes under `tests/backend/` plus `conftest.py` and `test_goemotions.py`. |
| Documentation | Current changes/new files under `docs/`, including API contract/OpenAPI, current and explicitly historical project-state documents, design/validation notes and this review. Root README does not link to private engineering reports. |
| Scientific evidence | All exact 38 paths in `frontend/research-sources.lock.json`, detailed below; retain the already tracked raw dataset manifests and aggregate diagnostics. |

`frontend/src/data/research.ts` is source and must be committed. `frontend/src/data/research.json` is generated and must not be added. The lockfile and exporter are source, not generated output.

## Required scientific sources: all 38

The authoritative exact path/hash inventory is [`frontend/research-sources.lock.json`](../frontend/research-sources.lock.json). All 38 files exist locally and match their approved SHA-256 values; `.gitignore` now allows every one without `-f`. All twelve provenance exceptions are explicit, not a wildcard allowing arbitrary model metadata.

**Ten scientific reports** under `reports/goemotions/`:

```text
multiseed_robustness_v001.json
multiseed_rare_label_analysis_v001.json
final_heldout_test_v001.json
final_model_freeze_v001.json
goemotions_research_final_v001.md
synthetic_pilot_v002_audit.json
multiseed_sampling_audit_v001.json
controlled_sampling_audit_v001.json
hybrid_sampling_audit_v001.json
annotation_audit_v001.json
```

**Four manifests/configuration sources**:

```text
data/processed/goemotions_v001/manifest.json
data/processed/synthetic_goemotions_v002/manifest.json
ml/configs/roberta_goemotions_multilabel_real_v001.json
ml/configs/synthetic/synthetic_emotion_generation_v002.txt
```

**Twenty-four metric/provenance files**: for each exact experiment ID below, retain both `ml/artifacts/metrics/<ID>_validation.json` and `ml/artifacts/models/<ID>/experiment_provenance.json`:

```text
roberta_goemotions_uniform_v001
roberta_goemotions_uniform_v001_seed123
roberta_goemotions_uniform_v001_seed456
roberta_goemotions_rare4x_v001
roberta_goemotions_rare4x_v001_seed123
roberta_goemotions_rare4x_v001_seed456
roberta_goemotions_synthetic_v002
roberta_goemotions_synthetic_v002_seed123
roberta_goemotions_synthetic_v002_seed456
roberta_goemotions_hybrid_v001
roberta_goemotions_hybrid_v001_seed123
roberta_goemotions_hybrid_v001_seed456
```

Additional exact exceptions retain existing scientific metadata, rather than misclassifying it as internal engineering work:

```text
data/raw/goemotions_raw_v001/manifest.json
data/raw/goemotions_simplified_v001/manifest.json
reports/goemotions/roberta_goemotions_multilabel_real_v001_diagnostics.json
reports/goemotions/roberta_token_lengths_v001.json
reports/goemotions/tfidf_goemotions_multilabel_v001_diagnostics.json
```

These five files were already tracked. The three diagnostics contain aggregate validation/length evidence; none is a new exception for row-level held-out predictions. The historical serving comparison/error CSVs and aggregate DAIR metrics also remain tracked because the active historical API uses them.

### Important: exact bytes and Git line endings

The local Git setting is `core.autocrlf=true`. Three pinned files already in HEAD have LF-normalized blob bytes, while the approved local files use CRLF:

```text
data/processed/goemotions_v001/manifest.json
reports/goemotions/annotation_audit_v001.json
ml/configs/roberta_goemotions_multilabel_real_v001.json
```

Their HEAD blobs do not match the approved SHA-256 hashes; their current working-tree files do. The difference was verified to be only CRLF versus LF. **No working-tree evidence bytes or pins were changed.** `.gitattributes` now specifies `-text` for each of the 38 files. The staging proposal explicitly renormalizes that narrow set under the new no-conversion policy so the checkpoint stores the approved bytes, even if Git's cached status initially hides the difference. The resulting three blob changes are byte-preservation fixes, not new scientific results. Do not run blanket renormalization on the repository or alter the hash lock.

## Excluded material

| Category | Excluded paths |
|---|---|
| Internal working reports/screenshots | Entire `reports/engineering/`, including prior audit/reviews, this task's logs and isolated test copy. No files under this directory are currently tracked. `.agents/` and `.codex/` are also ignored. |
| Reproducible frontend output | `frontend/src/data/research.json`, `frontend/public/research/`, `dist/`, build caches and TypeScript build info. These are generated by normal dev/build. |
| Environments/dependencies/tests | `.venv/`, `venv/`, `node_modules/`, coverage, Python caches, Playwright results/reports and logs. |
| Secrets/local configuration | `.env`, `.env.*` except explicitly named example templates; editor-local files. No secret values belong in the checkpoint. |
| Datasets and predictions | Raw/interim/processed rows, `data/synthetic/`, `ml/artifacts/predictions/`, and `reports/goemotions/final_heldout_predictions_v001.jsonl`. Only the exact dataset manifests above are exceptions. |
| Model/training artifacts | `ml/artifacts/models/` contents except the twelve named provenance files; checkpoints/tokenizers/optimizer states/alternate weights and weight formats elsewhere. The selected ~479 MiB model stays local. |
| Other reports | Reports not explicitly allowed as scientific evidence remain ignored; do not force-add all `reports/`. |

The selected local directory remains untouched:

```text
ml/artifacts/models/roberta_goemotions_rare4x_v001/best_model/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  training_args.bin
```

There is no public download/provisioning script or release asset for this bundle. The README documents separate provisioning, exact hash verification and degraded behavior without it. No weights were uploaded, moved or added to LFS.

## Already tracked artifacts recommended for untracking

`.gitignore` does not remove tracked files. `git ls-files -ci --exclude-standard` currently identifies **23 tracked paths** in four historical directories:

| Directory | Tracked contents | Recommendation |
|---|---|---|
| `DAIR-MODEL-OLD/ml/artifacts/models/` | `.gitkeep`, `baseline_tfidf_dair_v001.joblib`, `numpy_mlp_dair_v001.npz`, `numpy_mlp_dair_v002.npz`, `textcnn_dair_v001.pt` | Untrack from the new checkpoint; keep local copies. |
| `DAIR-MODEL-OLD/ml/artifacts/tokenizers/` | `.gitkeep`, `numpy_mlp_dair_v001_tfidf.joblib`, `numpy_mlp_dair_v002_tfidf.joblib`, `vocab_dair_v001.json`, `vocab_dair_v001.metadata.json` | Untrack; only historical live models need these. |
| `DAIR-MODEL-OLD/ml/artifacts/predictions/` | `.gitkeep`; test and validation prediction CSVs for `baseline_tfidf_dair_v001`, `numpy_mlp_dair_v001`, `numpy_mlp_dair_v002`, `textcnn_dair_v001` | Untrack row-level exports; keep local research copies. |
| `DAIR-MODEL-OLD/ml/artifacts/logs/` | `.gitkeep`, `numpy_mlp_dair_v001_epochs.csv`, `numpy_mlp_dair_v002_epochs.csv`, `textcnn_dair_v001_epochs.csv` | Untrack generated training logs. |

This is a separate, explicit supervisor choice. Until approved untracking is executed, these 23 paths remain part of a checkout of the existing index/HEAD despite being ignored. They are not newly staged by the source commands below. Untracking does not remove them from historical commits; no history rewrite is proposed.

**Keep** `DAIR-MODEL-OLD/ml/artifacts/comparisons/dair_v001_model_comparison.csv`, `errors/textcnn_dair_v001_errors.csv`, `errors/textcnn_dair_v001_confusion_pairs.csv`, and the three model `*_metrics.json` files. Unlike the row-level prediction export directories, these six files are required to preserve the user-facing historical APIs. They include historical example text/saved predictions intentionally; the checkpoint does not claim to contain only aggregate DAIR data.

## Proposed staging commands — not executed

Run from the repository root after reviewing the changes. These are PowerShell commands. Do not use `git add -f`, `git add .` or a blanket `git rm --cached .`.

```powershell
git status --short --untracked-files=all
git diff --check

$scientificLock = Get-Content frontend/research-sources.lock.json -Raw | ConvertFrom-Json
$scientificSources = @($scientificLock.PSObject.Properties.Name)
if ($scientificSources.Count -ne 38) { throw 'Expected exactly 38 research sources' }
foreach ($path in $scientificSources) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
    git check-ignore --no-index -q -- $path
    if ($LASTEXITCODE -eq 0) { throw "Required source is ignored: $path" }
    if ($LASTEXITCODE -ne 1) { throw "Ignore check failed: $path" }
}

# Source/documentation and the pre-existing App.css deletion.
git add -A -- .gitattributes .gitignore .env.example README.md requirements.txt pyproject.toml backend frontend docs ml/src ml/scripts ml/configs tests

# Exact evidence, no model-directory bulk add.
git add -- $scientificSources
git add --renormalize -- $scientificSources

# Review content, statistics, whitespace and excluded paths before any commit.
git diff --cached --stat
git diff --cached --name-status
git diff --cached --check
git diff --cached

$unexpected = @(git diff --cached --name-only | Where-Object {
    $_ -match '^reports/engineering/' -or
    $_ -match '^frontend/(src/data/research\.json|public/research/|dist/|node_modules/)' -or
    $_ -match '^ml/artifacts/models/.+/best_model/' -or
    $_ -match '^reports/goemotions/final_heldout_predictions_v001\.jsonl$'
})
if ($unexpected.Count) { throw "Unexpected staged files: $($unexpected -join ', ')" }
```

The check above intentionally fails rather than resetting anything. Inspect the entire staged diff; the path checks supplement, not replace, review. After staging, verify **indexed evidence bytes**, not only the working tree:

```powershell
@'
import hashlib, json, subprocess
from pathlib import Path
pins = json.loads(Path('frontend/research-sources.lock.json').read_text(encoding='utf-8'))
for path, expected in pins.items():
    blob = subprocess.check_output(['git', 'show', ':' + path])
    assert hashlib.sha256(blob).hexdigest() == expected, path
assert not subprocess.check_output(['git', 'ls-files', 'reports/engineering']).strip()
print('All 38 indexed source hashes match; no internal engineering reports tracked.')
'@ | python -
```

**Only if the supervisor approves the four historical-directory removals from the index**, additionally execute:

```powershell
# --cached keeps every local file; these commands were NOT executed by the agent.
git rm -r --cached -- DAIR-MODEL-OLD/ml/artifacts/models DAIR-MODEL-OLD/ml/artifacts/tokenizers DAIR-MODEL-OLD/ml/artifacts/predictions DAIR-MODEL-OLD/ml/artifacts/logs
git diff --cached --name-status
git diff --cached --check
```

Suggested concise commit message:

```text
Complete GoEmotions Reading Studio and preserve reproducible research sources
```

No commit command is executed or required by this review. Supervisor review precedes any commit/push.

## Remaining clean-checkout limitations

- This work prepares files for a future checkpoint; current HEAD still lacks the completed source/evidence. A clone of current HEAD is not yet a clone of these unstaged changes.
- Static research dev/build no longer depends on private reports, pre-existing generated output, weights or raw data. A clean-file copy with `npm ci` and normal build was tested; no temporary Git repository, alternate index or commit was created.
- Python requirements are unpinned, and the existing local research environment has a `datasets`/`huggingface-hub` constraint conflict. No package versions or environment locks were changed. The README does not promise a certified clean Python install.
- Live inference needs the original frozen five-file model bundle, provided separately. No public distribution was invented.
- Full backend regression also needs the local validation JSONL and selected-model validation predictions. Full browser regression needs the live API/model and installed Chrome; it assumes localhost ports 5173/8000 and free QA ports 5174/65530. Model-dependent tests are not claimed to pass in a bare clone.
- Historical Compare requires the six retained serving artifacts. Archived source/tests may depend on older import layouts; keeping the archive is not a claim that every historical training entry point works unchanged.
- Existing documentation may link to private historical engineering evidence; those files remain private. The root README presents the project without relying on those reports.

## Verification results

Fresh checks completed for this task:

| Check | Result |
|---|---|
| Research source completeness and SHA-256 | All 38 present and unchanged. |
| `git check-ignore --no-index` | All 38 scientific paths eligible; internal reports, generated exports, model weights, row-level data, environments and build outputs excluded. |
| `git check-attr text` | All 38 pinned originals have `text: unset`, preserving their exact bytes. |
| Standard frontend lint/build | Passed. Build generated and verified 12 validation runs, one saved held-out summary and 38 sources. |
| Isolated clean-file install/build | `npm ci` and the normal build passed using the existing lock. The copy started without generated research output, weights, dataset rows, private reports or node_modules. |
| Frontend regression | **64 passed in 3.6 minutes**, including the two tests using the relocated fixture. |
| Automated accessibility | **12 scans, zero reported violations**; not a full conformance claim. |
| Backend regression | **52 passed in 17.15 seconds**, using existing local artifacts. No held-out evaluation run. |
| Frozen model integrity | Exact file hashes, freeze metadata and 28-label mapping passed; model files untouched. |
| `git diff --check` | Passed. Git also prints informational LF/CRLF conversion warnings for non-pinned source files. |
| Index preservation | `git ls-files -s` unchanged from task start; no tracked internal engineering reports. |
| Scope preservation | Only the six expected existing files changed relative to the prior audit baseline; three new intended files are `.gitattributes`, the geometry fixture and this review. Backend, research sources, package lock, model and frontend design unchanged. |
| README/review local links | All relative file links resolve. |

The final status has 140 changed/untracked entries, predominantly the pre-existing completed work and newly exposed scientific evidence. Some line-ending-only staged changes will become explicit when the supervisor runs the narrow renormalization command. The source copy was verified without staging or creating a commit; a literal clone of a future commit cannot be certified before that commit exists.

Diagnostic output stays under ignored `reports/engineering/git-checkpoint/`, including `checkpoint-verification.json`, `line-endings.json`, `git-attributes.txt`, `build.log`, `clean-install.log`, `clean-build.log`, `frontend-tests.log`, `backend-tests.log`, `final-checks.json`, and before/after index/status inventories. It is not part of the intended checkpoint. Read-only Git inspection was performed; the status below means no Git mutations were performed.

**STATUS: READY FOR SUPERVISOR GIT REVIEW — NO GIT OPERATIONS PERFORMED**
