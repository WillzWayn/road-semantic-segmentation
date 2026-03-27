# Repository Alignment Plan and Execution Tasks

This task plan is designed for AI execution.
Each phase includes:
- Goals
- Sequential tasks (`[S]`)
- Parallelizable tasks (`[P]`)
- Exit criteria

---

## Phase 1 -> Critical (Public-Ready Foundation)

### Goal
Make the repository safe, reproducible, and ready to be public.

### Tasks
- [S] Audit repository for generated artifacts, caches, logs, checkpoints, and private/sensitive files.
- [S] Finalize `.gitignore` rules for ML project outputs:
  - `outputs/`
  - `logs/`
  - `modal_downloads/`
  - `**/__pycache__/`
  - `*.pyc`
  - large local datasets if not intended for git
- [S] Remove tracked generated files that should not be versioned.
- [S] Add `LICENSE` (recommended: MIT).
- [S] Update `pyproject.toml` project metadata:
  - accurate `description`
  - author/maintainer fields
  - repository/homepage links
- [S] Add a short security/compliance note in `README.md`:
  - no secrets in repo
  - dataset licensing responsibility
- [S] Validate quickstart path from clean environment (`uv sync` + one runnable command).

### Parallel opportunities
- [P] In parallel while cleanup is happening:
  - Draft `LICENSE`
  - Draft metadata updates for `pyproject.toml`
  - Draft README compliance note

### Exit criteria
- Working tree no longer includes unwanted generated artifacts.
- Core setup and one runnable command work from fresh install.
- Repo has legal/license baseline and publish-safe structure.

---

## Phase 2 -> Portfolio First Showcase

### Goal
Maximize recruiter/visitor impact quickly (clarity + visual proof + engineering quality).

### Tasks
- [S] Redesign `README.md` to portfolio-first structure:
  - project pitch and problem statement at top
  - key results near top
  - concise architecture overview
  - quick demo path in minutes
- [S] Create a compact results table (U-Net, DeepLabV3, baselines; IoU; notes).
- [S] Add visual assets under `static/` and reference them in README:
  - sample inputs/labels/predictions
  - training curves
- [S] Add usage sections with direct commands:
  - run baseline
  - run training
  - run prediction
- [S] Create minimal CI under `.github/workflows/ci.yml`:
  - dependency install
  - lint/import checks
  - lightweight tests/smoke checks
- [S] Add initial tests under `tests/`:
  - model forward-pass shape test
  - dataset loading/split smoke test
  - metric utility test (IoU)

### Parallel opportunities
- [P] README rewrite and results table creation can run in parallel with CI setup.
- [P] Test scaffolding can run in parallel with visual asset curation.
- [P] Command examples can be validated in parallel with docs polishing.

### Exit criteria
- New visitor understands value and can run a demo quickly.
- Repo displays both ML results and software-engineering credibility.

---

## Phase 3 -> University Showcase (Docs Folder)

### Goal
Provide academically structured documentation and reproducible evidence.

### Tasks
- [S] Create `docs/` folder and core files:
  - `docs/report.md`
  - `docs/experiments.md`
  - `docs/limitations.md`
  - `docs/references.md`
- [S] In `docs/report.md`, document:
  - problem definition
  - method and architecture
  - training setup
  - evaluation protocol
  - conclusions
- [S] In `docs/experiments.md`, document:
  - hyperparameters
  - seed/split strategy
  - checkpoint provenance
  - metric snapshots and comparisons
- [S] In `docs/limitations.md`, include:
  - known failure modes
  - data limitations
  - bias/risk notes
  - future improvements
- [S] In `docs/references.md`, include:
  - dataset citation/license
  - model/paper references
  - tools/libraries used
- [S] Link docs from `README.md` for easy navigation.

### Parallel opportunities
- [P] Each docs file can be drafted independently by different AI agents.
- [P] References collection can run in parallel with experiment write-up.

### Exit criteria
- Repository has a clear academic artifact suitable for university presentation.
- Technical claims are backed by documented experiment details.

---

## Phase 4 -> Publish, Promote, and Maintain

### Goal
Convert the project from assignment output into a durable portfolio asset.

### Tasks
- [S] Prepare release baseline:
  - define `v1.0.0` scope
  - write release notes (what changed, key metrics, visuals)
- [S] Configure GitHub repository profile:
  - About section
  - topics/tags
  - social preview image
- [S] Add maintenance docs:
  - `ROADMAP.md`
  - issue templates
  - contribution expectations (`CONTRIBUTING.md` optional but recommended)
- [S] Publish and promote:
  - make repo public
  - pin on GitHub profile
  - post concise technical summary (e.g., LinkedIn/GitHub)
- [S] Define follow-up technical backlog:
  - augmentation ablation
  - threshold calibration automation
  - experiment tracking integration (MLflow/W&B)
  - optional demo app (Streamlit/Gradio)

### Parallel opportunities
- [P] Roadmap drafting can run in parallel with release notes.
- [P] Issue template creation can run in parallel with GitHub profile updates.
- [P] Follow-up backlog definition can run in parallel with public launch prep.

### Exit criteria
- Public release is complete, discoverable, and professionally presented.
- Project has a clear next-iteration plan and maintenance path.

---

## Suggested AI Execution Strategy

- Use one AI agent per phase when possible.
- Inside each phase, run `[P]` items concurrently and gate merge on `[S]` completion.
- After each phase:
  - run validation checks
  - commit with phase-specific message
  - open/merge PR before starting next phase
- Recommended branch strategy:
  - `phase-1-critical`
  - `phase-2-portfolio`
  - `phase-3-university-docs`
  - `phase-4-publish-maintain`

## Definition of Done (Overall)

- The repository is safe and clean for public visibility.
- Portfolio presentation is strong and demo-friendly.
- University documentation is complete and evidence-based.
- Release and maintenance workflow is established.
