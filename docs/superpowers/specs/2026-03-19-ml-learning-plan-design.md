# ML Engineer Learning Plan — Design Spec

## Context

Student with strong DL/architecture background (MLP, CNN, RNN, GRU, Transformer, GPT, GAN, NeuMF — all implemented from scratch in PyTorch). Weak spots: practical data work, Python fundamentals (OOP, data structures), SQL (basic JOINs only), no portfolio projects, no Kaggle experience, no interview prep.

**Goal:** Prepare for ML Engineer (product + research hybrid) positions within 1 year.
**Time budget:** 4 hours/week (~200 hours total).
**Learning style:** Mix of exercises (Python) and projects (ML).

## Principles

Adopted from `.claude/Test.md` methodology (Andrey Zhogov's approach to escaping "eternal student" mode):
- **Just-in-Time learning** — theory only when needed for a project
- **80/20** — focus on skills that appear in job postings, skip exotic topics
- **Every concept backed by code** — no "dead knowledge"

Not adopted:
- Junior DS 6-month timeline — we have 1 year and target MLE, not DS
- CatBoost/sklearn as center — PyTorch is the student's strength, leverage it
- Basic Python from zero — student has working knowledge, needs sharpening not restart

## Plan Structure

### Phase 1: Ramp-up (Months 1-3)

**Python track (~1.5h/week):**
- Weeks 1-4: Data structures — list/dict/set/tuple, complexity, `collections` (defaultdict, Counter, deque)
- Weeks 5-8: OOP — classes, inheritance, dunder methods (`__init__`, `__repr__`, `__len__`, `__getitem__`), connection to PyTorch (Dataset, Module)
- Weeks 9-12: Generators, decorators, context managers, exception handling
- Format: Codewars 8-6 kyu / LeetCode Easy, 2-3 problems per session

**ML track (~2.5h/week):**
- First Kaggle project: tabular data classification (full cycle)
- **Constraint: pick from Kaggle "Getting Started" or "Playground" tier. Avoid datasets >1GB or complex multi-file formats.**
- Weeks 1-3: Pick competition, load data, EDA in Pandas (distributions, missing values, correlations)
- Weeks 4-6: Feature engineering, preprocessing, baseline (LogReg / RandomForest via sklearn)
- Weeks 7-9: CatBoost/XGBoost, hyperparameter tuning, cross-validation
- Weeks 10-12: Final submission, error analysis, formatted notebook with conclusions

**Phase 1 deliverables:**
- Kaggle submission with meaningful score
- Confident solving Python Easy-level problems
- One formatted notebook in `works/kaggle-tabular/`

### Phase 2: Momentum (Months 4-6)

**Python track (~1h/week, including SQL from week 17):**
- Weeks 13-16: LeetCode Easy→Medium — hashmaps, two pointers, basic recursion (interview-relevant)
- Weeks 17-20: Practical OOP — build custom Dataset/DataLoader-like classes
- Weeks 21-24: Standard library — `itertools`, `functools`, `pathlib`, `typing`

**SQL track (~0.5h/week, from Python budget):**
- Weeks 17-24: SQL practice on SQLBolt / HackerRank SQL / Mode Analytics
- Focus: subqueries, CTEs, window functions (ROW_NUMBER, RANK, LAG/LEAD)
- This closes the SQL gap early instead of cramming in Phase 4

**ML track (~3h/week):**
- Second project: DL on real-world (messy) data via PyTorch
- Options: NLP (non-toy dataset), CV (medical/satellite/defect with business context), or tabular+DL (TabNet/embeddings)
- Key requirement: data must be "uncomfortable" — missing values, class imbalance, noisy labels
- Weeks 13-16: Pick task, load data, EDA, strategy for imbalance
- Weeks 17-20: Build pipeline — Dataset, DataLoader, augmentations, baseline PyTorch model
- Weeks 21-24: Experiments — architectures, lr schedulers, early stopping, logging in MLflow

**New skill: experiment discipline**
- Track every experiment (what changed, what result)
- Use MLflow (local) for experiment tracking
- Be able to answer "why is this model better" with substance

**Risk: PC upgrade and environment migration**
- Student upgrading from DDR3 — new hardware expected during this phase
- Budget 1-2 sessions for conda/pip setup from scratch
- Document setup in checklist for future reproducibility
- Verify CUDA/GPU if available
- Save `environment.yml`

**Phase 2 deliverables:**
- DL project on messy real data with MLflow tracking
- Python Medium-level problems solved confidently
- Environment setup documented

### Phase 3: Portfolio (Months 7-9)

**Python track (~0.5h/week):**
- Maintenance mode: 1-2 Medium problems/week
- Skip if confident, redirect time to ML

**ML track (~3.5h/week):**

Sequential, not parallel:

**Weeks 25-31: Own project (not Kaggle)**
- Self-formulated problem (not given by competition)
- Self-sourced data (API, scraping, open sources)
- Self-chosen metric with justification
- Potential topics:
  - Recommendation system on own data (existing `recSys/` with MovieLens can be extended)
  - NLP task on Russian language (stands out in portfolio)
  - Audio classification (would need to start mostly from scratch despite `audio_processing/` existing — it currently only contains GAN experiments)
- Must be full cycle: data → model → result → conclusions → presentation

**Weeks 32-33: Deployment mini-project**
- Wrap one of the existing models (from Phase 1, 2, or 3) in a FastAPI endpoint
- Write a Dockerfile, run locally
- This directly produces a portfolio artifact and covers the deployment skill gap

**Weeks 34-36: GitHub portfolio formatting**
- Each project (phases 1-3) gets:
  - README: problem description, approach, results
  - Clean reproducible code (not raw 200-cell notebook)
  - `requirements.txt` or `environment.yml`
  - Result visualizations (charts, metric tables)
- **Git hygiene checklist:**
  - `.gitignore` for model weights (.pth), data files, cache, notebook outputs
  - Meaningful commit messages
  - No large binaries committed (note: `recSys/best_model.pth` is already committed — clean up)
- **Existing code (`archs/`, `recSys/`):** decide per-project — either refactor into portfolio with README, or leave as learning artifacts. Not everything needs to be portfolio-ready.

**Phase 3 deliverables:**
- 2-3 formatted projects on GitHub with READMEs
- One model deployed locally via FastAPI + Docker
- Clean git history and proper `.gitignore`
- Experience with self-sourced data and self-formulated problems

### Phase 4: Market Entry (Months 10-12)

**Python track:** On-demand, fill gaps found during interview prep.

**Main focus: interview prep and applications**

**Weeks 37-39: Buffer / catch-up from Phase 3.** Use for finishing portfolio formatting, filling gaps, or rest.

**1. Market analysis (weeks 40-41):**
- Collect 15-20 MLE/Junior ML Engineer/DS job postings (HH, LinkedIn, Habr Career)
- Extract recurring requirements and tech stacks
- Gap analysis against own skills
- Gaps = study plan for remaining weeks

**2. Typical interview questions (weeks 42-48):**
- ML theory: bias-variance, regularization, metrics (precision/recall/F1 — when to use which), overfitting, cross-validation
- DL: backpropagation (explain clearly), batch norm why, dropout why, learning rate selection
- Python: OOP tasks, generators, algorithm complexity
- SQL: review and deepen window functions, CTEs (foundation built in Phase 2)
- System design (basic): "how would you deploy a model" — student has hands-on experience from Phase 3 deployment mini-project

**Time split for overlapping weeks:**
- Weeks 42-45: 100% interview question prep
- Weeks 46-48: 50% mock interviews, 50% question prep
- Weeks 48-50: 50% mock interviews, 50% applications
- Weeks 50-52: 100% applications + gap filling from interview feedback

**3. Mock interviews (weeks 46-50):**
- Practice with AI mentor as interviewer
- Practice explaining projects: "tell me what you did and why"
- Timing: fit project description into 5 minutes

**4. Applications (weeks 48-52):**
- Resume with portfolio projects
- Start applying before "fully ready" — interviews teach by themselves
- Interview feedback → adjust weak areas

**Phase 4 deliverables:**
- Resume with 2-3 projects
- Confidence in typical interview questions
- First real applications and interviews

## Optional Track: Mathematics (Yandex Practicum)

**Scope:** Linear algebra, probability/statistics, calculus — full course.
**Timing:** Start in phase 2-3 (months 4-9) when Python track frees up time.
**Time budget:** Additional hours on top of the 4h/week main plan. This is NOT a replacement for project time.
**Priority:** Supporting track, never blocks main plan. If choice between math lecture and project work — project wins.
**Trigger to start:** When DL project work raises questions like "why does this loss function behave this way."

## Repository Structure

```
works/
├── progress/
│   ├── phase1.md        — phase 1 weekly log
│   ├── phase2.md        — phase 2 weekly log
│   ├── phase3.md        — phase 3 weekly log
│   └── phase4.md        — phase 4 weekly log
├── kaggle-tabular/      — first project (phase 1)
├── dl-project/          — second project (phase 2)
└── own-project/         — own project (phase 3)
```

**Log format (per phase file):**
```markdown
# Phase N — Name

## Week X (YYYY-MM-DD)
### Python
- Solved X problems on Codewars/LeetCode, topics: ...
- What was hard: ...

### ML
- Chose competition / Did: ...

### Notes
- (what to explore next session)
```

AI mentor fills in the log after each session based on student's report.

## Success Criteria

After 1 year, the student should:
1. Have 2-3 polished projects on GitHub with READMEs
2. Solve Python Medium-level problems confidently
3. Explain ML concepts and own project decisions clearly
4. Have basic SQL (including window functions)
5. Have started applying and attending interviews
6. Know MLflow for experiment tracking
7. Be able to deploy a model at basic level (FastAPI + Docker)
