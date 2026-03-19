# ML Engineer Learning Plan — Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare for ML Engineer (product + research hybrid) positions within 1 year at 4h/week.

**Architecture:** Four sequential phases — Python fundamentals + Kaggle → DL on messy data + SQL → portfolio + deployment → interviews + applications. Each phase builds on the prior. Progress tracked in `works/progress/phaseN.md`.

**Spec:** `docs/superpowers/specs/2026-03-19-ml-learning-plan-design.md`

---

## Setup

### Task 0: Repository and environment setup

**Files:**
- Create: `works/progress/phase1.md`
- Create: `works/kaggle-tabular/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create progress log for Phase 1**

```markdown
# Phase 1 — Ramp-up (Months 1-3)
# Python (~1.5h/week) + Kaggle tabular (~2.5h/week)
```

- [ ] **Step 2: Create project directory for first Kaggle project**

Create `works/kaggle-tabular/` directory.

- [ ] **Step 3: Update .gitignore**

Add rules to prevent committing data files, model weights, and notebook outputs:
```
# Data
*.csv
*.parquet
*.h5
data/

# Model weights
*.pth
*.pt
*.pkl
*.joblib

# Notebook outputs
.ipynb_checkpoints/

# Environment
.env
__pycache__/
```

- [ ] **Step 4: Register on platforms (if not already)**

- [ ] Kaggle account (kaggle.com)
- [ ] Codewars account (codewars.com) or LeetCode (leetcode.com)

---

## Phase 1: Ramp-up (Months 1-3)

### Task 1: Python — Data Structures (Weeks 1-4, ~1.5h/week)

**Resources:**
- Codewars: filter by tags `Lists`, `Data Structures`, `Fundamentals`, rank 8-6 kyu
- Python docs: `collections` module

- [ ] **Week 1: Lists and tuples**
- Solve 2-3 Codewars problems involving list manipulation (slicing, comprehensions, sorting)
- Mini-exercise: implement a function that takes a list of numbers and returns a dict with keys `mean`, `median`, `mode` — using only built-in Python (no numpy)
- Key concept: mutable vs immutable, when to use tuple vs list

- [ ] **Week 2: Dicts and sets**
- Solve 2-3 problems involving dict/set operations
- Mini-exercise: given two lists of words, find words that appear in both (use set operations, not loops)
- Key concept: O(1) lookup in dict/set vs O(n) in list. When does it matter?

- [ ] **Week 3: collections module**
- Solve 2 problems, then refactor solutions using `defaultdict`, `Counter`, `deque`
- Mini-exercise: count word frequencies in a text file using `Counter`, get top-10
- Key concept: choosing the right data structure reduces code complexity

- [ ] **Week 4: Complexity and review**
- Solve 2-3 problems. Before coding, predict the time complexity of your solution
- Mini-exercise: write two versions of a duplicate-finder (one O(n^2) with nested loops, one O(n) with set). Measure with `time.time()` on a list of 100,000 elements
- Key concept: Big-O is not academic — it determines if your data pipeline finishes in seconds or hours

### Task 2: Kaggle — EDA and data understanding (Weeks 1-3, ~2.5h/week)

**Files:**
- Create: `works/kaggle-tabular/eda.ipynb`

**Resources:**
- Kaggle "Getting Started" competitions: Titanic, House Prices, Spaceship Titanic
- Pandas docs: https://pandas.pydata.org/docs/

- [ ] **Week 1: Choose competition and load data**
- Browse Kaggle Getting Started / Playground competitions
- Pick one (recommendation: Spaceship Titanic — modern, tabular, binary classification, good size)
- Download data via Kaggle API or manually
- Load into Pandas, run `df.info()`, `df.describe()`, `df.head()`
- Write down: how many rows, columns, what is the target, what types of features

- [ ] **Week 2: EDA — distributions and missing values**
- Check missing values: `df.isnull().sum()` — which columns have gaps? How many?
- Visualize distributions of numerical features (histograms via matplotlib)
- Visualize categorical features (value_counts + bar plots)
- Write down: initial hypotheses — which features seem useful for prediction?

- [ ] **Week 3: EDA — correlations and feature relationships**
- Correlation matrix for numerical features (`df.corr()` + seaborn heatmap)
- Cross-tabulations for categorical vs target (e.g., survival rate by cabin deck)
- Identify: multicollinearity, features that strongly correlate with target
- Save EDA notebook with conclusions as markdown cells

### Task 3: Python — OOP (Weeks 5-8, ~1.5h/week)

**Resources:**
- Codewars: filter by `Object-oriented Programming`, rank 7-6 kyu
- PyTorch source: look at `torch.utils.data.Dataset` and `torch.nn.Module` signatures

- [ ] **Week 5: Classes and __init__**
- Solve 2 OOP problems on Codewars
- Mini-exercise: write a `DataStats` class that takes a list of numbers in `__init__` and has methods `.mean()`, `.std()`, `.summary()`
- Key concept: encapsulation — bundling data with operations on that data

- [ ] **Week 6: Dunder methods**
- Solve 2 problems involving `__repr__`, `__eq__`, `__lt__`
- Mini-exercise: write a `Sample` class with `__len__` (number of features), `__getitem__` (access feature by index), `__repr__` (human-readable string). This mirrors how PyTorch `Dataset` works
- Key concept: dunder methods let your objects work with Python built-ins (len(), [], print())

- [ ] **Week 7: Inheritance**
- Solve 1-2 problems
- Mini-exercise: write a base class `BasePreprocessor` with method `.transform(df)` that raises `NotImplementedError`. Write two subclasses: `FillNaPreprocessor` and `ScalePreprocessor` that implement `.transform()`
- Key concept: this is exactly how sklearn transformers and PyTorch modules work

- [ ] **Week 8: Putting it together**
- Mini-exercise: write a simple `TabularDataset(Dataset)` class for your Kaggle data that implements `__len__` and `__getitem__`. Load it into a `DataLoader` and iterate one batch
- Key concept: now you understand why PyTorch's Dataset API looks the way it does

### Task 4: Kaggle — Feature engineering and baseline (Weeks 4-6, ~2.5h/week)

**Files:**
- Create: `works/kaggle-tabular/baseline.ipynb`

- [ ] **Week 4: Handle missing values and encode categoricals**
- Decide strategy per column: drop, fill with median/mode, create "is_missing" flag
- Encode categorical features: `pd.get_dummies()` or `LabelEncoder`
- Result: clean DataFrame ready for modeling

- [ ] **Week 5: First baseline — LogisticRegression**
- Train/test split (or use Kaggle's test set)
- Fit `LogisticRegression` from sklearn
- Evaluate: accuracy, precision, recall, F1, confusion matrix
- Submit to Kaggle — record your first score
- Key question: is this score better than random? Better than always predicting the majority class?

- [ ] **Week 6: Second baseline — RandomForest**
- Fit `RandomForestClassifier`
- Compare metrics to LogReg
- Look at `feature_importances_` — do they match your EDA hypotheses?
- Key question: which features matter? Does the model agree with your intuition?

### Task 5: Python — Generators, decorators, context managers (Weeks 9-12, ~1.5h/week)

**Resources:**
- Codewars: filter by `Generators`, `Decorators`, rank 6 kyu
- Python docs: `contextlib`

- [ ] **Week 9: Generators**
- Solve 2 problems involving `yield`
- Mini-exercise: write a generator that reads a large CSV file in chunks (without loading it all into memory). Compare memory usage to `pd.read_csv()` on a large file
- Key concept: generators for memory-efficient data processing pipelines

- [ ] **Week 10: Decorators**
- Solve 2 problems
- Mini-exercise: write a `@timer` decorator that prints how long a function takes. Apply it to your model training function
- Key concept: decorators are used in Flask/FastAPI routes (`@app.get("/")`), PyTorch hooks, and testing

- [ ] **Week 11: Context managers and exception handling**
- Mini-exercise: write a context manager `TrainingContext` that sets `torch.set_grad_enabled(True)` on enter and `False` on exit (similar to `torch.no_grad()`)
- Practice: add try/except to your data loading code — handle `FileNotFoundError`, `pd.errors.ParserError`
- Key concept: resource management and graceful error handling

- [ ] **Week 12: Review and self-assessment**
- Go back to Week 1 problems — can you solve them faster and cleaner?
- Try 2 new 6 kyu problems without hints
- Honest assessment: which topics need more practice in Phase 2?

### Task 6: Kaggle — Boosting, tuning, and final submission (Weeks 7-12, ~2.5h/week)

**Files:**
- Create: `works/kaggle-tabular/boosting.ipynb`
- Create: `works/kaggle-tabular/README.md` (at the end)

- [ ] **Week 7: CatBoost / XGBoost baseline**
- Install CatBoost or XGBoost
- Fit with default parameters
- Compare to LogReg and RandomForest — how much did score improve?

- [ ] **Week 8: Cross-validation**
- Implement k-fold CV (k=5) using `sklearn.model_selection.cross_val_score` or `StratifiedKFold`
- Record mean and std of the metric across folds
- Key concept: single train/test split can be misleading. CV gives a more honest estimate

- [ ] **Week 9: Hyperparameter tuning**
- Try `GridSearchCV` or `RandomizedSearchCV` on 3-5 key hyperparameters (learning_rate, max_depth, n_estimators, etc.)
- Compare tuned score to default
- Key question: how much did tuning help? Was it worth the compute time?

- [ ] **Week 10: Feature engineering round 2**
- Based on error analysis: which samples does the model get wrong?
- Try creating new features (interactions, aggregations, binning)
- Rerun model — did new features help?

- [ ] **Week 11: Error analysis and final submission**
- Confusion matrix: which classes are confused?
- Look at worst predictions — what's special about those samples?
- Make final Kaggle submission
- Record final score in progress log

- [ ] **Week 12: Format and document**
- Clean up notebooks: remove dead code, add markdown headings and conclusions
- Write `works/kaggle-tabular/README.md`: problem description, approach, key findings, final score
- This is your first portfolio piece

---

## Phase 2: Momentum (Months 4-6)

### Task 7: Environment migration (when PC is upgraded)

**Files:**
- Create: `works/progress/phase2.md`
- Create: `works/progress/env_setup_checklist.md`

- [ ] **Step 0: Create Phase 2 progress log**

Create `works/progress/phase2.md` with header:
```markdown
# Phase 2 — Momentum (Months 4-6)
# Python+SQL (~1h/week) + DL project (~3h/week)
```

- [ ] **Step 1: Document current environment before migration**
- Run `conda env export > environment.yml` (or `pip freeze > requirements.txt`)
- Note Python version, CUDA version (if applicable), OS

- [ ] **Step 2: Set up new PC**
- Install Python (3.10+ recommended)
- Install conda or use venv
- Install: pytorch, pandas, numpy, scikit-learn, matplotlib, seaborn, catboost/xgboost, mlflow, jupyter
- Verify CUDA/GPU: `python -c "import torch; print(torch.cuda.is_available())"`

- [ ] **Step 3: Document the setup**
- Write `works/progress/env_setup_checklist.md` with exact steps taken
- Save `environment.yml`
- Verify all Phase 1 notebooks still run

### Task 8: Python — LeetCode Medium (Weeks 13-16, ~1h/week, full hour since SQL starts week 17)

**Resources:**
- LeetCode: "Top Interview Questions" → Easy/Medium
- Focus tags: `Hash Table`, `Two Pointers`, `String`, `Array`

- [ ] **Week 13: Hashmap problems**
- Solve 2 LeetCode problems using dict for O(1) lookup (e.g., Two Sum, Group Anagrams)
- Key pattern: "have I seen this before?" → use a dict

- [ ] **Week 14: Two pointers**
- Solve 2 problems (e.g., Valid Palindrome, Container With Most Water)
- Key pattern: two pointers converging from both ends or slow/fast

- [ ] **Week 15: String manipulation**
- Solve 2 problems involving string processing
- Relevance: text preprocessing in NLP, parsing log files

- [ ] **Week 16: Basic recursion**
- Solve 2 problems (e.g., tree traversal, flatten nested list)
- Key concept: recursion appears in tree-based models, nested data structures

### Task 9: DL Project — Data selection and EDA (Weeks 13-16, ~3h/week)

**Files:**
- Create: `works/dl-project/eda.ipynb`

- [ ] **Week 13: Choose task and dataset**
- Options:
  - NLP: Toxic comment classification (Jigsaw), Russian text classification, sentiment on non-English data
  - CV: Plant disease detection, chest X-ray, satellite imagery classification
  - Tabular+DL: Same domain as Phase 1 but with embeddings/TabNet approach
- Requirements: real-world messiness (imbalance, noise, missing data)
- Download data, initial `df.info()` / image sample inspection

- [ ] **Week 14: EDA focused on challenges**
- Class distribution: how imbalanced? (plot counts per class)
- Data quality: corrupted images? Mislabeled samples? Missing fields?
- Data size: will it fit in memory? Need chunking or generators?

- [ ] **Week 15: Strategy document**
- Write down in notebook: how will you handle imbalance? (oversampling, class weights, focal loss)
- What augmentations make sense for this data type?
- What baseline metric to beat? (majority class accuracy, random baseline)

- [ ] **Week 16: Data split strategy**
- Implement train/val/test split with stratification
- Verify class distributions are preserved in each split
- If temporal data — use time-based split, not random

### Task 10: Python — Practical OOP + SQL start (Weeks 17-20)

**Python (~0.5h/week):**

- [ ] **Week 17: Custom Dataset class (concept)**
- Study how `torch.utils.data.Dataset` works — read the PyTorch source for `__getitem__` and `__len__`
- Sketch on paper: what your Dataset class needs to do for your DL project data
- Note: the actual implementation happens in Task 11, Week 17 — here focus on understanding the OOP pattern

- [ ] **Week 18: Custom training utilities**
- Write an `EarlyStopping` class (you have one in `archs/Earlystop.py` — rewrite from scratch, understand every line)
- Write a `MetricTracker` class that stores and plots loss/metric curves

- [ ] **Week 19: Refactor DL project code**
- Extract reusable components into classes: `Trainer`, `Config` (dataclass)
- Practice: your code should be importable, not just a notebook

- [ ] **Week 20: Standard library deep dive**
- Use `pathlib.Path` instead of string paths in your project
- Use `functools.partial` to create pre-configured functions
- Use `typing` for function signatures in your utility classes

**SQL (~0.5h/week, weeks 17-20):**

- [ ] **Week 17: SQLBolt lessons 1-6 (review basics and JOINs)**
- [ ] **Week 18: SQLBolt lessons 7-12 (subqueries) + intro to CTEs (WITH clause)**
- [ ] **Week 19: Window functions — ROW_NUMBER, RANK, DENSE_RANK**
- [ ] **Week 20: Window functions — LAG, LEAD, running totals**

### Task 11: DL Project — Model pipeline (Weeks 17-20, ~3h/week)

**Files:**
- Create: `works/dl-project/train.py`
- Create: `works/dl-project/model.py`
- Create: `works/dl-project/dataset.py`

- [ ] **Week 17: DataLoader pipeline**
- Dataset class (from Task 10 week 17)
- DataLoader with proper `num_workers`, `pin_memory` (if GPU available)
- Augmentations (torchvision.transforms for CV, or text augmentations for NLP)
- Verify: iterate one batch, print shapes, visualize a few samples

- [ ] **Week 18: Baseline model**
- Pick simplest reasonable architecture (ResNet18 pretrained for CV, LSTM/small transformer for NLP)
- Train for 5 epochs, record loss curve
- Evaluate on validation set — is it learning at all?

- [ ] **Week 19: MLflow setup + experiment discipline**
- Install MLflow, run `mlflow ui` locally
- Log: hyperparameters, train/val loss per epoch, final metrics
- Compare: run the same model with 2 different learning rates, compare in MLflow UI
- **Start an experiment log habit:** for every run, write down in a notebook cell: (1) hypothesis — what you changed and why, (2) result — what happened, (3) conclusion — what to try next. This is the core skill, not just the tooling

- [ ] **Week 20: Training loop improvements**
- Add learning rate scheduler (CosineAnnealingLR or ReduceLROnPlateau)
- Add gradient clipping if needed
- Add `torch.amp` (automatic mixed precision) if GPU available

### Task 12: DL Project — Experiments and SQL (Weeks 21-24)

**ML (~3h/week):**

- [ ] **Week 21: Architecture experiments**
- Try 2-3 different architectures (e.g., ResNet18 vs ResNet50 vs EfficientNet, or LSTM vs GRU vs Transformer)
- Log all runs in MLflow
- Compare: accuracy, training time, model size

- [ ] **Week 22: Handling imbalance**
- Experiment with: class weights, oversampling (RandomOverSampler), focal loss
- Log results in MLflow
- Key question: which approach improved recall for minority class?

- [ ] **Week 23: Final tuning**
- Best architecture + best imbalance strategy
- Tune top 3 hyperparameters
- Record best validation metrics

- [ ] **Week 24: Wrap up and document**
- Save best model checkpoint
- Write experiment summary: what worked, what didn't, why
- Clean up code, ensure reproducibility
- Start `works/dl-project/README.md`

**SQL (~0.5h/week, weeks 21-24):**

- [ ] **Week 21: HackerRank SQL Medium problems (2-3 problems)**
- [ ] **Week 22: Practice CTEs with real-world examples**
- [ ] **Week 23: Complex window function queries**
- [ ] **Week 24: Self-assessment — can you write a query with JOIN + CTE + window function?**

---

## Phase 3: Portfolio (Months 7-9)

### Task 13: Own project — Problem formulation and data (Weeks 25-28, ~3.5h/week)

**Files:**
- Create: `works/own-project/` directory
- Create: `works/own-project/data_collection.ipynb` or `works/own-project/scraper.py`
- Create: `works/progress/phase3.md`

- [ ] **Week 25: Choose problem and justify it**
- Pick a problem YOU care about (not "what looks good on resume")
- Write 1 paragraph: what is the problem, who would use the solution, what metric defines success
- Options from spec: extend recSys with own data, Russian NLP, audio classification
- Bonus if it combines your DL strength with practical value

- [ ] **Week 26: Data sourcing**
- Find or collect data (Kaggle datasets, public APIs, web scraping with requests+BeautifulSoup, or open government data)
- If scraping: be respectful (rate limiting, robots.txt)
- Initial load and sanity check: how many samples, features, quality

- [ ] **Week 27: EDA on own data**
- Apply EDA skills from Phases 1-2
- Document data quality issues and how you'll handle them
- Key difference from Kaggle: nobody cleaned this data for you

- [ ] **Week 28: Data pipeline**
- Build robust data loading pipeline (Dataset + DataLoader)
- Handle edge cases you discovered in EDA
- Train/val/test split with justification

### Task 14: Own project — Model and results (Weeks 29-31, ~3.5h/week)

**Files:**
- Create: `works/own-project/train.py`
- Create: `works/own-project/model.py`

- [ ] **Week 29: Baseline model**
- Start simple (even logistic regression / simple NN)
- Establish baseline metric
- Log in MLflow

- [ ] **Week 30: Iterate on model**
- Apply learnings from Phase 2: better architecture, handling imbalance, tuning
- 2-3 experiment runs logged in MLflow
- Pick best approach

- [ ] **Week 31: Results and conclusions**
- Final evaluation on test set
- Visualize results: confusion matrix, example predictions (good and bad)
- Write conclusions: what worked, what didn't, what would you do with more time/data
- Begin `works/own-project/README.md`

### Task 15: Deployment mini-project (Weeks 32-33, ~3.5h/week)

**Files:**
- Create: `works/own-project/api.py` (or separate `works/deployment/`)
- Create: `works/own-project/Dockerfile`
- Create: `works/own-project/requirements.txt`

- [ ] **Week 32: FastAPI endpoint**
- Install FastAPI + uvicorn
- Write a single POST endpoint that takes input data and returns model prediction
- Test locally with `curl` or Python `requests`
- Handle: input validation (pydantic), model loading, error responses

- [ ] **Week 33: Docker**
- Write `Dockerfile`: base image, install dependencies, copy code, expose port
- Build and run locally: `docker build -t ml-model . && docker run -p 8000:8000 ml-model`
- Test: hit the endpoint from outside the container
- Document in README: how to build and run

### Task 16: GitHub portfolio formatting (Weeks 34-36, ~3.5h/week)

- [ ] **Week 34: Git hygiene**
- Create proper `.gitignore` (model weights, data, cache, checkpoints)
- Clean up `recSys/best_model.pth` from git history (or add note that it's legacy)
- Review commit history — any sensitive data?
- Decide: which existing directories (`archs/`, `recSys/`) to include in portfolio

- [ ] **Week 35: READMEs and documentation**
- Each project gets README with:
  - Problem description (1 paragraph)
  - Approach (what you tried, what worked)
  - Results (metrics, key visualizations embedded as images)
  - How to run (setup instructions, commands)
- Keep it concise — recruiter spends 30 seconds

- [ ] **Week 36: Final polish**
- Verify all `requirements.txt` / `environment.yml` are up to date
- Run each project from scratch to verify reproducibility
- Clean up notebook outputs, remove debug cells
- Final review: does your GitHub profile page look professional?

---

## Phase 4: Market Entry (Months 10-12)

### Task 17: Buffer and catch-up (Weeks 37-39)

- [ ] **Weeks 37-39: Finish any incomplete work from Phase 3**
- Portfolio formatting if not done
- Optional: ask a friend or peer to review one of your project READMEs — fresh eyes catch what you miss
- Optional: update all `environment.yml` / `requirements.txt` files
- REST if needed — burnout at 4h/week is real too
- Optional: start browsing job postings casually

### Task 18: Market analysis (Weeks 40-41, 4h/week)

**Files:**
- Create: `works/progress/phase4.md`
- Create: `works/progress/job_analysis.md`

- [ ] **Week 40: Collect job postings**
- Find 15-20 MLE / Junior ML Engineer / DS postings on HH, LinkedIn, Habr Career
- For each: note required skills, tools, experience level
- Save in `works/progress/job_analysis.md`

- [ ] **Week 41: Gap analysis**
- Tally: which skills appear most often?
- Compare to your current skills — mark green (have), yellow (partial), red (missing)
- Red items = study plan for weeks 42-48
- Common gaps to expect: Docker (you have it), cloud (AWS/GCP basics), A/B testing, SQL (you have it)

### Task 19: Interview prep — theory (Weeks 42-45, 4h/week)

- [ ] **Week 42: ML theory**
- Prepare answers for: bias-variance tradeoff, overfitting (what is it, how to fight), regularization (L1 vs L2), metrics (when precision > recall, when F1, when AUC-ROC)
- Format: write answers in your own words, then check against a reference
- Practice explaining to AI mentor

- [ ] **Week 43: DL theory**
- Prepare: backpropagation (be able to explain chain rule on a whiteboard), batch norm (why and where), dropout (why and where), learning rate selection strategies
- Connect to your projects: "in my Kaggle project, I used X because..."

- [ ] **Week 44: Python interview questions**
- Practice: mutable vs immutable, GIL, generators vs iterators, `__new__` vs `__init__`, decorators
- Solve 2-3 LeetCode Medium problems under time pressure (30 min each)

- [ ] **Week 45: SQL + System Design**
- SQL: solve 3-5 Medium problems on LeetCode/HackerRank (window functions, CTEs, complex JOINs)
- System Design: prepare a 5-min talk on "how I would deploy an ML model in production" (based on your Phase 3 deployment experience)

### Task 20: Mock interviews and applications (Weeks 46-52)

- [ ] **Weeks 46-48: Mock interviews with AI mentor**
- Session 1: ML theory questions (20 min)
- Session 2: "Tell me about your project" — practice for each of your 2-3 projects (5 min each)
- Session 3: Python coding problem live (30 min)
- Session 4: SQL query problem + system design (30 min)
- After each: note weak spots, review before next session

- [ ] **Week 48: Resume**
- Write resume with:
  - Skills section: Python, PyTorch, Pandas, SQL, MLflow, FastAPI, Docker, sklearn, CatBoost
  - Projects section: 2-3 projects with 2-sentence descriptions and links
  - Education / courses (include Yandex Practicum math if completed)

- [ ] **Weeks 48-52: Applications**
- Start applying — even to "reach" positions
- Track: where applied, when, status
- After each interview: write down questions asked, how you answered, what to improve
- Feed interview experience back into study plan

- [ ] **Week 52: Retrospective**
- Review progress logs from all 4 phases
- Compare to success criteria from spec
- Decide next steps: continue applying, deepen a skill, or adjust target role

---

## Optional: Mathematics (Yandex Practicum)

**Not scheduled in main plan.** Additional hours on top of 4h/week.

- [ ] **Trigger: start when DL project raises math questions (Phase 2-3)**
- [ ] **Rule: if choosing between math lecture and project work, project wins**
- [ ] **Suggested pace: 1-2 lectures per week alongside main plan**
