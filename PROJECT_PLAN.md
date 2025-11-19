# NETWORK INTRUSION DETECTION SYSTEM - PORTFOLIO PROJECT

## PROJECT OVERVIEW
You are helping build a production-quality Network Intrusion Detection System using anomaly detection algorithms. This is a portfolio project for job applications in AI/ML engineering.

## CORE OBJECTIVES
1. Implement 4 anomaly detection algorithms: Isolation Forest, One-Class SVM, Autoencoder, GANomaly/VAE
2. Achieve F1 > 0.90 on CICIDS2017 dataset
3. Build explainability layer using SHAP
4. Create deployable API + dashboard
5. Produce professional documentation

## CRITICAL SUCCESS FACTORS
- **COMPLETE > AMBITIOUS**: Finish 3 methods well rather than 5 methods poorly
- **MEASURE EVERYTHING**: Update claude.md after EVERY interaction with progress
- **FAIL FAST**: If GANomaly shows instability, pivot to VAE immediately
- **PRODUCTION READY**: Every component should be deployable, not just notebook code

## PROJECT STRUCTURE
```
network-intrusion-detection/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── claude.md                    # CRITICAL: Update after every interaction
├── PROJECT_PLAN.md              # This file
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_classical.ipynb
│   ├── 03_autoencoder.ipynb
│   ├── 04_ganomaly.ipynb
│   └── 05_comparison.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── api/
│   └── dashboard/
├── models/
├── results/
└── tests/
```

---

## IMPLEMENTATION PHASES

### PHASE 1: RAPID PROTOTYPING

**Focus: Get all algorithms working with solid evaluation**

#### Setup & Data
- Download CICIDS2017 dataset (start with Monday data)
- Create EDA notebook with class distribution and feature analysis
- Build preprocessing pipeline (handle missing values, normalization, train/test split)
- **Update claude.md with dataset statistics**

#### Isolation Forest
- Implement with contamination parameter tuning
- Evaluate: Precision, Recall, F1, AUC
- Extract feature importance
- **Update claude.md with IF metrics and top features**

#### One-Class SVM
- Implement with kernel selection (linear, RBF)
- Hyperparameter tuning: nu, gamma
- Compare performance with IF
- Benchmark inference speed
- **Update claude.md with OCSVM metrics and comparison**

#### Autoencoder
- Experiment with architectures (simple, medium, deep)
- Train on normal traffic only
- Optimize reconstruction threshold (test multiple percentiles)
- Evaluate on mixed traffic with per-attack-type analysis
- Generate latent space visualizations
- **Update claude.md with architecture choice and performance**

#### GANomaly/VAE
- Design architecture adapted for tabular data (80 features, not images)
- Implement and train on normal traffic
- **PIVOT POINT**: If training shows instability, switch to VAE
- Evaluate anomaly detection performance
- **Update claude.md with training results and pivot decision if needed**

#### Comprehensive Comparison
- Create comparison notebook with all methods
- Generate metrics table, ROC curves, confusion matrices
- Build per-attack-type heatmap showing which methods work best for which attacks
- **Update claude.md with final comparison and best method recommendations**

**Phase 1 Success Criteria:**
- All algorithms implemented and evaluated
- F1 > 0.88 for at least one method
- Clear understanding of method strengths/weaknesses

---

### PHASE 2: PRODUCTION ENGINEERING

**Focus: Make it deployable and explainable**

#### Explainability with SHAP
- Implement SHAP TreeExplainer for Isolation Forest
- Implement SHAP KernelExplainer for Autoencoder (sample subset for speed)
- Generate force plots and summary plots
- Create function returning top-3 contributing features per prediction
- **Update claude.md with explainability implementation**

#### FastAPI Endpoint
- Design API schema with Pydantic models
- Implement POST /predict endpoint accepting 80 features
- Response format: {anomaly_score, is_anomaly, confidence, top_features, shap_values}
- Add input validation and error handling
- Write API documentation with examples
- **Update claude.md with API design and test results**

#### Streamlit Dashboard
- Design interface: file upload (batch) and manual input (single prediction)
- Display predictions with anomaly score visualization
- Embed SHAP force plots
- Add model comparison view (switch between methods)
- Enable downloadable results
- **Update claude.md with dashboard features**

#### Docker Containerization
- Create Dockerfile with multi-stage build
- Create docker-compose.yml for easy deployment
- Configure volume mounts and environment variables
- Test: container should start API + dashboard
- **Update claude.md with Docker setup**

#### Basic Monitoring
- Log all predictions with timestamp and features
- Calculate feature statistics and compare to training distribution
- Implement simple drift detection (alert if features shift >20%)
- **Update claude.md with monitoring approach**

**Phase 2 Success Criteria:**
- Working API with explanations
- Functional dashboard
- Docker deployment working
- Basic drift detection implemented

---

### PHASE 3: POLISH & DOCUMENTATION

**Focus: Professional presentation**

#### Results Analysis
- Run comprehensive evaluation across all methods
- Generate visualizations: speed vs accuracy, memory usage, per-attack performance
- Create decision guide for method selection
- **Update claude.md with final evaluation**

#### Documentation
- Write README with problem statement, methodology, results, architecture diagram
- Add docstrings to all functions
- Create ARCHITECTURE.md explaining system design
- Include limitations section and future work
- **Update claude.md with documentation status**

#### Demo Preparation
- Prepare labeled demo scenarios (normal, port scan, DDoS, botnet)
- Test scenarios through dashboard
- Record screen capture demonstrating detection with explanations
- **Update claude.md with demo scenarios**

#### Blog Post
- Write article covering: problem, approach, results, insights, production considerations
- Include key visualizations
- Focus on lessons learned and technical decisions
- **Update claude.md with blog draft status**

#### Final Polish
- Code cleanup and refactoring
- Remove hardcoded paths and dead code
- Add missing tests
- Final integration testing
- **Update claude.md with completion checklist**

**Phase 3 Success Criteria:**
- Professional README with results
- Demo video ready
- Blog post drafted
- All code clean and tested

---

## CLAUDE.MD STRUCTURE

Update this file **after every interaction** with your progress.
```markdown
# Network Intrusion Detection System - Progress Journal

**Last Updated:** [TIMESTAMP]
**Current Phase:** [Phase name]
**Overall Status:** [Brief summary]

---

## Component Status

| Component | Status | Key Metric | Notes |
|-----------|--------|------------|-------|
| Isolation Forest | [Status] | F1: X.XX | [Brief note] |
| One-Class SVM | [Status] | F1: X.XX | [Brief note] |
| Autoencoder | [Status] | F1: X.XX | [Brief note] |
| GANomaly/VAE | [Status] | F1: X.XX | [Brief note] |
| API | [Status] | - | [Brief note] |
| Dashboard | [Status] | - | [Brief note] |
| Docker | [Status] | - | [Brief note] |
| Documentation | [Status] | - | [Brief note] |

**Status codes:** ✅ Complete | 🚧 In Progress | ⏳ Pending | ⚠️ Blocked

---

## Latest Updates

### [DATE/TIME] - [Component/Task Name]
**Status:** [Complete/In Progress/Blocked]

**What was done:**
- [Action 1]
- [Action 2]

**Results/Metrics:**
- [Key finding or metric]

**Decisions made:**
- [Decision and reasoning]

**Issues encountered:**
- [Problem and solution/workaround]

**Next steps:**
- [What to work on next]

---

### [Previous Update]
[Same format]

---

## Performance Summary

### Algorithm Comparison
| Method | Precision | Recall | F1 | AUC | Inference Speed |
|--------|-----------|--------|-----|-----|----------------|
| IF | X.XX | X.XX | X.XX | X.XX | Xms/1000 |
| OCSVM | X.XX | X.XX | X.XX | X.XX | Xms/1000 |
| AE | X.XX | X.XX | X.XX | X.XX | Xms/1000 |
| GAN/VAE | X.XX | X.XX | X.XX | X.XX | Xms/1000 |

### Per-Attack Detection (F1 Scores)
| Attack Type | IF | OCSVM | AE | GAN/VAE | Best |
|-------------|-----|-------|-----|---------|------|
| DDoS | X.XX | X.XX | X.XX | X.XX | [Method] |
| PortScan | X.XX | X.XX | X.XX | X.XX | [Method] |
| Botnet | X.XX | X.XX | X.XX | X.XX | [Method] |
| Web Attacks | X.XX | X.XX | X.XX | X.XX | [Method] |

---

## Critical Decisions Log

**[Decision Title]**
- **When:** [Date/context]
- **Decision:** [What was decided]
- **Rationale:** [Why]
- **Impact:** [What changed]

---

## Active Issues

**[Issue Title]**
- **Impact:** High/Medium/Low
- **Description:** [What's wrong]
- **Attempted solutions:** [What's been tried]
- **Current plan:** [Next approach]

---

## Key Insights

- [Technical insight 1]
- [Technical insight 2]
- [Lesson learned 1]
- [Best practice discovered]

---

## TODO - Current Focus

**High Priority:**
- [ ] Task 1
- [ ] Task 2

**Medium Priority:**
- [ ] Task 3
- [ ] Task 4

**Backlog:**
- [ ] Future enhancement 1
- [ ] Future enhancement 2

---

## Links & Resources

- GitHub Repo: [URL]
- Demo Video: [URL]
- Blog Post: [URL]
- Key Paper: [Title + URL]
```

---

## QUALITY GATES

Before moving to next major component, verify:

**After Classical Methods (IF + OCSVM):**
- ✅ Both achieve F1 > 0.85
- ✅ Feature importance documented
- ✅ Comparison complete

**After Autoencoder:**
- ✅ F1 > 0.88
- ✅ Architecture justified
- ✅ Latent space visualized

**After All Algorithms:**
- ✅ 3-4 methods working
- ✅ Comparison notebook complete
- ✅ Best method per attack type identified

**After API:**
- ✅ Returns predictions with explanations
- ✅ SHAP working
- ✅ Documentation complete

**After Dashboard:**
- ✅ File upload working
- ✅ Visualizations displaying correctly
- ✅ Docker container running

**Before Publishing:**
- ✅ README professional quality
- ✅ Demo video polished
- ✅ All tests passing

---

## PIVOT TRIGGERS

**Immediate pivots:**
- GANomaly unstable after reasonable tuning attempts → Switch to VAE
- F1 < 0.80 after algorithm implementation → Revisit preprocessing
- API latency > 500ms → Optimize inference
- Docker build repeatedly failing → Simplify dependencies

**When stuck >2 interactions on same issue:**
- Document problem in claude.md
- Propose 2-3 alternative approaches
- Choose simplest viable option

---

## GETTING HELP

When blocked, update claude.md with:
1. What you're trying to achieve
2. What you've tried (with code snippets)
3. Current error/unexpected behavior
4. Your hypothesis about the issue
5. Proposed next steps (2-3 options)

---

## INTERACTION PROTOCOL

**At the end of EVERY interaction, you must:**

1. **Update claude.md** with:
   - What was completed this interaction
   - Any metrics/results obtained
   - Decisions made and reasoning
   - Issues encountered and solutions
   - Next immediate steps

2. **Verify progress against quality gates**
   - Are we meeting success criteria?
   - Do we need to pivot?

3. **State clearly what to work on next**
   - Specific next task
   - Expected outcome
   - Potential blockers to watch for

**Example end of interaction:**
```
I've updated claude.md with the Isolation Forest implementation results (F1: 0.89).
Next interaction should focus on implementing One-Class SVM with RBF kernel.
Potential issue to watch: OCSVM may be slow on full dataset, be ready to subsample if needed.
```

---

## SUCCESS METRICS

**Technical:**
- F1 > 0.90 for best method
- Inference < 100ms per 1000 samples
- All 4 algorithms evaluated
- API functional with explanations

**Documentation:**
- README clearly explains approach and results
- Code well-structured and commented
- Demo video shows working system
- Blog post explains technical decisions

**Portfolio Readiness:**
- Project looks professional on GitHub
- Can discuss all design decisions
- Have 3+ key insights to share
- Clear differentiation from typical projects

---

## START HERE - FIRST ACTIONS

1. Create project structure
2. Copy this to PROJECT_PLAN.md
3. Initialize claude.md from template
4. Download CICIDS2017 Monday data
5. Update claude.md with initial status
6. Begin with EDA notebook

**Remember:** Update claude.md after completing these initial setup tasks before moving to algorithm implementation.

---

## FINAL NOTES

**Philosophy:** This project demonstrates your ability to:
- Execute complex ML projects end-to-end
- Make justified technical decisions
- Build production-ready systems
- Communicate results effectively

**Scope flexibility:** If timeline pressures emerge:
- Can deliver with 3 methods instead of 4 (drop GANomaly)
- Can simplify dashboard (basic Streamlit without fancy features)
- Cannot skip: documentation, deployment, explainability

**The non-negotiables:**
- Update claude.md every interaction
- Pass quality gates before moving forward
- Pivot when stuck (don't grind on broken approaches)
- Produce professional documentation

Ready to build!
