# XAI-Project-Algoritmos-Avanzados
Proyecto Equipo 3

cat > README.md <<'EOF'
# XAI Causal Anomaly Explainer (Midterm 2 – v1)

## Core idea (do not lose focus)
This repository implements a **repository-centered XAI system** where **detection is only a trigger** and the main output is an **explanation object**:
- a minimal causal subgraph (nodes + edges),
- ranked suspicious components,
- short causal paths,
- a fidelity score (coverage vs explanation size).

The goal is not “just anomaly detection”, but **algorithmic explanation of anomalous events** under a causal computation model (DAG).

## What’s included (v1)
- Synthetic DAG generation + linear-Gaussian simulation
- Anomaly injection (spike / drift / drop)
- Simple z-score detector (trigger)
- XAI explanation:
  - local causal residual scoring (per-node regression on parents)
  - minimal explanation via greedy set-cover
  - optional metaheuristic search (light evolutionary-style refinement)
- M2 robustness:
  - perturbations: noise + missingness
  - regularization: EMA smoothing
  - metric: Jaccard stability of explanation nodes
  - baseline vs regularized experiment + plot

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

## What this repo does (XAI-first)
Detection is only a trigger. The main output is an **explanation object** for an anomalous event:
- ranked suspicious nodes (residual-based),
- a minimal causal subgraph (nodes + edges),
- a fidelity score.

## How to run
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_experiment

## Outputs
The run creates: outputs/<timestamp>/
- results.json (includes explanation_baseline + robustness experiments)
- stability_plot.png (Jaccard stability: baseline vs regularized)

## Explanation Object (in results.json)
Key fields:
- explanation_baseline.explanation_nodes
- explanation_baseline.explanation_edges
- explanation_baseline.fidelity
- explanation_baseline.method
- robustness[*].jaccard_baseline vs robustness[*].jaccard_regularized
