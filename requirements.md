# Installation Requirements (lambda-synthesis-experiments)

**Role:** research experiments (USS: GPU-accelerated neuro-symbolic lambda synthesis).  
**Authoritative:** `src/` experiment code + `requirements_experiments.txt`.  
**Non-authoritative / local-only:** `uss-venv/` (huge local env; should remain untracked).

## Prerequisites

- Python 3
- For GPU runs: NVIDIA GPU + CUDA 12.x (per `lambda-synthesis-experiments/README.md`)

## Install (fresh)

From `lambda-synthesis-experiments/`:

```bash
python3 -m venv uss-venv
source uss-venv/bin/activate
pip install -U pip
pip install -r requirements_experiments.txt
```

## Notes

- Treat performance claims in the README as hypotheses until reproduced on the
  stated hardware (Ada SM89); record benchmark provenance in `roadmap-todos.md`.

