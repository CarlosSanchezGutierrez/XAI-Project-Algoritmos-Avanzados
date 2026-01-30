cat > src/eval_metrics.py <<'EOF'
from .robustness import jaccard

def stability_jaccard(expl_a: list, expl_b: list):
    return jaccard(set(expl_a), set(expl_b))
EOF
