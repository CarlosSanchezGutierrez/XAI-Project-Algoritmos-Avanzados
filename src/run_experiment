cat > src/run_experiment.py <<'EOF'
import os
import json
import time
import yaml
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from .dag import make_random_dag
from .simulate import simulate_linear_gaussian, inject_anomaly
from .detect import fit_zscore_baseline, detect_event
from .explain import explain_event
from .robustness import add_noise, apply_missing, ema_smooth_scores
from .eval_metrics import stability_jaccard

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["seed"])

    g = make_random_dag(
        n_nodes=cfg["data"]["n_nodes"],
        edge_prob=cfg["data"]["edge_prob"],
        rng=rng
    )

    df = simulate_linear_gaussian(
        g=g,
        n_steps=cfg["data"]["n_steps"],
        noise_std=cfg["data"]["noise_std"],
        rng=rng
    )

    df_anom, true_target = inject_anomaly(
        df=df,
        t0=cfg["anomaly"]["t0"],
        kind=cfg["anomaly"]["kind"],
        strength=cfg["anomaly"]["strength"],
        target=cfg["anomaly"]["target"],
        rng=rng
    )

    n_train = int(cfg["data"]["train_frac"] * len(df_anom))
    train = df_anom.iloc[:n_train]
    mu, sig = fit_zscore_baseline(train)

    t_event, suspicious, z = detect_event(
        df=df_anom,
        mu=mu,
        sig=sig,
        threshold=cfg["detection"]["zscore_threshold"],
        min_nodes_flagged=cfg["detection"]["min_nodes_flagged"]
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join("outputs", stamp)
    ensure_dir(outdir)

    meta = {
        "true_anomaly_target": true_target,
        "detected_t_event": t_event,
        "detected_suspicious_nodes": suspicious
    }

    if t_event is None:
        meta["status"] = "no_event_detected"
        with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("No event detected. Exiting.")
        return

    # === Baseline explanation (sin regularización) ===
    expl_base = explain_event(
        g=g,
        train=train,
        df=df_anom,
        t_event=t_event,
        suspicious_nodes=suspicious,
        topk=cfg["explanation"]["topk_suspicious"],
        max_expl_nodes=cfg["explanation"]["max_expl_nodes"],
        alpha=cfg["explanation"]["fidelity_alpha"],
        use_metaheuristic=cfg["explanation"]["use_metaheuristic"],
        rng=rng
    )

    # === M2: Robustness experiment ===
    rob = cfg["robustness"]
    results = {
        "config": cfg,
        "meta": meta,
        "explanation_baseline": expl_base,
        "robustness": []
    }

    if rob["enabled"]:
        # Regularized variant: aplicamos smoothing al z-score y usamos suspicious_nodes del smoothed
        # (nota: es una simplificación válida para M2 versión 1)
        z_df = pd.DataFrame(z.values, columns=z.columns)
        z_sm = ema_smooth_scores(z_df, alpha=rob["ema_alpha"])

        def suspicious_from_z(zrow, thr):
            return list(zrow.index[(zrow.abs() >= thr).values])

        # para estabilidad, comparamos explicaciones bajo perturbación vs baseline (y vs regularized)
        for sigma in rob["noise_levels"]:
            for miss in rob["missing_rates"]:
                df_p = apply_missing(add_noise(df_anom, sigma, rng), miss, rng)

                # baseline expl con datos perturbados
                t_event_p, susp_p, z_p = detect_event(df_p, mu, sig, cfg["detection"]["zscore_threshold"], cfg["detection"]["min_nodes_flagged"])
                if t_event_p is None:
                    continue

                expl_p = explain_event(
                    g=g,
                    train=train,
                    df=df_p,
                    t_event=t_event_p,
                    suspicious_nodes=susp_p,
                    topk=cfg["explanation"]["topk_suspicious"],
                    max_expl_nodes=cfg["explanation"]["max_expl_nodes"],
                    alpha=cfg["explanation"]["fidelity_alpha"],
                    use_metaheuristic=cfg["explanation"]["use_metaheuristic"],
                    rng=rng
                )

                # regularized suspicious nodes (smoothing)
                z_p_df = pd.DataFrame(z_p.values, columns=z_p.columns)
                z_p_sm = ema_smooth_scores(z_p_df, alpha=rob["ema_alpha"])
                susp_p_reg = suspicious_from_z(z_p_sm.iloc[t_event_p], cfg["detection"]["zscore_threshold"])

                expl_p_reg = explain_event(
                    g=g,
                    train=train,
                    df=df_p,
                    t_event=t_event_p,
                    suspicious_nodes=susp_p_reg,
                    topk=cfg["explanation"]["topk_suspicious"],
                    max_expl_nodes=cfg["explanation"]["max_expl_nodes"],
                    alpha=cfg["explanation"]["fidelity_alpha"],
                    use_metaheuristic=cfg["explanation"]["use_metaheuristic"],
                    rng=rng
                )

                jac_base = stability_jaccard(expl_base["explanation_nodes"], expl_p["explanation_nodes"])
                jac_reg  = stability_jaccard(expl_base["explanation_nodes"], expl_p_reg["explanation_nodes"])

                results["robustness"].append({
                    "noise_sigma": sigma,
                    "missing_rate": miss,
                    "t_event": t_event_p,
                    "jaccard_baseline": float(jac_base),
                    "jaccard_regularized": float(jac_reg),
                    "fid_baseline": float(expl_p["fidelity"]),
                    "fid_regularized": float(expl_p_reg["fidelity"])
                })

    # guardar JSON
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # plot simple M2: baseline vs regularized jaccard
    if results["robustness"]:
        df_r = pd.DataFrame(results["robustness"])
        df_r = df_r.sort_values(["noise_sigma","missing_rate"])

        plt.figure()
        plt.plot(df_r["jaccard_baseline"].values, marker="o")
        plt.plot(df_r["jaccard_regularized"].values, marker="o")
        plt.title("Stability (Jaccard) — baseline vs regularized")
        plt.xlabel("scenario index (noise/missing grid)")
        plt.ylabel("Jaccard vs baseline explanation")
        plt.legend(["baseline", "regularized"])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "stability_plot.png"), dpi=160)
        plt.close()

    # imprimir resumen
    print("=== META ===")
    print(meta)
    print("\n=== EXPLANATION OBJECT ===")
    print("method:", expl_base["method"])
    print("fidelity:", expl_base["fidelity"])
    print("nodes:", expl_base["explanation_nodes"])
    print("edges:", expl_base["explanation_edges"])

    if results["robustness"]:
        df_r = pd.DataFrame(results["robustness"])
        print("\n=== M2 ROBUSTNESS SUMMARY ===")
        print("avg jaccard baseline:", df_r["jaccard_baseline"].mean())
        print("avg jaccard regularized:", df_r["jaccard_regularized"].mean())

if __name__ == "__main__":
    main()
EOF
