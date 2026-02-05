import os, json, time, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dag import make_random_dag, plot_dag
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

    g = make_random_dag(cfg["data"]["n_nodes"], cfg["data"]["edge_prob"], rng)
    plot_dag(g, title="Generated Causal DAG")
    df = simulate_linear_gaussian(g, cfg["data"]["n_steps"], cfg["data"]["noise_std"], rng)

    df_anom, true_target = inject_anomaly(
        df, cfg["anomaly"]["t0"], cfg["anomaly"]["kind"],
        cfg["anomaly"]["strength"], cfg["anomaly"]["target"], rng
    )

    n_train = int(cfg["data"]["train_frac"] * len(df_anom))
    train = df_anom.iloc[:n_train]
    mu, sig = fit_zscore_baseline(train)

    t_event, suspicious, z = detect_event(
        df_anom, mu, sig,
        cfg["detection"]["zscore_threshold"],
        cfg["detection"]["min_nodes_flagged"]
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
        with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("No event detected.")
        return

    expl_base = explain_event(
        g, train, df_anom, t_event, suspicious,
        cfg["explanation"]["topk_suspicious"],
        cfg["explanation"]["max_expl_nodes"],
        cfg["explanation"]["fidelity_alpha"],
        cfg["explanation"]["use_metaheuristic"],
        rng
    )

    results = {
        "config": cfg,
        "meta": meta,
        "explanation_baseline": expl_base,
        "robustness": []
    }

    rob = cfg["robustness"]
    if rob["enabled"]:
        for sigma in rob["noise_levels"]:
            for miss in rob["missing_rates"]:
                df_p = apply_missing(add_noise(df_anom, sigma, rng), miss, rng)

                t_p, susp_p, z_p = detect_event(
                    df_p, mu, sig,
                    cfg["detection"]["zscore_threshold"],
                    cfg["detection"]["min_nodes_flagged"]
                )
                if t_p is None:
                    continue

                expl_p = explain_event(
                    g, train, df_p, t_p, susp_p,
                    cfg["explanation"]["topk_suspicious"],
                    cfg["explanation"]["max_expl_nodes"],
                    cfg["explanation"]["fidelity_alpha"],
                    cfg["explanation"]["use_metaheuristic"],
                    rng
                )

                z_p_df = pd.DataFrame(z_p.values, columns=z_p.columns)
                z_p_sm = ema_smooth_scores(z_p_df, alpha=rob["ema_alpha"])
                susp_p_reg = list(
                    z_p_sm.columns[
                        (z_p_sm.iloc[t_p].abs() >= cfg["detection"]["zscore_threshold"]).values
                    ]
                )

                expl_p_reg = explain_event(
                    g, train, df_p, t_p, susp_p_reg,
                    cfg["explanation"]["topk_suspicious"],
                    cfg["explanation"]["max_expl_nodes"],
                    cfg["explanation"]["fidelity_alpha"],
                    cfg["explanation"]["use_metaheuristic"],
                    rng
                )

                jac_base = stability_jaccard(expl_base["explanation_nodes"], expl_p["explanation_nodes"])
                jac_reg  = stability_jaccard(expl_base["explanation_nodes"], expl_p_reg["explanation_nodes"])

                results["robustness"].append({
                    "noise_sigma": sigma,
                    "missing_rate": miss,
                    "t_event": t_p,
                    "jaccard_baseline": float(jac_base),
                    "jaccard_regularized": float(jac_reg),
                    "fid_baseline": float(expl_p["fidelity"]),
                    "fid_regularized": float(expl_p_reg["fidelity"])
                })

    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results["robustness"]:
        df_r = pd.DataFrame(results["robustness"]).sort_values(["noise_sigma", "missing_rate"])
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

    print(meta)
    print("method:", expl_base["method"])
    print("fidelity:", expl_base["fidelity"])
    print("nodes:", expl_base["explanation_nodes"])
    print("edges:", expl_base["explanation_edges"])

if __name__ == "__main__":
    main()
