# tuner.py
# Generate + tune a synthetic TB dataset so your Naive Bayes model matches your rule-based testcases.

import os
import numpy as np
import pandas as pd

from data_generator import (
    TESTS,
    create_dataset,
    evaluate_df,
)


# ---------------------------
# 4) Tuning routine
# ---------------------------
def tune_params(
    iters: int = 300,
    seed: int = 42,
    tb_prevalence: float = 0.55,
    n_eval: int = 4000,
) -> dict:
    base = dict(
        tb_systemic_frac=0.30,
        p_cough_tb_pulm=0.95,
        p_cough_tb_sys=0.75,
        p_cough_notb=0.35,
        p_c2_tb_pulm=0.85,
        p_c2_tb_sys=0.20,
        p_c2_notb=0.20,
        p_blood_tb_pulm=0.50,
        p_blood_tb_sys=0.10,
        p_blood_notb=0.05,
        p_bp_tb_pulm=0.55,
        p_bp_tb_sys=0.25,
        p_bp_notb=0.15,
        p_cp_tb_pulm=0.45,
        p_cp_tb_sys=0.20,
        p_cp_notb=0.10,
        p_fever_tb_pulm=0.55,
        p_fever_tb_sys=0.80,
        p_fever_notb=0.20,
        p_ns_tb_pulm=0.50,
        p_ns_tb_sys=0.75,
        p_ns_notb=0.15,
        p_fat_tb_pulm=0.60,
        p_fat_tb_sys=0.75,
        p_fat_notb=0.30,
        p_w_tb_pulm=0.65,
        p_w_tb_sys=0.25,
        p_w_notb=0.10,
        p_loa_tb_pulm=0.55,
        p_loa_tb_sys=0.35,
        p_loa_notb=0.15,
        p_conta_tb_pulm=0.55,
        p_conta_tb_sys=0.20,
        p_conta_notb=0.05,
    )

    search_space = {
        "tb_systemic_frac": (0.15, 0.55),
        "p_c2_tb_sys": (0.02, 0.40),
        "p_w_tb_sys": (0.02, 0.45),
        "p_conta_tb_sys": (0.02, 0.45),
        "p_fever_notb": (0.10, 0.45),
        "p_ns_notb": (0.08, 0.40),
        "p_fat_notb": (0.15, 0.50),
        "p_blood_tb_sys": (0.02, 0.25),
        "p_cp_tb_sys": (0.05, 0.35),
    }

    rng = np.random.default_rng(0)

    def sample_params():
        p = base.copy()
        for k, (lo, hi) in search_space.items():
            p[k] = float(rng.uniform(lo, hi))

        # keep anchors stable
        p["p_c2_tb_pulm"] = 0.85
        p["p_c2_notb"] = 0.20
        p["p_cough_tb_pulm"] = 0.95
        p["p_cough_notb"] = 0.35
        p["p_blood_tb_pulm"] = 0.50
        p["p_blood_notb"] = 0.05
        return p

    best_score = -1
    best_params = base.copy()

    # Phase A: random search
    for _ in range(iters):
        params = sample_params()
        df_try = create_dataset(n=n_eval, tb_prevalence=tb_prevalence, seed=seed, params=params)
        score, _ = evaluate_df(df_try)

        if score > best_score:
            best_score = score
            best_params = params

        if best_score == len(TESTS):
            break

    # Phase B: small grid refinement for contact_with_TB (often the last mismatch)
    p_tb_pulm_grid = [0.20, 0.30, 0.40, 0.45, 0.55]
    p_notb_grid = [0.05, 0.10, 0.15, 0.20, 0.25]
    p_tb_sys_grid = [0.02, 0.05, 0.10, 0.15]

    best_objective = None
    best_params2 = best_params

    for p_pulm in p_tb_pulm_grid:
        for p_notb in p_notb_grid:
            for p_sys in p_tb_sys_grid:
                params = best_params.copy()
                params["p_conta_tb_pulm"] = p_pulm
                params["p_conta_notb"] = p_notb
                params["p_conta_tb_sys"] = p_sys

                df_try = create_dataset(n=n_eval, tb_prevalence=tb_prevalence, seed=seed, params=params)
                score, rows = evaluate_df(df_try)

                conta_row = [r for r in rows if r[0] == "c,c2,conta"][0]
                p_conta = float(conta_row[3])
                pred_conta = conta_row[2]
                penalty = 0 if pred_conta == "Moderate" else 1
                objective = (score, -penalty, -abs(p_conta - 0.5))

                if (best_objective is None) or (objective > best_objective):
                    best_objective = objective
                    best_params2 = params

    return best_params2


def generate_tuned_csv(
    out_csv: str = "data/tb_training_data_tuned.csv",
    n_final: int = 8000,
    tb_prevalence: float = 0.55,
    seed: int = 42,
    iters: int = 300,
) -> str:
    params = tune_params(iters=iters, seed=seed, tb_prevalence=tb_prevalence, n_eval=4000)
    df_final = create_dataset(n=n_final, tb_prevalence=tb_prevalence, seed=seed, params=params)
    score, rows = evaluate_df(df_final)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_final.to_csv(out_csv, index=False)

    print("✅ Tuned dataset generated")
    print("Output:", out_csv)
    print("TB prevalence:", f"{df_final['TB'].mean():.2%}")
    print("Passed:", f"{score}/{len(TESTS)}")
    print("\nTestcase results:")
    for toks, exp, pred, p, ok in rows:
        print(f"  {toks:<14}  p={p:.3f}  pred={pred:<8}  exp={exp:<8}  {'✅' if ok else '❌'}")

    print("\nKey tuned params:")
    for k in [
        "tb_systemic_frac",
        "p_c2_tb_sys",
        "p_w_tb_sys",
        "p_fever_notb",
        "p_ns_notb",
        "p_fat_notb",
        "p_conta_tb_pulm",
        "p_conta_tb_sys",
        "p_conta_notb",
    ]:
        print(f"  {k}: {params[k]}")

    return out_csv


def main():
    generate_tuned_csv(
        out_csv="data/tb_training_data_tuned.csv",
        n_final=8000,
        tb_prevalence=0.55,
        seed=42,
        iters=300,
    )


if __name__ == "__main__":
    main()
