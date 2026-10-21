"""Exploratory analysis and preprocessing of the Wisconsin Diagnostic Breast Cancer data.

One reproducible pipeline covering: distributions and feature relationships,
descriptive statistics, min-max normalisation, smoothing by equal-frequency
binning, IQR-based outlier analysis (overall and per diagnosis), and the effect
of mean imputation on a feature's distribution.

    python src/eda.py --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"


def load() -> pd.DataFrame:
    d = load_breast_cancer()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df["diagnosis"] = np.where(d.target == 1, "benign", "malignant")
    return df


def describe(x: pd.Series) -> dict:
    q1, q3 = x.quantile([0.25, 0.75])
    return {
        "mean": x.mean(), "harmonic_mean": stats.hmean(x), "median": x.median(), "mode": x.mode().iloc[0],
        "std": x.std(), "skewness": stats.skew(x), "min": x.min(), "Q1": q1, "Q3": q3, "max": x.max(),
    }


def iqr_mask(x: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = x.quantile([0.25, 0.75])
    return x.between(q1 - k * (q3 - q1), q3 + k * (q3 - q1))


def bin_smooth(x: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Equal-frequency binning, smoothed by bin means, medians and nearest boundaries."""
    order = np.argsort(x)
    out = {k: np.empty_like(x, dtype=float) for k in ("mean", "median", "boundary")}
    for idx in np.array_split(order, n_bins):
        v = x[idx]
        lo, hi = v.min(), v.max()
        out["mean"][idx] = v.mean()
        out["median"][idx] = np.median(v)
        out["boundary"][idx] = np.where(v - lo <= hi - v, lo, hi)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load()
    report: dict = {"n_samples": len(df), "class_counts": df["diagnosis"].value_counts().to_dict()}

    # 1. Distribution and relationships ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for lab, c in (("benign", BLUE), ("malignant", ORANGE)):
        ax.hist(df.loc[df.diagnosis == lab, "mean radius"], bins=30, alpha=0.6, color=c, label=lab)
    ax.set(xlabel="mean radius", ylabel="count", title="Mean radius by diagnosis")
    ax.legend(frameon=False)
    fig.savefig(args.out / "radius_by_diagnosis.png", dpi=150, bbox_inches="tight")

    feats = ["mean perimeter", "mean smoothness", "mean compactness", "mean concavity"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, f in zip(axes, feats):
        for lab, c in (("benign", BLUE), ("malignant", ORANGE)):
            s = df[df.diagnosis == lab]
            ax.scatter(s[f], s["mean texture"], s=8, alpha=0.5, color=c, label=lab)
        ax.set_xlabel(f)
    axes[0].set_ylabel("mean texture")
    axes[0].legend(frameon=False)
    fig.savefig(args.out / "texture_relationships.png", dpi=150, bbox_inches="tight")

    corr = df.drop(columns="diagnosis").iloc[:, :10].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(10), [c.replace("mean ", "") for c in corr.columns], rotation=60, ha="right")
    ax.set_yticks(range(10), [c.replace("mean ", "") for c in corr.columns])
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Correlation of the ten 'mean' features")
    fig.savefig(args.out / "correlation.png", dpi=150, bbox_inches="tight")
    report["corr_radius_perimeter"] = corr.loc["mean radius", "mean perimeter"]
    report["corr_radius_area"] = corr.loc["mean radius", "mean area"]

    # 2. Descriptive statistics and normalisation ------------------------------
    r = df["mean radius"]
    report["mean_radius_stats"] = describe(r)
    norm = (r - r.min()) / (r.max() - r.min())
    report["normalised_range"] = [norm.min(), norm.max()]

    # 3. Smoothing by binning -------------------------------------------------
    area = df["mean area"].to_numpy()
    sm = bin_smooth(area, args.bins)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for ax, (name, v) in zip(axes, [("original", area), *sm.items()]):
        ax.hist(v, bins=30, color=BLUE)
        ax.set_title(f"{name}" + ("" if name == "original" else f" ({args.bins} bins)"))
        ax.set_xlabel("mean area")
    fig.savefig(args.out / "binning_smoothing.png", dpi=150, bbox_inches="tight")
    report["binning_rmse"] = {k: float(np.sqrt(np.mean((v - area) ** 2))) for k, v in sm.items()}

    # 4. Outliers --------------------------------------------------------------
    keep = iqr_mask(df["mean perimeter"])
    report["perimeter_outliers_removed"] = int((~keep).sum())
    tex_b = df.loc[df.diagnosis == "benign", "mean texture"]
    tex_all = df["mean texture"]
    report["texture_outliers"] = {"pooled": int((~iqr_mask(tex_all)).sum()),
                                  "within_benign": int((~iqr_mask(tex_b)).sum())}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([df.loc[df.diagnosis == "malignant", "mean perimeter"], df.loc[df.diagnosis == "benign", "mean perimeter"]],
               labels=["malignant", "benign"])
    ax.set(ylabel="mean perimeter", title="Mean perimeter by diagnosis")
    fig.savefig(args.out / "perimeter_boxplot.png", dpi=150, bbox_inches="tight")
    # are the "outliers" errors or simply malignant tumours?
    report["perimeter_outliers_malignant_share"] = float((df.loc[~keep, "diagnosis"] == "malignant").mean())

    # 5. Mean imputation --------------------------------------------------------
    t = tex_all.copy()
    masked = t.between(17, 20)
    imputed = t.mask(masked).fillna(t.mask(masked).mean())
    report["imputation"] = {
        "values_masked": int(masked.sum()), "std_before": t.std(), "std_after": imputed.std(),
        "fill_value": float(t.mask(masked).mean()),
    }
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(t, bins=40, alpha=0.6, color=GREY, label="original")
    ax.hist(imputed, bins=40, alpha=0.6, color=ORANGE, label="after masking 17–20 and mean imputation")
    ax.set(xlabel="mean texture", ylabel="count", title="Mean imputation creates an artificial spike")
    ax.legend(frameon=False)
    fig.savefig(args.out / "mean_imputation.png", dpi=150, bbox_inches="tight")

    clean = json.loads(json.dumps(report, default=float))
    (args.out / "summary.json").write_text(json.dumps(clean, indent=2))
    print(json.dumps(clean, indent=2))


if __name__ == "__main__":
    main()
