#!/usr/bin/env python3
"""Plot a clearly labeled partial Spin(23) scan against the candidate ansatz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS=("#0072B2","#D55E00","#009E73")


def _identity_limits(x: np.ndarray,y: np.ndarray) -> tuple[float,float]:
    lower=float(min(np.min(x),np.min(y)))
    upper=float(max(np.max(x),np.max(y)))
    padding=0.04*max(upper-lower,1.0e-12)
    return lower-padding,upper+padding


def _ecdf(values: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    ordered=np.sort(values)
    return ordered,np.arange(1,ordered.size+1)/ordered.size


def _point_indices(point_ids: pd.Series) -> np.ndarray:
    """Recover the within-family manifest index encoded in each point ID."""
    try:
        return np.asarray([int(value.split("-")[1]) for value in point_ids],dtype=int)
    except (IndexError,ValueError):
        return np.arange(len(point_ids))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_csv",type=Path)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--tag",default="partial")
    args=parser.parse_args()

    frame=pd.read_csv(args.merged_csv)
    if frame.empty:
        raise SystemExit("the merged scan contains no rows")
    if frame["point_id"].duplicated().any():
        raise SystemExit("the merged scan contains duplicate point_id values")
    output_dir=args.output_dir or args.merged_csv.parent/"plots"
    output_dir.mkdir(parents=True,exist_ok=True)

    plt.rcParams.update({
        "font.family":"serif",
        "font.serif":["Computer Modern Roman","CMU Serif","DejaVu Serif"],
        "mathtext.fontset":"cm",
        "axes.labelsize":12,
        "axes.titlesize":12,
        "xtick.labelsize":10,
        "ytick.labelsize":10,
        "legend.fontsize":10,
    })

    # Direct numerical-versus-ansatz comparison for both complex components.
    figure,axes=plt.subplots(2,3,figsize=(11.2,6.8),constrained_layout=True)
    for channel in range(1,4):
        numerical_re=frame[f"M{channel}_re"].to_numpy()
        numerical_im=frame[f"M{channel}_im"].to_numpy()
        candidate_re=frame[f"candidate_M{channel}_re"].to_numpy()
        candidate_im=frame[f"candidate_M{channel}_im"].to_numpy()
        relative=frame[f"formula_rel_residual_M{channel}"].to_numpy()
        for row,(candidate,numerical,part) in enumerate((
            (candidate_re,numerical_re,r"\operatorname{Re}"),
            (candidate_im,numerical_im,r"\operatorname{Im}"),
        )):
            axis=axes[row,channel-1]
            limits=_identity_limits(candidate,numerical)
            axis.plot(limits,limits,color="black",linewidth=1.2,zorder=1)
            axis.scatter(
                candidate,numerical,s=12,marker="o",color=COLORS[channel-1],
                alpha=0.55,edgecolors="none",rasterized=True,zorder=2,
            )
            axis.set_xlim(limits); axis.set_ylim(limits)
            axis.set_aspect("equal",adjustable="box")
            axis.set_xlabel(rf"${part}\,M_{channel}^{{\rm ansatz}}$")
            axis.set_ylabel(rf"${part}\,M_{channel}^{{\rm numerical}}$")
            if row == 0:
                axis.set_title(
                    rf"$M_{channel}$: max relative residual "
                    rf"${np.max(relative):.2e}$"
                )
            axis.grid(alpha=0.18,linewidth=0.6)
    figure.suptitle(
        f"Spin(23) 1-to-3 ansatz check: {len(frame)} completed scan points",
        fontsize=14,
    )
    identity_base=output_dir/f"spin23_{args.tag}_ansatz_identity"
    figure.savefig(identity_base.with_suffix(".png"),dpi=240,bbox_inches="tight")
    figure.savefig(identity_base.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(figure)

    # Residual size, truncation diagnostic, and empirical residual distribution.
    figure,axes=plt.subplots(1,3,figsize=(13.0,3.9),constrained_layout=True)
    all_relative=[]
    all_ratios=[]
    point_indices=_point_indices(frame["point_id"])
    for channel in range(1,4):
        relative=frame[f"formula_rel_residual_M{channel}"].to_numpy()
        absolute=frame[f"formula_abs_residual_M{channel}"].to_numpy()
        q_change=frame[f"estimated_abs_error_M{channel}"].to_numpy()
        ratio=absolute/np.maximum(q_change,np.finfo(float).tiny)
        all_relative.append(relative)
        all_ratios.append(ratio)
        label=rf"$M_{channel}$"
        axes[0].scatter(
            point_indices,relative,s=10,alpha=0.55,
            edgecolors="none",color=COLORS[channel-1],label=label,rasterized=True,
        )
        axes[1].scatter(
            q_change,absolute,s=10,alpha=0.55,
            edgecolors="none",color=COLORS[channel-1],label=label,rasterized=True,
        )
        x_ecdf,y_ecdf=_ecdf(relative)
        axes[2].step(x_ecdf,y_ecdf,where="post",color=COLORS[channel-1],label=label)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Core Sobol-point index")
    axes[0].set_ylabel(r"$|M_i^{\rm numerical}-M_i^{\rm ansatz}|/|M_i^{\rm ansatz}|$")
    axes[0].grid(alpha=0.18,which="both",linewidth=0.6)
    axes[0].legend(frameon=False)

    finite_positive=np.concatenate([
        frame[f"estimated_abs_error_M{channel}"].to_numpy()
        for channel in range(1,4)
    ])
    absolute=np.concatenate([
        frame[f"formula_abs_residual_M{channel}"].to_numpy()
        for channel in range(1,4)
    ])
    lower=min(np.min(finite_positive),np.min(absolute))
    upper=max(np.max(finite_positive),np.max(absolute))
    axes[1].plot([lower,upper],[lower,upper],color="black",linewidth=1.2,label="equal")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlim(lower,upper); axes[1].set_ylim(lower,upper)
    axes[1].set_aspect("equal",adjustable="box")
    axes[1].set_xlabel(r"Adjacent-$q$ absolute change")
    axes[1].set_ylabel("Ansatz absolute residual")
    axes[1].grid(alpha=0.18,which="both",linewidth=0.6)

    axes[2].axvline(1.0e-6,color="black",linestyle="--",linewidth=1.2)
    axes[2].set_xscale("log")
    axes[2].set_xlim(left=min(np.concatenate(all_relative))*.8)
    axes[2].set_ylim(0,1.01)
    axes[2].set_xlabel("Relative ansatz residual")
    axes[2].set_ylabel("Empirical cumulative fraction")
    axes[2].grid(alpha=0.18,which="both",linewidth=0.6)
    axes[2].legend(frameon=False)

    figure.suptitle(
        f"Partial scan residual diagnostics ({len(frame)} core/production points)",
        fontsize=14,
    )
    residual_base=output_dir/f"spin23_{args.tag}_ansatz_residuals"
    figure.savefig(residual_base.with_suffix(".png"),dpi=240,bbox_inches="tight")
    figure.savefig(residual_base.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(figure)

    relative=np.concatenate(all_relative)
    ratios=np.concatenate(all_ratios)
    summary={
        "partial":True,
        "completed_points":int(len(frame)),
        "families":frame.groupby(["family","setting"]).size().to_dict(),
        "all_channel_relative_residual":{
            "median":float(np.median(relative)),
            "p90":float(np.percentile(relative,90)),
            "p99":float(np.percentile(relative,99)),
            "max":float(np.max(relative)),
        },
        "fraction_ansatz_residual_below_adjacent_q_change":float(np.mean(ratios <= 1)),
        "fraction_ansatz_residual_below_twice_adjacent_q_change":float(np.mean(ratios <= 2)),
    }
    serializable=dict(summary)
    serializable["families"]={
        f"{family}/{setting}":int(count)
        for (family,setting),count in summary["families"].items()
    }
    summary_path=output_dir/f"spin23_{args.tag}_ansatz_summary.json"
    summary_path.write_text(json.dumps(serializable,indent=2)+"\n")
    print(json.dumps(serializable,indent=2))
    print(identity_base.with_suffix(".png"))
    print(residual_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
