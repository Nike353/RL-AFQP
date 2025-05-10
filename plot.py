#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(qp_path="qp_log.npz", l1_path="l1_log.npz"):
    # --- load logs ---
    if not os.path.isfile(qp_path):
        raise FileNotFoundError(f"Baseline log not found: {qp_path}")
    if not os.path.isfile(l1_path):
        raise FileNotFoundError(f"L1 log not found:      {l1_path}")

    qp = np.load(qp_path)
    l1 = np.load(l1_path)

    # Baseline QP data
    t_qp   = qp["time"]            # (N,)
    grf_qp = qp["grf_nominal"]     # (N,12) or (N,) if only fz logged
    fz0_qp = grf_qp[:,2] if grf_qp.ndim>1 else grf_qp

    # L1 data
    t_l1     = l1["time"]            # (M_raw,)
    grf_nom  = l1["grf_nominal"]     # (M_raw,12)
    grf_corr = l1["grf_corrected"]   # (M_raw,12)
    d_hat    = l1["d_hat"]           # (M_raw,6)
    delta    = l1["delta"]           # (M_raw,6)
    fz0_nom  = grf_nom[:,2]
    fz0_corr = grf_corr[:,2]

    # --- 1) Align lengths of each series ---
    # baseline
    n_b = min(len(t_qp), len(fz0_qp))
    t_qp   = t_qp[:n_b]
    fz0_qp = fz0_qp[:n_b]

    # L1
    n_l = min(len(t_l1), len(fz0_nom), len(fz0_corr),
              len(d_hat), len(delta), grf_nom.shape[0], grf_corr.shape[0])
    t_l1     = t_l1[:n_l]
    fz0_nom  = fz0_nom[:n_l]
    fz0_corr = fz0_corr[:n_l]
    d_hat    = d_hat[:n_l, :]
    delta    = delta[:n_l, :]

    # also trim the full GRF arrays before reshaping
    grf_nom  = grf_nom[:n_l, :]
    grf_corr = grf_corr[:n_l, :]

    # --- 1) Foot‑0 vertical force ---
    plt.figure()
    plt.plot(t_qp,    fz0_qp,   label="QP only")
    plt.plot(t_l1,    fz0_nom,  label="L1 nominal")
    plt.plot(t_l1,    fz0_corr, '--', label="L1 corrected")
    plt.xlabel("Time [s]")
    plt.ylabel("Foot 0 $f_z$ [N]")
    plt.title("Vertical GRF on Foot 0")
    plt.legend()
    plt.grid(True)

    # --- 2) Disturbance estimates (\hat d vs \tilde d) ---
    names = ["aₓ","a_y","a_z","αₓ","α_y","α_z"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t_l1, d_hat[:, i],   label=r"$\hat d$")
        ax.plot(t_l1, delta[:, i],   '--', label=r"$\tilde d$")
        ax.set_title(names[i])
        ax.set_xlabel("Time [s]")
        if i % 3 == 0:
            ax.set_ylabel("Acceleration [unit/s²]")
        ax.legend()
        ax.grid(True)
    fig.suptitle("L₁ Disturbance Estimates vs Filtered Estimates")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 3) Norm of disturbance estimate vs filtered ---
    norm_hat   = np.linalg.norm(d_hat,  axis=1)
    norm_delta = np.linalg.norm(delta, axis=1)
    plt.figure()
    plt.plot(t_l1, norm_hat,   label=r"$\|\hat d\|$")
    plt.plot(t_l1, norm_delta, '--', label=r"$\|\tilde d\|$")
    plt.xlabel("Time [s]")
    plt.ylabel("2‑Norm [unit/s²]")
    plt.title("Disturbance Estimate Norm")
    plt.legend()
    plt.grid(True)

    # --- 4) Heatmap of GRF differences across all 4 feet ---
    # Now grf_nom and grf_corr are (n_l, 12), so safe to reshape into (n_l,4,3)
    M = n_l
    feet_nom  = grf_nom.reshape(M, 4, 3)
    feet_corr = grf_corr.reshape(M, 4, 3)
    fz_diff   = feet_corr[:, :, 2] - feet_nom[:, :, 2]  # (M,4)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        fz_diff.T,
        aspect='auto',
        origin='lower',
        extent=[t_l1[0], t_l1[-1], 0, 4]
    )
    plt.colorbar(label=r"$\Delta f_z$")
    plt.yticks(
        [0.5, 1.5, 2.5, 3.5],
        ["Foot0", "Foot1", "Foot2", "Foot3"]
    )
    plt.xlabel("Time [s]")
    plt.title("Corrected – Nominal $f_z$ per Foot")
    plt.grid(False)

    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot QP vs L1 logs")
    p.add_argument(
        "--baseline", "-b",
        default="qp_log.npz",
        help="npz file with QP baseline log"
    )
    p.add_argument(
        "--l1", "-l",
        default="l1_log.npz",
        help="npz file with L1 adaptive log"
    )
    args = p.parse_args()
    main(args.baseline, args.l1)
