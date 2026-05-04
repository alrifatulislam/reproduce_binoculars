"""Generate all figures and tables from saved JSON results."""
import argparse, os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
mpl.rcParams["figure.dpi"] = 200; sns.set(style="darkgrid")

def J(p): 
    with open(p) as f: return json.load(f)

def plot_fig1(r, path):
    ds_order = ["News", "Creative Writing", "Student Essay"]
    dets = list(next(iter(r.values())).keys())
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bw = 0.8 / len(dets); x = np.arange(len(ds_order))
    for i, d in enumerate(dets):
        vals = [r.get(ds, {}).get(d, {}).get("tpr_at_fpr", 0) if isinstance(r.get(ds, {}).get(d, {}), dict) else 0 for ds in ds_order]
        bars = ax.bar(x + i*bw - 0.4 + bw/2, vals, bw, label=d, color=colors[i % len(colors)])
        for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax.set_ylabel("TPR @ 0.01% FPR"); ax.set_xticks(x); ax.set_xticklabels(ds_order)
    ax.set_ylim(0, 1.15); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(dets), fontsize=8)
    ax.set_title("Figure 1: Detection of ChatGPT Text", fontweight='bold')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.show(); print(f"Saved {path}")

def plot_fig2(r, path):
    ds_order = ["News", "Creative Writing", "Student Essay"]
    mk = {'Binoculars (Ours)': 'o', 'Fast-DetectGPT': 's'}
    co = {'Binoculars (Ours)': '#d62728', 'Fast-DetectGPT': '#2ca02c'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for idx, ds in enumerate(ds_order):
        ax = axes[idx]; ax.set_title(ds, fontweight='bold')
        for det in r.get(ds, {}):
            xs = sorted([int(k) for k in r[ds][det]]); ys = [r[ds][det][str(t)] for t in xs]
            ax.plot(xs, ys, marker=mk.get(det, 'o'), color=co.get(det, 'gray'), label=det, lw=2, ms=8)
        ax.set_xlabel("Document Size (tokens)"); ax.set_xticks([128, 256, 512]); ax.set_ylim(-0.05, 1.05)
        if idx == 0: ax.set_ylabel("TPR @ 0.01% FPR")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.suptitle("Figure 2: Document Size Impact", fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.show(); print(f"Saved {path}")

def plot_fig3(r, path):
    co = {'Binoculars (Ours)': '#d62728', 'Fast-DetectGPT': '#2ca02c', 'DetectGPT': '#1f77b4', 'DNA-GPT': '#ff7f0e'}
    ls = {'Binoculars (Ours)': '-', 'Fast-DetectGPT': '--', 'DetectGPT': '-.', 'DNA-GPT': ':'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for idx, ds in enumerate(["PubMed", "CC News", "CNN"]):
        ax = axes[idx]; ax.set_title(ds, fontweight='bold'); ax.set_xscale('log')
        for det, d in r.get(ds, {}).items():
            ax.plot(d["fpr"], d["tpr"], color=co.get(det, 'gray'), ls=ls.get(det, '-'), label=f'{det} (AUC={d["auc"]:.3f})', lw=2)
        ax.set_xlabel("FPR"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc='lower right')
        if idx == 0: ax.set_ylabel("TPR")
    fig.suptitle("Figure 3: Detecting LLaMA-2-13B", fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.show(); print(f"Saved {path}")

def plot_fig7(r, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r["original"], bins=40, alpha=0.6, label="Original Essay", color="#1f77b4")
    ax.hist(r["corrected"], bins=40, alpha=0.6, label="Corrected Essay", color="#ff7f0e")
    ax.axvline(0.9015, color='red', ls='--', lw=2, label="Binoculars Threshold")
    ax.set_xlabel("Binoculars Score"); ax.set_ylabel("Count"); ax.legend()
    ax.set_title("Figure 7: ESL Essay Score Distribution", fontweight='bold')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.show(); print(f"Saved {path}")

def plot_fig8(r, path):
    styles = [k for k in r.keys()]
    fnrs = [r[k]["false_negative_rate"] * 100 for k in styles]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(styles, fnrs, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    for b, v in zip(bars, fnrs): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1f}%', ha='center', fontsize=9)
    ax.set_ylabel("False Negative Rate (%)"); ax.set_ylim(0, max(fnrs)*1.3 + 5)
    ax.set_title("Figure 8: Modified Prompting Strategies", fontweight='bold')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.show(); print(f"Saved {path}")

def plot_ext1(r, path):
    models = list(r.keys())
    accs = [r[m]["accuracy"] * 100 for m in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, accs, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    for b, v in zip(bars, accs): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontsize=9)
    ax.set_ylabel("Detection Accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Extension 1: Binoculars on Newer LLMs", fontweight='bold')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.show(); print(f"Saved {path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dir", default="results"); a = p.parse_args()
    d = a.dir
    for name, fn in [("figure1_results.json", lambda r: plot_fig1(r, f"{d}/figure1.png")),
                     ("figure2_results.json", lambda r: plot_fig2(r, f"{d}/figure2.png")),
                     ("figure3_results.json", lambda r: plot_fig3(r, f"{d}/figure3.png")),
                     ("figure7_results.json", lambda r: plot_fig7(r, f"{d}/figure7.png")),
                     ("figure8_results.json", lambda r: plot_fig8(r, f"{d}/figure8.png")),
                     ("ext1_results.json",    lambda r: plot_ext1(r, f"{d}/ext1.png"))]:
        fp = os.path.join(d, name)
        if os.path.exists(fp): print(f"\n>>> {name}"); fn(J(fp))
