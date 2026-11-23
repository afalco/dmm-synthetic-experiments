"""
make_figures_v2.py -- draws the two figures of Section 8 from the CSV files in ./results_v2
(run run_experiments_v2.py first). Output: ./figures_v2/*.pdf
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter, FuncFormatter
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results_v2"); FIG=os.path.join(HERE,"figures_v2"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":8,"axes.titlesize":8,"axes.labelsize":8,"xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":6,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.titlelocation":"left"})
def letter(ax,s): ax.text(-0.18,1.06,s,transform=ax.transAxes,fontsize=10,fontweight="bold",va="bottom")

def fig_theory():
    A1=pd.read_csv(f"{RES}/A1_rate.csv"); A2=pd.read_csv(f"{RES}/A2_dimension.csv"); A3=pd.read_csv(f"{RES}/A3_rare.csv"); A4=pd.read_csv(f"{RES}/A4_imbalance.csv")
    cols={5.0:"#1b9e77",20.0:"#7570b3",80.0:"#d95f02"}
    fig,axs=plt.subplots(1,4,figsize=(7.2,2.15),constrained_layout=True)
    ax=axs[0]
    for mb,g in A1.groupby("mbar"):
        ax.errorbar(g.n,g.op_mean,yerr=g.op_sd,fmt="o-",ms=2.5,lw=1,color=cols[mb])
        ax.plot(g.n,g.upper,"--",lw=0.8,color=cols[mb])
        ax.text(g.n.iloc[-1]*1.6,g.op_mean.iloc[-1],fr"$\bar m$={int(mb)}",color=cols[mb],va="center",fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(150,1.2e6); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("sample size $n$"); ax.set_ylabel(r"$\|\rho_{\mathcal{D}}-\rho_{\mathcal{D},\star}\|_2$"); ax.set_title("Error decays as $n^{-1/2}$")
    ax=axs[1]; g1=A2[A2.design.str.startswith("q")]; g2=A2[A2.design.str.startswith("mbar")]
    ax.errorbar(g1.d,g1.op_mean,yerr=g1.op_sd,fmt="o-",ms=2.5,lw=1,color="#444444")
    ax.errorbar(g2.d,g2.op_mean,yerr=g2.op_sd,fmt="s-",ms=2.5,lw=1,color="#d95f02")
    ax.text(820,g1.op_mean.iloc[-1]*0.82,r"$\bar m$ fixed",fontsize=6,color="#444444",ha="right",va="top")
    ax.text(820,g2.op_mean.iloc[-1]*1.05,r"$q$ fixed",fontsize=6,color="#d95f02",ha="right",va="bottom")
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_major_locator(FixedLocator([50,100,200,400,800]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{int(v)}")); ax.margins(0.06)
    ax.set_xlabel("ambient dimension $d$"); ax.set_ylabel("operator error"); ax.set_title(r"Only $\bar m=d/q$ matters")
    ax=axs[2]
    ax.plot(A3["frac_cells_below_1e-3"],A3.op_mean,"o-",ms=3,lw=1,color="#7570b3"); ax.axhline(A3.upper.iloc[0],ls="--",lw=0.8,color="#7570b3")
    ax.text(0.17,A3.upper.iloc[0]*0.93,"Thm 5.24 bound",fontsize=6,color="#7570b3",va="top"); ax.set_ylim(0,0.19)
    ax.set_xlabel(r"fraction of cells with $p<10^{-3}$"); ax.set_ylabel("operator error"); ax.set_title("No positivity needed")
    ax=axs[3]
    ax.plot(A4.pi_min,A4.op_mean,"o-",ms=3,lw=1,color="#444444"); ax.plot(A4.pi_min,A4.st_mean,"s-",ms=3,lw=1,color="#d95f02")
    ax.text(0.011,A4.op_mean.iloc[-1]+0.012,"operator error",fontsize=6,color="#444444",va="bottom")
    ax.text(0.0135,A4.st_mean.iloc[-1],r"$\sin\Theta$",fontsize=6,color="#d95f02",va="center")
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter()); ax.margins(0.06)
    ax.set_xlabel(r"minority proportion $\pi_{\min}$"); ax.set_ylabel("error"); ax.set_title("Imbalance acts via gap")
    for a,l in zip(axs,"abcd"): letter(a,l)
    fig.savefig(f"{FIG}/fig_theory_validation.pdf",bbox_inches="tight",pad_inches=0.03); plt.close(fig)

def fig_classifier():
    A5=pd.read_csv(f"{RES}/A5_classifier.csv"); c2={"fixed h=0.2":"#1b9e77","h=s/5":"#d95f02"}
    fig,axs=plt.subplots(1,2,figsize=(5.2,2.15),constrained_layout=True)
    for reg,g in A5.groupby("regime"):
        fixed=reg.startswith("fixed")
        lab="fixed $h=0.2$ (target $g^\\star_h$)" if fixed else r"$h=\varsigma/5$ (target $g^{\rm lat}$)"
        axs[0].plot(g.n,g.agree_smoothed if fixed else g.agree_latBayes,"o-",ms=3,lw=1,color=c2[reg],label=lab)
        axs[1].errorbar(g.n,g.risk_mean,yerr=g.risk_sd,fmt="o-",ms=3,lw=1,color=c2[reg])
        axs[1].axhline(g.L_h.iloc[0] if fixed else g.L_lat.iloc[0],ls="--",lw=0.8,color=c2[reg])
    for a in axs: a.set_xscale("log"); a.xaxis.set_minor_formatter(NullFormatter()); a.set_xlabel("sample size $n$")
    axs[0].set_ylabel("fraction of atoms on target"); axs[0].set_title("Classifier recovers its target rule"); axs[0].legend(frameon=False,loc="lower right")
    axs[1].set_ylabel("misclassification risk"); axs[1].set_title("Risk converges to the target risk")
    axs[1].text(1.5e5,A5.L_h.iloc[0]+0.003,r"$L(g^\star_h)$",fontsize=6,color=c2["fixed h=0.2"])
    axs[1].text(1.5e5,A5.L_lat.iloc[0]+0.003,r"$L^{\rm lat}=L^\star$",fontsize=6,color=c2["h=s/5"])
    for a,l in zip(axs,"ab"): letter(a,l)
    fig.savefig(f"{FIG}/fig_classifier_regimes.pdf",bbox_inches="tight",pad_inches=0.03); plt.close(fig)

if __name__=="__main__":
    fig_theory(); fig_classifier(); print("figures written to", FIG)
