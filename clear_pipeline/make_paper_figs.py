"""make_paper_figs.py — MSWiM (Topaz) figures on the 18-min CU + random DU data.

Final results use the 18-min CU anomaly capture (CU_*_random) and the random DU
capture (DU_*_random), all v0 no-net_ratio, frozen p99.9. Produces:
  fig_pr_roc.png        PR+ROC pooled, CU vs DU (+ prints AP/AUC)
  fig_det_heatmap.png   per-stress-type heatmaps (CPU/MEM/NET), CU & DU x topology, scale 0.90-1.0
  fig_ablation_cu.png   CU ablation (Full + A1..A5) re-evaluated on 18-min CU
  fig_traffic_72h.png   aggregate DU traffic over the real 72h normal trace
Also prints the ANY (system-level) detection F1 per topology for the LaTeX table.
"""
import csv, io, contextlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import run_experiment_v0_baseline as v0
from model_calibrated import CalibratedTopoAR
from run_noratio_ablations import slice_noratio, slice_noratio_noNnorm, ABLATIONS

FIG = Path("/home/somya/workspace/mswim2026/figures")
TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]   # N=2, N=1, N=3
NLAB = ["N=2", "N=1", "N=3"]
K = v0.COLD_START_K
CU_CFG=[("CPU","CU_CPU_random_STRESS",1),("MEM","CU_MEM_random_STRESS",2),("NET","CU_NET_random_STRESS",3)]
DU_CFG=[("CPU","DU_CPU_random_STRESS",1),("MEM","DU_MEM_random_STRESS",2),("NET","DU_NET_random_STRESS",3)]

def patch(model_cls=CalibratedTopoAR, slice_fn=slice_noratio, ckpt="v0_noratio_random_", save=False):
    v0._CU_FEAT_NAMES=["cpu","mem_pct","mem_bytes","net_tx","net_rx","net_diff"]
    v0._DU_FEAT_NAMES=(["cpu","mem_pct","mem_bytes","fs_writes","net_tx","net_rx"]+[f"pci_{i}" for i in range(22)]+["net_diff"])
    v0._RC_FEAT_GROUPS={1:{"CU":{0},"DU":{0}},2:{"CU":{2},"DU":{2}},3:{"CU":{3,4,5},"DU":{4,5,28}}}
    v0.slice_features=slice_fn; v0.CalibratedTopoAR=model_cls; v0.CKPT_PREFIX=ckpt
    v0.CU_THRESHOLD_PCT=99.9; v0.DU_THRESHOLD_PCT=99.9; v0.SAVE_ERRORS=save

def evalone(bd,stype,topo):
    v0.BASE_DIR=Path(bd); v0.STRESS_TYPE=stype
    train=[x for x in v0.ALL_TOPOS if x!=topo]
    with contextlib.redirect_stdout(io.StringIO()):
        return v0.run_one(train,topo)

# ---------- collect Full results + scores (SAVE_ERRORS) ----------
def collect_full():
    patch(save=True)
    F={}  # (ent,st,topo)-> dict(cu/du metric)
    cu_s,cu_y,du_s,du_y=[],[],[],[]
    for st,bd,stype in CU_CFG:
        sl="_".join(Path(bd).name.split("_")[:2]+([v0.DATA_VARIANT] if v0.DATA_VARIANT else []))
        for t in TOPOS:
            m=evalone(bd,stype,t); F[("CU",st,t)]=m
            z=np.load(Path(v0.__file__).parent/"exp_runs"/f"{sl}_my_model"/f"recon_errors_{t}_f6.npz")
            s=v0.lift_score(z["cu_sqerr"][K:],z["cu_feat_norm"]); y=(z["cu_stress"][K+1:]==stype).astype(int)
            n=min(len(s),len(y)); cu_s.append(s[:n]); cu_y.append(y[:n])
    for st,bd,stype in DU_CFG:
        sl="_".join(Path(bd).name.split("_")[:2]+([v0.DATA_VARIANT] if v0.DATA_VARIANT else []))
        for t in TOPOS:
            m=evalone(bd,stype,t); F[("DU",st,t)]=m
            z=np.load(Path(v0.__file__).parent/"exp_runs"/f"{sl}_my_model"/f"recon_errors_{t}_f6.npz")
            du=z["du_sqerr"]; Nd=du.shape[1]
            s=np.stack([v0.lift_score(du[K:,i,:],z["du_feat_norm"]) for i in range(Nd)],axis=1)
            y=(z["du_stress"][K+1:]==stype).astype(int); n=min(len(s),len(y))
            du_s.append(s[:n].ravel()); du_y.append(y[:n].ravel())
    return F,(np.concatenate(cu_s),np.concatenate(cu_y),np.concatenate(du_s),np.concatenate(du_y))

def fig_prroc(cu_s,cu_y,du_s,du_y):
    cu_ap=average_precision_score(cu_y,cu_s); du_ap=average_precision_score(du_y,du_s)
    f1,t1,_=roc_curve(cu_y,cu_s); f2,t2,_=roc_curve(du_y,du_s); cu_auc=auc(f1,t1); du_auc=auc(f2,t2)
    fig,(axp,axr)=plt.subplots(1,2,figsize=(9,3.6))
    for s,y,l,c in [(cu_s,cu_y,f"CU (AP={cu_ap:.3f})","#1f77b4"),(du_s,du_y,f"DU (AP={du_ap:.3f})","#2ca02c")]:
        pr,rc,_=precision_recall_curve(y,s); axp.plot(rc,pr,c,lw=2,label=l)
    axp.set_xlabel("Recall");axp.set_ylabel("Precision");axp.set_title("Precision-Recall");axp.set_ylim(0,1.02);axp.legend(loc="lower left",fontsize=9);axp.grid(alpha=.3)
    for s,y,l,c in [(cu_s,cu_y,f"CU (AUC={cu_auc:.3f})","#1f77b4"),(du_s,du_y,f"DU (AUC={du_auc:.3f})","#2ca02c")]:
        fpr,tpr,_=roc_curve(y,s); axr.plot(fpr,tpr,c,lw=2,label=l)
    axr.plot([0,1],[0,1],"k--",lw=.7,alpha=.5);axr.set_xlabel("FPR");axr.set_ylabel("TPR");axr.set_title("ROC");axr.legend(loc="lower right",fontsize=9);axr.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG/"fig_pr_roc.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"AUC -> CU AP={cu_ap:.3f} AUC={cu_auc:.3f} | DU AP={du_ap:.3f} AUC={du_auc:.3f}")

# number of DUs per held-out topology, and the panel titles
NDU={"cu0_du0du1":2,"cu1_du2":1,"cu2_du3du4du5":3}
PANEL_TITLE={"cu0_du0du1":"cu0+du0,du1  (N=2)","cu1_du2":"cu1+du2  (N=1)",
             "cu2_du3du4du5":"cu2+du3,du4,du5  (N=3)"}
# binned F1 colour scheme (light=worse, dark green=best); <0.92 flagged salmon
F1_BOUNDS=[0.0,0.92,0.96,0.98,0.988,1.0001]
F1_COLORS=["#edf8e9","#c7e9c0","#a1d99b","#41ab5d","#006d2c"]
F1_LABELS=["$<0.92$","$0.92$--$0.96$","$0.96$--$0.98$","$0.98$--$0.988$","$\\geq 0.988$"]

def fig_heatmaps(F):
    # one panel per held-out topology; rows = CU, DU0, DU1, ...; cols = CPU/MEM/NET.
    # CU row comes from the CU-stress experiment, DU_i rows from the DU-stress experiment.
    cmap=ListedColormap(F1_COLORS); norm=BoundaryNorm(F1_BOUNDS,cmap.N)
    nmax=max(NDU.values())+1                                   # CU + max DUs (=4)
    fig=plt.figure(figsize=(10,3.4)); gs=fig.add_gridspec(nmax,3,hspace=0.0,wspace=0.35)
    anytab={}
    for k,t in enumerate(TOPOS):
        ndu=NDU[t]; ents=["CU"]+[f"DU{i}" for i in range(ndu)]
        keys=["CU"]+[f"DU_{i}" for i in range(ndu)]
        M=np.array([[ (F[("CU",st,t)][k_][ "f1"] if k_=="CU" else F[("DU",st,t)][k_]["f1"])
                      for st in ["CPU","MEM","NET"]] for k_ in keys])
        ax=fig.add_subplot(gs[0:ndu+1,k])
        ax.imshow(M,cmap=cmap,norm=norm,aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(["CPU","MEM","NET"],fontsize=9)
        ax.set_yticks(range(ndu+1)); ax.set_yticklabels(ents,fontsize=9)
        ax.set_title(PANEL_TITLE[t],fontsize=10)
        for i in range(ndu+1):
            for j in range(3):
                ax.text(j,i,f"{M[i,j]:.3f}",ha="center",va="center",fontsize=8,
                        color="white" if M[i,j]>=0.98 else "black")
    # ANY per topo for the system-level table
    for st in ["CPU","MEM","NET"]:
        anytab[st]={"CU":[F[("CU",st,t)]["ANY"]["f1"] for t in TOPOS],
                    "DU":[F[("DU",st,t)]["ANY"]["f1"] for t in TOPOS]}
    # discrete legend across the bottom
    handles=[Patch(facecolor=c,edgecolor="0.5",label=l) for c,l in zip(F1_COLORS,F1_LABELS)]
    fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=8,frameon=False,
               title="Detection F1",title_fontsize=9,bbox_to_anchor=(0.5,-0.06))
    fig.savefig(FIG/"fig_det_heatmap.png",dpi=150,bbox_inches="tight"); plt.close()
    print("saved fig_det_heatmap.png (per-topology, per-entity)")
    return anytab

def fig_ablation():
    NO=["cu1_du2","cu0_du0du1","cu2_du3du4du5"]; cols=[f"{s}\nN={n}" for s in ["CPU","MEM","NET"] for n in [1,2,3]]
    cfg={"CPU":("CU_CPU_random_STRESS",1),"MEM":("CU_MEM_random_STRESS",2),"NET":("CU_NET_random_STRESS",3)}
    rows=["Full Topaz","A1 -topo norm","A2 fully-shared","A3 mean-pool","A4 stateless MLP","A5 -hidden LN"]
    variants=[("full",CalibratedTopoAR,slice_noratio,"v0_noratio_random_")]
    for v in ["A1","A2","A3","A4","A5"]:
        mc,sf=ABLATIONS[v]; variants.append((v,mc,sf,f"abl_{v}_noratio_random_"))
    M=[]
    for name,mc,sf,ck in variants:
        patch(mc,sf,ck)
        r=[]
        for st in ["CPU","MEM","NET"]:
            bd,stype=cfg[st]
            for t in NO:
                r.append(evalone(bd,stype,t)["CU"]["f1"])
        M.append(r)
    M=np.array(M)
    fig,ax=plt.subplots(figsize=(8,3.6)); im=ax.imshow(M,cmap="Greens",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(9)); ax.set_xticklabels(cols,fontsize=8); ax.set_yticks(range(6)); ax.set_yticklabels(rows)
    for i in range(6):
        for j in range(9):
            ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8,
                    color="white" if M[i,j]>=0.7 else "black")
    fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02,label="CU F1")
    fig.tight_layout(); fig.savefig(FIG/"fig_ablation_cu.png",dpi=150,bbox_inches="tight"); plt.close()
    print("saved fig_ablation_cu.png")

def fig_traffic():
    # Faithful workload figure: reconstruct the OFFERED aggregate UE load the
    # traffic generator actually drove, by replaying its per-cycle RNG exactly.
    # The driven load -- not the interface counters -- is what carries the diurnal
    # pattern, so this is the honest representation of the 72-hour workload.
    GLOBAL=[11.0,8.1,5.6,3.6,2.7,1.9,3.0,5.0,7.1,11.9,
            12.4,12.3,13.0,12.7,12.4,12.2,12.0,13.0,14.0,15.0]
    WEIGHTS={
      "srsue0":[3,3,2,2,1,1,2,2,3,3,3,3,3,3,3,3,3,3,3,3],
      "srsue1":[2,2,2,1,1,1,1,2,2,2,2,2,3,2,2,2,2,3,3,3],
      "srsue2":[1]*20,
      "srsue3":[2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
      "srsue4":[3,2,2,1,1,1,2,3,3,3,3,3,3,3,3,3,3,3,3,3],
      "srsue5":[1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2]}
    UL={"srsue0":0.4,"srsue1":0.7,"srsue2":0.5,"srsue3":0.6,"srsue4":0.3,"srsue5":0.5}
    WN,GS,UN,CN=0.30,0.20,0.10,0.15
    def offered_cycle(seed):
        rng=np.random.default_rng(seed)
        scale=float(rng.uniform(1-GS,1+GS))
        for nm in WEIGHTS:               # consume RNG in the exact generator order
            [max(0.1,w*float(rng.uniform(1-WN,1+WN))) for w in WEIGHTS[nm]]
            float(np.clip(UL[nm]+rng.uniform(-UN,UN),0.1,0.9))
        noisy=[gb*float(rng.uniform(1-CN,1+CN)) for gb in GLOBAL]
        return [c*scale*1000/900 for c in noisy]   # aggregate offered Mbps per 15-min slot
    n_slots=int(72/0.25)             # 288 fifteen-minute slots = 72 h
    mbps=[]; seed=0
    while len(mbps)<n_slots:
        mbps+=offered_cycle(seed); seed+=1
    mbps=np.array(mbps[:n_slots]); hrs=np.arange(n_slots)*0.25
    fig,ax=plt.subplots(figsize=(11,3.0))
    ax.step(hrs,mbps,where="post",color="#1f77b4",lw=0.9)
    ax.fill_between(hrs,mbps,step="post",color="#1f77b4",alpha=0.15)
    ax.set_xlabel("Time (hours)",fontsize=12)
    ax.set_ylabel("Offered aggregate\nUE traffic (Mbps)",fontsize=11)
    ax.set_title("72-hour heterogeneous workload (offered load; 5-hour diurnal cycle, per-cycle randomized)",fontsize=11)
    ax.grid(alpha=.3); ax.set_xlim(0,72); ax.set_ylim(0,mbps.max()*1.1)
    fig.tight_layout(); fig.savefig(FIG/"fig_traffic_72h.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"saved fig_traffic_72h.png (offered load, 72h, {n_slots} slots, peak {mbps.max():.1f} Mbps)")

if __name__=="__main__":
    F,sc=collect_full()
    fig_prroc(*sc)
    anytab=fig_heatmaps(F)
    fig_traffic()
    fig_ablation()
    print("\n=== ANY detection F1 per topology (for table) ===")
    for st in ["CPU","MEM","NET"]:
        print(f"CU {st}:", " & ".join(f"{x:.3f}" for x in anytab[st]["CU"]))
    for st in ["CPU","MEM","NET"]:
        print(f"DU {st}:", " & ".join(f"{x:.3f}" for x in anytab[st]["DU"]))
    print("ALL DONE")
