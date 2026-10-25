import aggregate as a, pandas as pd, numpy as np
a.DIRS=["out_v4","colab_out"]; df=a.load(); s,seed=a.summary(df)
M=s.xs('mean',axis=1,level=1); S=s.xs('std',axis=1,level=1)
P=pd.read_csv("paired.csv")
NM={"rpgm":"RPGM","gm":"GM","rwp":"RWP"}
DN={"LET":"LET","KF-LET":"KF-LET","SNR-trend":"SNR trend","GBT":"GBT","MLP":"MLP","LSTM":"LSTM","StaticGNN":"StaticGNN","StaticGNN-noagg":"StaticGNN w/o agg.","ST-GNN":"ST-GNN"}
f=lambda x,p=2: f"{x:.{p}f}".replace("-","$-$") if x<0 else f"{x:.{p}f}"
out={}
# Table analytical
# simpler pos: build separately
L=[]
for mob in ["rpgm","gm","rwp"]:
    for k,(st,lab) in enumerate([("disk","Disk"),("main","A2A")]):
        for j,m in enumerate(["LET","KF-LET"]):
            r=M.loc[(st,mob,m)]; sd=S.loc[(st,mob,m)]
            pos=f"{r.pos_rate:.3f}" if j==0 else ""
            L.append(f"{NM[mob] if k==0 and j==0 else ''} & {lab if j==0 else ''} & {m} & {r.MAE:.2f} $\\pm$ {sd.MAE:.2f} & {r.RMSE:.2f} & {f(r.R2,3)} & {r.AUC:.3f} & {r.AUPRC:.3f} $\\pm$ {sd.AUPRC:.3f} & {pos}\\\\")
    L.append("\\midrule")
out["analytical"]="\n".join(L[:-1])
# Table learned (main)
order=["LET","KF-LET","SNR-trend","GBT","MLP","LSTM","StaticGNN","StaticGNN-noagg","ST-GNN"]
L=[]
for mob in ["rpgm","gm","rwp"]:
    g=M.loc["main",mob]
    best={"MAE":g.MAE.idxmin(),"RMSE":g.RMSE.idxmin(),"R2":g.R2.idxmax(),"AUC":g.AUC.idxmax(),"AUPRC":g.AUPRC.idxmax()}
    B=lambda m,c,t: f"\\textbf{{{t}}}" if best[c]==m else t
    for j,m in enumerate(order):
        r=g.loc[m]; sd=S.loc[("main",mob,m)]; ss=seed.loc[("main",mob,m)]
        sseed="--" if np.isnan(ss.MAE) else f"{ss.MAE:.2f}"
        L.append(f"{NM[mob] if j==0 else ''} & {DN[m]} & {B(m,'MAE',f'{r.MAE:.2f} $\\pm$ {sd.MAE:.2f}')} & {sseed} & {B(m,'RMSE',f'{r.RMSE:.2f}')} & {B(m,'R2',f(r.R2,3))} & {B(m,'AUC',f'{r.AUC:.3f}')} & {B(m,'AUPRC',f'{r.AUPRC:.3f} $\\pm$ {sd.AUPRC:.3f}')}\\\\")
    L.append("\\midrule")
out["learned"]="\n".join(L[:-1])
# censored all
L=[]
for mob in ["rpgm","gm","rwp"]:
    g=M.loc["main",mob]
    best={"MAE_all":g.MAE_all.idxmin(),"Cindex":g.Cindex.idxmax(),"AUPRC_2":g.AUPRC_2.idxmax(),"AUPRC":g.AUPRC.idxmax(),"AUPRC_10":g.AUPRC_10.idxmax()}
    B=lambda m,c,t: f"\\textbf{{{t}}}" if best[c]==m else t
    for j,m in enumerate(order):
        r=g.loc[m]
        L.append(f"{NM[mob] if j==0 else ''} & {DN[m]} & {B(m,'MAE_all',f'{r.MAE_all:.2f}')} & {B(m,'Cindex',f'{r.Cindex:.3f}')} & {B(m,'AUPRC_2',f'{r.AUPRC_2:.3f}')} & {B(m,'AUPRC',f'{r.AUPRC:.3f}')} & {B(m,'AUPRC_10',f'{r.AUPRC_10:.3f}')}\\\\")
    L.append("\\midrule")
out["censored"]="\n".join(L[:-1])
# paired key table
def pr(st,mob,x,y,met,p=2):
    r=P[(P.setting==st)&(P.mobility==mob)&(P.a==x)&(P.b==y)&(P.metric==met)].iloc[0]
    return f"{f(r['diff'],p)} [{f(r.lo,p)}, {f(r.hi,p)}]"
L=[]
for mob in ["rpgm","gm","rwp"]:
    L.append(f"{NM[mob]} & {pr('main',mob,'LSTM','LET','AUPRC',3)} & {pr('main',mob,'LSTM','LET','MAE')} & {pr('main',mob,'LSTM','LET','MAE_all')} & {pr('main',mob,'StaticGNN','StaticGNN-noagg','MAE')} & {pr('main',mob,'StaticGNN','StaticGNN-noagg','AUPRC',3)}\\\\")
out["paired"]="\n".join(L)
# robustness
L=[]
SN={"motion":"Motion-driven shadowing","N20":"$N=20$","N50":"$N=50$","std4":"$\\sigma_X=4$~dB"}
for st in ["motion","N20","N50","std4"]:
    for j,mob in enumerate(["rpgm","gm","rwp"]):
        g=M.loc[st,mob]; lr=g.loc[["GBT","LSTM","StaticGNN"]]
        rg=lambda c,p=2: f"{lr[c].min():.{p}f}--{lr[c].max():.{p}f}"
        L.append(f"{SN[st] if j==0 else ''} & {NM[mob]} & {g.loc['LET'].MAE:.2f} & {rg('MAE')} & {g.loc['LET'].MAE_all:.2f} & {rg('MAE_all')} & {g.loc['LET'].AUPRC:.3f} & {rg('AUPRC',3)}\\\\")
    L.append("\\midrule")
out["robust"]="\n".join(L[:-1])
t=pd.read_csv("timing_dir/timing_cpu.csv")
L=[]
for _,r in t.iterrows():
    nm={"STGNN":"ST-GNN"}.get(r.method,r.method)
    par="--" if np.isnan(r.params) else f"{int(r.params):,}"
    L.append(f"{nm} & {par} & {r.median_ms:.2f} ({r.p95_ms:.2f})\\\\")
out["timing"]="\n".join(L)
import json; json.dump(out,open("tables.json","w"),indent=1)
for k,v in out.items(): print("=== ",k); print(v)
