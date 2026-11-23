"""
run_experiments_v2.py -- regenerates every result of Section 8 (v2) into ./results_v2.
Usage:  python run_experiments_v2.py   (requires numpy, scipy, pandas, scikit-learn; network for OpenML)
Seeds and configurations are exactly those reported in the paper.
"""
import os, time, itertools, numpy as np, pandas as pd, scipy.sparse as sp, warnings
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import dmm_lib_v2 as L
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results_v2"); DATA=os.path.join(HERE,"data_v2")
os.makedirs(RES,exist_ok=True); os.makedirs(DATA,exist_ok=True)

# ---------------- Block A: stability / minimax validation (exact multinomial simulation of F)
def sample_counts(M,n,rng):
    ny=rng.multinomial(n,M.priors); F=np.zeros((M.d,M.k))
    for c in range(M.k):
        if ny[c]: F[:,c]=np.concatenate([rng.multinomial(ny[c],pb[c]) for pb in M.p])
    return F
def one_cell(M,n,reps,r,rng):
    R=M.rho_D_star(); w=np.sort(np.linalg.eigvalsh(R))[::-1]; gap=w[r-1]-w[r]; U=L.top_eigvecs(R,r); op=[]; st=[]
    for _ in range(reps):
        A=np.sqrt(sample_counts(M,n,rng)); Rh=A@A.T/np.trace(A.T@A)
        op.append(np.linalg.norm(Rh-R,2)); st.append(L.sin_theta(L.top_eigvecs(Rh,r),U))
    mbar=M.d/M.q; kp=np.sum(np.sqrt(M.priors)); nu=np.sum(np.sqrt(M.priors*(1-M.priors))); ub=(kp*np.sqrt(mbar-1)+nu)/np.sqrt(n)
    return dict(n=n,d=M.d,q=M.q,mbar=mbar,k=M.k,pi_min=M.priors.min(),gap=gap,op_mean=np.mean(op),op_sd=np.std(op),st_mean=np.mean(st),st_sd=np.std(st),
                upper=ub,lower=np.sqrt(3*np.log(2))/512*np.sqrt((mbar-1)/n),st_upper=min(1,2*ub/gap))
def block_A():
    rng=np.random.default_rng(2026)
    A1=[one_cell(L.dirichlet_model([m]*10,4,0.3,seed=1),n,50,3,rng) for m in [5,20,80] for n in [250,500,1000,2000,4000,8000,16000,32000,64000]]
    pd.DataFrame(A1).to_csv(f"{RES}/A1_rate.csv",index=False)
    A2=[{**one_cell(L.dirichlet_model([10]*q,4,0.3,seed=2),4000,50,3,rng),"design":"q varies (mbar=10)"} for q in [5,10,20,40,80]]
    A2+=[{**one_cell(L.dirichlet_model([m]*10,4,0.3,seed=2),4000,50,3,rng),"design":"mbar varies (q=10)"} for m in [5,10,20,40,80]]
    pd.DataFrame(A2).to_csv(f"{RES}/A2_dimension.csv",index=False)
    A3=[]
    for alpha in [1.0,0.3,0.1,0.03,0.01]:
        M=L.dirichlet_model([20]*10,4,alpha,seed=3); row=one_cell(M,4000,50,3,rng); P=M.profiles()
        A3.append({**row,"alpha":alpha,"p_min":P.min(),"frac_cells_below_1e-3":np.mean(P<1e-3)})
    pd.DataFrame(A3).to_csv(f"{RES}/A3_rare.csv",index=False)
    A4=[one_cell(L.dirichlet_model([10]*10,2,0.5,priors=[1-p,p],seed=4),4000,100,2,rng) for p in [0.5,0.3,0.2,0.1,0.05,0.02,0.01]]
    pd.DataFrame(A4).to_csv(f"{RES}/A4_imbalance.csv",index=False)
    # A5: classifier regimes on an exactly computable model
    ms=[3,3,4]; k=3; r=2; pi=[.5,.3,.2]; r_=np.random.default_rng(3)
    P=[[r_.dirichlet(np.ones(m)*0.7) for m in ms] for _ in range(k)]; q=len(ms)
    atoms=list(itertools.product(*[range(m) for m in ms])); Xa=L.onehot(np.array(atoms),ms).astype(float); Xt=Xa/np.sqrt(q)
    pxy=np.array([[pi[y]*np.prod([P[y][b][t[b]] for b in range(q)]) for y in range(k)] for t in atoms])
    M=L.BlockModel([[P[y][b] for y in range(k)] for b in range(q)],np.array(pi)); Zs=Xt@L.top_eigvecs(M.rho_D_star(),r)
    D=np.linalg.norm(Zs[:,None]-Zs[None],axis=2); vs=np.min(D[D>1e-9]); glat=pxy.argmax(1); Lstar=1-pxy.max(1).sum()
    K0=(2*np.pi)**(-r/2); Kf=lambda u: K0*np.exp(-0.5*np.sum(u**2,-1)); pa=pxy/pxy.sum(0); out=[]
    for regime,h in [("fixed h=0.2",0.2),("h=s/5",vs/5)]:
        Ks=Kf((Zs[:,None]-Zs[None])/h); ss=Ks@pxy; gh=ss.argmax(1); Lh=1-pxy[np.arange(len(gh)),gh].sum()
        for n in [300,1000,3000,10000,30000,100000,300000,1000000]:
            ag_h=[];ag_l=[];risk=[];sup=[]
            for _ in range(40):
                ny=rng.multinomial(n,pi); C=np.column_stack([rng.multinomial(ny[c],pa[:,c]) for c in range(k)])
                A=np.sqrt(Xa.T@C); lam,V=np.linalg.eigh(A.T@A); o=np.argsort(lam)[::-1][:r]; Zh=Xt@((A@V[:,o])/np.sqrt(lam[o]))
                sh=Kf((Zh[:,None]-Zh[None])/h)@C/n; g=sh.argmax(1)
                sup.append(np.abs(sh-ss).max()); ag_h.append(np.mean(g==gh)); ag_l.append(np.mean(g==glat)); risk.append(1-pxy[np.arange(len(g)),g].sum())
            out.append(dict(regime=regime,n=n,h=h,sup_mean=np.mean(sup),agree_smoothed=np.mean(ag_h),agree_latBayes=np.mean(ag_l),
                            risk_mean=np.mean(risk),risk_sd=np.std(risk),L_h=Lh,L_lat=Lstar))
    pd.DataFrame(out).to_csv(f"{RES}/A5_classifier.csv",index=False)

# ---------------- Block B: synthetic benchmarks S1-S4 (10 replications, Scott bandwidth)
def collect(rows,setting,res,rep):
    for m,mt in res.items(): rows.append({**setting,"method":m,"rep":rep,**mt})
def trial(M,n,r,seed):
    C,X,y=L.sample(M,n,np.random.default_rng(10_000+seed))
    tr,te=train_test_split(np.arange(n),test_size=0.3,stratify=y,random_state=seed)
    return L.run_methods(X[tr],y[tr],C[tr],X[te],y[te],C[te],M.k,M.q,M.m,r,seed=seed)
def block_B(REPS=10,DEL=0.3):
    S1=[]; [collect(S1,{"delta":dl},trial(L.synthetic_model(20,[6]*20,3,list(range(5)),dl,seed=s),5000,3,s),s) for dl in [0.0,0.2,0.4,0.6,0.8,1.0] for s in range(REPS)]
    pd.DataFrame(S1).to_csv(f"{RES}/S1.csv",index=False)
    S2=[]
    for sc in [1,4,16,64]:
        ml=[5]*15; ml[5:10]=[5*sc]*5
        for s in range(REPS): collect(S2,{"scale":sc,"d":sum(ml)},trial(L.synthetic_model(15,ml,3,list(range(5)),DEL,seed=s),4000,3,s),s)
    pd.DataFrame(S2).to_csv(f"{RES}/S2.csv",index=False)
    S3=[]
    for a in [0.0,0.5,1.0,2.0,4.0]:
        for s in range(REPS):
            M=L.synthetic_model(15,[6]*15,3,list(range(5)),DEL,noise_blocks=int(round(a*15)),seed=s); collect(S3,{"alpha":a,"d":M.d},trial(M,4000,3,s),s)
    pd.DataFrame(S3).to_csv(f"{RES}/S3.csv",index=False)
    S4=[]
    for p0 in [0.5,0.8,0.9,0.95]:
        for s in range(REPS):
            M=L.synthetic_model(20,[6]*20,2,list(range(6)),DEL,priors=[p0,1-p0],seed=s)
            C,X,y=L.sample(M,6000,np.random.default_rng(10_000+s)); tr,te=train_test_split(np.arange(6000),test_size=0.3,stratify=y,random_state=s); res={}
            for name,norm,pri in [("DMM count, ML",False,False),("DMM count, MAP",False,True),("DMM CN, ML",True,False),("DMM CN, MAP",True,True)]:
                e=L.DMMEmbed(2,normalized=norm).fit(X[tr],y[tr],2,M.q)
                res[name]=L.metrics(y[te],L.LatentKDE(use_priors=pri).fit(e.transform(X[tr]),y[tr],2).predict(e.transform(X[te])))
            res.update(L.run_methods(X[tr],y[tr],C[tr],X[te],y[te],C[te],2,M.q,M.m,2,methods=["HellPCA+KDE","DCA+KDE","LDA-shrink","CatNB","LogReg","RF"],seed=s))
            collect(S4,{"p_major":p0},res,s)
    pd.DataFrame(S4).to_csv(f"{RES}/S4.csv",index=False)

# ---------------- Block C: real data (OpenML)
IDS={"car":40975,"mushroom":24,"nursery":26,"kr-vs-kp":3,"vote":56,"soybean":42,"splice":46,"tic-tac-toe":50}
def encode(ds):
    Xdf=ds.data.drop(columns=[c for c in ds.data.columns if c.lower() in ("instance_name","instance")],errors="ignore"); y=ds.target.astype(str)
    codes=[];ms=[]
    for c in Xdf.columns:
        cat=pd.Categorical(Xdf[c].astype("string").fillna("NA")); codes.append(cat.codes.astype(np.int64)); ms.append(len(cat.categories))
    C=np.column_stack(codes); keep=np.array(ms)>1
    return C[:,keep],list(np.array(ms)[keep]),y
def fetch(i):
    for parser in ("auto","liac-arff"):
        for attempt in range(4):
            try: return fetch_openml(data_id=i,as_frame=True,data_home=DATA,parser=parser)
            except TypeError: break
            except Exception: time.sleep(5*attempt+2)
    raise RuntimeError(f"could not fetch OpenML {i}")
def block_C():
    REAL=[]
    for name,i in IDS.items():
        C,ms,y=encode(fetch(i)); vc=y.value_counts(); ok=y.isin(vc[vc>=10].index).values; C=C[ok]; y=pd.Categorical(y[ok]).codes.astype(int)
        X=L.onehot(C,ms); k=len(np.unique(y)); q=len(ms); r=min(k,4)
        for s in range(10):
            tr,te=train_test_split(np.arange(len(y)),test_size=0.3,stratify=y,random_state=s)
            res=L.run_methods(X[tr],y[tr],C[tr],X[te],y[te],C[te],k,q,ms,r,seed=s)
            e=L.DMMEmbed(r).fit(X[tr],y[tr],k,q); res["DMM+KDE (ML)"]=L.metrics(y[te],L.LatentKDE(use_priors=False).fit(e.transform(X[tr]),y[tr],k).predict(e.transform(X[te])))
            collect(REAL,{"dataset":name,"n":len(y),"q":q,"d":sum(ms),"k":k},res,s)
    pd.DataFrame(REAL).to_csv(f"{RES}/real_data.csv",index=False)
    # Amazon Employee Access (sparse)
    C,ms,y=encode(fetch(4135)); y=pd.Categorical(y).codes.astype(int); n,q=C.shape; d=sum(ms); off=np.concatenate([[0],np.cumsum(ms)[:-1]])
    X=sp.csr_matrix((np.ones(n*q,dtype=np.float32),(np.repeat(np.arange(n),q),(C+off).ravel())),shape=(n,d))
    def counts(Xtr,ytr): return np.column_stack([np.asarray(Xtr[ytr==c].sum(0)).ravel() for c in range(2)])
    AMZ=[]
    for s in range(5):
        tr,te=train_test_split(np.arange(n),test_size=0.3,stratify=y,random_state=s); Xtr,Xte,ytr,yte=X[tr],X[te],y[tr],y[te]; res={}
        A=np.sqrt(counts(Xtr,ytr)); lam,V=np.linalg.eigh(A.T@A); o=np.argsort(lam)[::-1][:2]; U=(A@V[:,o])/np.sqrt(np.maximum(lam[o],1e-12))
        Ztr=np.asarray(Xtr@U)/np.sqrt(q); Zte=np.asarray(Xte@U)/np.sqrt(q)
        for nm,pri in [("DMM+KDE",True),("DMM+KDE (ML)",False)]: res[nm]=L.metrics(yte,L.LatentKDE(use_priors=pri).fit(Ztr,ytr,2).predict(Zte))
        F=counts(Xtr,ytr).T; P=F/F.sum(); rr=P.sum(1); c=P.sum(0); kp=c>0
        S=(P[:,kp]-np.outer(rr,c[kp]))/np.sqrt(np.outer(rr,c[kp])); _,_,Vt=np.linalg.svd(S,full_matrices=False); G=np.zeros((d,1)); G[kp]=Vt[:1].T/np.sqrt(c[kp])[:,None]
        Ztr=np.asarray(Xtr@G)/q; Zte=np.asarray(Xte@G)/q
        for nm,pri in [("DCA+KDE",True),("DCA+KDE (ML)",False)]: res[nm]=L.metrics(yte,L.LatentKDE(use_priors=pri).fit(Ztr,ytr,2).predict(Zte))
        res["CatNB"]=L.metrics(yte,L.CategoricalNB(alpha=1.0,min_categories=np.asarray(ms)).fit(C[tr],ytr).predict(C[te]))
        res["LogReg"]=L.metrics(yte,L.LogisticRegression(max_iter=3000).fit(Xtr,ytr).predict(Xte))
        res["RF"]=L.metrics(yte,L.RandomForestClassifier(300,random_state=s,n_jobs=-1).fit(Xtr,ytr).predict(Xte))
        collect(AMZ,{"dataset":"amazon","n":n,"q":q,"d":d,"k":2},res,s)
    pd.DataFrame(AMZ).to_csv(f"{RES}/real_amazon.csv",index=False)

if __name__=="__main__":
    block_A(); block_B(); block_C()
