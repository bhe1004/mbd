import numpy as np, dataclasses, sys
from exper.config import load_config
from exper.plant import FrankaPlant
from exper.training import get_model
from exper.backends import build_backend
from exper.planner import Schedule
from exper.trial import run_trial
out=open("exper/wcheck.log","w")
def log(m): out.write(m+"\n"); out.flush()
cfg=load_config("exp_b")
p=FrankaPlant(cfg)
tg=p.targets(cfg.task.num_targets, cfg.task.target_seed)
sch=Schedule("anneal",cfg.planner.stages,cfg.planner.num_samples,cfg.planner.sigma_start,cfg.planner.sigma_end)
wc=cfg.replace(data=dataclasses.replace(cfg.data, white=True))
for label,c,tag in [("coherent",cfg,""),("white",wc,"white_excitation")]:
    m=get_model(c,p,"bilinear",0,tag=tag,verbose=False)
    b=build_backend("bilinear",p,cfg.planner,m)
    res=[run_trial(cfg,p,b,sch,tg[i],condition=label,model_seed=0,target_idx=i,rng_seed=1000+i) for i in range(5)]
    log(f"{label:9s}: strict {sum(r.reached_strict for r in res)}/5  reach {sum(r.reached for r in res)}/5  "
        f"final {[round(r.final_err,3) for r in res]}  min_err {[round(r.min_err,3) for r in res]}")
log("DONE")
