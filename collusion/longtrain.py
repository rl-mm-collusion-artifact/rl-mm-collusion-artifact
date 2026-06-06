"""Longer-training check: does low Delta at N=4,5 survive doubled training?"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from collusion.sweep import run_cell, _spec

def build():
    specs=[]
    periods={4:5_000_000, 5:10_000_000}
    for n in (4,5):
        for s in range(12):
            sp=_spec("longtrain", n, 0.5, 0.2, 1.0, s)
            sp["periods"]=periods[n]
            specs.append(sp)
    return specs

if __name__=="__main__":
    specs=build()
    print(f"running {len(specs)} long-training cells ...", flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=24) as ex:
        futs=[ex.submit(run_cell,s) for s in specs]
        for f in as_completed(futs): rows.append(f.result())
    df=pd.DataFrame(rows); df.to_csv("collusion/results/longtrain.csv",index=False)
    print(df.groupby("n_makers")[["collusion_index","mean_spread"]].agg(["mean","std"]).round(4).to_string(), flush=True)
    print("wrote collusion/results/longtrain.csv", flush=True)
