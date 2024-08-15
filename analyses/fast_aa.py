import scanpy as sc
import numpy as np
import pandas as pd
from py_pcha import PCHA

print("read in data")
my_data = sc.read_h5ad("current_merged_sc_peaks.h5ad")
print(my_data.shape) ## should be 28k x ~200k

print("run archetypal analysis")

cell_names = my_data.obs_names
data_subset = my_data.layers["pearson_norm"].T
print(data_subset.shape) ## should be num_features by num_cells

for k in [8, 12, 15]:
    XC, S, C, SSE, varexpl = PCHA(data_subset, noc=k)
    print("SSE: %s"%SSE)
    print("variance explained: %s"%varexpl)

    pd.DataFrame(XC).to_csv("aa_outs/k%s_archetypes.csv"%k)
    pd.DataFrame(S, columns=cell_names).to_csv("aa_outs/k%s_s_mat.csv"%k)
    pd.DataFrame(C, index=cell_names).to_csv("aa_outs/k%s_c_mat.csv"%k)
 
    print("done with archetypal analysis for k=%s"%k)

print("done")
