import scanpy as sc
import numpy as np
import pandas as pd
import pickle
from plotnine import *

from scipy.stats import kstest, pearsonr, spearmanr

print("read in my anndata")
my_ad = sc.read_h5ad("../../../current_merged_sc_peaks.h5ad")

sat_var = pd.read_csv("../sat_var.csv", index_col = 0)
wherry_var = pd.read_csv("../wherry_var.csv", index_col = 0)

archetype_naming = {
    "k8_aa_0": "Cd8 naive arch",
    "k8_aa_1": "Cd8 mem arch",
    "k8_aa_2": "Cd4 Tfh arch",
    "k8_aa_3": "Cd8 dys arch",
    "k8_aa_4": "T cell activation arch",
    "k8_aa_5": "Cd8 eff arch",
    "k8_aa_6": "Cd4 Treg arch",
    "k8_aa_7": "Cd4 naive-mem arch"
}

y_matrix = my_ad.obs[archetype_naming.keys()]
y_matrix.columns = archetype_naming.values()


from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

def train_model(peaks, name, m = LinearRegression):
    X = my_ad[:, peaks].layers["pearson_norm"]
    print("running for %s"%name)
    print(X.shape)
    train_X, test_X, train_Y, test_Y = train_test_split(X, y_matrix, test_size = 0.2)
    model = m()
    print("now start training model")
    model.fit(np.array(train_X), np.array(train_Y))
    print("finished training")
    print("save model")
    with open("model_%s.pckl"%name, 'wb') as file:
        pickle.dump(model, file)
    print("save test data to check functionality")
    test_Y.to_csv("test_Y.%s.csv"%name)
    print("now calculate pred Y") 
    pred_Y = model.predict(test_X)
    pd.DataFrame(pred_Y).to_csv("raw_pred_Y.%s.csv"%name)
    scaled_pred_Y = minmax_scale(pred_Y, axis = 1)
    final_pred_Y = np.matmul(np.diag(1/scaled_pred_Y.sum(1)), scaled_pred_Y)
    print("check all predictions are scaled right")
    print(np.all(np.isclose(final_pred_Y.sum(1), 1)))
    pd.DataFrame(final_pred_Y).to_csv("scaled_pred_Y.%s.csv"%name)
    pd.DataFrame(peaks).to_csv("peaks_%s.csv"%name)
    print("done for %s"%name)


train_model(np.intersect1d(my_ad.var_names, sat_var.index), "sat")
train_model(np.intersect1d(my_ad.var_names, wherry_var.index), "wherry", RidgeCV)
