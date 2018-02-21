import numpy as np
import pandas as pd
import copy
from sklearn import linear_model

def train_model(tr_df, vecs):
    lrms = [] # one model per confidence value (24 ih total)
    for c in range(4, 28):
        lrm = linear_model.LinearRegression()
        lrm.fit(vecs, tr_df.iloc[:,c].values)
        lrms.append(lrm)
    return lrms

def make_predictions(lrms, ts_df, ts_doc_vecs):
    pred_obj = {}
    heading_names = list(ts_df.columns)
    for c in range(0, 24):
        pred_obj[heading_names[c+4]] = lrms[c].predict(ts_doc_vecs)
    return pd.DataFrame(pred_obj, index=ts_df.id)