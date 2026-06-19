import pandas as pd
import numpy as np
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import plot_gain
import pdb

def train(data, train_params):
    train = data['train']
    valid = data['test']
    train = train.to_pandas()
    valid = valid.to_pandas()
    test = valid[0:100]
    valid = valid[100:]

    feat_names = ['f0', 'f1', 'f2','f3', 'f4', 'f5','f5', 'f7', 'f8','f9', 'f10', 'f11']

    clf = UpliftRandomForestClassifier(control_name='0')
    clf.fit(
        train[feat_names].values,
        train['treatment'].astype(str).values,
        train['conversion'].values,
        valid[feat_names].values,
        valid['treatment'].astype(str).values,
        valid['conversion'].values
        )

    p = clf.predict(test[feat_names].values)

    # pdb.set_trace()


    # define model
    return 'tree_model'
