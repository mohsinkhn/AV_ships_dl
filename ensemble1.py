import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


from config import ROOT, TRAIN_FILE, TEST_FILE


def read_data(root):
    train_df = pd.read_csv(str(Path(root) / TRAIN_FILE))
    test_df = pd.read_csv(str(Path(root) / TEST_FILE))
    return train_df, test_df


if __name__=="__main__":
    train_df, test_df = read_data(ROOT)
    cvlist = list(StratifiedKFold(8, random_state=12345786).split(train_df,
                                                                  train_df.category))

    densenet_preds = np.load("test_preds_densenet.npy")
    inception_preds = np.load("test_preds_inception.npy")

    e_preds = 0.3*inception_preds + 0.7*densenet_preds

    y_test_preds = np.argmax(e_preds, axis=1)

    sub = test_df[["image"]]
    sub["category"] = y_test_preds + 1
    sub.to_csv("sube1.csv", index=False)

