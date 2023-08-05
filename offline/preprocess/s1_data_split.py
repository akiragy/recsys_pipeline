import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from data_exchange_center.constants import *
from data_exchange_center.paths import RAW_RATING_PATH, OFFLINE_IMP_PATH, ONLINE_IMP_PATH


def load_data():
    rating_data = pd.read_csv(RAW_RATING_PATH, sep="::", engine="python", header=None, names=[USERID, ITEMID, RATING, TS])
    return rating_data


def split_data(x):
    if x["rn"] >= x["u_cnt"] - ONLINE_LAST_N:  # last 10 for online serving
        return 2
    if x["rn"] < (x["u_cnt"] - ONLINE_LAST_N) * (1 - OFFLINE_TEST_RATIO):  # first 80% for offline train
        return 0
    return 1  # last 20% for offline test


if __name__ == "__main__":
    rating_data = load_data()

    # convert into implicit feedback
    rating_data[LABEL] = rating_data[RATING].map(lambda x: 1 if x > 3 else 0)
    rating_data = rating_data.drop(RATING, axis=1)

    # offline-online split
    tmp = rating_data.groupby(USERID)[ITEMID].count().rename("u_cnt")
    rating_data = pd.merge(rating_data, tmp, on=USERID)
    rating_data = rating_data.sort_values([USERID, TS], ascending=[True, True]).reset_index(drop=True)
    rating_data["rn"] = rating_data.groupby(USERID).cumcount()
    rating_data[ISTEST] = rating_data.apply(split_data, axis=1)
    offline_imp = rating_data.loc[rating_data[ISTEST] != 2].drop(["rn", "u_cnt"], axis=1).reset_index(drop=True)
    online_imp = rating_data.loc[rating_data[ISTEST] == 2].drop(["rn", ISTEST, "u_cnt"], axis=1).reset_index(drop=True)
    print(f"saved: offline_imp: {offline_imp.shape}, online_imp: {online_imp.shape}")
    offline_imp.to_csv(OFFLINE_IMP_PATH, index=False)
    online_imp.to_csv(ONLINE_IMP_PATH, index=False)