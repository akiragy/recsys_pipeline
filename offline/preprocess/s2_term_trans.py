import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import copy
import pandas as pd
import pickle
from data_exchange_center.constants import *
from data_exchange_center.paths import RAW_ITEM_PATH, OFFLINE_IMP_PATH, USER_FILTER_PATH, IMP_TERM_PATH, USER_TERM_PATH, ITEM_TERM_PATH


def load_data():
    item_data = pd.read_csv(RAW_ITEM_PATH, sep="::", engine="python", header=None, names=[ITEMID, TITLE, GENRES], encoding="ISO-8859-1")
    offline_imp = pd.read_csv(OFFLINE_IMP_PATH)
    offline_joined = pd.merge(offline_imp, item_data, on=ITEMID, how="left", right_index=False)
    return item_data, offline_imp, offline_joined


def update_genre_dict(_genres, _label, genre_dict, type):
    assert type in ("add", "sub")
    for _genre in _genres.split("|"):
        genre = GENRES_MAPPING.get(_genre, GENRES_MAPPING.get(EMPTY_KEY))
        if genre not in genre_dict:
            genre_dict[genre] = [0, 0]
        genre_dict[genre][0] += 1 * (-1 if type == "sub" else 1)
        genre_dict[genre][1] += _label * (-1 if type == "sub" else 1)
    return genre_dict


if __name__ == "__main__":
    item_data, offline_imp, offline_joined = load_data()

    # impression filter
    user_filter_dict = offline_imp.groupby(USERID)[ITEMID].agg(list).to_dict()
    pickle.dump(user_filter_dict, open(USER_FILTER_PATH, 'wb'), protocol=4)

    # imp and user term
    vals = offline_joined[[USERID, LABEL, GENRES]].values.tolist()
    imp_dict = dict()
    user_dict = dict()
    pre_userid = -1
    i = 0
    for userid, label, genres in vals + [[-1, -1, -1]]:
        if userid != pre_userid:  # update user term
            if pre_userid != -1:
                new_genres, new_label = dequeue[-1]
                genre_dict = update_genre_dict(new_genres, new_label, genre_dict, "add")
                user_dict[pre_userid] = copy.deepcopy(genre_dict)
            dequeue = []
            genre_dict = dict()
            if userid == -1:
                break
        if len(dequeue) > LAST_N_GENRE_CNT:
            timeout_genres, timeout_label = dequeue.pop(0)
            genre_dict = update_genre_dict(timeout_genres, timeout_label, genre_dict, "sub")
        if len(dequeue) > 0:
            new_genres, new_label = dequeue[-1]
            genre_dict = update_genre_dict(new_genres, new_label, genre_dict, "add")
        dequeue.append((genres, label))
        imp_dict[i] = copy.deepcopy(genre_dict)  # update imp term
        i += 1
        pre_userid = userid
    pickle.dump(imp_dict, open(IMP_TERM_PATH, 'wb'), protocol=4)
    pickle.dump(user_dict, open(USER_TERM_PATH, 'wb'), protocol=4)

    # item term
    vals = item_data[[ITEMID, GENRES]].values
    item_dict = dict()
    for itemid, genres in vals:
        item_dict[itemid] = set()
        for _genre in genres.split("|"):
            genre = GENRES_MAPPING.get(_genre, GENRES_MAPPING.get(EMPTY_KEY))
            item_dict[itemid].add(genre)
    pickle.dump(item_dict, open(ITEM_TERM_PATH, 'wb'), protocol=4)
    print(f"saved: imp_dict: {len(imp_dict)}, user_dict: {len(user_dict)}, item_dict: {len(item_dict)}")