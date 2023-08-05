import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from redis import Redis, ConnectionPool
import pickle
import tqdm
from data_exchange_center.constants import *
from data_exchange_center.paths import USER_FILTER_PATH, USER_TERM_PATH, USER_VECTOR_PATH


def load_data():
    user_filter_dict = pickle.load(open(USER_FILTER_PATH, "rb"))
    user_dict = pickle.load(open(USER_TERM_PATH, "rb"))
    user_vector = pickle.load(open(USER_VECTOR_PATH, "rb"))
    return user_filter_dict, user_dict, user_vector


if __name__ == '__main__':
    user_filter_dict, user_dict, user_vector = load_data()
    pool = ConnectionPool(host="localhost", port=6379, decode_responses=True)
    r = Redis(connection_pool=pool)

    # insert data
    for userid in tqdm.tqdm(user_dict):
        genre_dict = user_dict[userid]
        cur_user_genre_list = []
        for genre in genre_dict:
            if genre_dict[genre][1] >= 3:
                cur_user_genre_list.append(genre)
        cur_user_vector = user_vector[userid]
        cur_user_filter = user_filter_dict[userid]

        cur_user_key = RECALL_REDIS_PREFIX + str(userid)
        r.hset(cur_user_key, RECALL_REDIS_TERM_KEY, ",".join([str(v) for v in cur_user_genre_list]))
        r.hset(cur_user_key, RECALL_REDIS_VECTOR_KEY, ",".join([str(v) for v in cur_user_vector]))
        r.hset(cur_user_key, RECALL_REDIS_FILTER_KEY, ",".join([str(v) for v in cur_user_filter]))

    # check
    test_userid = 1234  # modify it to another id for further testing
    test_user_key = "user" + str(test_userid)
    print("userid:", test_userid)
    print("keys: ", r.hkeys(test_user_key))
    print("term: ", r.hget(test_user_key, RECALL_REDIS_TERM_KEY))
    print("vector: ", r.hget(test_user_key, RECALL_REDIS_VECTOR_KEY))
    print("filter: ", r.hget(test_user_key, RECALL_REDIS_FILTER_KEY))