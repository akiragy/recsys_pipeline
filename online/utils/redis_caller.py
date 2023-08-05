from redis import Redis, ConnectionPool
import logging
from data_exchange_center.constants import *
from online.utils.dto import *


class RedisCaller:
    def __init__(self):
        pool = ConnectionPool(host="localhost", port=6379, decode_responses=True)
        self.redis_obj = Redis(connection_pool=pool)

    def call(self, rec_data: RecData) -> None:
        userid = rec_data.user_info.userid
        test_user_key = RECALL_REDIS_PREFIX + str(userid)
        user_term = self.redis_obj.hget(test_user_key, RECALL_REDIS_TERM_KEY)
        user_vector = self.redis_obj.hget(test_user_key, RECALL_REDIS_VECTOR_KEY)
        user_filter = self.redis_obj.hget(test_user_key, RECALL_REDIS_FILTER_KEY)

        if user_term is None or len(user_term) == 0:
            rec_data.user_info.user_term = []
            logging.warning(f"user {userid} term empty")
        else:
            rec_data.user_info.user_term = [int(v) for v in user_term.split(",")]

        if user_vector is None or len(user_vector) == 0:
            rec_data.user_info.user_vector = [1] * RECALL_EMB_DIM
            logging.warning(f"user {userid} vector empty")
        else:
            rec_data.user_info.user_vector = [float(v) for v in user_vector.split(",")]

        if user_filter is None or len(user_filter) == 0:
            rec_data.user_info.user_filter = []
            logging.warning(f"user {userid} filter empty")
        else:
            rec_data.user_info.user_filter = [int(v) for v in user_filter.split(",")]

        logging.info(f"******user {userid}, redis done******")

