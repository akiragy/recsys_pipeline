from elasticsearch import Elasticsearch
import logging
from data_exchange_center.constants import *
from online.utils.dto import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ESCaller:
    def __init__(self):
        self.es_obj = Elasticsearch(f"https://elastic:{ES_KEY}@localhost:9200", verify_certs=False)

    def call(self, rec_data: RecData) -> None:
        user_term = rec_data.user_info.user_term
        user_vector = rec_data.user_info.user_vector
        user_filter = rec_data.user_info.user_filter

        term_query = {
            "bool": {
                "must": {
                    "terms": {
                        GENRES: user_term,
                        "boost": 0.1
                    }
                },
                "filter": {
                    "bool": {
                        "must_not": [
                            {
                                "terms": {
                                    ITEMID: user_filter
                                }
                            }
                        ]
                    }
                }
            }
        }

        vector_query = {
            "field": ITEM_VECTOR,
            "query_vector": user_vector,
            "k": rec_data.config.recall_size,
            "num_candidates": MAX_ITEMID,
            "boost": 0.9,
            "filter": {
                "bool": {
                    "must_not": {
                        "terms": {
                            ITEMID: user_filter
                        }
                    }
                }
            }
        }

        cnt = 0
        res = self.es_obj.search(index=ITEM_ES_INDEX, knn=vector_query, query=term_query,
                                 size=rec_data.config.recall_size)
        for i in range(len(res["hits"]["hits"])):
            cnt += 1
            itemid = res["hits"]["hits"][i]["_source"]["itemid"]
            item_info = ItemInfo()
            item_info.itemid = itemid
            rec_data.item_list.append(item_info)
        logging.info(f"******user {rec_data.user_info.userid}, es done, recall_cnt {cnt}******")
