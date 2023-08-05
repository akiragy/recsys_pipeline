import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pickle
import time
from data_exchange_center.constants import *
from data_exchange_center.paths import ITEM_TERM_PATH, ITEM_VECTOR_PATH
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_data():
    item_dict = pickle.load(open(ITEM_TERM_PATH, "rb"))
    item_vector = pickle.load(open(ITEM_VECTOR_PATH, "rb"))
    return item_dict, item_vector


if __name__ == '__main__':
    item_dict, item_vector = load_data()
    time.sleep(10)
    es = Elasticsearch(f"https://elastic:{ES_KEY}@localhost:9200", verify_certs=False)

    # create index
    if not es.indices.exists(index=ITEM_ES_INDEX):
        print(ITEM_ES_INDEX, "need create")
        create_mapping = {
            "properties": {
                ITEMID: {
                    "type": "long"
                },
                GENRES: {
                    "type": "long"
                },
                ITEM_VECTOR: {
                    "type": "dense_vector",
                    "dims": RECALL_EMB_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
        es.indices.create(index=ITEM_ES_INDEX, mappings=create_mapping)
        print("create done")
        es.options(ignore_status=400)

    # delete existing data
    all_query = {
        "match_all": {}
    }
    es.delete_by_query(index=ITEM_ES_INDEX, query=all_query)

    # insert data
    action = [{
        "_index": ITEM_ES_INDEX,
        ITEMID: i,
        GENRES: list(item_dict[i]),
        ITEM_VECTOR: item_vector[i]
    } for i in item_dict]
    helpers.bulk(client=es, actions=action)
    time.sleep(10)
    print("all data count: ", es.search(index=ITEM_ES_INDEX, query=all_query)["hits"]["total"]["value"])

    # check term index
    test_genre = 2  # modify it to another genre for further testing
    cnt_true = 0  # test_genre true count
    for i in item_dict:
        if test_genre in item_dict[i]:
            cnt_true += 1
    term_query = {
        "terms": {
            GENRES: [test_genre],
            "boost": 0.1
        }
    }
    res = es.search(index=ITEM_ES_INDEX, query=term_query)
    cnt_hit = res["hits"]["total"]["value"]  # test_genre hit count
    print(f"check term index with genre {test_genre}: true {cnt_true}, hit {cnt_hit}. ",
          "test passed!" if cnt_true == cnt_hit else "test failed!")

    # check vector index
    test_itemid = 2333  # modify it to another id for further testing
    query_vector = item_vector[test_itemid]  # item 1 as query vector
    vector_query = {
        "field": ITEM_VECTOR,
        "query_vector": query_vector,
        "k": 20,
        "num_candidates": 500,
        "boost": 0.9
    }
    res = es.search(index=ITEM_ES_INDEX, knn=vector_query)
    hit_1_itemid = res["hits"]["hits"][0]["_source"]["itemid"]
    print(f"check vector index with id {test_itemid}: true {test_itemid}, hit {hit_1_itemid}. ",
          "test passed!" if hit_1_itemid == test_itemid else "test failed!")

    # combined query
    print("knn and term", es.search(index=ITEM_ES_INDEX, knn=vector_query, query=term_query))