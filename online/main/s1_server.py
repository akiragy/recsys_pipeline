import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flask import Flask, request
import json
import logging
from data_exchange_center.constants import *
from online.utils.dto import *
from online.utils.redis_caller import RedisCaller
from online.utils.es_caller import ESCaller
from online.utils.feast_caller import FeastCaller
from online.utils.triton_caller import TritonCaller
logging.getLogger().setLevel(logging.INFO)


# init
redis_caller = RedisCaller()
es_caller = ESCaller()
triton_caller = TritonCaller()
feast_caller = FeastCaller()
app = Flask(__name__)


@app.route('/ml/rec', methods=['POST'])
def ml_rec():
    # parse request
    rec_data = build_rec_data(json.loads(request.data))

    # core
    recall(rec_data)
    rank(rec_data)

    # build response
    res = {"item_list": []}
    for i in range(min(rec_data.config.response_size, len(rec_data.item_list))):
        d = {
            "itemid": rec_data.item_list[i].itemid,
            "score": rec_data.item_list[i].score
        }
        res["item_list"].append(eval(str(d)))
    return res


def build_rec_data(rec_request: json) -> RecData:
    rec_data = RecData()
    rec_data.user_info.userid = rec_request["userid"]
    return rec_data


def recall(rec_data: RecData) -> None:
    redis_caller.call(rec_data)
    es_caller.call(rec_data)


def rank(rec_data: RecData) -> None:
    feast_caller.call(rec_data)
    triton_caller.call(rec_data)



