import requests
import json
import pandas as pd
import logging
from data_exchange_center.constants import *
from online.utils.dto import *


class FeastCaller:
    def __init__(self):
        self.url = "http://localhost:6566/get-online-features"

    def call(self, rec_data: RecData) -> None:
        userid = rec_data.user_info.userid
        user_feature_request = {
            "feature_service": "ml_user",
            "entities": {
                USERID: [userid]
            }
        }
        response = json.loads(
            requests.post(self.url, data=json.dumps(user_feature_request)).text)

        user_feat_names = response["metadata"]["feature_names"]
        if len(user_feat_names) != len(response["results"]):
            logging.error(f"******feast user feature num error, user {userid}******")
            rec_data.user_info.user_feature = {}
            return

        user_feature = pd.DataFrame()
        for i, col in enumerate(user_feat_names):
            user_feature[col] = response["results"][i]["values"]
        rec_data.user_info.user_feature = user_feature.iloc[0].to_dict()
        logging.info(f"******user {userid}, feast done******")

    def get_all_item_feature(self):
        item_feature_request = {
            "feature_service": "ml_item",
            "entities": {
                ITEMID: [v for v in range(MAX_ITEMID+1)]
            }
        }
        response = json.loads(
            requests.post(self.url, data=json.dumps(item_feature_request)).text)

        item_feat_names = response["metadata"]["feature_names"]
        if len(item_feat_names) != len(response["results"]) or len(item_feat_names) == 0:
            logging.error(f"******feast item feature num error, cannot start server******")
            return None
        all_item_feature = pd.DataFrame()
        for i, col in enumerate(item_feat_names):
            all_item_feature[col] = response["results"][i]["values"]
        logging.info("****** feast item feature prefetch done******")
        return all_item_feature