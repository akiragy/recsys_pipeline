import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from datetime import datetime

from data_exchange_center.paths import USER_ENTITY_PATH, ITEM_ENTITY_PATH, USER_ENTITY_FEAST_PATH, ITEM_ENTITY_FEAST_PATH

user_entity = pd.read_csv(USER_ENTITY_PATH)
item_entity = pd.read_csv(ITEM_ENTITY_PATH)
tstz = pd.to_datetime(datetime.today()).tz_localize('Asia/Shanghai')  # time zone is required
user_entity["timestamp_field"] = tstz
user_entity["created_timestamp_column"] = tstz
item_entity["timestamp_field"] = tstz
item_entity["created_timestamp_column"] = tstz

user_entity.to_parquet(USER_ENTITY_FEAST_PATH)
item_entity.to_parquet(ITEM_ENTITY_FEAST_PATH)