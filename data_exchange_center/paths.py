from data_exchange_center.constants import *

DATA_CENTER_PATH = "../../data_exchange_center"

# ml-1m raw data
RAW_USER_PATH = DATA_CENTER_PATH + "/offline/ml-1m/users.dat"
RAW_ITEM_PATH = DATA_CENTER_PATH + "/offline/ml-1m/movies.dat"
RAW_RATING_PATH = DATA_CENTER_PATH + "/offline/ml-1m/ratings.dat"

# split into offline and online
OFFLINE_IMP_PATH = DATA_CENTER_PATH + "/offline/common/offline_imp.csv"
ONLINE_IMP_PATH = DATA_CENTER_PATH + "/offline/common/online_imp.csv"

# term for recall and pit-join for rank
USER_FILTER_PATH = DATA_CENTER_PATH + "/offline/common/user_filter_term.pkl"
IMP_TERM_PATH = DATA_CENTER_PATH + "/offline/common/imp_term.pkl"
USER_TERM_PATH = DATA_CENTER_PATH + "/offline_to_online/recall/user_term.pkl"
ITEM_TERM_PATH = DATA_CENTER_PATH + "/offline_to_online/recall/item_term.pkl"

# vector for recall
RECALL_MODEL_PTH_PATH = DATA_CENTER_PATH + "/offline/recall/fm.pt"
USER_VECTOR_PATH = DATA_CENTER_PATH + "/offline_to_online/recall/user_vector.pkl"
ITEM_VECTOR_PATH = DATA_CENTER_PATH + "/offline_to_online/recall/item_vector.pkl"

# online feature fetch for rank
USER_ENTITY_PATH = DATA_CENTER_PATH + "/offline_to_online/rank/user_entity.csv"
ITEM_ENTITY_PATH = DATA_CENTER_PATH + "/offline_to_online/rank/item_entity.csv"
FEAT_META_PATH = DATA_CENTER_PATH + "/offline_to_online/rank/feat_meta.pkl"
USER_ENTITY_FEAST_PATH = DATA_CENTER_PATH + "/online/feast/user_entity.parquet"
ITEM_ENTITY_FEAST_PATH = DATA_CENTER_PATH + "/online/feast/item_entity.parquet"

# offline train for rank
TRAIN_DATA_PATH = DATA_CENTER_PATH + "/offline/rank/train_data.csv"
TEST_DATA_PATH = DATA_CENTER_PATH + "/offline/rank/test_data.csv"
RANK_MODEL_PTH_PATH = DATA_CENTER_PATH + "/offline/rank/deepfm.pt"

# online model for rank
RANK_MODEL_TRITON_PATH = DATA_CENTER_PATH + "/online/triton/ml_rec/1/model/model.onnx"
