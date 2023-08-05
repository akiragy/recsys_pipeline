# previous columns
USERID = "userid"
ITEMID = "itemid"
RATING = "rating"
TS = "ts"

GENDER = "gender"
AGE = "age"
OCCUPATION = "occupation"
ZIPCODE = "zipcode"

TITLE = "title"
GENRES = "genres"


# new columns
QUERYID = "queryid"
LABEL = "label"
ISTEST = "istest"
ONLINE_LAST_N = 10
OFFLINE_TEST_RATIO = 0.2
LAST_N_GENRE_CNT = 10
USER_VECTOR = "user_vector"
ITEM_VECTOR = "item_vector"


# mapping
EMPTY_KEY = "empty_key"
GENRES_MAPPING = {
    EMPTY_KEY: 0,
    "Action": 1,
    "Adventure": 2,
    "Animation": 3,
    "Children's": 4,
    "Comedy": 5,
    "Crime": 6,
    "Documentary": 7,
    "Drama": 8,
    "Fantasy": 9,
    "Film-Noir": 10,
    "Horror": 11,
    "Musical": 12,
    "Mystery": 13,
    "Romance": 14,
    "Sci-Fi": 15,
    "Thriller": 16,
    "War": 17,
    "Western": 18
}

GENDER_MAPPING = {
    EMPTY_KEY: 0,
    "M": 1,
    "F": 2
}

AGE_MAPPING = {
    EMPTY_KEY: 0,
    1: 1,
    18: 2,
    25: 3,
    35: 4,
    45: 5,
    50: 6,
    56: 7
}

OCCUPATION_MAPPING = {
    EMPTY_KEY: 0
}
for i in range(21):
    OCCUPATION_MAPPING[i] = i + 1


# config
RECALL_EMB_DIM = 16
RECALL_REDIS_PREFIX = "user"
RECALL_REDIS_TERM_KEY = "term"
RECALL_REDIS_VECTOR_KEY = "vector"
RECALL_REDIS_FILTER_KEY = "filter"
ITEM_ES_INDEX = "item_index"
MAX_USERID = 6040
MAX_ITEMID = 3952
ES_KEY = "XvQTkEiJj47=ZSA-epOG"