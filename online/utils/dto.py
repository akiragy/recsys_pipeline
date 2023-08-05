class UserInfo:
    def __init__(self):
        self.userid = None
        self.user_term = None
        self.user_vector = None
        self.user_filter = None
        self.user_feature = None


class ItemInfo:
    def __init__(self):
        self.itemid = None
        self.item_feature = None
        self.score = None


class ConfigInfo:
    def __init__(self):
        self.recall_size = 200
        self.rank_partition_size = 128
        self.response_size = 50
        assert self.response_size <= self.recall_size
        assert self.rank_partition_size <= 256


class RecData:
    def __init__(self):
        self.user_info = UserInfo()
        self.item_list = []
        self.config = ConfigInfo()
