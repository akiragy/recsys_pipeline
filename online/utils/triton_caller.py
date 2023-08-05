import tritonclient.grpc as triton_client
import numpy as np
import logging
import pickle
from data_exchange_center.parse_feat_meta import get_feature_def
from data_exchange_center.paths import FEAT_META_PATH
from online.utils.dto import *
from online.utils.feast_caller import FeastCaller


class TritonCaller:
    def __init__(self):
        self.url = "localhost:8001"

        feat_meta = pickle.load(open(FEAT_META_PATH, "rb"))
        _, _, _, sparse_id_feat, sparse_side_feat, dense_feat = get_feature_def(feat_meta)

        self.sparse_id_feat = sparse_id_feat
        self.sparse_side_feat = sparse_side_feat
        self.dense_feat = dense_feat
        self.all_item_feature = FeastCaller().get_all_item_feature()
        assert self.all_item_feature is not None

    def call(self, rec_data: RecData) -> None:
        all_feature = rec_data.user_info.user_feature

        n_items = len(rec_data.item_list)
        for i in range(n_items):
            rec_data.item_list[i].item_feature = self.all_item_feature.iloc[rec_data.item_list[i].itemid].to_dict()

        has_none_col = set()

        partition_size = rec_data.config.rank_partition_size
        mb_iter = n_items // partition_size if n_items % partition_size == 0 else n_items // partition_size + 1
        for mb in range(mb_iter):
            samples = []
            # sample building
            for i in range(mb * partition_size, min((mb + 1) * partition_size, n_items)):
                sample = []
                item_feature = rec_data.item_list[i].item_feature
                all_feature.update(item_feature)
                for col in self.sparse_id_feat + self.sparse_side_feat:
                    if all_feature[col] is None:
                        has_none_col.add(col)
                        sample.append(0)
                    else:
                        sample.append(all_feature[col])
                for col in self.dense_feat:
                    if all_feature[col] is None:
                        has_none_col.add(col)
                        sample.append(0.0)
                    else:
                        sample.append(all_feature[col])
                samples.append(sample)

            # online prediction
            client = triton_client.InferenceServerClient(url=self.url)
            samples_np = np.array(samples)
            input_name = "ml_input"
            inputs = [triton_client.InferInput(input_name, samples_np.shape, "FP32")]
            inputs[0].set_data_from_numpy(samples_np.astype(np.float32))
            output_name = "ml_output"
            outputs = [triton_client.InferRequestedOutput(output_name)]

            response = client.infer(model_name="ml_rec",
                                    inputs=inputs,
                                    outputs=outputs)
            online_pred = response.as_numpy(output_name)
            for i in range(mb * partition_size, min((mb + 1) * partition_size, n_items)):
                rec_data.item_list[i].score = online_pred[i - mb * partition_size][0]

        if len(has_none_col) != 0:
            logging.warning(f"triton user {rec_data.user_info.userid} has none col!, {has_none_col}")

        from operator import attrgetter
        rec_data.item_list = sorted(rec_data.item_list, key=attrgetter('score'), reverse=True)
        logging.info(f"******user {rec_data.user_info.userid}, triton done******")


