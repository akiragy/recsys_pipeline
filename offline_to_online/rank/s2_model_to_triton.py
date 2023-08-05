import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import pickle
from data_exchange_center.paths import FEAT_META_PATH, RANK_MODEL_PTH_PATH, RANK_MODEL_TRITON_PATH
from data_exchange_center.parse_feat_meta import build_test_samples


def load_data():
    feat_meta = pickle.load(open(FEAT_META_PATH, "rb"))
    model = torch.load(RANK_MODEL_PTH_PATH)
    return feat_meta, model


if __name__ == "__main__":
    feat_meta, model = load_data()
    samples = build_test_samples(feat_meta)
    torch.onnx.export(model, torch.Tensor(samples), RANK_MODEL_TRITON_PATH,
                      input_names=["ml_input"], output_names=["ml_output"], opset_version=12,
                      dynamic_axes={
                          "ml_input": {0: "batch"},
                          "ml_output": {0: "batch"}
                      })
