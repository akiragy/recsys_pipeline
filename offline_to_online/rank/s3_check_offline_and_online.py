import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import numpy as np
import pickle
import tritonclient.grpc as triton_client
from data_exchange_center.parse_feat_meta import build_test_samples
from data_exchange_center.paths import FEAT_META_PATH, RANK_MODEL_PTH_PATH
np.set_printoptions(precision=6)


def load_data():
    feat_meta = pickle.load(open(FEAT_META_PATH, "rb"))
    model = torch.load(RANK_MODEL_PTH_PATH)
    return feat_meta, model


if __name__ == "__main__":

    # build test samples
    feat_meta, model = load_data()
    samples = build_test_samples(feat_meta)

    # offline prediction
    offline_pred = model(torch.Tensor(samples)).detach().numpy()

    # online prediction
    client = triton_client.InferenceServerClient(url="localhost:8001")
    samples = np.array(samples)
    input_name = "ml_input"
    inputs = [triton_client.InferInput(input_name, samples.shape, "FP32")]
    inputs[0].set_data_from_numpy(samples.astype(np.float32))
    output_name = "ml_output"
    outputs = [triton_client.InferRequestedOutput(output_name)]

    response = client.infer(model_name="ml_rec",
                           inputs=inputs,
                           outputs=outputs)
    online_pred = response.as_numpy(output_name).reshape([-1])
    print(f"offline_pred: {offline_pred}, online_pred: {online_pred}")
