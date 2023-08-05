import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import torch
import pickle
from data_exchange_center.constants import *
from data_exchange_center.paths import FEAT_META_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, RANK_MODEL_PTH_PATH
from data_exchange_center.parse_feat_meta import get_feature_def
from offline.torchfm.model import *
from offline.torchfm.train import *


def load_data():
    feat_meta = pickle.load(open(FEAT_META_PATH, "rb"))
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    return feat_meta, train_data, test_data


def get_dataloader(train_data, test_data, sparse_id_feat, sparse_side_feat, dense_feat):
    train_data = train_data[sparse_id_feat + sparse_side_feat + dense_feat + [LABEL]]
    test_data = test_data[sparse_id_feat + sparse_side_feat + dense_feat + [LABEL]]
    print(f"load feat: {sparse_id_feat + sparse_side_feat + dense_feat}")
    train_dataset = MLDataSet(train_data, sparse_id_feat, sparse_side_feat, dense_feat, LABEL)
    test_dataset = MLDataSet(test_data, sparse_id_feat, sparse_side_feat, dense_feat, LABEL)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    print(len(train_dataloader), len(test_dataloader))
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    # param
    device = args.device
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    epoch = args.epoch
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    # data
    feat_meta, train_data, test_data = load_data()
    sparse_id_dims, sparse_side_dims, dense_dim, sparse_id_feat, sparse_side_feat, dense_feat = get_feature_def(feat_meta)
    train_dataloader, test_dataloader = get_dataloader(train_data, test_data, sparse_id_feat, sparse_side_feat, dense_feat)

    # model
    model = DeepFactorizationMachineModel(sparse_id_dims, sparse_side_dims, dense_dim, embed_dim=16, embed_dim_side=2, mlp_dims=(4,), dropout=0.2).to(device)
    print(model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=5, save_path=RANK_MODEL_PTH_PATH)

    # train
    for i in range(epoch):
        torch_train(model, optimizer, train_dataloader, criterion, device)
        auc = torch_test(model, test_dataloader, device)
        print(f"epoch {i}: test auc: {auc}")
        if not early_stopper.is_continuable(model, auc):
            print(f"test: best auc: {early_stopper.best_criterion}")
            break
