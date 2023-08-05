import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
import pickle
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from data_exchange_center.constants import *
from data_exchange_center.paths import OFFLINE_IMP_PATH, USER_VECTOR_PATH, ITEM_VECTOR_PATH, RECALL_MODEL_PTH_PATH


def load_data():
    offline_imp = pd.read_csv(OFFLINE_IMP_PATH)
    return offline_imp


class MFModel(torch.nn.Module):
    def __init__(self, user_cnt, item_cnt):
        super().__init__()
        self.P = torch.nn.Embedding(user_cnt, RECALL_EMB_DIM)
        self.Q = torch.nn.Embedding(item_cnt, RECALL_EMB_DIM)
        torch.nn.init.xavier_uniform_(self.P.weight.data)
        torch.nn.init.xavier_uniform_(self.Q.weight.data)

    def forward(self, x):
        x = x.to(torch.int32)
        user, item = x[:, 0], x[:, 1]
        u_emb = self.P(user)
        i_emb = self.Q(item)
        y = torch.sum(torch.multiply(u_emb, i_emb), dim=1)
        return torch.sigmoid(y)


if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    offline_imp = load_data()
    train_data = offline_imp[offline_imp[ISTEST] == 0].reset_index(drop=True)
    test_data = offline_imp[offline_imp[ISTEST] == 1].reset_index(drop=True)

    user_cnt, item_cnt = MAX_USERID+1, MAX_ITEMID+1
    rating_matrix = np.zeros((user_cnt, item_cnt))
    for u, i, l in train_data[[USERID, ITEMID, LABEL]].values:
        if l == 1:
            rating_matrix[u][i] = 1

    train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    epoch = 10
    batch_size = 1024
    model = MFModel(user_cnt, item_cnt)
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    model.train()

    for ep in range(epoch):
        for i in tqdm.tqdm(range(len(train_data) // batch_size + 1)):
            train_indices = range(i*batch_size, min((i+1)*batch_size, len(train_data)))
            x, target = torch.Tensor(train_data.loc[train_indices,[USERID, ITEMID]].values), \
                        torch.Tensor(train_data.loc[train_indices,LABEL].values)
            preds = model(x)
            loss = loss_func(preds, target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()

        targets, predicts = list(), list()
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_data) // batch_size + 1)):
                test_indices = range(i * batch_size, min((i+1) * batch_size, len(test_data)))
                x, target = torch.Tensor(test_data.loc[test_indices, [USERID, ITEMID]].values), \
                            torch.Tensor(test_data.loc[test_indices, LABEL].values)
                preds = model(x)
                targets.extend(target.tolist())
                predicts.extend(preds.tolist())
        auc = roc_auc_score(targets, predicts)
        print(f"epoch: {ep}, test auc: {auc}")
    torch.save(model, RECALL_MODEL_PTH_PATH)

    # save vectors
    user_vector = model.P.weight.data.numpy()
    item_vector = model.Q.weight.data.numpy()
    print(f"user_vector: {user_vector.shape}, item_vector: {item_vector.shape}")
    pickle.dump(user_vector.tolist(), open(USER_VECTOR_PATH, 'wb'), protocol=4)
    pickle.dump(item_vector.tolist(), open(ITEM_VECTOR_PATH, 'wb'), protocol=4)