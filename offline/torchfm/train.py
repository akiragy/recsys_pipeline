import torch
import tqdm
from sklearn.metrics import roc_auc_score


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_criterion = 0
        self.save_path = save_path

    def is_continuable(self, model, criterion):
        if criterion > self.best_criterion:
            self.best_criterion = criterion
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def torch_train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        x, target = fields.to(device), target.to(device)
        y = model(x)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def torch_test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            x, target = fields.to(device), target.to(device)
            y = model(x)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


class MLDataSet(torch.utils.data.DataLoader):
    def __init__(self, feat_df, sparse_id_df, sparse_side_df, dense_fea, label_col):
        self.x = feat_df[sparse_id_df + sparse_side_df + dense_fea].values
        self.y = feat_df[label_col].values

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item] if self.y is not None else None
