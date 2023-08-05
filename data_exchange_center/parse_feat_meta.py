from data_exchange_center.constants import *


def get_feature_def(feat_meta):
    sparse_id_dims = []
    sparse_side_dims = []
    user_sparse, item_sparse, user_dense, item_dense = feat_meta["sparse"]["user"], feat_meta["sparse"]["item"], \
                                                       feat_meta["dense"]["user"], feat_meta["dense"]["item"]
    # parse id features
    sparse_id_dims.append(user_sparse[USERID][1])
    sparse_id_dims.append(item_sparse[ITEMID][1])
    sparse_id_feat = [USERID, ITEMID]
    del user_sparse[USERID], item_sparse[ITEMID]

    # parse other sparse features
    for col in user_sparse:
        sparse_side_dims.append(user_sparse[col][1])
    for col in item_sparse:
        sparse_side_dims.append(item_sparse[col][1])
    sparse_side_feat = list(user_sparse.keys()) + list(item_sparse.keys())

    # parse dense features
    dense_dim = len(user_dense) + len(item_dense)
    dense_feat = list(user_dense.keys()) + list(item_dense.keys())
    # print(f"sparse_id_dims: {sparse_id_dims}, sparse_side_dims: {sparse_side_dims}, dense_dim: {dense_dim}, sparse_id_feat: {sparse_id_feat}, sparse_side_feat: {sparse_side_feat}, dense_feat: {dense_feat}")
    return sparse_id_dims, sparse_side_dims, dense_dim, sparse_id_feat, sparse_side_feat, dense_feat


def build_test_samples(feat_meta):
    sparse_id_dims, sparse_side_dims, dense_dim, _, _, _ = get_feature_def(feat_meta)
    sample1, sample2 = [], []
    for col in sparse_id_dims:
        sample1.append(0)
        sample2.append(col - 1)
    for col in sparse_side_dims:
        sample1.append(0)
        sample2.append(col - 1)
    for i in range(dense_dim):
        sample1.append(0.)
        sample2.append(1.)
    return [sample1, sample2]
