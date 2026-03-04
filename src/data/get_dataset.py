import os
from collections.abc import Iterable
from pathlib import Path
import scipy.sparse as sp
import pandas as pd
import numpy as np
import json

import torch
from torch_geometric.utils import dense_to_sparse

from src.data.wrapper import wrap_traffic_dataset
from src.data.utils import (
    generate_regression_task,
    generate_split, StandardScaler
)


def get_connectivity(adj_matrix_path):
    adj = sp.load_npz(adj_matrix_path)
    edge_indices, edge_values = dense_to_sparse(torch.tensor(adj.toarray()))
    edge_values = 1 / edge_values  # edge weights are [0, 1], convert to float
    return edge_indices, edge_values


def get_raw_data(dataset_path, split_ratio, n_hist, n_pred, norm):
    assert norm, 'Traffic data should be normalized for better performance'

    X_s, y_s = list(), list()
    scaler = None
    for split in ['train', 'val', 'test']:
        data_path = list(Path(dataset_path).glob(f'{split}*_hist{n_hist}_pred{n_pred}.npz'))

        if data_path:
            data = np.load(data_path[0])
            X_s.append(data['x'])
            y_s.append(data['y'])
            if split == 'train':
                scaler = StandardScaler(mean=data['mean'], std=data['std'])
        else:
            print(f"preprocessed data not found at {dataset_path}, generating new data")
            h5_path = list(Path(dataset_path).glob('*.h5'))
            add_time_in_day, add_time_in_week = None, None
            if h5_path:
                print(f'Loading data from {h5_path[0]}')
                df = pd.read_hdf(h5_path[0])
                add_time_in_day, add_time_in_week = True, False
                features, targets = generate_regression_task(
                    df, n_hist, n_pred,
                    add_time_in_day=add_time_in_day,
                    add_day_in_week=add_time_in_week,
                )
                features_fill, targets_fill = generate_regression_task(
                    df, n_hist, n_pred,
                    add_time_in_day=add_time_in_day,
                    add_day_in_week=add_time_in_week,
                    replace_drops=True,
                )

                (
                    (train_x, val_x, test_x,
                     train_y, val_y, test_y, scaler),
                    train_idx, val_idx, test_idx,
                ) = generate_split(
                    (features, features_fill),
                    (targets, targets_fill),
                    split_ratio,
                    norm
                )
            else:
                data_csv_path = os.path.join(dataset_path, 'vel.csv')
                print(f'Loading data from {data_csv_path}')
                # process X and get node features
                X = pd.read_csv(data_csv_path).to_numpy()  # shape [time slices, nodes]
                if len(X.shape) == 2:
                    X = np.expand_dims(X, 1)
                X = X.transpose((0, 2, 1)).astype(np.float32)
                features, targets = generate_regression_task(
                    X, n_hist, n_pred
                )
                features_fill, targets_fill = generate_regression_task(
                    X, n_hist, n_pred, replace_drops=True
                )

                (
                    (train_x, val_x, test_x,
                     train_y, val_y, test_y, scaler),
                    train_idx, val_idx, test_idx,
                ) = generate_split(
                    (features, features_fill),
                    (targets, targets_fill),
                    split_ratio,
                    norm
                )

            suffix = ''
            if add_time_in_day is not None:
                if add_time_in_day:
                    suffix += '_day'
                if add_time_in_week:
                    suffix += '_week'
            suffix += f'_hist{n_hist}_pred{n_pred}'
            np.savez_compressed(
                os.path.join(dataset_path, f'train{suffix}.npz'),
                x=train_x,
                y=train_y,
                idx=train_idx,
                mean=scaler.mean,
                std=scaler.std
            )
            np.savez_compressed(
                os.path.join(dataset_path, f'val{suffix}.npz'),
                x=val_x,
                y=val_y,
                idx=val_idx
            )
            np.savez_compressed(
                os.path.join(dataset_path, f'test{suffix}.npz'),
                x=test_x,
                y=test_y,
                idx=test_idx
            )
            return train_x, val_x, test_x, train_y, val_y, test_y, scaler

    return X_s + y_s + [scaler]


def get_dataset(
        mode: str = 'pretrain',
        data_dir: str = None,
        dataset_name: str = 'pems-bay',
        n_hist: int = 12,
        n_pred: int = 12,
        split_ratio=(20, 10),
        graph_token=True,
        seed=0,
        task='pred',
        norm=True,
):
    assert mode in [
        "pretrain",
        "finetune",
        "valid",
        "test",
        "debug"
    ]
    assert task == 'pred', 'no classification task implemented'
    if not data_dir:
        data_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(data_dir, 'traffic')

    dataset_path = os.path.join(data_dir, dataset_name)
    train_x, val_x, test_x, train_y, val_y, test_y, scaler = get_raw_data(
        dataset_path,
        split_ratio,
        n_hist,
        n_pred,
        norm
    )
    if scaler:
        print(f'Using normalization with {str(scaler)}')
    else:
        print('*** warning: no normalization! ***')
    edge_indices, edge_values = get_connectivity(os.path.join(dataset_path, 'adj.npz'))

    dataset = wrap_traffic_dataset(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        edge_indices=edge_indices.numpy(),
        edge_values=edge_values.numpy(),
        graph_token=graph_token,
        seed=seed,
        scaler=scaler
    )

    INFO = {
        'train_dataset': dataset.train_data,
        'valid_dataset': dataset.valid_data,
        'test_dataset': dataset.test_data,
    }

    print(f' > {dataset_name} loaded!')
    print(INFO)
    print(f' > dataset info ends')
    return INFO
