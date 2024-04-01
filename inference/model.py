# Copyright 2022 Intel Corporation
# SPDX-License-Identifier: MIT
#
"""This module implements AutoSteer's inference mode."""
import numpy as np
import torch
import torch.optim
import joblib
import os
from utils.custom_logging import logger
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from inference import net

CUDA = torch.cuda.is_available()
print(f'Use CUDA: {CUDA}')

EPS = 1e-15
PADDED_Y_VALUE = -1.0

def listMLE(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    mask = y_true == padded_value_indicator
    y_pred[mask] = float("-inf")
    max_pred_values, _ = y_pred.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = y_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    # observation_loss = torch.log(cumsums+EPS) - preds_sorted_by_true_minus_max
    observation_loss = cumsums - preds_sorted_by_true_minus_max
    observation_loss[mask] = 0.0
    return torch.mean(torch.sum(observation_loss, dim=1))

def listMLE_legacy(y_pred, y_true, eps=EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]
    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
    mask = y_true_sorted == padded_value_indicator
    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    observation_loss[mask] = 0.0
    return torch.mean(torch.sum(observation_loss, dim=1))

def _nn_path(base):
    return os.path.join(base, 'nn_weights')


def _x_transform_path(base):
    return os.path.join(base, 'x_transform')


def _y_transform_path(base):
    return os.path.join(base, 'y_transform')


def _channels_path(base):
    return os.path.join(base, 'channels')


def _n_path(base):
    return os.path.join(base, 'n')


def _inv_log1p(x):
    return np.exp(x) - 1


class BaoData:
    def __init__(self, data):
        assert data
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx]['tree'], self.__data[idx]['target']


def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    # pylint: disable=not-callable
    targets = torch.tensor(np.array(targets))
    return trees, targets


class BaoRegressionModel:
    """This class represents the Bao regression model used to predict execution times of query plans"""

    def __init__(self, plan_preprocessor):
        self.__net = None

        log_transformer = preprocessing.FunctionTransformer(np.log1p, _inv_log1p, validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([('log', log_transformer), ('scale', scale_transformer)])

        self.__tree_transform = plan_preprocessor
        self.__in_channels = None
        self.__n = 0

    def num_items_trained_on(self):
        return self.__n

    def load(self, path):
        logger.info('Load Bao regression model from directory %s ...', path)
        with open(_n_path(path), 'rb') as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), 'rb') as f:
            self.__in_channels = joblib.load(f)

        self.__net = net.BaoNet(self.__in_channels)
        self.__net.load_state_dict(torch.load(_nn_path(path)))
        self.__net.eval()

        with open(_y_transform_path(path), 'rb') as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), 'rb') as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)

        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), 'wb') as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), 'wb') as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), 'wb') as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), 'wb') as f:
            joblib.dump(self.__n, f)

    def fit(self, x_train, y_train, x_test, y_test, ltr=False):
        if ltr:
            for qid in y_train:
                y_train[qid] = np.array(y_train[qid])
                y_train[qid] = self.__pipeline.fit_transform(y_train[qid].reshape(-1, 1)).astype(np.float32)

            for qid in y_test:
                y_test[qid] = np.array(y_test[qid])
                y_test[qid] = self.__pipeline.fit_transform(y_test[qid].reshape(-1, 1)).astype(np.float32)

            fit_array = []
            for i in x_train:
                fit_array.extend(x_train[i])
            for i in x_test:
                fit_array.extend(x_test[i])

            self.__tree_transform.fit(fit_array)

            for qid in x_test:
                x_test[qid] = self.__tree_transform.transform(x_test[qid])
            for qid in x_train:
                x_train[qid] = self.__tree_transform.transform(x_train[qid])
            pairs_train = [DataLoader(list(zip(x_train[k], y_train[k])),batch_size=len(x_train[k]), shuffle=True, collate_fn=collate) for k in x_train]
            pairs_test = [DataLoader(list(zip(x_test[k], y_test[k])),batch_size=len(x_test[k]), shuffle=True, collate_fn=collate) for k in x_test]
            for inp, _ in pairs_train[0]:
                in_channels = inp[0][0].shape[0]
                logger.info('Initial input channels: %s', in_channels)
                break

            self.__net = net.BaoNet(in_channels)
            self.__in_channels = in_channels
            if CUDA:
                self.__net = self.__net.cuda()

            optimizer = torch.optim.Adam(self.__net.parameters())
            loss_fn = listMLE
            # loss_fn = listMLE_legacy

            training_losses = []
            test_losses = []
            for epoch in range(100):
                training_loss_accum = 0
                test_loss_accum = 0
                for dataset_train in pairs_train:
                    for x, y in dataset_train:
                        if CUDA:
                            y = y.cuda()
                        y_pred = self.__net(x)
                        loss = loss_fn(y_pred, y)
                        training_loss_accum += loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                for dataset_test in pairs_test:
                    for x, y in dataset_test:
                        if CUDA:
                            y = y.cuda()
                        y_pred = self.__net(x)
                        test_loss_accum += loss_fn(y_pred, y).item()

                training_loss_accum /= len(pairs_train)
                test_loss_accum /= len(pairs_test)
                training_losses.append(training_loss_accum)
                test_losses.append(test_loss_accum)
                if epoch % 1 == 0:
                    logger.info('Epoch %s\ttrain. loss\t%.4f', epoch, training_loss_accum)
                    logger.info('Epoch %s\ttest loss\t%.4f', epoch, test_loss_accum)

                # stopping condition
                if len(training_losses) > 10 and training_losses[-1] < 0.1:
                    last_two = np.min(training_losses[-2:])
                    if last_two > training_losses[-10] or (training_losses[-10] - last_two < 0.0001):
                        logger.info('Stopped training from convergence condition at epoch %s', epoch)
                        break
            else:
                logger.info('Stopped training after max epochs')
            return training_losses, test_losses

        else:
            if isinstance(y_train, list):
                y_train = np.array(y_train)

            if isinstance(y_test, list):
                y_test = np.array(y_test)

            self.__n = len(x_train)

            # transform the set of trees into feature vectors using a log
            self.__pipeline.fit(y_train.reshape(-1, 1))
            y_train = self.__pipeline.transform(y_train.reshape(-1, 1)).astype(np.float32)
            y_test = self.__pipeline.transform(y_test.reshape(-1, 1)).astype(np.float32)

            self.__tree_transform.fit(x_train + x_test)
            x_train = self.__tree_transform.transform(x_train)
            x_test = self.__tree_transform.transform(x_test)

            pairs = list(zip(x_train, y_train))
            pairs_test = list(zip(x_test, y_test))

            dataset_train = DataLoader(pairs, batch_size=32, shuffle=True, collate_fn=collate)
            dataset_test = DataLoader(pairs_test, batch_size=32, shuffle=True, collate_fn=collate)

            # determine the initial number of channels
            for inp, _ in dataset_train:
                in_channels = inp[0][0].shape[0]
                logger.info('Initial input channels: %s', in_channels)
                break

            self.__net = net.BaoNet(in_channels)
            self.__in_channels = in_channels
            if CUDA:
                self.__net = self.__net.cuda()

            optimizer = torch.optim.Adam(self.__net.parameters())
            loss_fn = torch.nn.MSELoss()

            training_losses = []
            test_losses = []
            for epoch in range(100):
                training_loss_accum = 0
                test_loss_accum = 0
                for x, y in dataset_train:
                    if CUDA:
                        y = y.cuda()
                    # print(len(x), x[0], y)
                    y_pred = self.__net(x)
                    loss = loss_fn(y_pred, y)
                    training_loss_accum += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for x, y in dataset_test:
                    if CUDA:
                        y = y.cuda()
                    y_pred = self.__net(x)
                    test_loss_accum += loss_fn(y_pred, y).item()

                training_loss_accum /= len(dataset_train)
                test_loss_accum /= len(dataset_test)
                training_losses.append(training_loss_accum)
                test_losses.append(test_loss_accum)
                if epoch % 1 == 0:
                    logger.info('Epoch %s\ttrain. loss\t%.4f', epoch, training_loss_accum)
                    logger.info('Epoch %s\ttest loss\t%.4f', epoch, test_loss_accum)

                # stopping condition
                if len(training_losses) > 10 and training_losses[-1] < 0.1:
                    last_two = np.min(training_losses[-2:])
                    if last_two > training_losses[-10] or (training_losses[-10] - last_two < 0.0001):
                        logger.info('Stopped training from convergence condition at epoch %s', epoch)
                        break
            else:
                logger.info('Stopped training after max epochs')
            return training_losses, test_losses

    def predict(self, x):
        """Predict one or more samples"""
        if not isinstance(x, list):
            x = [x]  # x represents one sample only
        # x = [ast.literal_eval(x_) if isinstance(x_, str) else x_ for x_ in x]
        x = self.__tree_transform.transform(x)
        self.__net.eval()
        prediction = self.__net(x).cpu().detach().numpy()
        return self.__pipeline.inverse_transform(prediction)
