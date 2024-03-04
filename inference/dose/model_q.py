# Copyright 2022 Intel Corporation
# SPDX-License-Identifier: MIT
#
"""This module implements AutoSteer's inference mode."""
import ast
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
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Flatten
from inference.dose.trainer import *
from inference.dose.util import Normalizer
from inference.dose.database_util import Encoding
import torch.nn as nn
from inference.dose.model import Dose

CUDA = torch.cuda.is_available()

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

def normalizer_path(base):
    return os.path.join(base, 'normalizer')

def encoder_path(base):
    return os.path.join(base, 'encoder')

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

    def fit(self, x_train, y_train, x_test, y_test):
        if isinstance(y_train, list):
            y_train = np.array(y_train)

        if isinstance(y_test, list):
            y_test = np.array(y_test)

        # Need to use ast here b/c some json plans use single quotes, but json lib expects double quotes
        # x_train = [ast.literal_eval(x) if isinstance(x, str) else x for x in x_train]
        # x_test = [ast.literal_eval(x) if isinstance(x, str) else x for x in x_test]

        self.__n = len(x_train)

        # transform the set of trees into feature vectors using a log
        y_train = self.__pipeline.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
        y_test = self.__pipeline.fit_transform(y_test.reshape(-1, 1)).astype(np.float32)

        self.__tree_transform.fit(x_train + x_test)
        x_train = self.__tree_transform.transform(x_train)
        x_test = self.__tree_transform.transform(x_test)

        pairs = list(zip(x_train, y_train))
        pairs_test = list(zip(x_test, y_test))

        dataset_train = DataLoader(pairs, batch_size=16, shuffle=True, collate_fn=collate)
        dataset_test = DataLoader(pairs_test, batch_size=16, shuffle=True, collate_fn=collate)

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

class LeafEmbeddingModel:
    def __init__(self, max_len, vocab_size):
        self.model = Sequential()
        self.model.add(SimpleRNN(128, return_sequences=True,
            activation='relu', input_shape=(max_len, vocab_size+1)))
        self.model.add(SimpleRNN(128, return_sequences=True, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        try:
            self.model.load_weights('./nn/model/embedding_model.keras')
            logger.info('Successfully loaded leaf embedding model from directory ./nn/model/embedding_model.keras ...')
        except:
            logger.info('Failed to load leaf embedding model from directory ./nn/model/embedding_model.keras, Retrain the model ...')
            self.__retrain()
    
    def __retrain(self):
        pass

    def get_model(self):
        return self.model

class Args:
    bs = 16
    lr = 0.001
    epochs = 40
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.2
    sch_decay = 0.6
    device = 'cpu'
    newpath = './results/full/cost/'
    to_predict = 'cost'

class DoseModel:

    def __init__(self, plan_preprocessor) -> None:
        self.__n = 0
        self.__tree_transform = plan_preprocessor
        self.__net = None
        self.normalizer = None
        self.encoder = None

    def num_items_trained_on(self):
        return self.__n
    
    def load(self, path):
        logger.info('Load Dose model from directory %s ...', path)
        self.__net = Dose()
        self.__net.load_state_dict(torch.load(_nn_path(path)))
        self.__net.eval()
        with open(normalizer_path(path), 'rb') as f:
            self.normalizer = joblib.load(f)
        with open(encoder_path(path), 'rb') as f:
            self.encoder = joblib.load(f)
        with open(_x_transform_path(path), 'rb') as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(normalizer_path(path), 'wb') as f:
            joblib.dump(self.normalizer, f)
        with open(encoder_path(path), 'wb') as f:
            joblib.dump(self.encoder, f)
        with open(_x_transform_path(path), 'wb') as f:
            joblib.dump(self.__tree_transform, f)

    
    def fit(self, x_train, y_train, x_test, y_test):
        if isinstance(y_train, list):
            y_train = np.array(y_train)

        if isinstance(y_test, list):
            y_test = np.array(y_test)

        self.__n = len(x_train)

        # transform the set of trees into feature vectors using a log
        min, max = (np.concatenate((y_train, y_test))).min(), (np.concatenate((y_train, y_test)).max())
        self.normalizer = Normalizer(min, max)  
        
        logger.info('fit_transforming training data ...')
        train_df = self.__tree_transform.fit_transform(x_train, y_train)
        logger.info('fit_transforming the training data finished')
        
        logger.info('fit_transforming testing data ...')
        test_df = self.__tree_transform.fit_transform(x_test, y_test)
        logger.info('fit_transforming the testing data finished')

        column_min_max_vals = {}
        with open('./data/tpcds_statistics.csv', 'r') as f:
            for line in f:
                line = line.strip().split(',')
                column_min_max_vals[line[0]] = (eval(line[1]), eval(line[2]))

        col2idx = self.__tree_transform.get_col_name('tpcds')
        self.encoder = Encoding(column_min_max_vals, col2idx)

        train_ds = PlanTreeDataset(train_df, self.encoder, self.normalizer)

        test_ds = PlanTreeDataset(test_df, self.encoder, self.normalizer)
        crit = nn.MSELoss()

        self.__net = Dose()
        _ = self.__net.to('cpu')
        args = Args()
        logger.info('Start training ...')
        training_losses, test_losses = train(self.__net, train_ds, test_ds, crit, self.normalizer,args)
        return training_losses, test_losses
    
    def predict(self, x):
        if not isinstance(x, list):
            x = [x]  # x represents one sample only
        # x = [ast.literal_eval(x_) if isinstance(x_, str) else x_ for x_ in x]
        x = self.__tree_transform.fit_transform(x)
        predict_ds = PlanTreeDataset(x, self.encoder, self.normalizer, eval=True)
        self.__net.eval()
        batch, batch_labels = collator(list(zip(*[i for i in predict_ds])))
        with torch.no_grad():
            prediction = self.__net(batch)
        return self.normalizer.unnormalize_labels(prediction)