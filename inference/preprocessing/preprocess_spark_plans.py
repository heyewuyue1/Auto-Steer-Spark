from inference.preprocessing.preprocessor import QueryPlanPreprocessor
from keras.models import Model
from inference.model import LeafEmbeddingModel
import re
import numpy as np
from utils.util import is_not_number
from utils.custom_logging import logger
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from inference.preprocessing.node import Node
from torch import nn
import torch

import os
import pickle

class SparkPlanPreprocessor(QueryPlanPreprocessor):
    def __normalize_data(self, val, column_name):
        min_val = self.min_max_vals[column_name][0]
        max_val = self.min_max_vals[column_name][1]
        val = float(val)
        if (val > max_val):
            val = max_val
        elif (val < min_val):
            val = min_val
        val = float(val)
        try:
            val_norm = (val - min_val) / (max_val - min_val)
        except ZeroDivisionError:
            val_norm = val
        return val_norm

    def __get_op_cols(self, forest):
        op_list = []
        column_list = []
        column_name_list = self.__get_col_name('tpcds')
        for i in range(len(forest)):
            for node in forest[i]:
                if node is not None:
                    if node.operator not in op_list and node.operator != 'root' and not node.operator.startswith('Scan csv'):
                        op_list.append(node.operator)
                    for key, v in node.data.items():
                        if 'Input' in key:
                            input = v[1:-1].split(', ')
                            for j in input:
                                if j not in column_list and j in column_name_list:
                                    column_list.append(j)
        return op_list, column_list

    def __postprocess_plan(self, plan) -> str:
        """Remove random ids from the explained query plan"""
        pattern = re.compile(r'#\d+L?|\[\d+]||\[plan_id=\d+\]')
        return re.sub(pattern, '', plan)

    def __postprocess_op(self, op) -> str:
        """Remove random ids from the explained query plan"""
        pattern = re.compile(r'\(.*\)|\*\ ')
        return re.sub(pattern, '', op)

    def __get_col_name(self, benchmark) -> list[str]:
        with open('./benchmark/schemas/' + benchmark + '.sql', 'r') as schema_text:
            col_name_list = []
            lines = schema_text.readlines()
            for line in lines:
                if 'create' not in line and 'USING' not in line and line != '\n':
                    col_name_list.append(line.split()[0])
            return col_name_list

    def __get_min_max_vals(self):
        column_min_max_vals = {}
        with open('./data/tpcds_statistics.csv') as f:
            lines = f.readlines()
            for line in lines:
                col_name, min_val, max_val = line.strip().split(',')
                column_min_max_vals[col_name] = (float(min_val), float(max_val))
            return column_min_max_vals
    
    def __featurize_nonleaf_operator(self, node):
        arr = np.zeros(len(self.op_list) + 1)
        arr[self.op_list.index(node.operator)] = 1
        stat = np.zeros(len(self.column_list))
        for key, val in node.data.items():
            if 'Input' in key:
                input = val[1:-1].split(', ')
                # logger.info(f'Non-leaf operator: {node.operator}, input: {input if len(input) < 5 else (str(input[:5]) + "...")}')
                for i in input:
                    if i in self.column_list:
                        stat[self.column_list.index(i)] = 1
        
        return np.concatenate((arr, stat, [0] * 64))

    def __featurize_null_operator(self):
        arr = np.zeros(len(self.op_list) + 1)
        arr[-1] = 1  # declare as null vector
        stat = np.zeros(len(self.column_list))
        return np.concatenate((arr, stat, [0] * 64))
    
    def __featurize_leaf_operator(self, tree, i):
        vocab_size = len(self.vocabulary)
        sentence = []
        cond = []
        for k in tree[i].data:
            if 'Output' in k:
                sentence.extend(tree[i].data[k][1:-1].split(', '))
        if 'Condition' in tree[i + 1].data:
            cond = tree[i + 1].data['Condition'].replace('(', ' ').replace(')', ' ').strip().split()
        for j in range(len(cond)):
            if not is_not_number(cond[j]):
                if cond[j - 2] in self.min_max_vals:
                    cond[j] = str(self.__normalize_data(eval(cond[j]), cond[j - 2]))
            else:
                if cond[j] in self.vocabulary:
                    sentence.append(cond[j])
        # logger.info(f'Sentence: {sentence if len(sentence) < 10 else (str(sentence[:10]) + "...")}')
        _s = []
        embed = nn.Embedding(vocab_size, 64)
        for word in sentence:
            if (is_not_number(word)):
                if word in self.vocab_dict:
                    # _tmp = np.column_stack((np.array([0]), self.vocab_dict[word]))
                    # _tmp = np.reshape(_tmp, (vocab_size+1))
                    _tmp = embed(torch.tensor(self.vocabulary.index(word)))
                    # assert (len(_tmp) == vocab_size+1)
                    _s.append(_tmp)
            else:
                _tmp = np.full(64, word)
                # assert (len(_tmp) == vocab_size+1)
                _s.append(_tmp)
        sentence = np.array(_s)
        if sentence.shape[0] > self.max_len:
            sentence = sentence[:self.max_len]
        if sentence.shape[0] < self.max_len:
            sentence = np.concatenate((sentence, np.zeros((self.max_len - sentence.shape[0], vocab_size+1))))
        sentence = sentence.reshape((1, self.max_len, 64))
        # intermediate_output = self.intermediate_layer_model.predict(sentence, verbose=0)
        logger.info(f'Embedded sequence: {sentence}')
        return np.concatenate((np.zeros(len(self.op_list) + len(self.column_list) + 1), sentence))

    def __featurize(self, tree, i):
        if tree[i].operator.startswith('Scan'):
            return self.__featurize_leaf_operator(tree, i)
        else:
            return self.__featurize_nonleaf_operator(tree[i]),\
            self.__featurize(tree, tree[i].lc) if tree[i].lc is not None else self.__featurize_null_operator(),\
            self.__featurize(tree, tree[i].rc) if tree[i].rc is not None else self.__featurize_null_operator()

    def __plan2tree(self, plan):
        if 'Subqueries' in plan:
            return None
        lines = plan.split('\n')
        node_num = eval(lines[1].split()[-1])
        tree = [None] * (node_num + 1)
        colon = 0  # count the colons to identify the depth of the tree
        join_stack = []
        tree[0] = Node(0, 'root')
        prev_idx = 0
        for node in lines[1: node_num + 1]:
            idx = eval(node.split()[-1])
            op = node.strip().split('- * ')[-1].split('- ')[-1].split(' (')[0]
            op = self.__postprocess_op(op)
            if colon <= node.count(':'):
                tree[prev_idx].lc = idx
            else:
                tree[join_stack[-1]].rc = idx
                join_stack.pop()
            if 'Join' in node or 'CartesianProduct' in node or 'Union' in node:
                join_stack.append(idx)
            tree[idx] = Node(idx, op)
            colon = node.count(':')
            prev_idx = idx
        cur_node = 0
        reused_op_id = 0
        for line in lines[node_num + 1:]:
            if line != '\n':
                if line.startswith('('):
                    cur_node = eval(line.split()[0])
                    reused_op_id = eval(line.split(': ')[-1][:-1]) if 'Reused' in line else 0
                    if reused_op_id != 0:
                        tree[cur_node].lc = tree[reused_op_id].lc
                        tree[cur_node].rc = tree[reused_op_id].rc
                        tree[cur_node].operator = tree[reused_op_id].operator
                        tree[cur_node].data = tree[reused_op_id].data
                else:
                    if reused_op_id == 0:
                        line = self.__postprocess_plan(line)
                        key = line.split(': ')[0].strip()
                        val = ''.join(line.split(': ')[1:]).strip()
                        tree[cur_node].data[key] = val
        return tree

    def __init__(self):
        super().__init__()
        self.min_max_vals = self.__get_min_max_vals()
        self.vocab_dict = {}
        self.vocabulary = []
        self.max_len = 0
        self.op_list, self.column_list = [], []

        with open('./data/vocab.txt', 'r') as f:
            lines = f.readlines()
            self.vocabulary = eval(lines[0])
            self.max_len = eval(lines[1])

        if os.path.isfile('./data/forest.pkl'):
            logger.info('__init__: loading forest...')
            with open('./data/forest.pkl', 'rb') as f:
                forest = pickle.load(f)
                logger.info('Loaded forest')
                self.op_list, self.column_list = self.__get_op_cols(forest)
        else:
            logger.info('No existing forest found, waiting for fit to build one')

        logger.info(f'Get op_list: {self.op_list}')
        logger.info(f'Get column_list: {self.column_list}')
        logger.info(f'Feature length: {len(self.op_list) + len(self.column_list) + 65}')
        logger.info(f'len(vocabulary): {len(self.vocabulary)}')
        vocab_size = len(self.vocabulary)
        # logger.info(f'vocabulary: {self.vocabulary}')
        _vocabulary = np.array(self.vocabulary)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(_vocabulary)
        encoded = to_categorical(integer_encoded)
        for v, e in zip(self.vocabulary, encoded):
            self.vocab_dict[v] = np.reshape(np.array(e), (1, vocab_size))
        logger.info(f'max_len: {self.max_len}')

    def fit(self, trees) -> None:
        if not os.path.isfile('./data/forest.pkl'):
            logger.info('Building forest...')
            forest = []
            for text in trees:
                tree = self.__plan2tree(text)
                if tree is not None:
                    forest.append(tree)
                with open('./data/forest.pkl', 'wb') as f:
                    pickle.dump(forest, f)
        with open('./data/forest.pkl', 'rb') as f:
            logger.info('fit: loading forest...')
            forest = pickle.load(f)
            self.op_list, self.column_list = self.__get_op_cols(forest)
        return forest
    
    def transform(self, trees) -> list:
        forest = []
        for i in range(len(trees)):
            logger.info(f'Processing plan {i}...')
            tree = self.__plan2tree(trees[i])
            if tree is not None:
                featurized_tree = self.__featurize(tree, tree[0].lc)
                forest.append(featurized_tree)
        return forest    
