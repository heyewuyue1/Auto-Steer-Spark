from inference.preprocessing.preprocessor import QueryPlanPreprocessor
import re
import numpy as np
from utils.custom_logging import logger
from inference.preprocessing.node import Node
from utils.config import read_config
import os
import pickle

class SparkPlanPreprocessor(QueryPlanPreprocessor):
    def __get_op(self, forest):
        op_list = []
        for tree in forest:
            for node in tree:
                if node is not None and node.operator not in op_list and node.operator != 'root':
                    op_list.append(node.operator)
        return op_list

    def __postprocess_plan(self, plan) -> str:
        """Remove random ids from the explained query plan"""
        pattern = re.compile(r'#\d+L?|\[\d+]||\[plan_id=\d+\]')
        return re.sub(pattern, '', plan)

    def __postprocess_op(self, op) -> str:
        """Remove random ids from the explained query plan"""
        pattern = re.compile(r'\(.*\)|\*\ ')
        op = re.sub(pattern, '', op)
        # if 'Scan' in op:
        #     return 'Scan'
        # else:
        return op

    def __get_col_name(self, benchmark) -> list[str]:
        with open('./benchmark/schemas/' + benchmark + '.sql', 'r') as schema_text:
            col_name_list = []
            lines = schema_text.readlines()
            if benchmark == 'job':
                for line in lines:
                    if 'USING' not in line and line != '\n':
                        if 'CREATE' in line:
                            table_name = line.split()[-2]
                        else:
                            col_name_list.append('job.' + table_name + '.' + line.split()[0])
            elif benchmark == 'tpcds':
                for line in lines:
                    if 'USING' not in line and 'create' not in line != '\n':
                        col_name_list.append(line.split()[0])
            return col_name_list
    
    def __featurize_not_null_operator(self, node):
        arr = np.zeros(len(self.op_list) + 1)
        arr[self.op_list.index(node.operator)] = 1
        stats = self.__extract_stats(node)
        stat_arr = np.zeros(len(self.column_list))
        for stat in stats:
            stat_arr[self.column_list.index(stat)] = 1
        return np.concatenate((arr, stat_arr))

    def __extract_stats(self, node):
        stats = []
        for key, val in node.data.items():
            if 'Input' in key:
                input = val[1:-1].split(', ')
                for i in input:
                    if i in self.column_list and i not in stats:
                        stats.append(i)
        return stats

    def __featurize_null_operator(self):
        arr = np.zeros(len(self.op_list) + 1)
        arr[-1] = 1  # declare as null vector
        stat = np.zeros(len(self.column_list))
        return np.concatenate((arr, stat))

    def __featurize(self, tree, i):
        logger.debug(f'Featurizing node {i} {tree[i].operator}')
        return self.__featurize_not_null_operator(tree[i]),\
        self.__featurize(tree, tree[i].lc) if tree[i].lc is not None else self.__featurize_null_operator(),\
        self.__featurize(tree, tree[i].rc) if tree[i].rc is not None else self.__featurize_null_operator()

    def __plan2tree(self, plan):
        if 'Subqueries' in plan:
            # logger.warning(f'Contain Subqueries, ignored, raw plan: {plan}')
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
                        # tree[cur_node].lc = tree[reused_op_id].lc
                        # tree[cur_node].rc = tree[reused_op_id].rc
                        tree[cur_node].operator = tree[reused_op_id].operator
                        tree[cur_node].data = tree[reused_op_id].data
                else:
                    if reused_op_id == 0:
                        line = self.__postprocess_plan(line)
                        key = line.split(': ')[0].strip()
                        val = ''.join(line.split(': ')[1:]).strip()
                        tree[cur_node].data[key] = val
        return tree

# 
    def __init__(self):
        super().__init__()
        self.benchmark = read_config()['DEFAULT']['BENCHMARK']
        self.op_list, self.column_list = [], []
        if os.path.isfile('./data/forest.pkl'):
            logger.info('__init__: loading forest...')
            with open('./data/forest.pkl', 'rb') as f:
                forest = pickle.load(f)
                logger.info('Loaded forest')
                self.op_list = self.__get_op(forest)
                self.column_list = self.__get_col_name(self.benchmark)
                logger.info(f'Get op_list: {self.op_list}')
                logger.info(f'Get column_list: {self.column_list}')
                logger.info(f'Feature length: {len(self.op_list) + len(self.column_list)}')
        else:
            logger.info('No existing forest found, waiting for fit to build one')

    def fit(self, trees) -> None:
        if not os.path.isfile('./data/forest.pkl'):
            logger.info('Building forest...')
            forest = []
            for i in range(len(trees)):
                logger.debug(f'Processing plan {i}')
                tree = self.__plan2tree(trees[i])
                if tree is not None:
                    forest.append(tree)
            with open('./data/forest.pkl', 'wb') as f:
                pickle.dump(forest, f)
        with open('./data/forest.pkl', 'rb') as f:
            logger.info('fit: loading forest...')
            forest = pickle.load(f)
            self.op_list = self.__get_op(forest)
            self.column_list = self.__get_col_name(self.benchmark)
            logger.info(f'Get op_list: {self.op_list}')
            logger.info(f'Get column_list: {self.column_list}')
            logger.info(f'Feature length: {len(self.op_list) + len(self.column_list)}')
        return forest
    
    def transform(self, trees) -> list:
        forest = []
        for i in range(len(trees)):
            logger.debug(f'Processing plan {i}')
            tree = self.__plan2tree(trees[i])
            if tree is not None:
                try:
                    featurized_tree = self.__featurize(tree, tree[0].lc)
                except Exception as e:
                    logger.error(f'Error in plan {i} message: {e} raw plan: {trees[i]}')
                    continue
                forest.append(featurized_tree)
        return forest    
