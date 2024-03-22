from inference.preprocessing.preprocessor import QueryPlanPreprocessor
import re
import numpy as np
from utils.custom_logging import logger
from inference.preprocessing.node import Node
from inference.preprocessing.preprocess_subquery import PlanToTree

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
        return op
    
    def __featurize_not_null_operator(self, node):
        '''featurize: op、column、info'''
        arr = np.zeros(len(self.op_list) + 1)
        arr[self.op_list.index(node.operator)] = 1
        info = self.__extract_stats(node)
        return np.concatenate((arr,info))

    def __extract_stats(self, node):
        '''提取explain内容中的node的信息：input、rowcount、size'''
        info = []
        for key, val in node.data.items():
            if 'Row count' in key:
                info.append(eval(val))
            elif 'sizeInBytes' in key:
                num = eval(val.split(' ')[0])
                if num == 0: # 写为0（忘记带单位了）
                    info.append(num)
                else:   # 有数（带单位）
                    unit = val.split(' ')[1]
                    if unit == 'B':
                        pass
                    elif unit == 'KiB':
                        num *= 1024
                    elif unit == 'MiB':
                        num *= 1024**2
                    elif unit == 'GiB':
                        num *= 1024**3
                    elif unit == 'TiB':
                        num *= 1024**4
                    elif unit == 'PiB':
                        num *= 1024**5
                    elif unit == 'EiB':
                        num *= 1024**6
                    else:
                        logger.warning('wrong size' + unit)
                    info.append(np.log1p(num))
        return info

    def __featurize_null_operator(self):
        '''np.concatenate((arr, stat, info))'''
        arr = np.zeros(len(self.op_list) + 1)
        arr[-1] = 1  # declare as null vector
        info = np.zeros(2)
        return np.concatenate((arr,info))

    def __featurize(self, tree, i):
        logger.debug(f'Featurizing node {i} {tree[i].operator}')
        return self.__featurize_not_null_operator(tree[i]),\
        self.__featurize(tree, tree[i].lc) if tree[i].lc is not None else self.__featurize_null_operator(),\
        self.__featurize(tree, tree[i].rc) if tree[i].rc is not None else self.__featurize_null_operator()

    def __plan2tree(self, plan):
        if 'Subqueries' in plan:
            EE = PlanToTree(plan)
            return EE
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
            if 'Scan' in op:
                op = 'Scan csv'
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
            if line != '\n' and line != '':
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
                        # logger.info('len= '+str(len(tree))+'cur_node='+str(cur_node)+' line:'+line)
        return tree

    def __init__(self):
        super().__init__()
        self.op_list= []
        self.forest = []

    def fit(self, trees) -> None:
        logger.info('Building forest...')
        for i in range(len(trees)):
            logger.debug(f'Processing plan {i}')
            tree = self.__plan2tree(trees[i])
            if tree is not None:
                self.forest.append(tree)
        self.op_list = self.__get_op(self.forest)
        logger.info(f'Get op_list: {self.op_list}')
        logger.info(f'Feature length: {len(self.op_list) + 2}')
        return self.forest
    
    def transform(self, trees) -> list:
        forest = []
        for i in range(len(trees)):
            logger.debug(f'Processing plan {i}')
            tree = self.__plan2tree(trees[i])
            if tree is not None:
                featurized_tree = self.__featurize(tree, tree[tree[0].lc].lc)
                forest.append(featurized_tree)
        return forest    
