import os
import pickle
from pprint import pprint
import re

class Node:
    def __init__(self) -> None:
        self.lc = None
        self.rc = None
        self.idx = None
        self.operator = None
        self.data = {}


    def __init__(self, idx, operator) -> None:
        self.lc = None
        self.rc = None
        self.idx = idx
        self.operator = operator
        self.data = {}


    def __str__(self) -> str:
        return '('+ str(self.idx) + ') ' + self.operator + '\n' \
            + 'Left Child: ' + str(self.lc) + '\n'\
            + 'Right Child: ' + str(self.rc) + '\n'\
            + '\n'.join([k + ': ' + v for k, v in self.data.items()]) + '\n' \
            + '\n'

raw_plan_path = './data/raw_plan/tpcds/'
f_list = os.listdir(raw_plan_path)
forest = []
tpcds_sub_queries = ['5.plan', '8.plan','13.plan','14.plan','23.plan','24.plan','25.plan','26.plan','47.plan','57.plan','61.plan']
f_list = list(set(f_list).difference(set(tpcds_sub_queries)))


def _postprocess_plan(plan) -> str:
    """Remove random ids from the explained query plan"""
    pattern = re.compile(r'#\d+L?|\(\d+\)|\[\d+]||\[plan_id=\d+\]')
    return re.sub(pattern, '', plan)

def _postprocess_op(op) -> str:
    """Remove random ids from the explained query plan"""
    pattern = re.compile(r'\(.*\)|\*\ ')
    return re.sub(pattern, '', op)

def get_col_name(benchmark) -> list[str]:
    with open('./benchmark/schemas/' + benchmark + '.sql', 'r') as schema_text:
        col_name_list = []
        lines = schema_text.readlines()
        for line in lines:
            if 'create' not in line and 'USING' not in line and line != '\n':
                col_name_list.append(line.split()[0])
        return col_name_list

for f_name in sorted(f_list):
    with open(raw_plan_path + f_name, 'r') as plan_file:
        lines = plan_file.readlines()
        node_num = eval(lines[1].split()[-1])
        tree = [None] * (node_num + 1)
        colon = 0  # count the colons to identify the depth of the tree
        join_stack = []
        tree[0] = Node(0, 'root')
        prev_idx = 0
        for node in lines[1: node_num + 1]:
            idx = eval(node.split()[-1])
            op = node.strip().split('- * ')[-1].split('- ')[-1].split(' (')[0]
            op = _postprocess_op(op)
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
                    reused_op_id = eval(line.split(': ')[-1][:-2]) if 'Reused' in line else 0
                    if reused_op_id != 0:
                        tree[cur_node].lc = tree[reused_op_id].lc
                        tree[cur_node].rc = tree[reused_op_id].rc
                        tree[cur_node].operator = tree[reused_op_id].operator
                        tree[cur_node].data = tree[reused_op_id].data
                else:
                    if reused_op_id == 0:
                        line = _postprocess_plan(line)
                        key = line.split(': ')[0].strip()
                        val = ''.join(line.split(': ')[1:]).strip()
                        tree[cur_node].data[key] = val
        if f_name == '7.plan':
            print(len(forest))
        forest.append(tree)

with open('./data/examples/preprocessed.plan', 'w') as file:
    for node in forest[0]:
        file.write(str(node))

with open('./nn/data/hinted_forest.pkl', 'wb') as file:
    pickle.dump(forest, file)


def get_op_cols():
    op_list = []
    column_list = []
    column_name_list = get_col_name('tpcds')
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

op_list, column_list = get_op_cols()
# pprint(op_list)
# pprint(column_list)

