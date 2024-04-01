from inference.preprocessing.preprocessor import QueryPlanPreprocessor
import re
from utils.custom_logging import logger
from inference.preprocessing.node import Node
import pandas as pd
import json
from inference.preprocessing.preprocess_subquery_backup import PlanToTree

class SparkPlanPreprocessor(QueryPlanPreprocessor):
    def get_col_name(self) -> dict:
        with open('./benchmark/schemas/tpcds_sf1.sql', 'r') as schema_text:
            col_name_list = []
            lines = schema_text.readlines()
            for line in lines:
                if 'USING' not in line and 'create' not in line != '\n':
                    col_name = line.split()[0]
                    alias = col_name.split('_')[0]
                    col_name_list.append(f'{alias}.{col_name}')
        col_name_dict = {col_name_list[i]: i for i in range(len(col_name_list))}
        col_name_dict['NA'] = len(col_name_list)
        return col_name_dict
        
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

    def __tree2dict(self, tree: list, i) -> dict:
        ret = {
            'Node Type': tree[i].operator if 'Scan' not in tree[i].operator else 'Scan',
            'Plans': []
        }
        if ret['Node Type'] == 'Filter':
            if 'Join' not in tree[tree[i].lc].operator.split('.')[-1] and 'Window' not in tree[tree[i].lc].operator.split('.')[-1]:
                ret['Relation Name'] = tree[tree[i].lc].operator.split('.')[-1]
            for k in tree[i].data:
                # 可能不需要这个if
                if 'Input' in k:
                    ret['Alias'] = tree[i].data[k].strip()[1:].split('_')[0]
            filters = tree[i].data['Condition'].replace('(',' ').replace(')',' ')
            ret['Filter'] = filters


        if 'Join' in ret['Node Type']:
            ret['Node Type'] = tree[i].operator.split(' ')[0]
            ret['Join Type'] = tree[i].data['Join type']
            try:
                ret['Hash Cond'] = f'{tree[i].data["Left keys"][1:].split("_")[0]}.{tree[i].data["Left keys"][1: -1]}={tree[i].data["Right keys"][1:].split("_")[0]}.{tree[i].data["Right keys"][1: -1]}'
            except:
                pass
        if tree[i].lc is not None and not (tree[i].operator == 'Filter' and 'Scan' in tree[tree[i].lc].operator):
            left = self.__tree2dict(tree, tree[i].lc)
            ret['Plans'].append(left)
        if tree[i].rc is not None:
            right = self.__tree2dict(tree, tree[i].rc)
            ret['Plans'].append(right)
        return ret
            
    def __plan2tree(self, plan: str) -> dict:
        if 'Subqueries' in plan:
            ee = PlanToTree(plan)
            return self.__tree2dict(ee, 0)
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
        return self.__tree2dict(tree, 0)

    def fit_transform(self, data, label=None) -> None:
        forest = []
        for i in range(len(data)):
            logger.debug(f'Processing plan {i}')
            tree = self.__plan2tree(data[i])
            if tree is not None:
                tree = {
                    "Plan": tree,
                }
                if label is not None:
                    tree['Execution Time'] = label[i]
                forest.append(json.dumps(tree))
        forest = pd.DataFrame(forest, columns=['json'])
        forest['id'] = forest.reset_index().index
        return forest

    
    def transform(self, trees) -> list:
        pass
