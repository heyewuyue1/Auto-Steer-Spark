import re
from utils.custom_logging import logger
from inference.preprocessing.node import Node

# def extract_Info(node):
#     stats = []
#     # for key, val in node.data.items():
#     #     if 'Input' in key:
            
#     return 

def postprocess_op( op) -> str:
        """Remove random ids from the explained query plan"""
        pattern = re.compile(r'\(.*\)|\*\ ')
        op = re.sub(pattern, '', op)
        # if 'Scan' in op:
        #     return 'Scan'
        # else:
        return op

def postprocess_plan( plan) -> str:
        """Remove random ids from the explained query plan"""
        pattern = re.compile(r'#\d+L?|\[\d+]||\[plan_id=\d+\]')
        return re.sub(pattern, '', plan)
#子树向主树转接
def preorder_traversal(tree,root,i):
    if i is None:
        return
    tree[root[i].idx] = Node(root[int(i)].idx,root[int(i)].operator)
    tree[int(i)].lc = root[int(i)].lc
    tree[int(i)].rc = root[int(i)].rc
    tree[int(i)].data = root[int(i)].data
    preorder_traversal(tree,root,root[int(i)].lc)
    preorder_traversal(tree,root,root[int(i)].rc)

def PlanToTree(plan):
    # 定义正则表达式
    pattern = r'AdaptiveSparkPlan \((\d+)\)'
    # 使用正则表达式查找所有匹配项
    matches = re.findall(pattern, plan)
    # 将匹配到的数字转换为整数并找到最大值
    max_number = max(map(int, matches))
    
    lines = plan.split('\n')
    parent_index = lines.index('===== Subqueries =====')
    # logger.info(plan)
    node_num = eval(lines[1].split()[-1])
    Htree = Str_Sub_Tree(max_number,node_num,lines[:parent_index],1)

    # 匹配"id="后面的数字
    pattern = r"Hosting operator id = (\d+)"
    # 用于存储提取出来的id及其索引位置
    id_indices = []
    for idx, string in enumerate(lines):
        match = re.search(pattern, string)
        if match:
            # 提取id和索引位置并存储到id_indices列表中
            id_value = match.group(1)
            id_indices.append((id_value, idx))    
    result = {}
    flag = {}
    num = len(id_indices)
    i = 0   
    for id_value,idx in id_indices:
        i +=1
        temp =[]
        N = -1 #用于记录子节点个数 
        if i<=num-1 :
            temp=lines[idx:id_indices[i][1]]
            for str in lines[idx:]:
                if str == '':
                    break
                N +=1        
        else:
            temp=lines[idx:]  
            for str in lines[idx:]:
                if str == '':
                    break
                N +=1   
    
        #处理不同子id对应相同的执行方案
        for k in result.keys():
            g = 0
            for q in result[k]:
                # if not isinstance(q, list) :
                if q[1] == temp[1]:
                    temp = q
                    g=1
                    break
                # else:
                #     break
            if g==1:
                break

        #处理相同子id对应不同的执行方案，这里
        
        if id_value in result.keys():
            result[id_value].append(temp)
        else:
            result[id_value] = [temp]
        
        if id_value in flag:
            if not isinstance(flag[id_value], list):
                flag[id_value] = [flag[id_value]]
            flag[id_value].append(N)
        else:
            flag[id_value] = N   

    for P_idx in flag.keys():
        if  isinstance(flag[P_idx], list):
            i = 0
            for q in flag[P_idx]:
                Tree = Str_Sub_Tree(max_number,int(q),result[P_idx][i],2)  #这里
                if Tree[0].lc is None:
                    continue
                if Htree[int(P_idx)].rc is None:
                    Htree[int(P_idx)].rc = Tree[Tree[0].lc].lc
                else:
                    # Htree[Tree[0].lc+1]=Node(Tree[0].lc+1,'AdaptiveSparkPlan')
                    Temp =Htree[int(P_idx)].rc
                    
                    Htree[Tree[0].lc] = Node(Tree[Tree[0].lc].idx,Tree[Tree[0].lc].operator)
                    Htree[Tree[0].lc].lc = Temp
                    Htree[Tree[0].lc].rc = Tree[Tree[0].lc].lc
                    Htree[Tree[0].lc].data = Tree[Tree[0].lc].data

                    Htree[int(P_idx)].rc = Htree[Tree[0].lc].idx
                preorder_traversal(Htree,Tree,Tree[Tree[0].lc].lc)
                i +=1
        else:
            Tree = Str_Sub_Tree(max_number,int(flag[P_idx]),result[P_idx][0],2) #这里
            if Htree[int(P_idx)].rc is None:
                Htree[int(P_idx)].rc = Tree[Tree[0].lc].lc
                preorder_traversal(Htree,Tree,Tree[Tree[0].lc].lc)
            else:
                # Htree[Tree[0].lc]=Node(Tree[0].lc+1,'AdaptiveSparkPlan')
                # Temp =Htree[int(P_idx)].rc
                # Htree[int(P_idx)].rc = Tree[0].lc+1
                # Htree[Tree[0].lc+1].lc = Temp
                # Htree[Tree[0].lc+1].rc = Tree[0].lc
                Temp =Htree[int(P_idx)].rc   
                Htree[Tree[0].lc] = Node(Tree[Tree[0].lc].idx,Tree[Tree[0].lc].operator)
                Htree[Tree[0].lc].lc = Temp
                Htree[Tree[0].lc].rc = Tree[Tree[0].lc].lc
                Htree[Tree[0].lc].data = Tree[Tree[0].lc].data

                Htree[int(P_idx)].rc = Htree[Tree[0].lc].idx
                preorder_traversal(Htree,Tree,Tree[Tree[0].lc].lc)
    

    return Htree
        
def Str_Sub_Tree(max_number,node_num,lines,isAdapt):
    tree = [None] * (max_number+1)
    colon = 0  # count the colons to identify the depth of the tree
    join_stack = []
    tree[0] = Node(0, 'root')
    prev_idx = 0
    for node in lines[isAdapt: node_num + 1]:
        # if str(node.split()[-1]) >='a' and str(node.split()[-1]) <='z':
        #     break
        # idx  = eval(node.split()[-1])
        pattern = r" \((\d+)\)"
            # cur_node = eval(line.split()[0])
        match = re.search(pattern,node)
        if not match:
            continue
        else:
            idx = int(match.group(1))
        op = node.strip().split('- * ')[-1].split('- ')[-1].split(' (')[0]
        op =postprocess_op(op)

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
        if line != '\n':
            if "AdaptiveSparkPlan" in line and isAdapt == 2:
                break
            if line.startswith('('):
                pattern = r"(\d+)"
                # cur_node = eval(line.split()[0])
                match = re.search(pattern,line)
                if not match:
                    continue
                else:
                    cur_node =int(match.group(1)) 
                reused_op_id = eval(line.split(': ')[-1][:-1]) if 'Reused' in line else 0
                if reused_op_id != 0:
                        # tree[cur_node].lc = tree[reused_op_id].lc
                        # tree[cur_node].rc = tree[reused_op_id].rc
                    tree[cur_node].operator = tree[reused_op_id].operator
                    tree[cur_node].data = tree[reused_op_id].data
            else:
                if reused_op_id == 0:
                    line = postprocess_plan(line)
                    key = line.split(': ')[0].strip()
                    val = ''.join(line.split(': ')[1:]).strip()
                    tree[cur_node].data[key] = val
                
    
    return tree
