import re
from utils.custom_logging import logger
dic={'TakeOrderedAndProject':'Sort','Scan':'Relation'}
l_p = ['Relation','Filter']
p_p = ['Join','Scan','Filter']

def get_lg_op(re_log):
    '''获得逻辑计划中join/filter/scan的位置'''
    lg_op = []
    for i in re_log:
            if i[0] in l_p or 'Join' in i[0]:
                lg_op.append(i)
    return lg_op

def re_physical_plan(explainf_result):
    '''正则提取explain formatted结果的plan部分和node信息部分'''
    match_phyplan = re.search(r'(.+?)(\n\n)(.+)', explainf_result, re.DOTALL)
    physical_plan = match_phyplan.group(1).strip()
    physical_node = match_phyplan.group(3).strip()
    return physical_plan, physical_node

def re_logical_plan(explaincost_result):
    '''对开启cbo得到的logical plan进行正则，提取：op、sizeInBytes、rowCount'''
    match = re.search(r'== Optimized Logical Plan ==(.+?)(== Physical Plan ==)', explaincost_result, re.DOTALL)
    optimized_logical_plan = match.group(1).strip()
    re_log = []
    lines = optimized_logical_plan.splitlines()
    for line in lines:
        pattern = r'Filter (.*?),|Join (.*?),|\+\- (.*?) |\:\- (.*?) '
        match = re.search(pattern, line)
        match2 = re.search(r'sizeInBytes=(.*?),',line)
        match3 = re.search(r'rowCount=(.*?)\)',line)
        if match:
            match4 = re.search('([ a-zA-Z]+)', match.group(0).strip())
            if match3 is not None:
                re_log.append([match4.group(0).strip(),match2.group(1),match3.group(1)])
            else:   # 信息里没有row count【不知道为什么】
                match2 = re.search(r'sizeInBytes=(.*?)\)',line)
                re_log.append([match4.group(0).strip(),match2.group(1),'0'])
            # 操作 + sizeInBytes + rowCount
    return re_log
# MiB = 2^20 字节，Kib = 2^10字节，B
# 没有收录 GlobalLimit

def get_num_node(physical_plan):
    '''获得物理计划中op为join/filter/scan的node的idx'''
    lines = physical_plan.splitlines()
    num_node=[]
    for k in range(1,len(lines)):
        idx = eval(lines[k].split()[-1])
        match = re.search('([ a-zA-Z._ ]+)\(', lines[k])
        if match:
            op = match.group(1).strip()
        if 'Scan' in op: op = 'Scan'
        if op in p_p or p_p[0] in op:
            num_node.append(idx)
    return num_node

def add_info(physical_node,num_node,lg_op):
    '''在node中添加Row count、sizeInBytes'''
    lines_n = physical_node.splitlines()
    for line in lines_n:
        if line.startswith('('):
            num = int(line.split()[0][1:-1])
            if num in num_node:     # op是Join/filter/scan
                k = num_node.index(num)
                row_count = 'Row count: '+ lg_op[k][2]
                size = 'sizeInBytes: '+ lg_op[k][1]
                line_idx = lines_n.index(line)
                lines_n.insert(line_idx + 1, row_count)
                lines_n.insert(line_idx + 2, size)
            else:   # 其他op设为0
                row_count = 'Row count: 0'
                size = 'sizeInBytes: 0 B'
                line_idx = lines_n.index(line)
                lines_n.insert(line_idx + 1, row_count)
                lines_n.insert(line_idx + 2, size)    
    physical_node = '\n'.join(lines_n)
    return physical_node



def get_explain(timed_result_f, timed_result_c):
    '''生成加了rowcount、size的物理计划'''
    physical_plan, physical_node = re_physical_plan(timed_result_f)
    re_log = re_logical_plan(timed_result_c)
    lg_op = get_lg_op(re_log)
    num_node = get_num_node(physical_plan)
    physical_node = add_info(physical_node,num_node,lg_op)

    result = physical_plan+'\n\n\n'+physical_node+'\n\n'
    
    # return tuple()+(result,)
    return result
