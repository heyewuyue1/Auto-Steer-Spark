import re
import numpy as np
from utils.custom_logging import logger
import json
dic={'TakeOrderedAndProject':'Sort','Scan':'Relation'}
l_p = ['Relation','Filter']
p_p = ['Join','Scan','Filter']

"""
with open('dic.txt', 'r') as file:
    # 从dic.txt文件里获取dic_table_columns,dic_column_stat
    dic_table_columns = eval(file.readline().strip())  # 读取第一行，并去除换行符
    dic_column_stat = eval(file.readline().strip())   # 读取第二行，并去除换行符
"""
def get_dic():
    file_name = "data/tpcds_sf1_stats/dic_column_stat.json"
    dic_column_stat = {}
    with open(file_name, "r") as file:
        dic_column_stat = json.load(file)

    file_name = "data/tpcds_sf1_stats/dic_table_columns.json"
    dic_table_columns = {}
    with open(file_name, "r") as file:
        dic_table_columns = json.load(file)
    return dic_table_columns,dic_column_stat


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
    '''在node中添加Rowcount、sizeInBytes、hist'''
    dic_table_columns,dic_column_stat = get_dic()
    lines_n = physical_node.splitlines()
    hist_len = []
    for line in lines_n:
        if line.startswith('('):
            add_row_size(line,lines_n,num_node,lg_op)
            add_hist(hist_len,dic_column_stat,line,lines_n)  # hist
    logger.info('max_hist_len: '+str(max(hist_len)))
    physical_node = '\n'.join(lines_n)
    return physical_node

def add_row_size(line,lines_n,num_node,lg_op):
    '''添加Rowcount、sizeInBytes'''
    line_idx = lines_n.index(line)
    num = int(line.split()[0][1:-1])
    if num in num_node:     # op是Join/filter/scan
        k = num_node.index(num)
        row_count = 'Row count: '+ lg_op[k][2]
        size = 'sizeInBytes: '+ lg_op[k][1]
        lines_n.insert(line_idx + 1, row_count)
        lines_n.insert(line_idx + 2, size)
    else:   # 其他op设为0
        row_count = 'Row count: 0'
        size = 'sizeInBytes: 0 B'
        lines_n.insert(line_idx + 1, row_count)
        lines_n.insert(line_idx + 2, size)


def add_hist(hist_len,dic_column_stat,line,lines_n):
    '''添加hist'''
    line_idx = lines_n.index(line)

    op = line.split(' ')[1]     # node
    if op != 'Filter':  # 非filter节点，添加空的hist
        hist_info = 'Hist: '+ '[]'
        lines_n.insert(line_idx + 1, hist_info)  # 添加到node下一行
        if op == 'AdaptiveSparkPlan':
            lines_n.insert(line_idx + 1, 'max_hist_len: '+ str(max(hist_len)))  # 记录每个物理计划最长的hist长度
    else: # 解析filter节点
        con_line = lines_n[line_idx + 4]    # condition行
        if 'Condition' not in con_line: 
            logger.info('错啦！！')
        # logger.info(con_line)
        dic_q_filter={}     # 存储filter对列的限制范围
        is_notnull = False  # 判断是否是isnotnull条件
        matches = re.findall(r'(\w+#?\d+\s*(?:[<>!]=?|==|=)\s*[\d.]+\b)|isnotnull\((\w+#?\d+)\)', con_line)
        for match in matches:   # 遍历每个限制条件
            is_notnull = False
            if match[0] != '':
                col,pred,num = match[0].split(' ')
                col = match[0].split('#')[0]
            elif match[1] != '':    # isnotnull
                col = match[1].split('#')[0]
                is_notnull = True

            # logger.info(col)
            if col not in dic_q_filter:
                dic_q_filter[col] = {}
            if not is_notnull:  # 非isnotnull条件
                if pred == '>=':
                    dic_q_filter[col]['min'] = num if 'min' not in dic_q_filter[col] else min(num,dic_q_filter[col]['min'])
                elif pred == '<=':
                    dic_q_filter[col]['max'] = num if 'max' not in dic_q_filter[col] else max(num,dic_q_filter[col]['max'])
                    # 目前是找最宽松的范围
                elif pred == '=':
                    dic_q_filter[col]['eq'] = num
                elif pred == '<':
                    dic_q_filter[col]['lt'] = num if 'lt' not in dic_q_filter[col] else min(num,dic_q_filter[col]['lt'])
                elif pred == '>':
                    dic_q_filter[col]['gt'] = num if 'gt' not in dic_q_filter[col] else max(num,dic_q_filter[col]['gt'])
                elif pred == '!=':
                    dic_q_filter[col]['ne'] = num
                else:
                    print('未处理的pred！',pred)
            else:   # isnotnull条件
                dic_q_filter[col]['notnull'] = 1

        logger.info(dic_q_filter)
        hist = []
        for c in dic_q_filter:  # 处理filter里的每列
            ne_f = False
            if c not in dic_column_stat:
                print('非原表列：',c)   # 可能是sum/count等新列
                continue
            data_type = dic_column_stat[c]['data_type']
            if data_type == 'date':     # date类型的数据有点问题-v-
                print('date型列：',c)
                continue
            if 'varchar' in data_type:
                print('字符型列：',c)
                continue
            if len(dic_column_stat[c]['height_bin']) == 1:
                print('histogram为空：',c,' type：',data_type)
                continue
            if len(dic_q_filter[c])==1 and 'notnull' in dic_q_filter[c]:    # 只有isnotnull条件，设为全1
                c_hist = np.ones(50)
            else:
                c_hist = np.zeros(50)
                # print(c,dic_column_stat[c])
                c_min = eval(dic_column_stat[c]['min'])
                c_max = eval(dic_column_stat[c]['max'])
                if 'min' in dic_q_filter[c]:
                    f_min = dic_q_filter[c]['min']
                    c_min = eval(f_min)
                if 'max' in dic_q_filter[c]:
                    f_max = dic_q_filter[c]['max']
                    c_max = eval(f_max)
                if 'eq' in dic_q_filter[c]:
                    eq = dic_q_filter[c]['eq']
                    c_min = eval(eq) -0.5
                    c_max = eval(eq) +0.5
                    # 先这样处理，后续再改
                if 'lt' in dic_q_filter[c]:
                    lt = dic_q_filter[c]['lt']  #less_than
                    c_max = eval(lt)-0.01
                if 'gt' in dic_q_filter[c]:
                    gt = dic_q_filter[c]['gt']  #greater-than
                    c_min = eval(gt)+0.01
                if 'ne' in dic_q_filter[c]:
                    ne = eval(dic_q_filter[c]['ne'])
                    ne_f = True
                for i in range(1,51):   # 遍历每个bin的信息
                    bin,bin_info = dic_column_stat[c]['height_bin'][i]
                    lower_bound,upper_bound,distinct_count = bin_info.split(',')
                    lower_bound = eval(lower_bound.split(':')[1])
                    upper_bound = eval(upper_bound.split(':')[1])
                    if ne_f:    # 如果是'!='的限制
                        if lower_bound > ne or upper_bound < ne:
                            c_hist[i-1] = 1
                    else:   # 其他限制，且该bin符合条件，设为相应比例值
                        if (lower_bound <= c_min <= c_max <= upper_bound):
                            c_hist[i-1] = (c_max-c_min)/(upper_bound-lower_bound) if upper_bound>lower_bound else 1
                        elif (lower_bound <= c_min <= upper_bound <= c_max):
                            c_hist[i-1] = (upper_bound-c_min)/(upper_bound-lower_bound) if upper_bound>lower_bound else 1
                        elif (c_min <= lower_bound <= upper_bound <= c_max):
                            c_hist[i-1] = 1
                        elif (c_min <= lower_bound <= c_max <= upper_bound):
                            c_hist[i-1] = (c_max-lower_bound)/(upper_bound-lower_bound) if upper_bound>lower_bound else 1
            # print(c,c_hist)
            hist.extend(c_hist)  # 拼接上该列的hist
        hist_len.append(len(hist))

        hist_info = 'Hist: '+ str(hist)  # 在filter上添加其涉及所有列的hist
        logger.info(hist_info)
        lines_n.insert(line_idx + 1, hist_info)   # 加上，idx是filer的index



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
