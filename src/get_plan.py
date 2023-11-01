from pyhive import hive
import os
from tqdm import tqdm
from time import time
import requests
from bs4 import BeautifulSoup

conn = hive.Connection(host='10.77.110.152', port=10000, username='hejiahao')
cursor = conn.cursor()
cursor.execute('USE tpcds')
f_list = os.listdir('./benchmark/queries/tpcds/')
query_idx = 243
for f_name in sorted(f_list):
    if f_name.endswith('.sql'):
        with open('./benchmark/queries/tpcds/' + f_name, 'r') as f:
            print('running SQL:', f_name)
            query = f.read()
            start = time()
            try:
                cursor.execute(query)
                end = time()
                dur = end - start
            except:
                dur = 10e9
            html = requests.get('http://10.77.110.152:4040/SQL/execution/?id=' + str(query_idx)).text
            print('crawling from: http://10.77.110.152:4040/SQL/execution/?id=' + str(query_idx))
            soup = BeautifulSoup(html, 'html.parser')
            node_list = soup.select('div.dot-file')[0].decode().split('\n')
            scan_list = {}
            for line in node_list:
                if 'Scan csv' in line:
                    idx = line.strip().split()[0]
                    meta_data = soup.select('div#plan-meta-data-' + idx)[0].text
                    table_filter = meta_data.split()[2].split('[')[0] + meta_data.split('PushedFilters: ')[1].split(', ReadSchema')[0]
                    try:
                        scan_list[table_filter] = eval(line.split('number of output rows: ')[1].split('&lt;')[0].replace(',',''))
                    except:
                        pass
            plan = soup.select('div#physical-plan-details pre')[0].text
            plan_list = plan.split('\n\n')
            for i in range(len(plan_list)):
                if 'Scan csv' in plan_list[i] and 'PushedFilters' in plan_list[i]:
                    k = plan_list[i].split()[3] + plan_list[i].split('PushedFilters: ')[1].split('\n')[0]
                    for filter in scan_list:
                        if filter.endswith('...'):
                            if filter[:-3] in k:
                                plan_list[i] += '\nNumber of output rows: ' + str(scan_list[filter])
                        else:
                            if filter == k:
                                plan_list[i] += '\nNumber of output rows: ' + str(scan_list[filter])
            raw_plan = '\n\n'.join(plan_list).strip() + '\nDuration: ' + str(dur)
            with open('./data/raw_plan/tpcds/' + f_name[:-4] + '.plan', 'w') as of:
                of.write(raw_plan)
            query_idx += 1
