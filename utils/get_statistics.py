from pyhive import hive

conn = hive.Connection(host='10.77.110.152', port=10000, username='hejiahao')
cursor = conn.cursor()
cursor.execute('USE tpcds')

schema_text = open('./benchmark/schemas/tpcds.sql', 'r')
output = open('./data/tpcds_statistics.csv', 'w')

lines = schema_text.readlines()
for line in lines:
    if 'create table' in line:
        table_name = line.split()[-2]
    if 'create table' not in line and 'USING' not in line and line != '\n' and ('decimal' in line or 'integer' in line):
        column_name = line.split()[0]
        print(table_name, column_name)
        cursor.execute(f"SELECT min({column_name}), max({column_name}) FROM {table_name}")
        min, max = cursor.fetchall()[0]
        alias = column_name.split('_')[0]
        output.write(f'{alias}.{column_name},{min},{max}\n')
output.close()
schema_text.close()
