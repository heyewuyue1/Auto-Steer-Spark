from pyhive import hive
import sys
import os


def read_sql_from_path(path) -> str:
    with open(path, 'r') as sql_file:
        query = sql_file.read()
        return query

conn = hive.Connection(host='202.112.113.146', port=10000, username='hejiahao')
cursor = conn.cursor()
if len(sys.argv) > 1:
    query = read_sql_from_path(sys.argv[1])
    cursor.execute(query)
    print(cursor.fetchall())
else:
    while 1:
        try:
            query = input(">>> ")
            if query == 'exit':
                break
            if query.startswith('explain'):
                cursor.execute('EXPLAIN FORMATTED '+ read_sql_from_path(query.split()[1]))
                print(cursor.fetchall()[0][0])
            if os.path.isfile(query):
                cursor.execute(read_sql_from_path(query))
                print(cursor.fetchall())
            else:
                cursor.execute(query)
                print(cursor.fetchall())
        except Exception as e:
            print(e)
            pass
# query = 'SET spark.sql.optimizer.excludedRules=spark.sql.catalyst.optimizer.EliminateSerialization'
# cursor.execute(query)
# print(cursor.fetchall())
# query = 'SET spark.sql.thriftServer.interruptOnCancel'
# cursor.execute(query)
# print(cursor.fetchall())