from pyhive import hive

conn = hive.Connection(host='10.77.110.152', port=10000, username='hejiahao')
cursor = conn.cursor()
query = '''SHOW DATABASES'''
cursor.execute(query)
print(cursor.fetchall())
# query = 'SET spark.sql.optimizer.excludedRules=spark.sql.catalyst.optimizer.EliminateSerialization'
# cursor.execute(query)
# print(cursor.fetchall())
# query = 'SET spark.sql.thriftServer.interruptOnCancel'
# cursor.execute(query)
# print(cursor.fetchall())