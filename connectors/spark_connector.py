# Copyright 2022 Intel Corporation
# SPDX-License-Identifier: MIT
#
"""This module provides a connection to a Spark cluster that is used for benchmarking"""
from pyhive import hive
import time
import re
from utils.custom_logging import logger
from connectors.connector import DBConnector
from utils.config import read_config
import configparser

EXCLUDED_RULES = 'spark.sql.optimizer.excludedRules'

def _postprocess_plan(plan) -> str:
    """Remove random ids from the explained query plan"""
    # pattern = re.compile(r'\[\d+]||\[plan_id=\d+\]')
    # plan = re.sub(pattern, '', plan)
    # lines = plan.split('\n')
    # is_scan = False
    # replace_dict = {}
    # for line in lines:
    #     if line.startswith('(') and 'Scan' in line:
    #         table_name = line.split()[-1]
    #         is_scan = True
    #     if is_scan and 'Output' in line:
    #         col_names = line.split('[')[-1][:-1].split(', ')
    #         for col_name in col_names:
    #             replace_dict[col_name] = table_name + '.' + col_name
    # for key, val in replace_dict.items():
    #     plan = plan.replace(key, val)
    # pattern = re.compile(r'#\d+L?')
    # plan = re.sub(pattern, '', plan)
    # logger.debug('Postprocessed plan: %s', plan)
    pattern = re.compile(r'#\d+L?|\[\d+]||\[plan_id=\d+\]')
    return re.sub(pattern, '', plan)
    # return plan

class SparkConnector(DBConnector):
    """This class implements the AutoSteer-G connector for a Spark cluster accepting SQL statements"""
    def __init__(self):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read('./config.cfg')
        defaults = self.config['DEFAULT']
        for i in range(5):
            try:
                self.conn = hive.Connection(host=defaults['THRIFT_SERVER_URL'], port=defaults['THRIFT_PORT'], username=defaults['THRIFT_USERNAME'], database=defaults['BENCHMARK'])
                logger.info('SparkSQL connector conntects to thrift server: ' + defaults['THRIFT_SERVER_URL'] + ':' + defaults['THRIFT_PORT'])
                break
            except:
                logger.warning(f'Atempt {i + 1} Failed to connect to thrift server, retrying...')
        self.cursor = self.conn.cursor()
    # 
    def execute(self, query) -> DBConnector.TimedResult:
        for i in range(3):
            try:
                begin = time.time_ns()
                self.cursor.execute(query)
                collection = self.cursor.fetchall()
                elapsed_time_usecs = int((time.time_ns() - begin) / 1_000)
                break
            except:
                if i == 2:
                    logger.fatal('Execution failed 3 times.')
                    raise
                else:
                    logger.warning('Execution failed %s times, try again...', str(i + 1))
        logger.info('QUERY RESULT %s', str(collection)[:100] if len(str(collection)) > 100 else collection)
        collection = 'EmptyResult' if len(collection) == 0 else collection[0]
        logger.debug('Hash(QueryResult) = %s', str(hash(str(collection))))
        return DBConnector.TimedResult(collection, elapsed_time_usecs)

    def explain(self, query) -> str:
        timed_result = self.execute(f'EXPLAIN FORMATTED {query}')
        return _postprocess_plan(timed_result.result[0])

    def set_disabled_knobs(self, knobs) -> None:
        """Toggle a list of knobs"""
        if len(knobs) == 0:
            self.cursor.execute(f'RESET {EXCLUDED_RULES}')
        else:
            formatted_knobs = [f'org.apache.spark.sql.catalyst.optimizer.{rule}' for rule in knobs]
            self.cursor.execute(f'SET {EXCLUDED_RULES}={",".join(formatted_knobs)}')

    def get_knob(self, knob: str) -> bool:
        """Get current status of a knob"""
        self.cursor.execute(f'SET {EXCLUDED_RULES}')
        excluded_rules = self.cursor.fetchall()[0]
        logger.info('Current excluded rules: %s', excluded_rules)
        if excluded_rules is None:
            return True
        else:
            return not knob in excluded_rules

    @staticmethod
    def get_name() -> str:
        return 'spark'

    @staticmethod
    def get_knobs() -> list:
        """Static method returning all knobs defined for this connector"""
        with open('data/knobs.txt', 'r', encoding='utf-8') as f:
            return [line.replace('\n', '') for line in f.readlines()]
        
    @staticmethod
    def get_plan_preprocessor():
        from inference.preprocessing.preprocess_spark_plans import SparkPlanPreprocessor as complex_preprocessor
        from inference.preprocessing.preprocess_simple import SparkPlanPreprocessor as simple_preprocessor
        return simple_preprocessor