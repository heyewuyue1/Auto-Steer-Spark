# Copyright 2022 Intel Corporation
# SPDX-License-Identifier: MIT
#
"""Run AutoSteer's training mode to explore alternative query plans"""
from typing import Type
import storage
import os
import sys

import connectors.connector
from connectors.spark_connector import SparkConnector
from utils.arguments_parser import get_parser
from utils.custom_logging import logger
from autosteer.dp_exploration import explore_optimizer_configs
from autosteer.query_span import run_get_query_span
from inference.train import train_tcnn
from tqdm import tqdm
from pyhive import hive
import configparser

config = configparser.ConfigParser()
config.read('config.cfg')
default = config['DEFAULT']

def approx_query_span_and_run(connector: Type[connectors.connector.DBConnector], benchmark: str, query: str):
    run_get_query_span(connector, benchmark, query)
    connector = connector()
    explore_optimizer_configs(connector, f'{benchmark}/{query}')


def inference_mode(connector, benchmark: str, retrain: bool, create_datasets: bool):
    train_tcnn(connector, benchmark, retrain, create_datasets)

def check_and_load_database():
    database = default['BENCHMARK']
    logger.info(f'check and load database {database}...')
    conn = hive.Connection(host=default['THRIFT_SERVER_URL'], port=default['THRIFT_PORT'], username=default['THRIFT_USERNAME'])
    cursor = conn.cursor()
    # cursor.execute(f'DROP DATABASE IF EXISTS {database} CASCADE')
    cursor.execute(f'CREATE DATABASE IF NOT EXISTS {database}')
    cursor.execute(f'USE {database}')
    with open(f'./benchmark/schemas/{database}.sql', 'r') as f:
        query = f.read()
        query = query.split(';')
        for q in query:
            if q.strip() != '':
                cursor.execute(q)

if __name__ == '__main__':
    args = get_parser().parse_args()
    ConnectorType = SparkConnector
    storage.TESTED_DATABASE = args.database
    check_and_load_database()
    if args.benchmark is None or not os.path.isdir(args.benchmark):
        logger.fatal('Cannot access the benchmark directory containing the sql files with path=%s', args.benchmark)
        sys.exit(1)

    storage.BENCHMARK_ID = storage.register_benchmark(args.benchmark)

    if (args.inference and args.training) or (not args.inference and not args.training):
        logger.fatal('Specify either training or inference mode')
        sys.exit(1)
    if args.inference:
        logger.info('Run AutoSteer\'s inference mode')
        inference_mode(ConnectorType, args.benchmark, args.retrain, args.create_datasets)
    elif args.training:
        logger.info('Run AutoSteer\'s training mode')
        f_list = sorted(os.listdir(args.benchmark))
        logger.info('Found the following SQL files: %s', f_list)
        for query in tqdm(f_list):
            logger.info('run Q%s...', query)
            approx_query_span_and_run(ConnectorType, args.benchmark, query)
        most_frequent_knobs = storage.get_most_disabled_rules()
        logger.info('Training ended. Most frequent disabled rules: %s', most_frequent_knobs)
