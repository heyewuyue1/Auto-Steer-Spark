# Auto-Steer-Spark

## Requirements

### Packages

- sqlite3
    - Statistics extension (we provide a download script: `sqlean-extensions/download.sh`)
- python3 (at least version 3.10)

### Python3 requirements

- Install python requirements using the file `pip3 install -r requirements.txt` (CUDA is not listed in requirements.txt, make sure you install them if you need them.)

## Run Auto-Steer

### Configuration
config file is `config.cfg`. refrences below
| key | info |
|BENCHMAK|the benchmark you want to use `job` or `tpcds`|
|THRIFT_SERVER_URL|jdbc server url|
|THRIFT_PORT|jdbc server port|
|THRIFT_USERNAME|your username|
|EXPLAIN_THREADS|how many thread to explain query plans|
|REPEATS|how many times to run a single query|

### Executing Auto-Steer

- run taining mode(example)
```commandline
python main.py --training --database job --benchmark ./benchmark/queries/job
```

- run inference mode(example)
```commandline
python main.py --inference --database job --benchmark benchmark/queries/job
```
1. By now, Auto-Steer persisted all generated training data (e.g. query plans and execution statistics) in a
   sqlite-database that can be found under `results/<database>.sqlite`.
2. The inference results can be found in the directory `evaluation`.
