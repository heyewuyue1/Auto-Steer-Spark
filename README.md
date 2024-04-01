# Auto-Steer-Spark

## Requirements

### Packages

- sqlite3
    - Statistics extension (we provide a download script: `sqlean-extensions/download.sh`)
    - 这个extension目前是linux-x64版本的，如果电脑是其他操作系统请自己找到对应的包安装
- python3 (at least version 3.10)

### Python3 requirements

- Install python requirements using the file `pip3 install -r requirements.txt` (CUDA is not listed in requirements.txt, make sure you install them if you need them.)

## Run Auto-Steer

### Configuration
config file is `config.cfg`. refrences below

| key               | info                                           |
| ----------------- | ---------------------------------------------- |
| BENCHMAK          | the benchmark you want to use `job` or `tpcds` |
| THRIFT_SERVER_URL | jdbc server url                                |
| THRIFT_PORT       | jdbc server port                               |
| THRIFT_USERNAME   | your username                                  |
| EXPLAIN_THREADS   | how many thread to explain query plans         |
| REPEATS           | how many times to run a single query           |

### Executing Auto-Steer

- run taining mode(example)
```commandline
python main.py --training --database tpcds_sf10 --benchmark ./benchmark/queries/tpcds
```

- run inference mode(example)
```commandline
python main.py --inference --database tpcds_sf10 --benchmark benchmark/queries/tpcds --retrain --create-datasets
```


## 每次运行前后需要关注的文件

| 文件               | 解释                                                         |
| ------------------ | ------------------------------------------------------------ |
| `data/forest.pkl`  | 第一次运行`inference`的时候创建的森林，如果训练数据发生改变需要删除，然后重新运行`inference` |
| `evaluation/*.csv` | 最近一次`inference`的实验结果，如果认为有意义，请及时重命名或保存至其他位置，否则会被下一次运行的结果覆盖 |
| `evaluation/*.pdf` | 最近一次`inference`的loss图像，同上                          |
| `results/*.sqlite` | `train`过程中产生的数据集会保存在指定的数据库中，两次指定同一个数据库会导致两次的数据合并在一个数据库里，一般需要每次指定不一样的，或者将上一次的删除再指定相同的数据库 |

## 其他重要文件

`data/knobs.txt`：每一行一个可以开关的knob，可以用于手动指定要探索的knobs

`nn/data/_data`：每一次`inference`只要指定了`--create-dataset`就会自动保存的训练用数据集，会自动覆盖，一般不用管。

`nn/model/_model`：每一次`inference`保存的模型参数，会自动覆盖，一般不用管。