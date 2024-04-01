import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

with open(args.path) as f:
    lines = f.readlines()
    best_possible = 0
    default = 0
    for line in lines:
        if 'y:' in line:
            last_best_possible = eval(line.split(':')[-1].split()[0])
            print('best', last_best_possible)
        if 'best' in line:
            last_default = last_best_possible / eval(line.split('-> ')[-1].split(':')[0])
            print('default', last_default)
            best_possible += last_best_possible
            default += last_default
    print(best_possible, default, best_possible / default)