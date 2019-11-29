#
# This file just serves to give a better intuition of how we conducted batch experiments
# It is not strictly a part of the DBH infrastructure, just an automation layer one level above it
#


import os
import copy

import dbh

shared = {
    'csv': 'dataset.csv',
    'label': 'BUG',
    'clean': False,
    'seed': 1337,
    'output': os.path.abspath('output'),
    'device': '/device:CPU:0',
    'log_device': False,
    'calc_completeness': True
}

data_steps = [

    # preprocess method

    # {
    #     'preprocess': [['labels', 'binarize']],
    # },
    # {
    #     'preprocess': [['features', 'normalize'], ['labels', 'binarize']],
    # },
    {
        'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
        'resample': 'down',
        'resample_amount': 100
    },

    # resample method/amount

    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'up',
    #     'resample_amount': 100
    # },
    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'up',
    #     'resample_amount': 75
    # },
    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'up',
    #     'resample_amount': 50
    # },
    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'up',
    #     'resample_amount': 25
    # },
    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'down',
    #     'resample_amount': 100
    # },
    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'down',
    #     'resample_amount': 75
    # },
    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'down',
    #     'resample_amount': 50
    # },
    # {
    #     'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    #     'resample': 'down',
    #     'resample_amount': 25
    # },
]

basic_strategy = [
    # ['knn', ''],
    # ['bayes', ''],
    # ['logistic', '--penalty l2 --solver saga --C 0.2 --tol 0.0001']
    # ['svm', ''],
    # ['forest', ''],
    # ['tree', ''],
    # ['tree', '--max-depth 5']
    ['linear', ''],
    # ['logistic', ''],
    # ['zeror', ''],
    # ['sdnnc', '--layers 3 --neurons 100 --batch 100 --epochs 5 --lr 0.1']
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 5']
]

chosen_prep = {
    'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
    'resample': 'up',
    'resample_amount': 50
}

strategy_steps = [
    # ['sdnnc', '--layers 2 --neurons 100 --batch 100 --epochs 5 --lr 0.1'],
    # ['sdnnc', '--layers 3 --neurons 100 --batch 100 --epochs 5 --lr 0.1'],
    # ['sdnnc', '--layers 4 --neurons 100 --batch 100 --epochs 5 --lr 0.1'],
    # ['sdnnc', '--layers 5 --neurons 100 --batch 100 --epochs 5 --lr 0.1'],
    # ['sdnnc', '--layers 5 --neurons 150 --batch 100 --epochs 5 --lr 0.1'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 5 --lr 0.1'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 2 --lr 0.1'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.1'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 20 --lr 0.1'],

    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.025'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.05'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.1'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.2'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.3'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.4'],
    # ['sdnnc', '--layers 5 --neurons 200 --batch 100 --epochs 10 --lr 0.5'],
    
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.025'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.05'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.2'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.3'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.4'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.5'],
    
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.5 --beta 0.0005'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.5 --beta 0.001'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.5 --beta 0.002'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.5 --beta 0.005'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.5 --beta 0.02'],

    # ['cdnnc', '--layers 4 --neurons 200 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 6 --neurons 200 --batch 100 --lr 0.1'],

    # ['cdnnc', '--layers 5 --neurons 150 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 5 --neurons 250 --batch 100 --lr 0.1'],

    # ['cdnnc', '--layers 5 --neurons 200 --batch 50 --lr 0.1'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 5 --neurons 200 --batch 150 --lr 0.1'],

    # ['cdnnc', '--layers 5 --neurons 300 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 5 --neurons 300 --batch 100 --lr 0.2'],
    # ['cdnnc', '--layers 5 --neurons 300 --batch 100 --lr 0.3'],
    # ['cdnnc', '--layers 6 --neurons 300 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 6 --neurons 300 --batch 100 --lr 0.2'],
    # ['cdnnc', '--layers 6 --neurons 300 --batch 100 --lr 0.3'],
    # ['cdnnc', '--layers 6 --neurons 350 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 6 --neurons 350 --batch 100 --lr 0.2'],
    # ['cdnnc', '--layers 6 --neurons 350 --batch 100 --lr 0.3'],
    # ['cdnnc', '--layers 7 --neurons 350 --batch 100 --lr 0.1'],
    # ['cdnnc', '--layers 7 --neurons 350 --batch 100 --lr 0.2'],
    # ['cdnnc', '--layers 7 --neurons 350 --batch 100 --lr 0.3'],

    # later tuning of sklearn algs, too

    # ['knn', '--n_neighbors 22'],
    # ['knn', '--n_neighbors 24'],
    # ['knn', '--n_neighbors 26'],
    # ['knn', '--n_neighbors 28'],
    # ['knn', '--n_neighbors 30'],
    # ['knn', '--n_neighbors 32'],
    # ['knn', '--n_neighbors 34'],
    # ['knn', '--n_neighbors 36'],
    # ['knn', '--n_neighbors 38'],
    # ['knn', '--n_neighbors 40'],
    # ['knn', '--n_neighbors 42'],
    # ['knn', '--n_neighbors 44'],
    # ['knn', '--n_neighbors 6 --weights distance'],
    # ['knn', '--n_neighbors 8 --weights distance'],
    # ['knn', '--n_neighbors 10 --weights distance'],
    # ['knn', '--n_neighbors 12 --weights distance'],
    # ['knn', '--n_neighbors 14 --weights distance'],
    # ['knn', '--n_neighbors 16 --weights distance'],
    # ['knn', '--n_neighbors 18 --weights distance'],
    # ['knn', '--n_neighbors 20 --weights distance'],

    # ['tree', '--max-depth 5'],
    # ['tree', '--max-depth 10'],
    # ['tree', '--max-depth 20'],
    # ['tree', '--max-depth 50'],
    # ['tree', '--max-depth 100'],
    # ['tree', '--criterion entropy --max-depth 5'],
    # ['tree', '--criterion entropy --max-depth 10'],
    # ['tree', '--criterion entropy --max-depth 20'],
    # ['tree', '--criterion entropy --max-depth 50'],
    # ['tree', '--criterion entropy --max-depth 100'],

    # Gauss is not parametric, so done :)

    # ['svm', '--kernel linear --C 0.1'],
    # ['svm', '--kernel linear --C 0.5'],
    # ['svm', '--kernel linear --C 1.0'],
    # ['svm', '--kernel linear --C 1.5'],
    # ['svm', '--kernel linear --C 2.0'],
    # ['svm', '--kernel poly --degree 2 --C 0.1'],
    # ['svm', '--kernel poly --degree 2 --C 0.5'],
    # ['svm', '--kernel poly --degree 2 --C 1.0'],
    # ['svm', '--kernel poly --degree 2 --C 1.5'],
    # ['svm', '--kernel poly --degree 2 --C 2.0'],
    # ['svm', '--kernel poly --degree 3 --C 0.1'],
    # ['svm', '--kernel poly --degree 3 --C 0.5'],
    # ['svm', '--kernel poly --degree 3 --C 1.0'],
    # ['svm', '--kernel poly --degree 3 --C 1.5'],
    # ['svm', '--kernel poly --degree 3 --C 2.0'],
    # ['svm', '--kernel rbf --C 0.1'],
    # ['svm', '--kernel rbf --C 0.5'],
    # ['svm', '--kernel rbf --C 1.0'],
    # ['svm', '--kernel rbf --C 1.5'],
    # ['svm', '--kernel rbf --C 2.0'],
    # ['svm', '--kernel rbf --C 2.2'],
    # ['svm', '--kernel rbf --C 2.4'],
    # ['svm', '--kernel rbf --C 2.6'],
    # ['svm', '--kernel rbf --C 2.8'],
    # ['svm', '--kernel rbf --C 3.0'],
    # ['svm', '--kernel rbf --C 4.0'],
    # ['svm', '--kernel rbf --C 5.0'],
    # ['svm', '--kernel sigmoid --C 0.1'],
    # ['svm', '--kernel sigmoid --C 0.5'],
    # ['svm', '--kernel sigmoid --C 1.0'],
    # ['svm', '--kernel sigmoid --C 1.5'],
    # ['svm', '--kernel sigmoid --C 2.0'],

    # ['svm', '--kernel rbf --C 2.6 --gamma 0.005'],
    # ['svm', '--kernel rbf --C 2.6 --gamma 0.01'],
    # ['svm', '--kernel rbf --C 2.6 --gamma 0.02'],
    # ['svm', '--kernel rbf --C 2.6 --gamma 0.05'],

    # ['forest', '--max-depth 10 --n-estimators 5'],
    # ['forest', '--max-depth 10 --n-estimators 10'],
    # ['forest', '--max-depth 10 --n-estimators 20'],
    # ['forest', '--max-depth 10 --n-estimators 50'],
    # ['forest', '--max-depth 10 --n-estimators 100'],
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 5'],
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 10'],
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 20'],
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 50'],
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 100'],


    # ['logistic', '--penalty l2 --solver newton-cg --C 5.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver lbfgs     --C 5.0 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver liblinear --C 5.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver liblinear --C 5.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver sag       --C 5.0 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver saga      --C 5.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver saga      --C 5.0 --tol 0.0001'],

    # ['logistic', '--penalty l2 --solver newton-cg --C 2.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver lbfgs     --C 2.0 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver liblinear --C 2.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver liblinear --C 2.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver sag       --C 2.0 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver saga      --C 2.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver saga      --C 2.0 --tol 0.0001'],

    # ['logistic', '--penalty l2 --solver newton-cg --C 1.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver lbfgs     --C 1.0 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver liblinear --C 1.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver liblinear --C 1.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver sag       --C 1.0 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver saga      --C 1.0 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver saga      --C 1.0 --tol 0.0001'],

    # ['logistic', '--penalty l2 --solver newton-cg --C 0.5 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver lbfgs     --C 0.5 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver liblinear --C 0.5 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver liblinear --C 0.5 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver sag       --C 0.5 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver saga      --C 0.5 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver saga      --C 0.5 --tol 0.0001'],

    # ['logistic', '--penalty l2 --solver newton-cg --C 0.2 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver lbfgs     --C 0.2 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver liblinear --C 0.2 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver liblinear --C 0.2 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver sag       --C 0.2 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver saga      --C 0.2 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver saga      --C 0.2 --tol 0.0001'],

    # ['logistic', '--penalty l2 --solver newton-cg --C 0.1 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver lbfgs     --C 0.1 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver liblinear --C 0.1 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver liblinear --C 0.1 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver sag       --C 0.1 --tol 0.0001'],
    # ['logistic', '--penalty l1 --solver saga      --C 0.1 --tol 0.0001'],
    # ['logistic', '--penalty l2 --solver saga      --C 0.1 --tol 0.0001'],

    # ['linear', ''],
    # ['bayes', ''],
]

def main():

    for data_step in data_steps:
        params = copy.deepcopy(shared)
        params = {**params, **data_step}
        params['strategy'] = basic_strategy
        dbh.main(params)

    # params = copy.deepcopy(shared)
    # params = {**params, **chosen_prep}
    # params['strategy'] = strategy_steps
    # dbh.main(params)
    


if __name__ == '__main__':
    main()