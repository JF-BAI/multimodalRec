# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BM3', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')

    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation')

    config_dict = {
        'gpu_id': 0,
        'sparse': 1,
        'lambda_coeff': 0.9,
        'item_topk': 10,


    }

    args, _ = parser.parse_known_args()
#调用 quick_start 函数：传递模型名称、数据集名称、配置字典、是否保存模型以及是否使用镜像梯度。
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)#-->quick_start.py


