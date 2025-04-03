# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)#{'gpu_id': 0, 'model': 'BM3', 'dataset': 'baby',len():3,} -->utils.configurator.py
    init_logger(config)
    logger = getLogger()#实例化日志器
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)#-->utils.dataset.py
    # print dataset statistics
    logger.info(str(dataset))#数据集信息：用户数，物品数，交互数，稀疏度

    train_dataset, valid_dataset, test_dataset = dataset.split()#划分数据集，并打印相关信息
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader封装到数据加载器中，用于机器学习模型的训练和评估。
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)#用于加载训练数据集train_dataset
    (valid_data, test_data) = (#用于加载验证数据集valid_dataset和测试数据集test_dataset。
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()#验证指标
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])#加载模型并初始化
        logger.info(model)#记录模型的信息

        # trainer loading and initialization加载训练器
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training开始训练，返回最好的验证结果和最好的测试结果
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))#将超参数组合与对应的最佳结果存储起来。

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

