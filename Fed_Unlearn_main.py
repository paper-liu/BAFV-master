# -*- coding: utf-8 -*-
"""
Created on 05 30 2024

@author: liu
"""
import os
import logging
import datetime
import json
import os.path

#%%
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time 

#ourself libs
from model_initiation import model_init
from data_preprocess import data_init, data_init_with_shadow
from FL_base import global_train_once
from FL_base import fedavg
from FL_base import test

from FL_base import FL_Train, FL_Retrain
from Fed_Unlearn_base import unlearning, unlearning_without_cali, federated_learning_unlearning
from membership_inference import train_attack_model, attack
from Utils_BAFV import *


"""Step 0. 初始化Federated Unlearning 的参数"""
class Arguments():
    def __init__(self):
        self.message = "test for debug"  # test for debug

        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10
        self.data_name = 'mnist'    # mnist, cifar10, fashion-mnist, purchase, adult
        self.global_epoch = 50      # 30
        self.local_epoch = 10       # 10
        
        # Model Training Settings
        self.local_batch_size = 128
        self.local_lr = 0.005
        self.test_batch_size = 128
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False
        
        # Federated Unlearning Settings
        self.unlearn_interval = 1
        self.forget_client_idx = 2
        self.if_retrain = False
        self.if_unlearning = False
        self.forget_local_epoch_ratio = 0.5
        # self.mia_oldGM = False

        # BAFV Settings
        self.if_imprint = True
        self.if_posttrain = False
        self.posttrain_epoch = 50
        self.unlearn_method = 'FedEraser'
        self.imprint_method = 'EnduraMark'
        self.poison_label_swap = 3
        self.poison_min_local_epoch = 10
        self.poison_max_local_epoch = 20
        self.poisoning_per_batch = 64
        self.adversarial_index = -1
        self.type = 'MNIST'
        # gap 2 size 1*4 base (0, 0)
        self.poison_pattern_0 = [[1, 0], [1, 1], [1, 2], [1,  3], [1,  4]]
        self.poison_pattern_1 = [[1, 7], [1, 8], [1, 9], [1, 10], [1, 11]]
        self.poison_pattern_2 = [[4, 0], [4, 1], [4, 2], [4,  3], [4,  4]]
        self.poison_pattern_3 = [[4, 7], [4, 8], [4, 9], [4, 10], [4, 11]]
        self.poison_pattern_4 = [[7, 0], [7, 1], [7, 2], [7,  3], [7,  4]]
        self.poison_pattern_5 = [[7, 7], [7, 8], [7, 9], [7, 10], [7, 11]]


def Federated_Unlearning():
    """Step 1. 设置 Federated Unlearning 的参数"""
    FL_params = Arguments()
    print(FL_params.message)

    torch.manual_seed(FL_params.seed)
    logdir = './logs'
    mkdirs(logdir)

    argument_path = 'experiment_argument-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join(logdir, argument_path), 'w') as f:
        json.dump(class_to_dict(FL_params), f)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path = 'experiment_argument-%s.log' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        filename=os.path.join(logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    print('* Forgetting Validation Algorithm For Federated Unlearning *')
    logger.info('* Forgetting Validation Algorithm For Federated Unlearning *')

    # kwargs for data loader
    print(60*'=')
    print("Step1. Federated Learning Settings \nWe use dataset: "+FL_params.data_name+(" for our Federated Unlearning experiment."))
    logger.info(60*'=')
    logger.info('Step1. Federated Learning Settings')
    logger.info('We use dataset: {} for our Federated Unlearning experiment'.format(FL_params.data_name))

    """Step 2. 构建联邦学习所需要的必要用户私有数据集，以及共有测试集"""
    print(60*'=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    logger.info(60*'=')
    logger.info('Step2. Client data loaded, testing data loaded!!!')
    logger.info('Initial Model loaded!!!')

    init_global_model = model_init(FL_params.data_name)
    client_all_loaders, test_loader = data_init(FL_params)

    print('init_global_model: ')
    logger.info('init_global_model: ')
    Mytest_poison(init_global_model, test_loader, FL_params)

    selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client, replace=False)
    client_loaders = list()
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])
    # client_all_loaders = client_loaders[selected_clients]
    # client_loaders, test_loader, shadow_client_loaders, shadow_test_loader = data_init_with_shadow(FL_params)

    """Step 3. 选择某一个用户的数据来遗忘，1.Federated Learning, 2.Unlearning, and 3.Unlearing without calibration"""
    print(60*'=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    logger.info(60*'=')
    logger.info('Step3. Fedearated Learning and Unlearning Training...')

    old_GMs, unlearn_GMs, uncali_unlearn_GMs, old_CMs = federated_learning_unlearning(init_global_model,
                                                                                      client_loaders,
                                                                                      test_loader,
                                                                                      FL_params,
                                                                                      logger=logger)
    if(FL_params.if_retrain == True):
        t1 = time.time()
        retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
        t2 = time.time()
        print("Time using = {} seconds".format(t2-t1))
        logger.info("Time using = {} seconds".format(t2-t1))

    print(60*'=')
    print("Step3.5. BACKDOOR Attack aganist GMS...")
    logger.info(60*'=')
    logger.info('Step3.5. BACKDOOR Attack aganist GMS...')
    normal_acc_list, normal_loss_list, backdoor_acc_list, forgetting_rate_list = trigger_performance_test(old_GMs,
                                                                                                          uncali_unlearn_GMs,
                                                                                                          unlearn_GMs,
                                                                                                          test_loader,
                                                                                                          FL_params)
    print(60 * '=')
    logger.info(60 * '=')
    print("normal_acc_list: {}".format(normal_acc_list))
    print("normal_loss_list: {}".format(normal_loss_list))
    print("EnduraMark_acc_list: {}".format(backdoor_acc_list))
    print("forgetting_rate_list: {}".format(forgetting_rate_list))
    logger.info("normal_acc_list: {}".format(normal_acc_list))
    logger.info("normal_loss_list: {}".format(normal_loss_list))
    logger.info("EnduraMark_acc_list: {}".format(backdoor_acc_list))
    logger.info("forgetting_rate_list: {}".format(forgetting_rate_list))

    print("average_forgetting_rate: {}".format(sum(forgetting_rate_list)/len(forgetting_rate_list)))
    logger.info("average_forgetting_rate: {}".format(sum(forgetting_rate_list)/len(forgetting_rate_list)))

    """Step 4  基于 target global model 在 client_loaders 和 test_loader 上的输出，构建成员推断攻击模型"""
    print(60*'=')
    print("Step4. Membership Inference Attack aganist GM...")
    logger.info(60*'=')
    logger.info('Step4. Membership Inference Attack aganist GM...')

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs[T_epoch]
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)

    print("\nEpoch = {}".format(T_epoch))
    print("Attacking against FL Standard")
    logger.info("Epoch = {}".format(T_epoch))
    logger.info("Attacking against FL Standard")

    target_model = old_GMs[T_epoch]
    (ACC_old, PRE_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
    if(FL_params.if_retrain == True):
        print("Attacking against FL Retrain")
        logger.info("Attacking against FL Retrain")
        target_model = retrain_GMs[T_epoch]
        (ACC_retrain, PRE_retrain) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Unlearn")
    logger.info("Attacking against FL Unlearn")
    target_model = unlearn_GMs[T_epoch]
    (ACC_unlearn, PRE_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)


if __name__ == '__main__':
    Federated_Unlearning()
