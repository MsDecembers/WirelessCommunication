import sys
import os
import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import torch

param_data = {}


def plot_rewards(rewards, ma_rewards, result_path, tag='train'):
    device = get_device()
    env_name = get_parameter('base', 'env_name')
    algorithm_name = get_parameter('base', 'algorithm_name')
    save = get_parameter('base', 'save')
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(device, algorithm_name, env_name))
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(result_path + "{}_rewards_curve".format(tag))
    # plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('episodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    # plt.show()


def save_results(rewards, ma_rewards, result_path, tag='train'):
    """ 保存奖励
    """
    np.save(result_path + '{}_rewards.npy'.format(tag), rewards)
    np.save(result_path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('奖励值保存完毕！')


def make_dir(*paths):
    """ 创建文件夹
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    """ 删除目录下所有空文件夹
    """
    for path in paths:
        dirs = os.listdir(path)
        for d in dirs:
            if not os.listdir(os.path.join(path, d)):
                os.removedirs(os.path.join(path, d))


def get_project_url():
    curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
    parent_path = os.path.dirname(curr_path)  # 父路径
    sys.path.append(parent_path)  # 添加路径到系统路径
    return parent_path


def get_current_time():
    # 获取当前时间
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_parameter(param_type, param_name):
    params = get_parameters(param_type)
    return params[param_name]


def get_parameters(param_type):
    if param_data:
        return param_data[param_type]
    # 打开文件
    curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
    parent_path = os.path.dirname(curr_path)  # 父路径
    path = parent_path + '/config.yaml'
    file = open(path, 'r', encoding='utf-8')
    param_data.update(yaml.load(file, Loader=yaml.FullLoader))
    return param_data[param_type]


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def update_epsilon():
    param_data['drl']['epsilon_start'] = 0.0
    param_data['drl']['epsilon_end'] = 0.0


def get_save_paths():
    env_name = get_parameter('base', 'env_name')
    result_path = get_project_url() + "/outputs/" + env_name + \
                  '/' + get_current_time() + '/results/'  # 保存结果的路径
    model_path = get_project_url() + "/outputs/" + env_name + \
                 '/' + get_current_time() + '/models/'  # 保存模型的路径
    return result_path, model_path
