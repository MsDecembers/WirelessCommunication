import sys
import os
import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

param_data = {}

def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path + "{}_rewards_curve".format(tag))
    plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('episodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """ 保存奖励
    """
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
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
    # 打开文件
    curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
    parent_path = os.path.dirname(curr_path)  # 父路径
    path = parent_path + '/config.yaml'
    file = open(path, 'r', encoding='utf-8')
    # 加载数据，转换后出来的是字典
    if not param_data:
        param_data.update(yaml.load(file, Loader=yaml.FullLoader))
    return param_data[param_type][param_name]



