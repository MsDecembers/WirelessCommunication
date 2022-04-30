import math
import random
import numpy as np
import itertools as it

from common import utils
from communication import channel_gain

np.random.seed(10)


class IRS_COMP_MISO:
    def __init__(self):
        self.bs_num = utils.get_parameter('model', 'bs_num')
        self.ue_num = utils.get_parameter('model', 'ue_num')
        self.irs_elements_num = utils.get_parameter('model', 'irs_elements_num')
        self.antenna_num = utils.get_parameter('model', 'antenna_num')
        self.bandwidth = utils.get_parameter('model', 'bandwidth')
        self.power_max = utils.get_parameter('model', 'power_max')
        self.rate_min = utils.get_parameter('model', 'rate_min')
        self.action_dim = self.bs_num ** self.ue_num
        self.state_dim = self.bs_num * self.ue_num * self.antenna_num
        # Todo
        self.precode_matrices = initial_precode_matrices(self.bs_num, self.ue_num, self.antenna_num)
        # Todo
        self.reflect_coefficient_matrix = np.identity(self.irs_elements_num, dtype=complex)
        self.coord_bs, self.coord_ue, self.coord_irs = get_coordinate()
        self.action_list = []
        self.action_list.extend(list(it.product(range(2), repeat=self.ue_num)))
        self.rewards = []

    def check_done(self):
        flag = False
        if len(self.rewards) % 100 == 0:
            rewards_var = np.var(self.rewards[-100:])
            if rewards_var < 30000:
                flag = True
        return flag

    def step(self, actions):
        # Todo
        action_1 = self.action_list[actions]
        action_2 = self.action_list[self.action_dim - actions - 1]
        actions = [action_1, action_2]

        reward = 0
        gains = channel_gain.get_total_channel_gain(self.coord_bs, self.coord_ue, self.coord_irs,
                                                    self.reflect_coefficient_matrix)

        flag_1, sum_rate = check_min_rate_constraint(actions, self.rate_min, self.precode_matrices, gains)
        flag_2 = check_max_power_constraint(actions, self.power_max, self.precode_matrices)
        if flag_1 and flag_2:
            reward = sum_rate
        else:
            reward = -10000
        next_state = gains.flatten()
        self.rewards.append(reward)
        done = self.check_done()
        return next_state, reward, done

    def reset(self):
        # 反射系数矩阵会变化，重新计算信道增益
        gains = channel_gain.get_total_channel_gain(self.coord_bs, self.coord_ue, self.coord_irs,
                                                    self.reflect_coefficient_matrix)
        state = gains.flatten()
        return state


def check_min_rate_constraint(actions, rate_min, precode_matrices, gains):
    ue_num = utils.get_parameter('model', 'ue_num')
    flag = True
    sum_rate = 0
    for ue_id in range(ue_num):
        limit_rate = get_limit_transmit_rate(precode_matrices, gains, ue_id, actions)
        sum_rate += limit_rate
        if limit_rate < rate_min:
            flag = False
            break
    return flag, sum_rate


def check_max_power_constraint(actions, power_max, precode_matrices):
    flag = True
    bs_num = utils.get_parameter('model', 'bs_num')
    powers = get_transmit_powers(precode_matrices, actions)
    for bs in range(bs_num):
        if powers[bs] > power_max:
            flag = False
    return flag


def set_seed(seed):
    np.random.seed(seed)


def get_coordinate():
    coord_bs = np.random.randn(2, 3)
    coord_ue = np.random.randn(3, 3)
    coord_irs = np.random.randn(1, 3)
    return coord_bs, coord_ue, coord_irs


def get_additive_white_gaussian_noise():
    """ 获取加性高斯白噪声功率

    :return: 白噪声功率
    """
    bandwidth = utils.get_parameter('model', 'bandwidth')
    noise_dbm = -174 + 10 * np.log10(bandwidth)
    noise = np.power(10, ((noise_dbm - 30) / 10))
    return noise


def get_limit_transmit_rate(precode_matrices, gains, ue_id, actions):
    """ 计算极限传输速率

    :param precode_matrices: 预编码矩阵，维度：【基站数，用户数，天线数】
    :param gains: 信道增益，维度：【基站数，用户数，天线数】
    :param ue_id: 计算速率的用户id
    :param actions: 动作，维度：【基站数，用户数】
    :return: 用户的极限传输速率
    """
    bandwidth = utils.get_parameter('model', 'bandwidth')
    noise = get_additive_white_gaussian_noise()
    signal = 0
    for bs in range(len(actions)):
        ue_idx = 0
        for connected in actions[bs]:
            if connected == 1:
                gain = gains[bs][ue_idx]
                precode_vector = precode_matrices[bs][ue_idx]
                power = np.power(np.linalg.norm(np.dot(gain, precode_vector.T)), 2)
                if ue_idx == ue_id:
                    signal += power
                else:
                    noise += power
            ue_idx += 1
    sinr = signal / noise
    return bandwidth * np.log2((1 + sinr))


def initial_precode_matrices(bs_num, ue_num, antenna_num):
    """ 初始化预编码矩阵

    :param bs_num: 基站个数
    :param ue_num: 用户个数
    :param antenna_num: 基站的天线个数
    :return: 预编码矩阵，维度：【基站个数，用户个数，天线个数】
    """
    lamb = 1
    precode_matrices = []
    for bs in range(bs_num):
        precode_matrix = []
        for ue in range(ue_num):
            precode_vector = np.zeros(shape=[antenna_num], dtype=np.complex)
            for i in range(antenna_num):
                theta = 2 * random.random() * math.pi
                precode_vector[i] = lamb * (math.cos(theta) + math.sin(theta) * 1j)
            precode_matrix.append(precode_vector)
        precode_matrices.append(precode_matrix)
    return precode_matrices


def get_transmit_powers(precode_matrices, actions):
    """ 获取基站发射总功率

    :param precode_matrices: 预编码矩阵，维度：【基站个数，用户个数，天线个数】
    :param actions: 动作，维度：【基站个数，用户个数】
    :return: 基站发射总功率，维度：【基站个数】
    """
    bs_num = utils.get_parameter('model', 'bs_num')
    ue_num = utils.get_parameter('model', 'ue_num')
    powers = []
    for bs in range(bs_num):
        power = 0
        for ue in range(ue_num):
            if actions[bs][ue] == 1:
                power += np.power(np.linalg.norm(precode_matrices[bs][ue]), 2)
        powers.append(power)
    return powers
