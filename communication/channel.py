import math
import random
import numpy as np

from common import utils


def initial_coordinate(bs_num, ue_num):
    """ 获取基站、用户、IRS的坐标

    :param bs_num: 基站数量
    :param ue_num: 用户数量
    :return: 基站、用户、IRS的坐标
    """
    # Todo
    coord_bs = np.random.randn(bs_num, 3)
    coord_ue = np.random.randn(ue_num, 3)
    coord_irs = np.random.randn(1, 3)
    return coord_bs, coord_ue, coord_irs


def get_channel_gain(coord_1, coord_2, path_loss, small_fading_style):
    """ 获取基站到用户的信道增益

    :param coord_1: 坐标1，维度：【m，3】
    :param coord_2: 坐标2，维度：【n，3】
    :param path_loss: 路径损耗系数
    :param small_fading_style: 小尺度衰落类型
    :return: 基站到用户的信道增益，维度：【m，n】
    """
    m = coord_1.shape[0]  # 基站数
    n = coord_2.shape[0]  # 用户数

    channel_gain = np.zeros((m, n), dtype=complex)
    for i in range(m):
        for j in range(n):
            if small_fading_style == utils.get_parameter('gain', 'Rayleigh'):
                small = np.sqrt(pow(np.random.normal(0, 1 / 2, 1), 2) + pow(np.random.normal(0, 1 / 2, 1), 2))
            else:
                small = np.sqrt(1 / 3) * np.sqrt(
                    pow(np.random.normal(0, 1 / 2, 1), 2) + pow(np.random.normal(0, 1 / 2, 1), 2)) \
                        + np.sqrt(2 / 3) * (
                                np.cos(np.random.rand() * 2 * np.pi) + np.sin(np.random.rand() * 2 * np.pi) * 1j)
            ad = np.array(coord_1[i, :] - coord_2[j, :]).reshape(1, 3)
            arr1 = np.linalg.norm(ad, axis=1, keepdims=True)  # 默认是2范数
            if arr1[0, 0] == 0:
                arr1[0, 0] = 0.000001
            channel_gain[i, j] = np.sqrt(0.001 * (arr1[0, 0]) ** (-path_loss)) * small[0]
    return channel_gain


def get_total_channel_gain(coord_bs, coord_ue, coord_irs, reflect_coefficient_matrix):
    """ 获取总的信道增益

    :param coord_bs: 基站坐标，维度：【基站个数，3】
    :param coord_ue: 用户个数，维度：【用户个数，3】
    :param coord_irs: IRS坐标，维度：【3】
    :param reflect_coefficient_matrix: IRS反射系数矩阵
    :return: IRS辅助的基站到用户信道增益，维度：【基站个数，用户数，天线数】
    """
    antenna_num = utils.get_parameter('model', 'antenna_num')
    small_fading_style = utils.get_parameter('gain', 'Rayleigh')
    irs_elements_num = utils.get_parameter('model', 'irs_elements_num')
    bs_ue_gain = get_channel_gain(coord_bs, coord_ue, small_fading_style, utils.get_parameter('gain', 'ue_bs_ple'))
    bs_ue_gain = np.expand_dims(bs_ue_gain, axis=1).repeat(antenna_num, axis=1)
    irs_ue_gain = get_channel_gain(coord_irs, coord_ue, small_fading_style, utils.get_parameter('gain', 'ue_irs_ple'))
    irs_ue_gain = irs_ue_gain.repeat(irs_elements_num, axis=0)
    bs_irs_gain = get_channel_gain(coord_bs, coord_irs, small_fading_style, utils.get_parameter('gain', 'irs_bs_ple'))
    bs_irs_gain = np.expand_dims(bs_irs_gain, axis=2).repeat(irs_elements_num, axis=1).repeat(antenna_num, axis=2)
    temp1 = np.dot(irs_ue_gain.T.conjugate(), reflect_coefficient_matrix)
    temp2 = np.dot(temp1, bs_irs_gain).transpose(1, 0, 2)
    total_gain = bs_ue_gain.transpose(0, 2, 1) + temp2
    return total_gain


def get_additive_white_gaussian_noise():
    """ 获取加性高斯白噪声功率

    :return: 白噪声功率
    """
    bandwidth = utils.get_parameter('model', 'bandwidth')
    noise_dbm = -174 + 10 * np.log10(bandwidth)
    noise = np.power(10, ((noise_dbm - 30) / 10))
    return noise


def get_limit_transmit_rate(precode_matrices, gains, actions):
    """ 计算极限传输速率

    :param precode_matrices: 预编码矩阵，维度：【基站数，用户数，天线数】
    :param gains: 信道增益，维度：【基站数，用户数，天线数】
    :param actions: 动作，用户选择的基站id，维度：【用户数】
    :return: 用户的极限传输速率，维度：【用户数】
    """
    bandwidth = utils.get_parameter('model', 'bandwidth')
    bs_num = utils.get_parameter('model', 'bs_num')
    ue_num = utils.get_parameter('model', 'ue_num')

    limit_rates = np.zeros(ue_num)

    noise = get_additive_white_gaussian_noise()
    signal = 0

    for ue_id in range(ue_num):
        for bs in range(bs_num):
            for ue in range(ue_num):
                gain = gains[bs][ue_id]
                precode_vector = precode_matrices[bs][ue]
                power = np.power(np.linalg.norm(np.dot(gain, precode_vector.T)), 2)
                if bs == actions[ue] and ue_id == ue:
                    signal += power
                else:
                    noise += power
        sinr = signal / noise
        limit_rates[ue_id] += bandwidth * np.log2((1 + sinr))

    return limit_rates


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


def initial_reflect_matrices(irs_elements_num):
    """ 初始化反射系数矩阵

    :param irs_elements_num: 反射元件个数
    :return: 反射系数矩阵，对角矩阵，维度：【反射元件个数，反射元件个数】
    """
    return np.identity(irs_elements_num, dtype=complex)


def get_transmit_powers(precode_matrices, actions):
    """ 获取基站发射总功率

    :param precode_matrices: 预编码矩阵，维度：【基站个数，用户个数，天线个数】
    :param actions: 动作，每个用户选取的基站id，维度：【用户个数】
    :return: 基站发射总功率，维度：【基站个数】
    """
    bs_num = utils.get_parameter('model', 'bs_num')
    ue_num = utils.get_parameter('model', 'ue_num')
    powers = np.zeros(bs_num)
    for ue in range(ue_num):
        for bs in range(bs_num):
            # 判断用户是否和该基站连接
            if actions[ue] == bs:
                powers[bs] += np.power(np.linalg.norm(precode_matrices[bs][ue]), 2)
    return powers


def get_content_ue_requests():
    """ 获取用户请求内容情况

    一个用户只能请求一个内容

    :return: 用户请求的内容id，维度：【用户数】
    """
    ue_num = utils.get_parameter('model', 'ue_num')
    content_num = utils.get_parameter('model', 'content_num')
    requests = np.random.randint(0, high=content_num, size=ue_num)
    return requests


if __name__ == '__main__':
    np.random.seed(1002)
    coord_BS = np.random.randn(2, 3)
    coord_UE = np.random.randn(3, 3)
    coord_Irs = np.random.randn(1, 3)
    r = np.identity(20, dtype=complex)
    Total_gain = get_total_channel_gain(coord_BS, coord_UE, coord_Irs, r)
    print(Total_gain)
