import numpy as np
from common import utils


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


if __name__ == '__main__':
    np.random.seed(1002)
    coord_BS = np.random.randn(2, 3)
    coord_UE = np.random.randn(3, 3)
    coord_Irs = np.random.randn(1, 3)
    r = np.identity(20, dtype=complex)
    Total_gain = get_total_channel_gain(coord_BS, coord_UE, coord_Irs, r)
    print(Total_gain)
