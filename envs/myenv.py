import collections

import numpy as np

from common import utils
from communication import channel


class IRS_COMP_MISO:
    def __init__(self):
        self.bs_num = utils.get_parameter('model', 'bs_num')
        self.ue_num = utils.get_parameter('model', 'ue_num')
        self.antenna_num = utils.get_parameter('model', 'antenna_num')
        self.irs_elements_num = utils.get_parameter('model', 'irs_elements_num')
        self.content_num = utils.get_parameter('model', 'content_num')
        self.bandwidth = utils.get_parameter('model', 'bandwidth')
        # Todo
        self.power_max = np.array(utils.get_parameter('model', 'power_max')).repeat(self.bs_num)
        # Todo
        self.rate_min = np.array(utils.get_parameter('model', 'rate_min')).repeat(self.ue_num)
        self.cache_max = np.array(utils.get_parameter('model', 'cache_max'))
        # Todo
        self.content_sizes = np.array(utils.get_parameter('model', 'content_sizes')).repeat(self.content_num)
        self.flink_capacity = np.array(utils.get_parameter('model', 'flink_capacity'))
        # Todo
        self.precode_matrices = channel.initial_precode_matrices(self.bs_num, self.ue_num, self.antenna_num)
        # Todo
        self.reflect_coefficient_matrix = channel.initial_reflect_matrices(self.irs_elements_num)
        # Todo
        self.coord_bs, self.coord_ue, self.coord_irs = channel.initial_coordinate(self.bs_num, self.ue_num)

        self.buffer = []  # 循环队列
        self.position = 0
        self.counter = 0
        self.capacity = utils.get_parameter('model', 'queue_capacity')
        self.reward_var_limit = utils.get_parameter('model', 'reward_var_limit')

    def check_done(self, reward):
        """ 判断是否结束本次Episode

        :param reward: 当前time step获得的奖励
        :return: 结束标志，True为结束，False为继续
        """
        flag = False
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = reward
        self.position = (self.position + 1) % self.capacity
        reward_var = np.var(self.buffer)
        if self.counter > self.capacity and reward_var < self.reward_var_limit:
            self.counter = 0
            flag = True
        return flag

    def step(self, actions):
        """ 计算奖励、获取下一个time step状态、获取结束标志

        :param actions: 动作，用户选择的基站id，内容存放的基站id，维度：【【用户数】，【内容数】】
        :return: next_state（维度：【基站数，用户数，天线数】）, reward（【标量】）, done（【Boolean】）
        """

        # 重新计算信道增益
        gains = channel.get_total_channel_gain(self.coord_bs, self.coord_ue, self.coord_irs,
                                               self.reflect_coefficient_matrix)
        ue_actions = actions['ue_actions']
        content_actions = actions['content_actions']
        # ue的校验
        flag_1, limit_rates = check_min_rate_constraint(self.precode_matrices, gains, self.rate_min, ue_actions)
        flag_2 = check_max_power_constraint(self.precode_matrices, self.power_max, ue_actions)
        # cache的校验
        flag_3, bs_caches = check_max_cache_constraint(self.cache_max, self.content_sizes, content_actions)

        requests = channel.get_content_ue_requests()
        cached_list = []  # 维度：【用户数】，类型：布尔
        lack_count = collections.defaultdict(int)
        for ue, bs in enumerate(ue_actions):
            request = requests[ue]
            # cached_judge = lambda x: True if len(x) > 0 else False
            # cached = cached_judge(set(bs_caches[bs]).intersection(request))
            cached = request in set(bs_caches[bs])
            if not cached:
                lack_count[bs] += 1
            cached_list.append(cached)

        # 计算延迟
        delays = []
        for ue, bs in enumerate(ue_actions):
            request = requests[ue]
            delay = self.content_sizes[request] / limit_rates[ue]
            if not cached_list[ue]:  # 基站没有缓存请求的内容，多加一部分延迟
                delay += self.content_sizes[request] * lack_count[bs] / self.flink_capacity[bs]
            delays.append(delay)

        reward = -5
        if flag_1 and flag_2 and flag_3:
            reward = -sum(delays)

        next_state = gains
        self.counter += 1
        done = self.check_done(reward)
        return next_state, reward, done

    def reset(self):
        # 反射系数矩阵会变化，重新计算信道增益
        gains = channel.get_total_channel_gain(self.coord_bs, self.coord_ue, self.coord_irs,
                                               self.reflect_coefficient_matrix)
        self.buffer = []  # 循环队列
        self.position = 0
        self.counter = 0
        # state = gains.flatten()
        return gains


def check_max_cache_constraint(cache_max, content_sizes, content_actions):
    bs_num = utils.get_parameter('model', 'bs_num')
    bs_caches = collections.defaultdict(list)  # 含空基站
    for content, bs in enumerate(content_actions):
        bs_caches[bs].append(content)
    caches_sum = []
    for bs in range(bs_num):
        cache_sum = sum(content_sizes[bs_caches[bs]])
        caches_sum.append(cache_sum)
    flag = (np.array(caches_sum) < cache_max).all()
    return flag, bs_caches


def check_min_rate_constraint(precode_matrices, gains, rate_min, actions):
    """ 【Constraint1】校验是否满足最低传输速率约束

    :param precode_matrices: 预编码矩阵，维度：【基站数，用户数，天线数】
    :param gains: 信道增益，维度：【基站数，用户数，天线数】
    :param actions: 动作，用户选择的基站id，维度：【用户数】
    :param rate_min: 各用户的最低传输速率，维度：【用户数】
    :return: 校验标志、各用户的极限传输速率（维度：【用户数】）
    """

    limit_rates = channel.get_limit_transmit_rate(precode_matrices, gains, actions)
    flag = (limit_rates >= rate_min).all()
    return flag, limit_rates


def check_max_power_constraint(precode_matrices, power_max, actions):
    """ 【Constraint2】校验是否满足最大发射功率约束

    :param precode_matrices: 预编码矩阵，维度：【基站数，用户数，天线数】
    :param power_max: 基站最大发射功率，维度，【基站数】
    :param actions: 动作，选择的基站id，【标量】
    :return: 校验标志
    """
    powers = channel.get_transmit_powers(precode_matrices, actions)
    flag = (powers <= power_max).all()
    return flag


def set_seed(seed):
    np.random.seed(seed)
