import numpy as np


class EnvConfig:
    def __init__(self):
        self.bs_num = 2  # 基站个数
        self.ue_num = 5  # 用户个数
        self.p_max = 50  # 基站最大发射功率
        self.irs_elements_num = 20  # IRS反射元件个数
        self.antenna_num = 5  # 基站天线个数
        self.reflect_phrase_max = 2 * np.pi  # 反射元件的最大相移
        self.r_min = 1  # 最低速率要求
        self.bw = 400  # 带宽




