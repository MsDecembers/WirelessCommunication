#!/usr/bin/env python
# coding=utf-8

import collections
import numpy as np
import torch

from common import utils
from drl import dqn
from envs import myenv


def env_agent_config():
    bs_num = utils.get_parameter('model', 'bs_num')
    ue_num = utils.get_parameter('model', 'ue_num')
    antenna_num = utils.get_parameter('model', 'antenna_num')
    content_num = utils.get_parameter('model', 'content_num')
    seed = utils.get_parameter('base', 'seed')

    env = myenv.IRS_COMP_MISO()

    ue_state_dim = bs_num * antenna_num
    ue_action_dim = bs_num
    content_state_dim = bs_num * antenna_num * ue_num
    content_action_dim = bs_num + 1

    ue_agents = [dqn.DQN('ue' + str(_), ue_state_dim, ue_action_dim) for _ in range(ue_num)]
    content_agents = [dqn.DQN('content' + str(_), content_state_dim, content_action_dim) for _ in range(content_num)]

    # 设置随机种子
    if seed != 0:
        torch.manual_seed(seed)
        myenv.set_seed(seed)
        np.random.seed(seed)
    return env, ue_agents, content_agents


def train(result_path, model_path):
    device = utils.get_device()
    env_name = utils.get_parameter('base', 'env_name')
    algorithm_name = utils.get_parameter('base', 'algorithm_name')
    train_eps = utils.get_parameter('base', 'train_eps')
    target_update = utils.get_parameter('drl', 'target_update')

    env, ue_agents, content_agents = env_agent_config()

    print('开始训练!')
    print(f'环境：{env_name}, 算法：{algorithm_name}, 设备：{device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            actions = collections.defaultdict(list)
            for ue_id, sub_agent in enumerate(ue_agents):
                action = sub_agent.choose_action(state[:, ue_id, :].flatten())
                actions['ue_actions'].append(action)
            for content_id, sub_agent in enumerate(content_agents):
                action = sub_agent.choose_action(state.flatten())
                actions['content_actions'].append(action)
            next_state, reward, done = env.step(actions)  # 更新环境，返回transition
            for ue_id, sub_agent in enumerate(ue_agents):
                sub_agent.memory.push(state[:, ue_id, :].flatten(), actions['ue_actions'][ue_id], reward,
                                      next_state[:, ue_id, :].flatten(), done)
                sub_agent.update()
            for content_id, sub_agent in enumerate(content_agents):
                sub_agent.memory.push(state.flatten(), actions['content_actions'][content_id], reward,
                                      next_state.flatten(), done)
                sub_agent.update()
            state = next_state
            ep_reward += reward  # 累加奖励
            if done:
                break
        if (i_ep + 1) % target_update == 0:  # 智能体目标网络更新
            for sub_agent in ue_agents:
                sub_agent.target_net.load_state_dict(sub_agent.policy_net.state_dict())
            for sub_agent in content_agents:
                sub_agent.target_net.load_state_dict(sub_agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, train_eps, ep_reward))
    print('完成训练！')

    # 保存模型
    for sub_agent in ue_agents:
        sub_agent.save(model_path)
    for sub_agent in content_agents:
        sub_agent.save(model_path)
    utils.save_results(rewards, ma_rewards, result_path, tag='train')  # 保存结果
    utils.plot_rewards(rewards, ma_rewards, result_path, tag="train")  # 画出结果


def test(result_path, model_path):
    env, ue_agents, content_agents = env_agent_config()
    # 导入模型
    for sub_agent in ue_agents:
        sub_agent.load(model_path)
    for sub_agent in content_agents:
        sub_agent.load(model_path)
    env_name = utils.get_parameter('base', 'env_name')
    algorithm_name = utils.get_parameter('base', 'algorithm_name')
    device = utils.get_device()
    test_eps = utils.get_parameter('base', 'test_eps')
    print('开始测试!')
    print(f'环境：{env_name}, 算法：{algorithm_name}, 设备：{device}')
    # 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0
    utils.update_epsilon()
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            actions = collections.defaultdict(list)
            for ue_id, sub_agent in enumerate(ue_agents):
                action = sub_agent.choose_action(state[:, ue_id, :].flatten())
                actions['ue_actions'].append(action)
            for content_id, sub_agent in enumerate(content_agents):
                action = sub_agent.choose_action(state.flatten())
                actions['content_actions'].append(action)
            next_state, reward, done = env.step(actions)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    utils.save_results(rewards, ma_rewards, result_path, tag='test')  # 保存结果
    utils.plot_rewards(rewards, ma_rewards, result_path, tag="test")  # 画出结果


def execute():
    result_path, model_path = utils.get_save_paths()
    utils.make_dir(result_path, model_path)
    train(result_path, model_path)
    test(result_path, model_path)


if __name__ == "__main__":
    execute()
