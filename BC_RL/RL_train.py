import gym
import torch
from stable_baselines3 import PPO

# 1. 创建环境和模型
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

# 2. 加载预训练权重
#    注意路径和 device 要和你保存时一致
actor_dict  = torch.load("clone_actor.pt",  map_location=model.device)
critic_dict = torch.load("clone_critic.pt", map_location=model.device)

# 3. 指定子模块加载
#    以下两个模块名都来源于 `ActorCriticPolicy._build_mlp_extractor`
model.policy.mlp_extractor.policy_net.load_state_dict(actor_dict, strict=False)
model.policy.mlp_extractor.value_net.load_state_dict(critic_dict, strict=False)

# 4. 继续 PPO 微调
model.learn(total_timesteps=10000)
model.save("model.pt")
