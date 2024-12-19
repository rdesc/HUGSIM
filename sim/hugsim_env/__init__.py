from gymnasium.envs.registration import register


register(
     id="hugsim_env/HUGSim-v0",
     entry_point="hugsim_env.envs:HUGSimEnv",
     max_episode_steps=400,
)