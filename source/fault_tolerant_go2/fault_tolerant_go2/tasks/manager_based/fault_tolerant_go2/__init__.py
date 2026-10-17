"""Register the fault-tolerant Go2 tasks."""

import gymnasium as gym

from . import agents

gym.register(
    id="FT-Go2-Flat-Locked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fault_tolerant_go2_env_cfg:FTGo2FlatLockedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FTGo2FlatLockedPPORunnerCfg",
    },
)

gym.register(
    id="FT-Go2-Flat-Locked-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fault_tolerant_go2_env_cfg:FTGo2FlatLockedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FTGo2FlatLockedPPORunnerCfg",
    },
)
