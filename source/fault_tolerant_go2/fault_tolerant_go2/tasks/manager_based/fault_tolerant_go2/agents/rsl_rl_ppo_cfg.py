"""PPO settings: identical to the stock Go2 flat runner so all three policies train the same way."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class FTGo2FlatLockedPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "ft_go2_flat_locked"


@configclass
class FTGo2FlatLockedHistPPORunnerCfg(FTGo2FlatLockedPPORunnerCfg):
    """Policy 3: same PPO settings as policy 2; separate logs folder."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "ft_go2_flat_locked_hist30"
