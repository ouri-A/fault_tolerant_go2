"""Fault-tolerant Go2: the stock Isaac Lab Go2 flat task plus one locked joint per episode."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from . import mdp


@configclass
class LockedJointEventCfg(EventCfg):
    """Stock locomotion events plus the joint lock.

    Declared last so it runs after the stock ``reset_robot_joints`` term.
    """

    lock_joint = EventTerm(
        func=mdp.lock_one_joint,
        mode="reset",
        params={"joint_names": None},  # None = any of the 12 joints, uniformly
    )


@configclass
class FTGo2FlatLockedEnvCfg(UnitreeGo2FlatEnvCfg):
    """Training config: a random joint is locked at the start of every episode."""

    events: LockedJointEventCfg = LockedJointEventCfg()


@configclass
class FTGo2FlatLockedEnvCfg_PLAY(FTGo2FlatLockedEnvCfg):
    """Viewing config: front-left knee locked on every robot, no randomisation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # same viewing setup as the stock Go2 flat Play config
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # fixed, easy-to-see failure
        self.events.lock_joint.params["joint_names"] = ["FL_calf_joint"]

        # camera follows robot 0 (used by --video), looking at the front-left side
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.5, 1.5, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.0)
