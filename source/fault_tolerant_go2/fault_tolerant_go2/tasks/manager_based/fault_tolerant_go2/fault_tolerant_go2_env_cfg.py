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
        # joint_names None = any of the 12 joints, uniformly.
        # healthy_fraction = share of episodes with no lock at all.
        params={"joint_names": None, "healthy_fraction": 0.2},
    )


@configclass
class FTGo2FlatLockedEnvCfg(UnitreeGo2FlatEnvCfg):
    """Training config: a random joint is locked at the start of 80% of episodes; 20% run healthy."""

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
        self.events.lock_joint.params["healthy_fraction"] = 0.0

        # robot spawns facing +x and walks straight ahead at 0.6 m/s, so the camera angle is predictable
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # camera follows robot 0 from its left side (+y), so the locked front-left leg is nearest.
        # Pulled back and raised so the velocity arrows (drawn 0.5 m above the base, pointing
        # forward ~1 m at 0.6 m/s) stay in frame; aimed between the robot and the arrows.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.4, 2.4, 0.9)
        self.viewer.lookat = (0.4, 0.0, 0.2)
