"""Actuator-failure events for the fault-tolerant Go2 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["lock_one_joint"]

# Half-width of the locked joint's range, in radians (~0.06 deg). Keeps lower < upper for PhysX
# and keeps the standing angle strictly inside the range, so nothing gets clamped.
LOCK_HALF_WIDTH = 1.0e-3


def lock_one_joint(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_names: list[str] | None = None,
    healthy_fraction: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Lock exactly one joint per reset environment at its standing angle.

    Runs in "reset" mode, after the stock joint reset. For each environment being reset it:
      1. restores healthy joint limits (undoes the previous episode's lock),
      2. picks one joint, uniformly from ``joint_names`` (or from all joints if None),
         and with probability ``healthy_fraction`` marks the episode healthy (no lock),
      3. puts the robot back in its standing pose,
      4. narrows the picked joint's limits to [theta - eps, theta + eps] so PhysX holds it
         rigid (skipped for healthy episodes),
      5. restores the nominal default joint positions.

    Step 5 matters: ``write_joint_position_limit_to_sim`` clamps ``default_joint_pos`` into the
    new limits, and the ``joint_pos`` observation is measured relative to that default. Left
    alone, it would tell the policy which joint failed.

    The locked joint index per env (-1 for a healthy episode) is stored in
    ``env.locked_joint_ids`` for logging/evaluation only. It is never put into the observation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # one-time buffers
    if not hasattr(env, "locked_joint_ids"):
        env.locked_joint_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        env._nominal_default_joint_pos = asset.data.default_joint_pos.clone()
    nominal_default = env._nominal_default_joint_pos
    healthy_limits = asset.data.default_joint_pos_limits

    # 1. healthy limits back on
    asset.write_joint_position_limit_to_sim(
        healthy_limits[env_ids].clone(), env_ids=env_ids, warn_limit_violation=False
    )
    asset.data.default_joint_pos[env_ids] = nominal_default[env_ids]

    # 2. choose one joint per env
    if joint_names is None:
        candidates = torch.arange(asset.num_joints, device=env.device)
    else:
        ids, _ = asset.find_joints(joint_names, preserve_order=True)
        candidates = torch.tensor(ids, dtype=torch.long, device=env.device)
    pick = torch.randint(len(candidates), (len(env_ids),), device=env.device)
    joint_idx = candidates[pick]
    healthy = torch.rand(len(env_ids), device=env.device) < healthy_fraction

    # 3. standing pose (the stock reset ran under the old lock and may have clamped it)
    joint_pos = nominal_default[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 4. lock (healthy episodes keep the limits restored in step 1)
    rows = torch.nonzero(~healthy).squeeze(-1)
    cols = joint_idx[rows]
    theta = joint_pos[rows, cols]
    limits = healthy_limits[env_ids].clone()
    limits[rows, cols, 0] = theta - LOCK_HALF_WIDTH
    limits[rows, cols, 1] = theta + LOCK_HALF_WIDTH
    asset.write_joint_position_limit_to_sim(limits, env_ids=env_ids, warn_limit_violation=False)

    # 5. no leak through the default-relative observation
    asset.data.default_joint_pos[env_ids] = nominal_default[env_ids]

    env.locked_joint_ids[env_ids] = torch.where(healthy, -1, joint_idx)
