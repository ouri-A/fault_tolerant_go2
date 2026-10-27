"""Evaluate a checkpoint on a fixed failure condition: how far does the Go2 walk, and does it fall.

Runs N robots with the same joint locked (or no lock with --healthy), all given the same fixed
velocity command (--vx forward, --vy sideways, default 0.6 m/s straight ahead), facing +x, for
--duration seconds. No pushes, no observation noise, no learning. Each robot gets exactly one
attempt: once it falls, its result is frozen.

Per robot it records the distance covered along the commanded direction (metres), the sideways
drift off the commanded line (metres, + = left of it), whether it fell, and the time of the fall.
With --vy 0 the distance is the x displacement, the same number as before the --vy option existed.
Results go to results/<label>.json and a summary is printed.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Distance/fall evaluation of an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="FT-Go2-Flat-Locked-Play-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=256, help="Number of robots.")
parser.add_argument("--duration", type=float, default=20.0, help="Evaluation length in seconds.")
parser.add_argument("--joint", type=str, default="FL_calf_joint", help="Joint to lock.")
parser.add_argument("--healthy", action="store_true", default=False, help="No lock at all (overrides --joint).")
parser.add_argument("--label", type=str, required=True, help="Name for the results file, e.g. stock_FL_calf.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--vx", type=float, default=0.6, help="Commanded forward speed (m/s).")
parser.add_argument("--vy", type=float, default=0.0, help="Commanded sideways speed (m/s), + = left.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import fault_tolerant_go2.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint (full path to a model_*.pt) is required")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # failure condition
    if args_cli.healthy:
        env_cfg.events.lock_joint.params["healthy_fraction"] = 1.0
    else:
        env_cfg.events.lock_joint.params["joint_names"] = [args_cli.joint]
        env_cfg.events.lock_joint.params["healthy_fraction"] = 0.0

    # fixed velocity command; heading stays at 0 (robot faces +x), as set in the Play config
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)

    # keep the time-out out of the measurement window, so every termination is a fall
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.duration + 1.0)

    resume_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    dt = base_env.step_dt
    num_steps = int(round(args_cli.duration / dt))
    device = base_env.device

    obs = env.get_observations()

    # check the failure condition actually applied
    locked = base_env.locked_joint_ids
    if args_cli.healthy:
        assert bool((locked == -1).all()), "expected no locked joint"
        condition = "healthy"
    else:
        jid = robot.joint_names.index(args_cli.joint)
        assert bool((locked == jid).all()), f"expected {args_cli.joint} (id {jid}) locked on every robot"
        condition = args_cli.joint

    start_xy = robot.data.root_pos_w[:, :2].clone()
    last_xy = start_xy.clone()
    alive = torch.ones(base_env.num_envs, dtype=torch.bool, device=device)
    fall_step = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=device)

    with torch.inference_mode():
        for step in range(num_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            dones = dones.bool()
            # robots that terminate this step have already been reset by the env: keep their last pose
            newly_fallen = alive & dones
            fall_step[newly_fallen] = step + 1
            alive &= ~dones
            last_xy[alive] = robot.data.root_pos_w[alive, :2]

    # unit vector along the command, and its left-hand normal
    speed = (args_cli.vx**2 + args_cli.vy**2) ** 0.5
    if speed == 0:
        raise ValueError("--vx and --vy are both 0: no commanded direction to measure along")
    along_dir = torch.tensor([args_cli.vx / speed, args_cli.vy / speed], device=device)
    left_dir = torch.tensor([-along_dir[1], along_dir[0]], device=device)
    disp = last_xy - start_xy
    distance = (disp @ along_dir).cpu()
    lateral = (disp @ left_dir).cpu()
    disp = disp.cpu()
    fell = (fall_step >= 0).cpu()
    fall_time = (fall_step.float() * dt).cpu()

    summary = {
        "label": args_cli.label,
        "checkpoint": resume_path,
        "condition": condition,
        "num_robots": base_env.num_envs,
        "duration_s": args_cli.duration,
        "command_vx_mps": args_cli.vx,
        "command_vy_mps": args_cli.vy,
        "command_mps": speed,
        "perfect_distance_m": speed * args_cli.duration,
        "seed": args_cli.seed,
        "mean_distance_m": distance.mean().item(),
        "std_distance_m": distance.std().item(),
        "median_distance_m": distance.median().item(),
        "mean_lateral_m": lateral.mean().item(),
        "mean_abs_lateral_m": lateral.abs().mean().item(),
        "fall_rate": fell.float().mean().item(),
        "mean_time_to_fall_s": fall_time[fell].mean().item() if fell.any() else None,
    }

    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"{k:>22}: {v:.3f}" if isinstance(v, float) else f"{k:>22}: {v}")
    print("=" * 60 + "\n")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"{args_cli.label}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "summary": summary,
                "distance_m": distance.tolist(),
                "lateral_m": lateral.tolist(),
                "dx_m": disp[:, 0].tolist(),
                "dy_m": disp[:, 1].tolist(),
                "fell": fell.tolist(),
                "fall_time_s": fall_time.tolist(),
            },
            f,
            indent=1,
        )
    print(f"[INFO] Results written to {os.path.abspath(out)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
