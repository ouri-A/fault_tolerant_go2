# Training runs

| Run | Date | Policy | Failed joint | Seed | Iters | Checkpoint | Result | Notes |
|---|---|---|---|---|---|---|---|---|
| stock | 2026-08-31 | 1 (healthy-only, stock Isaac Lab Go2 flat) | none (trained) | default | 1500 | `~/isaaclab_ws/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-08-31_21-17-48/model_1499.pt` | FL_calf locked eval: mean 0.72 m / 12 m, median 1.08 m, 17% end behind start, 0% falls | `results/stock_FL_calf.json` |
| policy2_scratch | 2026-10-17 | 2 (failures, no history) | random, 20% healthy | 42 | 1000 | `logs/rsl_rl/ft_go2_flat_locked/2026-10-17_16-31-17_policy2_scratch/model_999.pt` | not converged; superseded by resume below | — |
| policy2_scratch_2000 | 2026-10-18 | 2 (failures, no history) | random, 20% healthy | 42 | 2000 (resumed from 999) | `logs/rsl_rl/ft_go2_flat_locked/2026-10-18_13-44-29_policy2_scratch_2000/model_1998.pt` | FL_calf locked eval: mean 12.10 m / 12 m, min 11.49 m, 0% falls | `results/policy2_FL_calf.json` |
