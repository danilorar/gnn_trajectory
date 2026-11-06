"""Phase D prep: MetaDrive + ScenarioNet smoke tests (CPU only). Run with the .venv_sim interpreter.

    .venv_sim/bin/python scripts/sim_check.py metadrive   # reset + a few steps, no rendering
    .venv_sim/bin/python scripts/sim_check.py convert     # 5 nuScenes v1.0-mini scenes -> sim/scenarios/nuscenes_mini5
    .venv_sim/bin/python scripts/sim_check.py replay      # replay one scene, top-down PNG + GIF -> outputs/sim/

nuScenes mini lives in data/nuscenes_mini (v1.0-mini tables only; maps -> data/maps).
"""

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = PROJECT_ROOT / "data/nuscenes_mini"
SCENARIO_DIR = PROJECT_ROOT / "sim/scenarios/nuscenes_mini5"
OUT_DIR = PROJECT_ROOT / "outputs/sim"


def check_metadrive(steps):
    """MetaDrive's basic loop: make an env, reset, take a few random steps, no rendering."""
    from metadrive.envs.metadrive_env import MetaDriveEnv

    env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, start_seed=0, log_level=50))
    try:
        obs, _ = env.reset()
        print(f"reset ok | observation {obs.shape}, action space {env.action_space}")
        total = 0.0
        for _ in range(steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            if terminated or truncated:
                env.reset()
        print(f"{steps} steps ok | return {total:.2f} | ego speed {env.agent.speed_km_h:.1f} km/h")
    finally:
        env.close()


def convert(n_scenes):
    """ScenarioNet's nuScenes converter (the calls of scenarionet/convert_nuscenes.py, 1 worker, first n scenes)."""
    from nuscenes import NuScenes
    from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario
    from scenarionet.converter.utils import write_to_directory

    nusc = NuScenes(version="v1.0-mini", dataroot=str(MINI_ROOT), verbose=False)
    scenes = nusc.scene[:n_scenes]
    print(f"converting {[s['name'] for s in scenes]}")
    write_to_directory(
        convert_func=convert_nuscenes_scenario, scenarios=scenes, output_path=str(SCENARIO_DIR),
        dataset_version="v1.0-mini", dataset_name="nuscenes", overwrite=True, num_workers=1,
        nuscenes=[nusc], past=[2], future=[6], prediction=[False], map_radius=[500],
    )


def replay(index, max_steps):
    """Replay one converted scene (logged ego + traffic) and save top-down frames."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame
    import numpy as np
    from PIL import Image
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.scenario.utils import get_number_of_scenarios

    n = get_number_of_scenarios(str(SCENARIO_DIR))
    env = ScenarioEnv(dict(use_render=False, agent_policy=ReplayEgoCarPolicy, num_scenarios=n,
                           data_directory=str(SCENARIO_DIR), horizon=max_steps, log_level=50))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        env.reset(seed=index)
        frames = []
        for t in range(max_steps):
            _, _, terminated, truncated, info = env.step([0, 0])
            frame = env.render(mode="topdown", window=False, screen_size=(800, 800), film_size=(3000, 3000),
                               scaling=4, draw_target_vehicle_trajectory=True)
            if t % 5 == 0:
                frames.append(np.swapaxes(np.asarray(frame), 0, 1))   # pygame (W, H, 3) -> image (H, W, 3)
            if terminated or truncated:
                break
        print(f"scenario {index}: {t + 1} steps replayed ({(t + 1) * 0.1:.1f} s), {n} scenarios in the database, "
              f"route completion {info.get('route_completion', float('nan')):.2f}")
        png, gif = OUT_DIR / f"replay_scenario{index}.png", OUT_DIR / f"replay_scenario{index}.gif"
        Image.fromarray(frames[len(frames) // 2]).save(png)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"saved {png} and {gif}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("metadrive").add_argument("--steps", type=int, default=20)
    sub.add_parser("convert").add_argument("--scenes", type=int, default=5)
    p = sub.add_parser("replay")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    if args.step == "metadrive":
        check_metadrive(args.steps)
    elif args.step == "convert":
        convert(args.scenes)
    else:
        replay(args.index, args.max_steps)
