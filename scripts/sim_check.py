"""Phase D prep: MetaDrive + ScenarioNet smoke tests (CPU only). Run with the .venv_sim interpreter.

    .venv_sim/bin/python scripts/sim_check.py metadrive   # reset + a few steps, no rendering
    .venv_sim/bin/python scripts/sim_check.py convert     # 5 nuScenes v1.0-mini scenes -> sim/scenarios/nuscenes_mini5
    .venv_sim/bin/python scripts/sim_check.py replay      # replay one scene, top-down PNG + GIF -> outputs/sim/
    .venv_sim/bin/python scripts/sim_check.py drive       # ego on a constant command, others replay the log

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


def drive(index, seconds, throttle, steering):
    """The ego under our control: a constant [steering, throttle] command while all other agents replay the log.

    This is where the planner plugs in later: replace the constant action with the planner's output.
    Crashes do not end the episode; off-road does in ScenarioEnv, but the loop ignores `terminated` and keeps
    stepping, so the full `seconds` are always driven and logged.
    Each second prints the ego position, speed, the logged ego position at the same time, and collision /
    off-road flags. Saves outputs/sim/drive_test.png (last frame) and drive_test.gif.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    from PIL import Image
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.scenario.utils import get_number_of_scenarios

    n = get_number_of_scenarios(str(SCENARIO_DIR))
    env = ScenarioEnv(dict(use_render=False, num_scenarios=n, data_directory=str(SCENARIO_DIR),
                           horizon=10_000, log_level=50, reactive_traffic=False,
                           crash_vehicle_done=False, crash_object_done=False,
                           crash_human_done=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        env.reset(seed=index)
        scenario = env.engine.data_manager.current_scenario
        sdc = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]
        dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]   # seconds per env.step
        steps = int(round(seconds / dt))
        start = np.asarray(env.agent.position)
        print(f"scene {scenario['metadata'].get('scenario_id', index)} | dt {dt:.2f} s | action "
              f"[steering {steering}, throttle {throttle}] for {seconds:.0f} s | start {start.round(1)}")
        print(" t [s] | ego x, y [m]      | speed [m/s] | logged ego x, y | offset [m] | crash | off-road")
        frames, crashed, off_road = [], False, False
        for t in range(1, steps + 1):
            _, _, _, _, info = env.step([steering, throttle])
            crashed |= bool(info.get("crash"))
            off_road |= bool(info.get("out_of_road"))
            frame = env.render(mode="topdown", window=False, screen_size=(800, 800), film_size=(3000, 3000),
                               scaling=4, draw_target_vehicle_trajectory=True)
            if t % 2 == 0:
                frames.append(np.swapaxes(np.asarray(frame), 0, 1))
            if t % int(round(1 / dt)) == 0:
                pos = np.asarray(env.agent.position)
                k = min(t, len(sdc["position"]) - 1)
                logged = np.asarray(sdc["position"][k][:2]) if sdc["valid"][k] else np.full(2, np.nan)
                print(f" {t * dt:5.1f} | {pos[0]:7.1f}, {pos[1]:7.1f} | {env.agent.speed:11.2f} | "
                      f"{logged[0]:7.1f}, {logged[1]:7.1f} | {np.linalg.norm(pos - logged):10.1f} | "
                      f"{str(bool(info.get('crash'))):5s} | {bool(info.get('out_of_road'))}")
        print(f"driven {np.linalg.norm(np.asarray(env.agent.position) - start):.1f} m | any crash {crashed} | "
              f"any off-road {off_road}")
        Image.fromarray(frames[-1]).save(OUT_DIR / "drive_test.png")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(OUT_DIR / "drive_test.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"saved {OUT_DIR / 'drive_test.png'} and {OUT_DIR / 'drive_test.gif'}")
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
    p = sub.add_parser("drive")
    p.add_argument("--index", type=int, default=0, help="scenario index (0 = scene-0061)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--throttle", type=float, default=0.2)
    p.add_argument("--steering", type=float, default=0.0)
    args = parser.parse_args()
    if args.step == "metadrive":
        check_metadrive(args.steps)
    elif args.step == "convert":
        convert(args.scenes)
    elif args.step == "drive":
        drive(args.index, args.seconds, args.throttle, args.steering)
    else:
        replay(args.index, args.max_steps)
