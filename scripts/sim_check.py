"""Phase D prep: MetaDrive + ScenarioNet smoke tests (CPU only). Run with the .venv_sim interpreter.

    .venv_sim/bin/python scripts/sim_check.py metadrive   # reset + a few steps, no rendering
    .venv_sim/bin/python scripts/sim_check.py convert     # 5 nuScenes v1.0-mini scenes -> sim/scenarios/nuscenes_mini5
    .venv_sim/bin/python scripts/sim_check.py replay      # replay one scene, top-down PNG + GIF -> outputs/sim/
    .venv_sim/bin/python scripts/sim_check.py drive       # ego on a constant command, others replay the log
    .venv_sim/bin/python scripts/sim_check.py match       # adapter (sim -> v2 tensors) vs the processed dataset
    .venv_sim/bin/python scripts/sim_check.py follow      # plan generator + tracker on the logged route, no predictor

nuScenes mini lives in data/nuscenes_mini (v1.0-mini tables only; maps -> data/maps).
"""

import argparse
import os
import sys
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


def match(scene_name, checkpoint, sample_limit):
    """Adapter match test on one mini scene that is also in the processed v2 dataset.

    For every processed sample of the scene: rebuild it from the ScenarioNet scenario with sim/adapter.py
    (history buffer at 10 Hz, keyframe = every 5th step) and compare agents, ego, lane segments, edges, and
    the model's predictions on both inputs. Then repeat one sample through a live MetaDrive replay
    (snapshot_from_env), whose map frame is shifted, to check the result does not depend on the sim frame."""
    import json
    import pickle
    import numpy as np
    import torch
    sys.path.insert(0, str(PROJECT_ROOT))
    from sim.adapter import (HistoryBuffer, add_connector_polygons, build_sample, snapshot_from_env,
                             snapshot_from_scenario, to_model_inputs)
    from utils.dataset_v2 import TrajectoryDatasetV2
    from utils.models_v2 import V2Predictor

    scenes = {s["name"]: s for s in json.loads((MINI_ROOT / "v1.0-mini/scene.json").read_text())}
    samples = {s["token"]: s for s in json.loads((MINI_ROOT / "v1.0-mini/sample.json").read_text())}
    keyframe, tok = {}, scenes[scene_name]["first_sample_token"]
    while tok:
        keyframe[tok] = len(keyframe)
        tok = samples[tok]["next"]
    scen_file = next((SCENARIO_DIR).rglob(f"*{scene_name}.pkl"))
    scenario = pickle.loads(scen_file.read_bytes())

    split = None
    for name in ("train", "val", "test"):
        data = pickle.loads((PROJECT_ROOT / f"data/processed_v2/nuscenes_{name}_prediction_samples_map.pkl").read_bytes())
        idx = [i for i, s in enumerate(data) if s["scene_token"] == scenes[scene_name]["token"]]
        if idx:
            split = name
            break
    print(f"{scene_name}: {len(idx)} processed samples in the {split} split, {len(keyframe)} keyframes, "
          f"scenario {scenario['length']} steps")
    idx = idx[:sample_limit] if sample_limit else idx

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest = json.loads(Path(checkpoint).with_suffix(".json").read_text())
    model = V2Predictor(state["decoder.anchors"].numpy(), predict_agents=manifest.get("predict_agents", 1),
                        ego_plan=manifest.get("ego_plan", False), target_split=manifest.get("target_split", False),
                        plan_queries=manifest.get("plan_queries", False)).eval()
    model.load_state_dict(state)
    ds = TrajectoryDatasetV2.__new__(TrajectoryDatasetV2)
    ds.samples, ds.max_agents, ds.max_lanes, ds.max_points = data, 64, 192, 21

    def predict(item):
        with torch.no_grad():
            traj, logits = model({k: v[None] for k, v in item.items()}, plan=True)
        return traj[0].numpy(), logits[0].softmax(-1).numpy()

    def compare(ref, new, ref_item, new_item):
        out = {}
        ids_ref, ids_new = list(ref["agent_instance_tokens"]), list(new["agent_instance_tokens"])
        out["agents"] = (len(ids_ref), len(ids_new), len(set(ids_ref) ^ set(ids_new)))
        common = [t for t in ids_ref if t in ids_new]
        a = [ids_ref.index(t) for t in common]
        b = [ids_new.index(t) for t in common]
        vr, vn = ref["agent_past_valid"][a], new["agent_past_valid"][b]
        out["valid_mismatch"] = int((vr != vn).sum())
        both = vr & vn
        out["past_err"] = float(np.abs(ref["agent_past"][a][both] - new["agent_past"][b][both]).max()) if both.any() else 0.0
        out["size_err"] = float(np.abs(ref["agent_size"][a] - new["agent_size"][b]).max())
        out["ego_past_err"] = float(np.abs(ref["ego_past"] - new["ego_past"]).max())
        out["ego_future_err"] = float(np.abs(ref["ego_future"] - new["ego_future"]).max())
        # lane segments as sets (the model does not depend on token order): same length, points within 1 cm
        pairs = {}
        for a_i, pa in enumerate(ref["lane_points"]):
            for b_i, pb in enumerate(new["lane_points"]):
                if b_i not in pairs.values() and len(pa) == len(pb) and np.abs(pa - pb).max() < 1e-2:
                    pairs[a_i] = b_i
                    break
        out["lanes"] = (len(ref["lane_points"]), len(new["lane_points"]), len(pairs))
        out["lane_err"] = max((float(np.abs(ref["lane_points"][a_] - new["lane_points"][b_]).max())
                               for a_, b_ in pairs.items()), default=0.0)
        out["conn_mismatch"] = int(sum(ref["lane_is_connector"][a_] != new["lane_is_connector"][b_]
                                       for a_, b_ in pairs.items()))
        inv = {b_: a_ for a_, b_ in pairs.items()}
        er = {(int(i), int(j)) for i, j in ref["lane_edges"] if i in pairs and j in pairs}
        en = {(inv[i], inv[j]) for i, j in new["lane_edges"] if i in inv and j in inv}
        out["edges"] = (len(er), len(en), len(er ^ en))
        tr, pr = predict(ref_item)
        tn, pn = predict(new_item)
        out["pred_err"] = float(np.abs(tr[0] - tn[0]).max())           # target, all modes, metres
        out["prob_err"] = float(np.abs(pr[0] - pn[0]).max())
        return out

    def ego_plan(k):
        sdc = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]
        steps = [STRIDE_ * (k + j) for j in range(1, 13)]
        plan = np.array([[*sdc["position"][s][:2], sdc["heading"][s]] if s < len(sdc["valid"]) else [0, 0, 0]
                         for s in steps])
        return plan, np.array([s < len(sdc["valid"]) and bool(sdc["valid"][s]) for s in steps])

    STRIDE_ = 5
    features = add_connector_polygons(scenario["map_features"], scenario["metadata"]["map"])
    rows = []
    for i in idx:
        ref = data[i]
        k = keyframe[ref["sample_token"]]
        buf = HistoryBuffer()
        for step in range(max(0, STRIDE_ * k - 20), STRIDE_ * k + 1):
            buf.push(snapshot_from_scenario(scenario, step))
        plan, plan_valid = ego_plan(k)
        new = build_sample(buf, features, ref["target_instance_token"], plan, plan_valid)
        rows.append((i, k, compare(ref, new, ds[i], to_model_inputs(new))))

    print(" sample  key | agents ref/new/diff | valid mism | past err | size err | ego past/fut err |"
          " lanes ref/new/shared | lane err | edges ref/new/diff | pred err [m] | prob err")
    for i, k, r in rows:
        print(f" {i:6d} {k:4d} | {r['agents'][0]:3d} {r['agents'][1]:3d} {r['agents'][2]:3d}      | {r['valid_mismatch']:10d} |"
              f" {r['past_err']:.1e}  | {r['size_err']:.1e}  | {r['ego_past_err']:.1e} {r['ego_future_err']:.1e}  |"
              f" {r['lanes'][0]:4d} {r['lanes'][1]:4d} {r['lanes'][2]:4d}       | {r['lane_err']:.1e}  |"
              f" {r['edges'][0]:4d} {r['edges'][1]:4d} {r['edges'][2]:3d}    | {r['pred_err']:.2e}    | {r['prob_err']:.1e}")
    worst = max(r["pred_err"] for _, _, r in rows)
    print(f"max target prediction difference over {len(rows)} samples: {worst:.2e} m | connector-flag mismatches "
          f"{sum(r['conn_mismatch'] for _, _, r in rows)} | agent-set differences {sum(r['agents'][2] for _, _, r in rows)} | "
          f"lane segments unmatched {sum(r['lanes'][0] + r['lanes'][1] - 2 * r['lanes'][2] for _, _, r in rows)} | "
          f"edge differences {sum(r['edges'][2] for _, _, r in rows)}")

    # Same sample through a live MetaDrive replay (shifted sim frame, snapshot_from_env)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.scenario.utils import get_number_of_scenarios
    n = get_number_of_scenarios(str(SCENARIO_DIR))
    i, k, _ = rows[len(rows) // 2]
    env = ScenarioEnv(dict(use_render=False, agent_policy=ReplayEgoCarPolicy, num_scenarios=n,
                           data_directory=str(SCENARIO_DIR), log_level=50))
    try:
        for index in range(n):
            env.reset(seed=index)
            if env.engine.data_manager.current_scenario["metadata"]["scenario_id"] == scene_name:
                break
        sim_scen = env.engine.data_manager.current_scenario
        buf = HistoryBuffer()
        buf.push(snapshot_from_env(env))
        for _ in range(STRIDE_ * k):
            env.step([0, 0])
            buf.push(snapshot_from_env(env))
        sdc = sim_scen["tracks"][sim_scen["metadata"]["sdc_id"]]["state"]
        steps = [STRIDE_ * (k + j) for j in range(1, 13)]
        plan = np.array([[*sdc["position"][s][:2], sdc["heading"][s]] for s in steps])
        live_features = add_connector_polygons(sim_scen["map_features"], sim_scen["metadata"]["map"])
        live = build_sample(buf, live_features, data[i]["target_instance_token"], plan)
        r = compare(data[i], live, ds[i], to_model_inputs(live))
        shift = np.asarray(scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]["position"][0][:2]) - \
            np.asarray(sdc["position"][0][:2])
        print(f"live MetaDrive replay, sample {i} (keyframe {k}, step {env.episode_step}), sim frame shifted by "
              f"{shift.round(1)} m: agents {r['agents']}, past err {r['past_err']:.1e}, ego past err "
              f"{r['ego_past_err']:.1e}, lanes {r['lanes']}, edges diff {r['edges'][2]}, "
              f"pred err {r['pred_err']:.2e} m")
    finally:
        env.close()


def follow(index, replan_steps=5):
    """Plan generator + tracker without a predictor: every 0.5 s generate the 35 candidate plans (checked and
    timed), but always execute "offset 0, speed = logged speed now"; the tracker follows it for 5 sim steps.
    Others replay the log. Reports the lateral error to the reference route, off-road / crash flags, and saves
    outputs/sim/follow_test.gif and follow_test.png (driven path vs route, lateral error and speed)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import time
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(PROJECT_ROOT))
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.scenario.utils import get_number_of_scenarios
    from sim.planner import Tracker, generate_plans, logged_speed, plans_to_ego_future, reference_from_scenario

    n = get_number_of_scenarios(str(SCENARIO_DIR))
    env = ScenarioEnv(dict(use_render=False, num_scenarios=n, data_directory=str(SCENARIO_DIR), horizon=10_000,
                           log_level=50, crash_vehicle_done=False, crash_object_done=False, crash_human_done=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        env.reset(seed=index)
        scenario = env.engine.data_manager.current_scenario
        ref = reference_from_scenario(scenario)
        steps = scenario["length"] - 1
        tracker, frames, log = Tracker(), [], []
        gen_ms, plan = [], None
        flags = {"out_of_road": 0, "crash": 0}
        for step in range(steps):
            a = env.agent
            x, y, yaw, v = float(a.position[0]), float(a.position[1]), float(a.heading_theta), float(a.speed)
            if step % replan_steps == 0:
                t0 = time.perf_counter()
                grid, grid_dense, meta = generate_plans(ref, (x, y, yaw, v))
                gen_ms.append(1000 * (time.perf_counter() - t0))
                if step == 0:
                    ego_future = plans_to_ego_future(grid, np.array([x, y]), yaw)
                    print(f"candidate plans: {len(meta)} = {len(set(m[0] for m in meta))} speeds x "
                          f"{len(set(m[1] for m in meta))} offsets | sparse {grid.shape}, as ego_future "
                          f"{ego_future.shape}, dense {grid_dense.shape}")
                v_log = logged_speed(scenario, step)
                _, chosen, _ = generate_plans(ref, (x, y, yaw, v), speeds=[v_log], offsets=[0.0])
                plan, plan_t, tracker = chosen[0], 0.0, tracker
            act = tracker.action(x, y, yaw, v, plan, plan_t)
            _, _, _, _, info = env.step(act)
            plan_t += 0.1
            s_, d_ = ref.project(np.asarray(env.agent.position))
            off_road, crash = bool(info.get("out_of_road")), bool(info.get("crash"))
            flags["out_of_road"] += off_road
            flags["crash"] += crash
            log.append((step + 1, *np.asarray(env.agent.position)[:2], env.agent.speed, logged_speed(scenario, step + 1),
                        s_, d_, off_road, crash))
            if step % 2 == 0:
                frame = env.render(mode="topdown", window=False, screen_size=(800, 800), film_size=(3000, 3000),
                                   scaling=4, draw_target_vehicle_trajectory=True)
                frames.append(np.swapaxes(np.asarray(frame), 0, 1))
        L = np.asarray(log, float)
        lat = np.abs(L[:, 6])
        print(f"{steps} steps ({steps / 10:.1f} s) | lateral error to route: max {lat.max():.2f} m, mean {lat.mean():.2f} m, "
              f"p95 {np.percentile(lat, 95):.2f} m | off-road steps {flags['out_of_road']} | crash steps {flags['crash']}")
        print(f"route progress {L[-1, 5]:.1f} m of {ref.length:.1f} m logged | speed error vs log: mean "
              f"{np.mean(np.abs(L[:, 3] - L[:, 4])):.2f} m/s | plan generation (35 plans): median {np.median(gen_ms):.1f} ms")
        for t in range(10, steps + 1, 20):
            r = L[t - 1]
            print(f"  t {r[0] / 10:5.1f} s | speed {r[3]:5.2f} (log {r[4]:5.2f}) m/s | lateral {r[6]:+.2f} m | "
                  f"off-road {bool(r[7])} | crash {bool(r[8])}")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(OUT_DIR / "follow_test.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        route = ref.xy[ref.s <= ref.length]
        ax[0].plot(route[:, 0], route[:, 1], color="0.6", lw=6, alpha=0.5, label="reference (logged route)")
        ax[0].plot(L[:, 1], L[:, 2], color="tab:green", lw=2, label="driven")
        ax[0].set_aspect("equal")
        ax[0].legend(loc="lower left", fontsize=8)
        ax[0].set_title("path")
        ax[1].plot(L[:, 0] / 10, L[:, 6], color="tab:blue")
        ax[1].axhline(0, color="0.5", lw=0.8)
        ax[1].set(xlabel="t [s]", ylabel="lateral error [m]", title=f"lateral error (max {lat.max():.2f} m)")
        ax[2].plot(L[:, 0] / 10, L[:, 4], color="0.5", label="logged")
        ax[2].plot(L[:, 0] / 10, L[:, 3], color="tab:green", label="driven")
        ax[2].set(xlabel="t [s]", ylabel="speed [m/s]", title="speed")
        ax[2].legend()
        for a_ in ax:
            a_.grid(alpha=0.3)
        fig.suptitle(f"{scenario['metadata']['scenario_id']}: offset 0, speed = logged speed, no predictor")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "follow_test.png", dpi=110)
        print(f"saved {OUT_DIR / 'follow_test.gif'} and {OUT_DIR / 'follow_test.png'}")
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
    p = sub.add_parser("follow")
    p.add_argument("--index", type=int, default=0, help="scenario index (0 = scene-0061)")
    p = sub.add_parser("match")
    p.add_argument("--scene", default="scene-0061")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/v2_d/v2_seed42_best.pth"))
    p.add_argument("--limit", type=int, default=0, help="only the first N samples (0 = all)")
    args = parser.parse_args()
    if args.step == "metadrive":
        check_metadrive(args.steps)
    elif args.step == "convert":
        convert(args.scenes)
    elif args.step == "follow":
        follow(args.index)
    elif args.step == "match":
        match(args.scene, args.checkpoint, args.limit)
    elif args.step == "drive":
        drive(args.index, args.seconds, args.throttle, args.steering)
    else:
        replay(args.index, args.max_steps)
