"""Phase D prep: MetaDrive + ScenarioNet smoke tests (CPU only). Run with the .venv_sim interpreter.

    .venv_sim/bin/python scripts/sim_check.py metadrive   # reset + a few steps, no rendering
    .venv_sim/bin/python scripts/sim_check.py convert     # 5 nuScenes v1.0-mini scenes -> sim/scenarios/nuscenes_mini
    .venv_sim/bin/python scripts/sim_check.py replay      # replay one scene, top-down PNG + GIF -> outputs/sim/
    .venv_sim/bin/python scripts/sim_check.py drive       # ego on a constant command, others replay the log
    .venv_sim/bin/python scripts/sim_check.py match       # adapter (sim -> v2 tensors) vs the processed dataset
    .venv_sim/bin/python scripts/sim_check.py follow      # plan generator + tracker on the logged route, no predictor
    .venv_sim/bin/python scripts/sim_check.py plan        # closed-loop planner (CV or world-model predictor) + expert

nuScenes mini lives in data/nuscenes_mini (v1.0-mini tables only; maps -> data/maps).
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = PROJECT_ROOT / "data/nuscenes_mini"
SCENARIO_DIR = PROJECT_ROOT / "sim/scenarios/nuscenes_mini"
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


def stream_table(root, table):
    import ijson
    with open(root / f"{table}.json", "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def subset_tables(names, out_root, source=PROJECT_ROOT / "data/v1.0-trainval"):
    """Stream-filter the trainval tables (ijson, low RAM) down to the scenes `names`; written as a
    v1.0-trainval table set under out_root (maps symlinked), which the devkit loads like the full one."""
    import json
    out = out_root / "v1.0-trainval"
    out.mkdir(parents=True, exist_ok=True)

    def save(table, rows):
        (out / f"{table}.json").write_text(json.dumps(rows))
        print(f"  {table}: {len(rows)}", flush=True)

    scenes = [r for r in stream_table(source, "scene") if r["name"] in set(names)]
    scene_tokens, log_tokens = {r["token"] for r in scenes}, {r["log_token"] for r in scenes}
    save("scene", scenes)
    samples = [r for r in stream_table(source, "sample") if r["scene_token"] in scene_tokens]
    sample_tokens = {r["token"] for r in samples}
    save("sample", samples)
    sd = [r for r in stream_table(source, "sample_data") if r["sample_token"] in sample_tokens]
    save("sample_data", sd)
    ego, cal = {r["ego_pose_token"] for r in sd}, {r["calibrated_sensor_token"] for r in sd}
    del sd
    save("ego_pose", [r for r in stream_table(source, "ego_pose") if r["token"] in ego])
    save("calibrated_sensor", [r for r in stream_table(source, "calibrated_sensor") if r["token"] in cal])
    anns = [r for r in stream_table(source, "sample_annotation") if r["sample_token"] in sample_tokens]
    inst = {r["instance_token"] for r in anns}
    save("sample_annotation", anns)
    del anns
    save("instance", [r for r in stream_table(source, "instance") if r["token"] in inst])
    save("log", [r for r in stream_table(source, "log") if r["token"] in log_tokens])
    save("map", [{**r, "log_tokens": [t for t in r["log_tokens"] if t in log_tokens]}
                 for r in stream_table(source, "map") if log_tokens & set(r["log_tokens"])])
    for table in ("category", "attribute", "visibility", "sensor"):
        save(table, list(stream_table(source, table)))
    if not (out_root / "maps").exists():
        (out_root / "maps").symlink_to(PROJECT_ROOT / "data/maps", target_is_directory=True)
    return len(scenes)


def convert_test(n_scenes, workers, which="test150"):
    """which="test150": official nuScenes val scenes (= the scenes of our test split; never used for training or
    model selection), minus the mini scenes already used to develop the planner, into sim/scenarios/nuscenes_test150
    (the first n_scenes in name order).
    which="dev100": n_scenes random scenes (seed 0) of our val split (model selection; official nuScenes train
    scenes), minus the mini scenes, into sim/scenarios/nuscenes_dev100: planner development, never test."""
    import json
    import pickle
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes import NuScenes
    from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario
    from scenarionet.converter.utils import write_to_directory

    splits = create_splits_scenes()
    dev = set(splits["mini_train"] + splits["mini_val"])
    test_tokens = {s["scene_token"] for s in pickle.loads(
        (PROJECT_ROOT / "data/processed_v2/nuscenes_test_prediction_samples_map.pkl").read_bytes())}
    token_to_name = {r["token"]: r["name"] for r in stream_table(PROJECT_ROOT / "data/v1.0-trainval", "scene")}
    test_names = {token_to_name[t] for t in test_tokens}
    if which == "dev100":
        import numpy as np
        val_tokens = {s["scene_token"] for s in pickle.loads(
            (PROJECT_ROOT / "data/processed_v2/nuscenes_val_prediction_samples_map.pkl").read_bytes())}
        pool = sorted(token_to_name[t] for t in val_tokens if token_to_name[t] not in dev)
        names = sorted(np.random.default_rng(0).choice(pool, min(n_scenes, len(pool)), replace=False).tolist())
        assert not set(names) & test_names and not set(names) & set(splits["val"])
        print(f"val-split scenes: {len(val_tokens)}, minus mini -> {len(pool)}; {len(names)} drawn (seed 0); "
              f"none in the test split or in official nuScenes val")
    else:
        names = sorted(n for n in splits["val"] if n not in dev)[:n_scenes]
        print(f"official val scenes: {len(splits['val'])}, minus {len(set(splits['val']) & dev)} mini scenes used in "
              f"development -> {len(names)} scenes; {sum(n in test_names for n in names)} of them have test-split "
              f"prediction samples, {sum(n not in test_names for n in names)} have none")
    root = PROJECT_ROOT / f"sim/nuscenes_{which}_tables"
    if not (root / "v1.0-trainval/scene.json").exists():
        subset_tables(names, root)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=False)
    scenes = [s for s in nusc.scene if s["name"] in set(names)]
    out = PROJECT_ROOT / f"sim/scenarios/nuscenes_{which}"
    write_to_directory(convert_func=convert_nuscenes_scenario, scenarios=scenes, output_path=str(out),
                       dataset_version="v1.0-trainval", dataset_name="nuscenes", overwrite=True, num_workers=workers,
                       nuscenes=[nusc] * workers, past=[2] * workers, future=[6] * workers,
                       prediction=[False] * workers, map_radius=[500] * workers)
    (out / "scene_list.json").write_text(json.dumps({"scenes": [s["name"] for s in scenes],
                                                     "with_test_samples": sorted(n for n in names if n in test_names),
                                                     "excluded_dev_mini": sorted(set(splits["val"]) & dev)}, indent=1))


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


def plan(predictor_names, scene_indices, gif_indices, viz_indices, checkpoint, out_json, data_dir=None, quiet=False,
         reactive="off", device="cuda"):
    """Closed-loop planners (CV and/or WM) on the converted mini scenes + the logged ego ("expert").
    During CV runs the WM runs in shadow mode (chosen plan only) so every at-fault collision can be compared
    with what the WM would have predicted."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import json
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT))
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.scenario.utils import get_number_of_scenarios
    from sim.closed_loop import WEIGHTS, CVPredictor, WorldModelPredictor, run_scene

    data_dir = Path(data_dir or SCENARIO_DIR)
    n = get_number_of_scenarios(str(data_dir))
    scene_indices = scene_indices or list(range(n))
    wm = WorldModelPredictor(checkpoint, device) if ("wm" in predictor_names or "cv" in predictor_names) else None
    runs = [(("CV" if p_ == "cv" else "WM") + " planner", CVPredictor() if p_ == "cv" else wm,
             wm if p_ == "cv" else None, {}, False) for p_ in predictor_names]
    runs.append(("expert (log)", None, None, dict(agent_policy=ReplayEgoCarPolicy), True))
    from sim.closed_loop import EGO_SIZE_ZOE
    # Ego = Renault Zoe size (the nuScenes ego); MetaDrive's default car is 4.52 x 1.85 m.
    # reactive="metadrive": MetaDrive's reactive_traffic (only vehicles spawning behind the ego, IDM at 40 km/h);
    # reactive="idm": our log-path IDM agents (closed_loop.ReactiveTraffic, docs/reactive_agents_design.md stage 1).
    base = dict(use_render=False, num_scenarios=n, data_directory=str(data_dir), horizon=10_000, log_level=50,
                crash_vehicle_done=False, crash_object_done=False, crash_human_done=False, reactive_traffic=reactive == "metadrive",
                vehicle_config=dict(vehicle_model="varying_dynamics", width=EGO_SIZE_ZOE[0], length=EGO_SIZE_ZOE[1]),
                agent_configs={"default_agent": dict(use_special_color=True, spawn_lane_index=None,
                                                     vehicle_model="varying_dynamics", width=EGO_SIZE_ZOE[0],
                                                     length=EGO_SIZE_ZOE[1])})
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, predictor, shadow, cfg, expert in runs:
        rows = []
        tag = label.split()[0].lower()
        env = None
        try:
            for i in scene_indices:
                # A fresh MetaDrive engine per scene: its physics keeps state across reset() (0.1 mm differences
                # in the ego path after earlier scenes), which can flip a close plan choice; with a new env the
                # result of a scene does not depend on which scenes ran before it.
                if env is not None:
                    env.close()
                env = ScenarioEnv({**base, **cfg})
                env.reset(seed=i)
                gif = out_dir / f"plan_{tag}_scene{i}.gif" if (i in gif_indices and not expert) else None
                viz = out_dir / f"planview_{tag}_scene{i}.gif" if (i in viz_indices and not expert) else None
                m, chosen = run_scene(env, predictor, gif=gif, expert=expert, shadow=shadow, viz=viz, device=device,
                                      reactive=reactive == "idm")
                m["chosen_plans"], m["index"] = chosen, i
                rows.append(m)
                print(f"  {label:14s} {m['scene']}: at-fault {m['at_fault']}, off-road steps {m['offroad_steps']}, "
                      f"progress {m['progress_ratio']:.2f}" + (f", plan {m['plan_ms']:.0f} ms" if not expert else ""),
                      flush=True)
        finally:
            if env is not None:
                env.close()
        results[label] = rows

    head = (f" {'scene':10s} | {'':14s} | at-fault | not at fault | off-road | progress | max |a| | max |jerk| "
            f"| plan ms (median / p95)")
    print(head)
    print("-" * len(head))
    for idx in range(len(scene_indices)):
        for label, rows in results.items():
            m = rows[idx]
            ms = f"{m['plan_ms']:.0f} / {m['plan_ms_p95']:.0f}" if label != "expert (log)" else ""
            print(f" {m['scene']:10s} | {label:14s} | {m['at_fault']:8d} | {m['not_at_fault']:12d} | "
                  f"{'yes' if m['offroad_steps'] else 'no':>8s} | {m['progress_ratio']:8.2f} | {m['max_acc']:8.2f} | "
                  f"{m['max_jerk']:10.2f} | {ms}")
    for label, rows in results.items():
        ms = (f"{np.median([m['plan_ms'] for m in rows]):.0f} (predict {np.median([m['predict_ms'] for m in rows]):.0f}"
              f" + cost {np.median([m['cost_ms'] for m in rows]):.0f})") if label != "expert (log)" else ""
        print(f" {'MEAN':10s} | {label:14s} | {np.mean([m['at_fault'] for m in rows]):8.2f} | "
              f"{np.mean([m['not_at_fault'] for m in rows]):12.2f} | {np.mean([m['offroad_steps'] > 0 for m in rows]):8.0%} | "
              f"{np.nanmean([m['progress_ratio'] for m in rows]):8.2f} | {np.mean([m['max_acc'] for m in rows]):8.2f} | "
              f"{np.mean([m['max_jerk'] for m in rows]):10.2f} | {ms}")
    print("nearby agents (vehicles / pedestrians within 30 m of the ego, summed over planning steps):")
    for label, rows in results.items():
        tot, wmn = sum(m["nearby_agents"] for m in rows), sum(m["nearby_wm"] for m in rows)
        if tot:
            print(f"  {label:14s}: {wmn}/{tot} = {wmn / tot:.0%} predicted by the WM, {1 - wmn / tot:.0%} fell back to CV"
                  + ("  (shadow WM)" if label.startswith("CV") else ""))
    print("at-fault collisions, and the WM prediction for that agent at the last planning step before:")
    for label, rows in results.items():
        for m in rows:
            for c in m["collided"]:
                if not c["at_fault"]:
                    continue
                line = f"  {label:14s} {m['scene']} t {c['t']:.1f} s: {c['category']} {c['object']}"
                if "wm_covered" in c:
                    line += f" | WM pred at t {c['wm_pred_t']:.1f} s ({c['wm_horizon_s']:.1f} s ahead), covered {c['wm_covered']}"
                if "wm_top_prob" in c:
                    line += (f", top mode p {c['wm_top_prob']:.2f} err {c['wm_err_top_m']:.1f} m, best mode err "
                             f"{c['wm_err_best_m']:.1f} m (p {c['wm_prob_best']:.2f}); dist to ego plan: predicted "
                             f"{c['wm_pred_dist_to_ego_plan_m']:.1f} m, actual {c['wm_actual_dist_to_ego_plan_m']:.1f} m")
                elif "wm_note" in c:
                    line += f", {c['wm_note']}"
                print(line)
    print("chosen-plan cost terms (mean over the scene):")
    for label, rows in results.items():
        if label.startswith("expert"):
            continue
        for m in rows:
            ch = m["chosen_plans"]
            print(f"  {label:14s} {m['scene']}: " + ", ".join(
                f"{k} {np.mean([c[k] for c in ch]):.2f}" for k in ("speed", "offset", "risk", "progress", "comfort",
                                                                   "offset2", "total")))
    out_json.write_text(json.dumps({"weights": WEIGHTS, "results": results}, indent=1, default=str))
    print(f"saved {out_json}")
    return results


def evaluate_test(data_dir, checkpoint, out_dir, seed=0, compare=None, reactive="off"):
    """Closed-loop test: CV and WM planners + the expert on every scene of data_dir (frozen weights), then
    paired statistics, a per-map breakdown, coverage / timing, the 5 worst scenes per planner and a spot-check
    of the collision classification (both with GIFs, from a second deterministic pass with rendering on)."""
    import json
    import numpy as np
    from scipy.stats import wilcoxon
    out_dir.mkdir(parents=True, exist_ok=True)
    res = plan(["cv", "wm"], [], [], [], checkpoint, out_dir / "results.json", data_dir, reactive=reactive)
    cv, wm, ex = res["CV planner"], res["WM planner"], res["expert (log)"]
    rng = np.random.default_rng(seed)

    def boot(d):
        d = np.asarray(d, float)
        means = rng.choice(d, (10_000, len(d)), replace=True).mean(1)
        return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def wil(d):
        d = np.asarray(d, float)
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    title = {"nuscenes_test150": "146 nuScenes val scenes = our test split, minus 4 dev mini scenes",
             "nuscenes_dev100": "100 scenes of our val split (planner development; never test)",
             "nuscenes_mini": "10 mini dev scenes"}.get(Path(data_dir).name, str(data_dir))
    lines = [f"# Closed-loop evaluation: CV vs WM planner ({title})",
             "", f"Checkpoint: `{checkpoint}`; weights frozen: col 500, progress 1, comfort 0.1, offset 0.5, veto early "
             "risk > 0.3 within 2 s. " + ("Reactive agents: log-path IDM (stage 1); a hit from behind within 2 s after the "
                                           "ego entered the agent's path counts as at fault (unsafe cut-in)."
                                           if reactive == "idm" else "Non-reactive agents (log replay)."), ""]
    def mean_row(rows, name):
        return (f"| {name} | {np.mean([m['at_fault'] for m in rows]):.3f} | {np.mean([m['at_fault'] > 0 for m in rows]):.1%} | "
                f"{np.mean([m['not_at_fault'] for m in rows]):.3f} | {np.mean([m['offroad_steps'] > 0 for m in rows]):.1%} | "
                f"{np.nanmean([m['progress_ratio'] for m in rows]):.3f} | {np.mean([m['max_acc'] for m in rows]):.2f} | "
                f"{np.mean([m['max_jerk'] for m in rows]):.2f} |")
    if compare is not None and Path(compare).exists():
        old = json.loads(Path(compare).read_text())["results"]
        lines += ["## Old vs new (old: MetaDrive ego 4.52 x 1.85 m, stopped-ego collisions at fault; "
                  "new: Zoe 4.08 x 1.73 m, collisions with ego speed < 0.5 m/s not at fault)", "",
                  "| | at-fault / scene old -> new | scenes with at-fault old -> new | rear/not-at-fault / scene old -> new | "
                  "progress old -> new |", "|---|---|---|---|---|"]
        for name in ("CV planner", "WM planner", "expert (log)"):
            o, n_ = old[name], res[name]
            f = lambda r, k: np.mean([m[k] for m in r])
            lines.append(f"| {name} | {f(o, 'at_fault'):.3f} -> {f(n_, 'at_fault'):.3f} | "
                         f"{np.mean([m['at_fault'] > 0 for m in o]):.1%} -> {np.mean([m['at_fault'] > 0 for m in n_]):.1%} | "
                         f"{f(o, 'not_at_fault'):.3f} -> {f(n_, 'not_at_fault'):.3f} | "
                         f"{np.nanmean([m['progress_ratio'] for m in o]):.3f} -> {np.nanmean([m['progress_ratio'] for m in n_]):.3f} |")
        lines.append("")
    lines += ["## Collision types (counts over all scenes)", "",
              "initial overlap = first contact within the first 0.5 s (boxes overlap from the start of the log);",
              "not at fault = rear-ended, or the ego stopped (< 0.5 m/s).", "",
              "| | at fault | not at fault | initial overlap |", "|---|---|---|---|"]
    for name, rows in (("CV planner", cv), ("WM planner", wm), ("expert (log)", ex)):
        lines.append(f"| {name} | {sum(m['at_fault'] for m in rows)} | {sum(m['not_at_fault'] for m in rows)} | "
                     f"{sum(m.get('initial_overlap', 0) for m in rows)} |")
    lines += ["", "## At-fault events: what the planner saw at the last 2 planning steps", "",
              "| planner | scene | t | agent | vetoed plans | chosen-plan risk | risk from this agent | zero-risk plans | "
              "first seen (s before) | diagnosis |", "|---|---|---|---|---|---|---|---|---|---|"]
    diag_count = {}
    for name, rows in (("CV", cv), ("WM", wm)):
        for m in rows:
            for c in m["collided"]:
                if c.get("kind") != "at_fault":
                    continue
                d = c.get("diagnosis", "?")
                diag_count[(name, d)] = diag_count.get((name, d), 0) + 1
                f2 = lambda xs: " / ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in xs)
                lines.append(f"| {name} | {m['scene']} | {c['t']:.1f} s | {c['category']} | {f2(c.get('diag_vetoed', []))} | "
                             f"{f2(c.get('diag_chosen_risk', []))} | {f2(c.get('diag_agent_risk', []))} | "
                             f"{f2(c.get('diag_zero_risk_plans', []))} | {c.get('first_seen_s_before')} | {d} |")
    lines += ["", "Values are for the two planning steps before the collision (earlier / later).", "",
              "| planner | no safe plan existed | safe plan existed but cost picked a risky one | zero predicted risk |",
              "|---|---|---|---|"]
    for name in ("CV", "WM"):
        lines.append(f"| {name} | {diag_count.get((name, 'no safe plan existed'), 0)} | "
                     f"{diag_count.get((name, 'safe plan existed but cost picked a risky one'), 0)} | "
                     f"{diag_count.get((name, 'zero predicted risk'), 0)} |")
    lines.append("")
    lines += ["## Near misses (means over scenes)", "", "| | min distance [m] | min TTC [s] (capped at 5) | "
              "scenes with TTC < 1 s | share of steps with distance < 1 m |", "|---|---|---|---|---|"]
    for name, rows in (("CV planner", cv), ("WM planner", wm), ("expert (log)", ex)):
        ttc = np.minimum([m["min_ttc_s"] for m in rows], 5.0)
        lines.append(f"| {name} | {np.mean([m['min_dist_m'] for m in rows]):.2f} | {ttc.mean():.2f} | "
                     f"{np.mean(ttc < 1.0):.1%} | {np.mean([m['share_dist_below_1m'] for m in rows]):.2%} |")
    lines += ["", "Distance = box to box (0 when touching); TTC = constant-velocity extrapolation with the planner's "
              "circle footprints, agents behind the ego in its lane skipped.", ""]
    lines += ["## Means over scenes", "", "| | at-fault / scene | scenes with at-fault | rear-ended / scene | scenes off-road | "
              "progress ratio | max \\|a\\| | max \\|jerk\\| |", "|---|---|---|---|---|---|---|---|",
              mean_row(cv, "CV planner"), mean_row(wm, "WM planner"), mean_row(ex, "expert (log)"), ""]
    d_af = [w["at_fault"] - c["at_fault"] for w, c in zip(wm, cv)]
    pairs = [(w["progress_ratio"], c["progress_ratio"]) for w, c in zip(wm, cv)
             if np.isfinite(w["progress_ratio"]) and np.isfinite(c["progress_ratio"])]
    d_pr = [w - c for w, c in pairs]
    d_rear = [w["not_at_fault"] - c["not_at_fault"] for w, c in zip(wm, cv)]
    stats = {}
    lines += ["## Paired WM - CV (same scenes)", "", "| metric | n scenes | mean WM - CV | bootstrap 95% CI | Wilcoxon p |",
              "|---|---|---|---|---|"]
    d_md = [w["min_dist_m"] - c["min_dist_m"] for w, c in zip(wm, cv)]
    d_ttc = [min(w["min_ttc_s"], 5.0) - min(c["min_ttc_s"], 5.0) for w, c in zip(wm, cv)]
    d_sh = [w["share_dist_below_1m"] - c["share_dist_below_1m"] for w, c in zip(wm, cv)]
    for name, d in (("at-fault collisions / scene", d_af), ("progress ratio", d_pr), ("rear-ended / scene", d_rear),
                    ("min distance [m]", d_md), ("min TTC [s] (capped at 5)", d_ttc),
                    ("share of steps with distance < 1 m", d_sh)):
        m_, lo, hi = boot(d)
        stats[name] = dict(n=len(d), mean=m_, ci=[lo, hi], wilcoxon_p=wil(d))
        lines.append(f"| {name} | {len(d)} | {m_:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {stats[name]['wilcoxon_p']:.3g} |")
    lines += ["", "Progress ratio excludes scenes where the logged ego does not move (n/a).", ""]
    if reactive == "idm":                  # pre-registered decision (docs/EXPERIMENTS.md, Amendment 1)
        def boot_ci(d, level):
            d = np.asarray(d, float)
            means = np.random.default_rng(0).choice(d, (10_000, len(d)), replace=True).mean(1)
            return float(np.percentile(means, (100 - level) / 2)), float(np.percentile(means, 100 - (100 - level) / 2))
        co = {}
        for name, d in (("at-fault collisions / scene", d_af), ("close-call share (steps < 1 m)", d_sh)):
            lo, hi = boot_ci(d, 97.5)
            co[name] = dict(mean=float(np.mean(d)), ci97_5=[lo, hi], wilcoxon_p=wil(d), below_zero=hi < 0)
        p_lo, p_hi = boot_ci(d_pr, 95)
        progress_ok = p_lo > -0.02
        safer = any(v["below_zero"] for v in co.values()) and progress_ok
        stats["preregistered"] = dict(co_primary=co, progress_ci95=[p_lo, p_hi], progress_ok=progress_ok, wm_safer=safer)
        lines += ["## Pre-registered decision (Amendment 1: co-primary endpoints, alpha = 0.025 each)", "",
                  "| co-primary endpoint | mean WM - CV | bootstrap 97.5% CI (seed 0) | Wilcoxon p | CI below 0 |",
                  "|---|---|---|---|---|"]
        for name, v in co.items():
            lines.append(f"| {name} | {v['mean']:+.4f} | [{v['ci97_5'][0]:+.4f}, {v['ci97_5'][1]:+.4f}] | "
                         f"{v['wilcoxon_p']:.3g} ({'<' if v['wilcoxon_p'] < 0.025 else '>='} 0.025) | {v['below_zero']} |")
        lines += ["", f"Progress ratio WM - CV: 95% CI [{p_lo:+.4f}, {p_hi:+.4f}]; not worse by > 0.02: {progress_ok}.", "",
                  f"**Decision: {'WM is safer' if safer else 'no detectable difference'}** "
                  "(rule: either co-primary 97.5% CI entirely below 0 and progress not worse by > 0.02).", "",
                  f"At-fault events: CV {sum(m['at_fault'] for m in cv)}, WM {sum(m['at_fault'] for m in wm)}.", ""]
    lines += ["## Per map", "", "| map | scenes | planner | at-fault / scene | rear-ended / scene | off-road | progress |",
              "|---|---|---|---|---|---|---|"]
    for mp in sorted({m["map"] for m in cv}):
        for name, rows in (("CV", cv), ("WM", wm), ("expert", ex)):
            r = [m for m in rows if m["map"] == mp]
            lines.append(f"| {mp} | {len(r)} | {name} | {np.mean([m['at_fault'] for m in r]):.3f} | "
                         f"{np.mean([m['not_at_fault'] for m in r]):.3f} | {np.mean([m['offroad_steps'] > 0 for m in r]):.1%} | "
                         f"{np.nanmean([m['progress_ratio'] for m in r]):.3f} |")
    cov = {n_: (sum(m["nearby_wm"] for m in r), sum(m["nearby_agents"] for m in r)) for n_, r in (("CV", cv), ("WM", wm))}
    lines += ["", "## Ego tracking error (driven vs planned position at the same time)", "",
              "| planner | mean [m] | p95 [m] | max [m] | lateral p95 [m] |", "|---|---|---|---|---|"]
    for name, rows in (("CV", cv), ("WM", wm)):
        lines.append(f"| {name} | {np.mean([m['track_err_mean_m'] for m in rows]):.2f} | "
                     f"{np.mean([m['track_err_p95_m'] for m in rows]):.2f} | {np.max([m['track_err_max_m'] for m in rows]):.2f} | "
                     f"{np.mean([m['track_lat_p95_m'] for m in rows]):.2f} |")
    lines += ["", "Means over scenes of the per-scene mean / p95; max = over all scenes.", ""]
    lines += ["", "## Coverage and timing", "",
              f"- WM coverage (vehicles / pedestrians within 30 m, all planning steps): WM planner {cov['WM'][0]}/{cov['WM'][1]}"
              f" = {cov['WM'][0] / max(cov['WM'][1], 1):.0%}; shadow WM during the CV runs {cov['CV'][0] / max(cov['CV'][1], 1):.0%}",
              f"- Median planning time per step: CV {np.median([m['plan_ms'] for m in cv]):.0f} ms, "
              f"WM {np.median([m['plan_ms'] for m in wm]):.0f} ms (p95 of scene medians: CV "
              f"{np.percentile([m['plan_ms'] for m in cv], 95):.0f}, WM {np.percentile([m['plan_ms'] for m in wm], 95):.0f})", ""]

    def worst(rows):
        key = lambda m: (-m["at_fault"], -(m["offroad_steps"] > 0), np.nan_to_num(m["progress_ratio"], nan=1.0))
        return sorted(rows, key=key)[:5]
    worst_sel = {"cv": worst(cv), "wm": worst(wm)}
    events = [(tag, m["index"], m["scene"], c) for tag, rows in (("cv", cv), ("wm", wm)) for m in rows for c in m["collided"]]
    af = [e for e in events if e[3]["at_fault"]]
    ra = [e for e in events if not e[3]["at_fault"]]
    spot = ([af[i] for i in rng.choice(len(af), min(5, len(af)), replace=False)] if af else []) + \
           ([ra[i] for i in rng.choice(len(ra), min(5, len(ra)), replace=False)] if ra else [])

    gif_dir = out_dir / "gifs"
    gif_dir.mkdir(exist_ok=True)
    rerun = {}
    for tag in ("cv", "wm"):
        idx = sorted({m["index"] for m in worst_sel[tag]} | {e[1] for e in spot if e[0] == tag})
        if idx:
            r = plan([tag], idx, idx, idx, checkpoint, gif_dir / f"rerun_{tag}.json", data_dir, reactive=reactive)
            rerun[tag] = {m["index"]: m for m in r[("CV" if tag == "cv" else "WM") + " planner"]}
    first = {"cv": {m["index"]: m for m in cv}, "wm": {m["index"]: m for m in wm}}
    same = all(rerun[t][i]["at_fault"] == first[t][i]["at_fault"] and rerun[t][i]["not_at_fault"] == first[t][i]["not_at_fault"]
               for t in rerun for i in rerun[t])
    lines += ["## 5 worst scenes per planner", "", f"Re-run with rendering for the GIFs; identical collision counts to "
              f"the main run: {same}.", "", "| planner | scene | map | at-fault | rear-ended | off-road | progress | GIFs |",
              "|---|---|---|---|---|---|---|---|"]
    for tag, rows in worst_sel.items():
        for m in rows:
            lines.append(f"| {tag.upper()} | {m['scene']} | {m['map']} | {m['at_fault']} | {m['not_at_fault']} | "
                         f"{'yes' if m['offroad_steps'] else 'no'} | {m['progress_ratio']:.2f} | "
                         f"`gifs/planview_{tag}_scene{m['index']}.gif`, `gifs/plan_{tag}_scene{m['index']}.gif` |")

    from PIL import Image
    spot_dir = out_dir / "spotcheck"
    spot_dir.mkdir(exist_ok=True)
    lines += ["", "## Spot-check of the collision classification", "",
              "Left: planning view at the last planning step before the event; right: MetaDrive top-down at the event.",
              "", "| # | planner | scene | t | object | classified | WM top-mode error just before | frames |",
              "|---|---|---|---|---|---|---|---|"]
    for n_, (tag, i, scene, c) in enumerate(spot, 1):
        step = int(round(c["t"] * 10))
        try:
            pv = Image.open(gif_dir / f"planview_{tag}_scene{i}.gif")
            pv.seek(min(max((step - 1) // 5, 0), pv.n_frames - 1))
            md = Image.open(gif_dir / f"plan_{tag}_scene{i}.gif")
            md.seek(min(max((step - 1) // 2, 0), md.n_frames - 1))
            a_, b_ = pv.convert("RGB"), md.convert("RGB").resize((pv.height, pv.height))
            pair = Image.new("RGB", (a_.width + b_.width, a_.height), "white")
            pair.paste(a_, (0, 0))
            pair.paste(b_, (a_.width, 0))
            name = f"{n_:02d}_{tag}_{scene}_t{c['t']:.1f}_{'atfault' if c['at_fault'] else 'rear'}.png"
            pair.save(spot_dir / name)
        except (FileNotFoundError, EOFError) as e:
            name = f"(no frame: {e})"
        err = f"{c['wm_err_top_m']:.1f} m" if "wm_err_top_m" in c else "–"
        lines.append(f"| {n_} | {tag.upper()} | {scene} | {c['t']:.1f} s | {c['category']} {c['object']} | "
                     f"{'at fault' if c['at_fault'] else 'rear-ended'} | {err} | `spotcheck/{name}` |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(dict(stats=stats, coverage=cov, spot=[(e[0], e[2], e[3]) for e in spot],
                                                          worst={t: [m["scene"] for m in r] for t, r in worst_sel.items()},
                                                          rerun_identical=same), indent=1, default=str))
    print("\n".join(lines))


def reactive_check(data_dirs, out_dir, idm_overrides, cases_per_set=20):
    """Stage-1 reactive agents, validation (docs/reactive_agents_design.md section 4), never on test.

    (a) The expert (logged ego) drives in reactive mode: for every reactive agent, the position error vs its log
        (Euclidean and along-track) at 1 / 3 / 5 / 8 s after it became active, the share of agents with error > 1 m,
        the share of agent-steps where a_IDM was binding, and agent-agent collisions not present in the log.
        Targets: median < 0.2 m and p95 < 1 m at 8 s; 0 new agent-agent collisions.
    (b) Sim follower test: scenes with a moving vehicle 2-15 m behind the ego in its lane (|lateral| <= 1.5 m,
        heading within 20 deg), ego >= 3 m/s, at a keyframe >= 2 s into the scene (one case per scene). The ego follows
        its logged route at the logged speed, then from t0 brakes to a stop at -4 m/s^2 (tracker). Reactive vs replay:
        collision with the follower, min gap, follower peak deceleration, reaction delay (first decel < -0.5 m/s^2),
        and whether the follower stays behind. Target: no collision in reactive mode."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import json
    import pickle
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT))
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.scenario.utils import get_number_of_scenarios, read_dataset_summary
    from sim import closed_loop as cl
    from sim.adapter import HistoryBuffer, snapshot_from_env
    from sim.planner import Tracker, generate_plans, logged_speed, reference_from_scenario

    cl.IDM.update(idm_overrides)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoe = dict(vehicle_model="varying_dynamics", width=cl.EGO_SIZE_ZOE[0], length=cl.EGO_SIZE_ZOE[1])
    base = dict(use_render=False, horizon=10_000, log_level=50, crash_vehicle_done=False, crash_object_done=False,
                crash_human_done=False, vehicle_config=zoe,
                agent_configs={"default_agent": dict(use_special_color=True, spawn_lane_index=None, **zoe)})
    result = {"idm": dict(cl.IDM), "sets": {}}
    for data_dir in data_dirs:
        name = Path(data_dir).name
        n = get_number_of_scenarios(str(data_dir))
        # (a) expert drives the log
        errs = {1: [], 3: [], 5: [], 8: []}
        along = {1: [], 3: [], 5: [], 8: []}
        ep_max, new_coll, log_coll, bind, n_agents, standoff = [], 0, 0, [], 0, 0.0
        expert_events = []
        for i in range(n):
            env = ScenarioEnv({**base, "num_scenarios": n, "data_directory": str(data_dir),
                               "agent_policy": ReplayEgoCarPolicy})
            try:
                env.reset(seed=i)
                m, _ = cl.run_scene(env, None, expert=True, reactive=True)
            finally:
                env.close()
            R = m["reactive"]
            expert_events += [(m["scene"], c["t"], c["category"], c["kind"]) for c in m["collided"]]
            n_agents += R["n_agents"]
            new_coll += R["new_agent_collisions"]
            log_coll += R["log_agent_collisions"]
            bind.append(R["idm_binding_share"])
            standoff = max(standoff, R["max_ego_standoff_s"])
            for oid, lst in R["errors"].items():
                d = {round(t, 1): (e, al) for t, e, al in lst}
                for tau in errs:
                    if float(tau) in d:
                        errs[tau].append(d[float(tau)][0])
                        along[tau].append(d[float(tau)][1])
                ep_max.append(max(e for _, e, _ in lst))
        a_res = {f"{tau}s": dict(n=len(errs[tau]), median=float(np.median(errs[tau])) if errs[tau] else None,
                                 p95=float(np.percentile(errs[tau], 95)) if errs[tau] else None,
                                 along_median=float(np.median(along[tau])) if along[tau] else None,
                                 share_above_1m=float(np.mean(np.asarray(errs[tau]) > 1.0)) if errs[tau] else None)
                 for tau in errs}
        a_res.update(agents=n_agents, episode_max_median=float(np.median(ep_max)) if ep_max else None,
                     episode_max_p95=float(np.percentile(ep_max, 95)) if ep_max else None,
                     share_agents_episode_err_above_1m=float(np.mean(np.asarray(ep_max) > 1.0)) if ep_max else None,
                     idm_binding_share=float(np.mean(bind)), new_agent_collisions=new_coll,
                     log_agent_collisions=log_coll, max_ego_standoff_s=standoff,
                     expert_at_fault=sum(e[3] == "at_fault" for e in expert_events),
                     expert_not_at_fault=sum(e[3] == "not_at_fault" for e in expert_events),
                     expert_events=expert_events,
                     pass_target=bool(a_res["8s"]["median"] is not None and a_res["8s"]["median"] < 0.2
                                      and a_res["8s"]["p95"] < 1.0 and new_coll == 0))
        print(f"[{name}] (a) expert drives the log: " + json.dumps(a_res), flush=True)

        # (b) follower cases from the log
        summ, ids, mapping = read_dataset_summary(str(data_dir))
        cases = []
        for i, f in enumerate(ids):
            sc = pickle.loads((Path(data_dir) / mapping[f] / f).read_bytes())
            sdc = sc["tracks"][sc["metadata"]["sdc_id"]]["state"]
            for t0 in range(20, sc["length"] - 40, 5):
                if not sdc["valid"][t0]:
                    continue
                ex, ey = sdc["position"][t0][:2]
                eh = sdc["heading"][t0]
                if logged_speed(sc, t0) < 3.0:
                    continue
                found = None
                for oid, tr in sc["tracks"].items():
                    if oid == sc["metadata"]["sdc_id"] or tr["type"] != "VEHICLE" or not tr["state"]["valid"][t0]:
                        continue
                    ag = cl.LogPathAgent(oid, tr)
                    if not ag.eligible:
                        continue
                    x, y = tr["state"]["position"][t0][:2]
                    c, s_ = np.cos(eh), np.sin(eh)
                    xr, yr = c * (x - ex) + s_ * (y - ey), -s_ * (x - ex) + c * (y - ey)
                    dh = abs(np.angle(np.exp(1j * (tr["state"]["heading"][t0] - eh))))
                    if -15 <= xr <= -2 and abs(yr) <= 1.5 and dh <= np.radians(20):
                        found = oid
                        break
                if found:
                    cases.append((i, t0, found))
                    break
            if len(cases) >= cases_per_set:
                break
        rows = []
        for i, t0, fid in cases:
            row = dict(scene_index=i, t0=t0 / 10, follower=fid[:8])
            for mode in ("reactive", "replay"):
                env = ScenarioEnv({**base, "num_scenarios": n, "data_directory": str(data_dir)})
                try:
                    env.reset(seed=i)
                    scen = env.engine.data_manager.current_scenario
                    row["scene"] = scen["metadata"]["scenario_id"]
                    ref = reference_from_scenario(scen)
                    rt = cl.ReactiveTraffic(env) if mode == "reactive" else None
                    buf, tracker = HistoryBuffer(), Tracker()
                    buf.push(snapshot_from_env(env, rt))
                    plan, plan_t, gaps, fv, hit, behind = None, 0.0, [], [], False, True
                    for step in range(min(scen["length"] - 1, t0 + 80)):
                        a = env.agent
                        st = (float(a.position[0]), float(a.position[1]), float(a.heading_theta), float(a.speed))
                        if step % 5 == 0 or (step == t0):
                            if step < t0:
                                _, d_, _ = generate_plans(ref, st, speeds=[logged_speed(scen, step)], offsets=[0.0])
                            else:
                                _, d_, _ = generate_plans(ref, st, speeds=[0.0], offsets=[0.0], a_max=4.0)
                            plan, plan_t = d_[0], 0.0
                        env.step(tracker.action(*st, plan, plan_t))
                        plan_t += cl.DT
                        if rt is not None:
                            pre = snapshot_from_env(env, rt)
                            rt.step(pre, buf.snapshots[-5] if len(buf.snapshots) >= 5 else None,
                                    np.asarray(env.agent.velocity[:2], float))
                        buf.push(snapshot_from_env(env, rt))
                        now = buf.now()
                        if step + 1 >= t0 and fid in now:
                            x, y, yaw, w_, l_, _ = now[fid]
                            ex, ey, eh = float(env.agent.position[0]), float(env.agent.position[1]), float(env.agent.heading_theta)
                            eb = cl.box(ex, ey, eh, cl.EGO_SIZE_ZOE[1], cl.EGO_SIZE_ZOE[0])
                            fb = cl.box(x, y, yaw, l_, w_)
                            gaps.append(eb.distance(fb))
                            hit |= eb.intersects(fb)
                            xr = np.cos(eh) * (x - ex) + np.sin(eh) * (y - ey)
                            behind &= xr < 0
                            fv.append((step + 1, rt.agents[fid].v if (rt and fid in rt.agents and rt.agents[fid].active)
                                       else None, (x, y)))
                finally:
                    env.close()
                # follower speed: reactive state, or 2 Hz positions for replay
                pos = np.array([p_ for _, _, p_ in fv])
                if mode == "reactive" and all(v is not None for _, v, _ in fv):
                    v = np.array([v for _, v, _ in fv])
                    acc = np.diff(v) / cl.DT
                    k = 1
                else:
                    v = np.linalg.norm(np.diff(pos[::5], axis=0), axis=1) / 0.5 if len(pos) > 5 else np.zeros(1)
                    acc = np.diff(v) / 0.5 if len(v) > 1 else np.zeros(1)
                    k = 5
                dec = np.flatnonzero(acc < -0.5)
                row[mode] = dict(collision=bool(hit), min_gap_m=float(min(gaps)) if gaps else None,
                                 peak_decel=float(-acc.min()) if len(acc) else 0.0,
                                 reaction_delay_s=float(dec[0] * cl.DT * k) if len(dec) else None,
                                 stays_behind=bool(behind))
            rows.append(row)
            print(f"[{name}] (b) {row}", flush=True)
        b_res = dict(cases=len(rows), reactive_collisions=sum(r["reactive"]["collision"] for r in rows),
                     replay_collisions=sum(r["replay"]["collision"] for r in rows),
                     reactive_min_gap_median=float(np.median([r["reactive"]["min_gap_m"] for r in rows])) if rows else None,
                     reactive_peak_decel_median=float(np.median([r["reactive"]["peak_decel"] for r in rows])) if rows else None,
                     reactive_reaction_delay_median=float(np.median([r["reactive"]["reaction_delay_s"] for r in rows
                                                                     if r["reactive"]["reaction_delay_s"] is not None]))
                     if any(r["reactive"]["reaction_delay_s"] is not None for r in rows) else None,
                     reactive_stays_behind=sum(r["reactive"]["stays_behind"] for r in rows))
        b_res["pass_target"] = bool(rows and b_res["reactive_collisions"] == 0)
        print(f"[{name}] (b) summary: {json.dumps(b_res)}", flush=True)

        # (d) the logged ego driven 10% slower (tracker on its logged route, 0.9 x logged speed)
        d_rows = []
        for i in range(n):
            env = ScenarioEnv({**base, "num_scenarios": n, "data_directory": str(data_dir)})
            try:
                env.reset(seed=i)
                m, _ = cl.run_scene(env, None, reactive=True, drive_log_scale=0.9)
            finally:
                env.close()
            R = m["reactive"]
            d_rows.append(dict(scene=m["scene"], new_overlaps=R["new_agent_collisions"], log_overlaps=R["log_agent_collisions"],
                               max_decel=R["max_decel"], rear_ends=sum(c["kind"] == "not_at_fault" for c in m["collided"]),
                               at_fault=sum(c["kind"] == "at_fault" for c in m["collided"]),
                               cut_ins=sum(c.get("unsafe_cut_in", False) for c in m["collided"])))
        d_res = dict(scenes=len(d_rows), new_agent_overlaps=sum(r["new_overlaps"] for r in d_rows),
                     log_agent_overlaps=sum(r["log_overlaps"] for r in d_rows),
                     max_decel=float(max(r["max_decel"] for r in d_rows)) if d_rows else 0.0,
                     rear_ends=sum(r["rear_ends"] for r in d_rows), ego_at_fault=sum(r["at_fault"] for r in d_rows),
                     scenes_with_rear_end=[r["scene"] for r in d_rows if r["rear_ends"]])
        d_res["pass_target"] = bool(d_res["new_agent_overlaps"] == 0 and d_res["rear_ends"] == 0
                                    and d_res["max_decel"] <= cl.IDM["b_max"] + 1e-6)
        print(f"[{name}] (d) ego 10% slower: {json.dumps(d_res)}", flush=True)
        result["sets"][name] = dict(a=a_res, b=b_res, b_cases=rows, d=d_res, d_rows=d_rows)
    tag = "_".join(f"{k}{v}" for k, v in idm_overrides.items()) or "doc"
    (out_dir / f"validation_{tag}.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"saved {out_dir / f'validation_{tag}.json'}")
    return result


def showcase(data_dir, index, checkpoint, out, reactive, follow=35.0):
    """Side-by-side GIF for the README: the same scene planned with constant-velocity predictions (left) and with the
    world model (right), camera following the ego, legend, collision counter. Development scenes only."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(PROJECT_ROOT))
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.scenario.utils import get_number_of_scenarios
    from sim import closed_loop as cl

    n = get_number_of_scenarios(str(data_dir))
    zoe = dict(vehicle_model="varying_dynamics", width=cl.EGO_SIZE_ZOE[0], length=cl.EGO_SIZE_ZOE[1])
    base = dict(use_render=False, num_scenarios=n, data_directory=str(data_dir), horizon=10_000, log_level=50,
                crash_vehicle_done=False, crash_object_done=False, crash_human_done=False, vehicle_config=zoe,
                agent_configs={"default_agent": dict(use_special_color=True, spawn_lane_index=None, **zoe)})
    tmp = OUT_DIR / "showcase"
    tmp.mkdir(parents=True, exist_ok=True)
    # the road user to highlight: the one the CV planner collides with (deterministic: fresh env per run)
    env = ScenarioEnv(base)
    try:
        env.reset(seed=index)
        m0, _ = cl.run_scene(env, cl.CVPredictor(), reactive=reactive)
    finally:
        env.close()
    hits = [c for c in m0["collided"] if c["kind"] == "at_fault"]
    if not hits:
        raise SystemExit("the CV planner has no at-fault collision in this scene; pick another --index")
    focus = hits[0]["object_id"]
    print(f"highlighted road user: {hits[0]['category']} {focus[:8]} (CV collision at {hits[0]['t']:.1f} s)")
    gifs, summary = [], []
    for pred, title in ((cl.CVPredictor(), "Planner with simple predictions\n(everyone keeps going straight)"),
                        (cl.WorldModelPredictor(checkpoint), "Planner with our world model\n(learned predictions)")):
        env = ScenarioEnv(base)
        try:
            env.reset(seed=index)
            path = tmp / f"{pred.name.lower()}_scene{index}.gif"
            m, _ = cl.run_scene(env, pred, viz=path, reactive=reactive, viz_title=title, viz_focus=focus)
        finally:
            env.close()
        gifs.append(Image.open(path))
        summary.append(f"{pred.name}: at fault {m['at_fault']}, progress {m['progress_ratio']:.2f}")
        print(f"{m['scene']} {summary[-1]}", flush=True)
    frames = []
    for i in range(min(g.n_frames for g in gifs)):
        parts = []
        for g in gifs:
            g.seek(i)
            parts.append(g.convert("RGB"))
        w, h = sum(p_.width for p_ in parts), max(p_.height for p_ in parts)
        canvas = Image.new("RGB", (w, h), "white")
        x = 0
        for p_ in parts:
            canvas.paste(p_, (x, 0))
            x += p_.width
        ImageDraw.Draw(canvas).line([(parts[0].width, 0), (parts[0].width, h)], fill=(160, 160, 160), width=2)
        frames.append(canvas.quantize(colors=128))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[450] * (len(frames) - 1) + [2500],
                   loop=0, optimize=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames) | " + " | ".join(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("metadrive").add_argument("--steps", type=int, default=20)
    p = sub.add_parser("convert")
    p.add_argument("--scenes", type=int, default=5)
    p.add_argument("--set", choices=("mini", "test150", "dev100"), default="mini",
                   help="mini: data/nuscenes_mini; test150: official val scenes minus the dev mini scenes; "
                        "dev100: 100 scenes of our val split (planner development)")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("replay")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)
    p = sub.add_parser("drive")
    p.add_argument("--index", type=int, default=0, help="scenario index (0 = scene-0061)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--throttle", type=float, default=0.2)
    p.add_argument("--steering", type=float, default=0.0)
    p = sub.add_parser("showcase")
    p.add_argument("--data", default=str(PROJECT_ROOT / "sim/scenarios/nuscenes_mini"))
    p.add_argument("--index", type=int, default=8, help="scenario index (8 = mini scene-1094)")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/v2_d_rerun/v2_seed42_best.pth"))
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "assets/closed_loop_cv_vs_wm.gif")
    p.add_argument("--reactive", choices=("off", "idm"), default="idm")
    p.add_argument("--follow", type=float, default=35.0, help="camera half-width around the ego [m]")
    p = sub.add_parser("reactive-check")
    p.add_argument("--data", nargs="+", default=[str(PROJECT_ROOT / "sim/scenarios/nuscenes_mini"),
                                                 str(PROJECT_ROOT / "sim/scenarios/nuscenes_dev100")])
    p.add_argument("--out", type=Path, default=OUT_DIR / "reactive_dev")
    p.add_argument("--idm", default="", help="IDM overrides, e.g. a_max=3.0,T=1.0 (tuning on dev100 only)")
    p.add_argument("--cases", type=int, default=20)
    p = sub.add_parser("evaluate")
    p.add_argument("--idm", default="", help="IDM overrides used for --reactive idm")
    p.add_argument("--compare", type=Path, help="results.json of an earlier run, for an old -> new table")
    p.add_argument("--reactive", choices=("off", "metadrive", "idm"), default="off",
                   help="metadrive: MetaDrive's reactive_traffic; idm: log-path IDM agents (stage 1)")
    p.add_argument("--data", default=str(PROJECT_ROOT / "sim/scenarios/nuscenes_test150"))
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/v2_d_rerun/v2_seed42_best.pth"))
    p.add_argument("--out", type=Path, default=OUT_DIR / "test150")
    p = sub.add_parser("plan")
    p.add_argument("--predictor", choices=("cv", "wm"), nargs="+", default=["cv"])
    p.add_argument("--scenes", type=int, nargs="*", default=[], help="scenario indices (default: all)")
    p.add_argument("--gif", type=int, nargs="*", default=[], help="scenario indices for a MetaDrive top-down GIF")
    p.add_argument("--viz", type=int, nargs="*", default=[8], help="scenario indices for the planning-view GIF")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/v2_d_rerun/v2_seed42_best.pth"))
    p.add_argument("--out", type=Path, help="JSON with metrics and chosen-plan cost terms")
    p.add_argument("--data", help="scenario folder (default: the mini scenes)")
    p.add_argument("--reactive", choices=("off", "metadrive", "idm"), default="off",
                   help="metadrive: MetaDrive's reactive_traffic; idm: log-path IDM agents (stage 1)")
    p.add_argument("--device", default="cuda", help="device for the WM and the collision cost (cuda / cpu)")
    p = sub.add_parser("follow")
    p.add_argument("--index", type=int, default=0, help="scenario index (0 = scene-0061)")
    p = sub.add_parser("match")
    p.add_argument("--scene", default="scene-0061")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/v2_d_rerun/v2_seed42_best.pth"))
    p.add_argument("--limit", type=int, default=0, help="only the first N samples (0 = all)")
    args = parser.parse_args()
    if args.step == "metadrive":
        check_metadrive(args.steps)
    elif args.step == "convert":
        convert_test(args.scenes, args.workers, args.set) if args.set in ("test150", "dev100") else convert(args.scenes)
    elif args.step == "plan":
        plan(args.predictor, args.scenes, args.gif, args.viz, args.checkpoint,
             args.out or OUT_DIR / f"plan_{'_'.join(args.predictor)}.json", args.data, reactive=args.reactive,
             device=args.device)
    elif args.step == "showcase":
        if "test150" in args.data:
            raise SystemExit("showcase is for development scenes only")
        showcase(Path(args.data), args.index, args.checkpoint, args.out, args.reactive == "idm", args.follow)
    elif args.step == "reactive-check":
        if any("test150" in d for d in args.data):
            raise SystemExit("reactive-check is for the mini / dev scenes only, never test")
        ov = {k: float(v) for k, v in (kv.split("=") for kv in args.idm.split(",") if kv)}
        reactive_check(args.data, args.out, ov, args.cases)
    elif args.step == "evaluate":
        if args.idm:
            from sim import closed_loop as _cl
            _cl.IDM.update({k: float(v) for k, v in (kv.split("=") for kv in args.idm.split(",") if kv)})
        evaluate_test(Path(args.data), args.checkpoint, args.out, compare=args.compare, reactive=args.reactive)
    elif args.step == "follow":
        follow(args.index)
    elif args.step == "match":
        match(args.scene, args.checkpoint, args.limit)
    elif args.step == "drive":
        drive(args.index, args.seconds, args.throttle, args.steering)
    else:
        replay(args.index, args.max_steps)
