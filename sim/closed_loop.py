"""Phase D step 4: closed-loop planner with a swappable predictor, plan costs and closed-loop metrics.

Every 0.5 s: 35 candidate plans (sim/planner.py) -> predictor -> cost per plan -> the cheapest plan is tracked
for 5 sim steps. Everyone except the ego replays the log (non-reactive), as in nuPlan's open-loop-agents setting.

Predictor interface:  predict(scene, plans) -> paths (P, A, K, 12, 2) world xy at t + 0.5 ... 6 s,
                                                probs (P, A, K), ids (A,) object ids
    scene = Scene(buffer, map_features, ego_state, step);  plans = (P, 12, 3) world (x, y, yaw)
CVPredictor:          constant speed and heading per agent, K = 1, prob 1, the same for every plan.
WorldModelPredictor:  the v2 model (e.g. run (d) epoch 28) through sim/adapter.py, K = 6, plan-dependent.

Cost per plan (lowest wins):
    col * risk + progress_w * (-progress) + comfort_w * comfort + offset_w * offset^2
    risk     = sum_agents sum_k p_k * 1[overlap at some t] / (1 + t_first), footprints of 3 circles (r = 1 m)
               along vehicles (1 circle r = 0.5 m for pedestrians / small objects), every 0.1 s;
               agents behind the ego in its lane (rear-end is the follower's job) are ignored.
    progress = distance along the route at 6 s up to the end of the logged route, minus any distance past it
               (the logged route ends where the expert stopped, so the plan should stop there too);  comfort = mean(|a|^2) + mean(|jerk|^2) of the plan rollout.
Off-road (closed loop) = the ego is not on any lane surface or touches a sidewalk.
Closed-loop max |acceleration| / |jerk| are computed from positions at the data's 2 Hz (0.5 s differences),
for the planner and the logged expert alike (the log is interpolated between 2 Hz keyframes, so 0.1 s
differences of the replayed speed show artificial spikes).
"""

from dataclasses import dataclass

import numpy as np

from sim.adapter import EGO_ID, HistoryBuffer, snapshot_from_env
from sim.planner import DT, OFFSETS, SPEEDS, Tracker, generate_plans, reference_from_scenario

WEIGHTS = dict(col=100.0, progress=1.0, comfort=0.1, offset=0.5)
VEHICLE_TYPES = ("vehicle.",)
STEP_TIMES = np.arange(1, 13) * 0.5            # model output times
DENSE_TIMES = np.arange(0, 61) * DT            # 0 ... 6 s


@dataclass
class Scene:
    buffer: HistoryBuffer
    map_features: dict
    ego: tuple            # (x, y, yaw, v)
    step: int


def agent_states(scene, radius=60.0):
    """Current (x, y, yaw, v, width, length, category) of every object within radius of the ego.
    Speed from the positions 0.5 s apart (the log interpolates between 2 Hz keyframes)."""
    now = scene.buffer.now()
    ex, ey = scene.ego[:2]
    old = scene.buffer.snapshots[-6] if len(scene.buffer.snapshots) >= 6 else None
    ids, states = [], []
    for i, s in now.items():
        if i == EGO_ID or np.hypot(s[0] - ex, s[1] - ey) > radius:
            continue
        v = 0.0
        if old is not None and i in old:
            v = float(np.hypot(s[0] - old[i][0], s[1] - old[i][1]) / 0.5)
        ids.append(i)
        states.append((s[0], s[1], s[2], v, s[3], s[4], s[5]))
    return ids, states


class CVPredictor:
    """Each agent keeps its current speed and heading; K = 1; identical for every plan."""
    name = "CV"

    def predict(self, scene, plans):
        ids, states = agent_states(scene)
        P, A = len(plans), len(ids)
        paths = np.zeros((P, A, 1, 12, 2))
        for a, (x, y, yaw, v, *_rest) in enumerate(states):
            paths[:, a, 0, :, 0] = x + v * np.cos(yaw) * STEP_TIMES
            paths[:, a, 0, :, 1] = y + v * np.sin(yaw) * STEP_TIMES
        return paths, np.ones((P, A, 1)), ids


class WorldModelPredictor:
    """The v2 model through the adapter. The target (slot 0) is the closest vehicle to the ego; the model
    predicts it and its 15 nearest neighbours under every plan (one batch, the plan as ego_future).
    Objects the model does not predict (farther away, or not vehicles / pedestrians) fall back to CV."""
    name = "WM"

    def __init__(self, checkpoint, device="cuda"):
        import json
        from pathlib import Path
        import torch
        from utils.models_v2 import V2Predictor
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        m = json.loads(Path(checkpoint).with_suffix(".json").read_text())
        self.model = V2Predictor(state["decoder.anchors"].cpu().numpy(), predict_agents=m["predict_agents"],
                                 ego_plan=True, target_split=m.get("target_split", False),
                                 plan_queries=m.get("plan_queries", False)).to(device).eval()
        self.model.load_state_dict(state)
        self.device, self.cv = device, CVPredictor()

    def predict(self, scene, plans):
        import torch
        from sim.adapter import build_sample, to_model_inputs
        from utils.lane_tokens import poses_to_local as to_local
        paths, probs, ids = self.cv.predict(scene, plans)
        self.last_wm_ids = set()                                            # ids the model predicted this call
        ids_cv, states = agent_states(scene)
        cands = [(np.hypot(s[0] - scene.ego[0], s[1] - scene.ego[1]), i) for i, s in zip(ids_cv, states)
                 if s[6].startswith(VEHICLE_TYPES)]
        if not cands:
            return np.repeat(paths, 6, axis=2), np.repeat(probs, 6, axis=2) / 6, ids
        target = min(cands)[1]
        sample = build_sample(scene.buffer, scene.map_features, target, ego_plan=plans[0])
        item = to_model_inputs(sample)
        P = len(plans)
        batch = {k: v[None].expand(P, *v.shape).contiguous().to(self.device) for k, v in item.items()}
        tx, ty, tyaw = scene.buffer.now()[target][:3]
        batch["ego_future"] = torch.as_tensor(np.stack([to_local(p, np.array([tx, ty]), tyaw) for p in plans]),
                                              dtype=torch.float32, device=self.device)
        with torch.no_grad():
            traj, logits = self.model(batch, plan=True)                     # (P, M, 6, 12, 2) target frame
        traj, pr = traj.cpu().numpy(), logits.softmax(-1).cpu().numpy()
        # slot -> object id, with the dataset's ordering (target, ego, neighbours by distance to the target)
        order_ids = sample["agent_instance_tokens"]
        dist = np.linalg.norm(sample["agent_past"][1:, -1, :2], axis=-1)
        neighbours = [order_ids[1 + j] for j in np.argsort(dist, kind="stable")][:62]
        slot_ids = {0: target, **{2 + j: n for j, n in enumerate(neighbours)}}
        c, s = np.cos(tyaw), np.sin(tyaw)
        world = np.stack([tx + c * traj[..., 0] - s * traj[..., 1], ty + s * traj[..., 0] + c * traj[..., 1]], -1)
        K = traj.shape[2]
        out_paths = np.repeat(paths, K, axis=2)                             # CV copies for everyone ...
        out_probs = np.repeat(probs, K, axis=2) / K
        index = {i: a for a, i in enumerate(ids)}
        for m_, slot in enumerate(self.model.slots):                        # ... replaced where the model predicts
            oid = slot_ids.get(slot)
            if oid in index and item["agent_mask"][slot]:
                out_paths[:, index[oid]] = world[:, m_]
                out_probs[:, index[oid]] = pr[:, m_]
                self.last_wm_ids.add(oid)
        return out_paths, out_probs, ids


def footprint(xy, yaw, length, width, vehicle):
    """Circle centres (..., C, 2) and radius: 3 circles along a vehicle, 1 for anything else."""
    if not vehicle:
        return xy[..., None, :], 0.5
    offs = np.array([-1.0, 0.0, 1.0]) * max(length / 2 - 1.0, 0.0)
    d = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    return xy[..., None, :] + offs[:, None] * d[..., None, :], 1.0


def densify(paths, current):
    """(P, A, K, 12, 2) at 0.5 s steps + current (A, 2) -> (P, A, K, 61, 2) at 0.1 s, and headings."""
    full = np.concatenate([np.broadcast_to(current[None, :, None, None], paths.shape[:3] + (1, 2)), paths], axis=3)
    t_in = np.concatenate([[0.0], STEP_TIMES])
    idx = np.clip(np.searchsorted(t_in, DENSE_TIMES, side="right") - 1, 0, len(t_in) - 2)
    w = ((DENSE_TIMES - t_in[idx]) / 0.5)[:, None]
    dense = full[..., idx, :] * (1 - w) + full[..., idx + 1, :] * w
    d = np.diff(dense, axis=3, append=dense[..., -1:, :] + (dense[..., -1:, :] - dense[..., -2:-1, :]))
    return dense, np.arctan2(d[..., 1], d[..., 0])


def plan_costs(scene, ref, dense_plans, meta, paths, probs, ids, ego_size, weights=WEIGHTS):
    """Cost terms per plan; dense_plans (P, 61, 4) world (x, y, yaw, v) from generate_plans."""
    P = len(dense_plans)
    _, states = agent_states(scene)
    by_id = dict(zip(*agent_states(scene)))
    ex, ey, eyaw = scene.ego[:3]
    c, s = np.cos(eyaw), np.sin(eyaw)
    keep = []
    for a, i in enumerate(ids):
        x, y, yaw = by_id[i][:3]
        xr, yr = c * (x - ex) + s * (y - ey), -s * (x - ex) + c * (y - ey)
        same_dir = np.cos(yaw - eyaw) > np.cos(np.radians(30))
        keep.append(not (xr < -1.0 and abs(yr) < 1.5 and same_dir))          # behind in the same lane -> ignore
    keep = np.asarray(keep, bool)

    risk = np.zeros(P)
    if len(ids) and keep.any():
        current = np.array([by_id[i][:2] for i in ids])
        dense, heading = densify(paths, current)                            # (P, A, K, 61, 2)
        ego_c, ego_r = footprint(dense_plans[:, :, :2], dense_plans[:, :, 2], ego_size[1], ego_size[0], True)
        for a in np.flatnonzero(keep):
            x, y, yaw, v, w_, l_, cat = by_id[ids[a]]
            veh = cat.startswith(VEHICLE_TYPES) or cat in ("VEHICLE", "CYCLIST")
            ag_c, ag_r = footprint(dense[:, a], heading[:, a], l_, w_, veh)  # (P, K, 61, C, 2)
            dist = np.linalg.norm(ego_c[:, None, :, :, None, :] - ag_c[:, :, :, None, :, :], axis=-1)
            hit = (dist.min(axis=(-1, -2)) < ego_r + ag_r)                  # (P, K, 61)
            first = np.where(hit.any(-1), hit.argmax(-1) * DT, np.inf)
            risk += (probs[:, a] * np.where(np.isfinite(first), 1.0 / (1.0 + first), 0.0)).sum(-1)

    s_now = min(ref.project(np.array(scene.ego[:2]))[0], ref.length)
    s_end = np.array([ref.project(p[-1, :2])[0] for p in dense_plans])
    progress = np.minimum(s_end, ref.length) - s_now - np.maximum(s_end - ref.length, 0.0)
    v = dense_plans[:, :, 3]
    acc_long = np.diff(v, axis=1) / DT
    yaw_rate = np.diff(np.unwrap(dense_plans[:, :, 2], axis=1), axis=1) / DT
    acc = np.hypot(acc_long, v[:, 1:] * yaw_rate)
    jerk = np.diff(acc_long, axis=1) / DT
    comfort = (acc ** 2).mean(1) + (jerk ** 2).mean(1)
    offset2 = np.array([m[1] for m in meta]) ** 2
    total = weights["col"] * risk - weights["progress"] * progress + weights["comfort"] * comfort + \
        weights["offset"] * offset2
    return total, dict(risk=risk, progress=progress, comfort=comfort, offset2=offset2)


def box(x, y, yaw, length, width):
    from shapely.geometry import Polygon
    c, s = np.cos(yaw), np.sin(yaw)
    pts = [(dx * length / 2, dy * width / 2) for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in pts])


def collisions(ego_pose, ego_size, snapshot):
    """Objects whose logged box overlaps the ego box now; each tagged at-fault or not.
    Not at fault (as in nuPlan): the other object touches the ego from behind (its centre is behind the ego's
    rear axle region, in roughly the same direction)."""
    ex, ey, eyaw = ego_pose
    eb = box(ex, ey, eyaw, ego_size[1], ego_size[0])
    c, s = np.cos(eyaw), np.sin(eyaw)
    out = []
    for i, st in snapshot.items():
        if i == EGO_ID:
            continue
        x, y, yaw, w_, l_, cat = st
        if np.hypot(x - ex, y - ey) > 15 or not eb.intersects(box(x, y, yaw, l_, w_)):
            continue
        xr = c * (x - ex) + s * (y - ey)
        rear = xr < -ego_size[1] / 4 and np.cos(yaw - eyaw) > 0
        out.append((i, cat, not rear))
    return out


AGENT_PREFIXES = ("vehicle.", "human.pedestrian")


def run_scene(env, predictor, weights=WEIGHTS, gif=None, expert=False, replan_steps=5, shadow=None, viz=None,
              nearby_radius=30.0):
    """One closed-loop episode (or the logged ego with expert=True). Returns metrics and the chosen-plan log.

    shadow: a second predictor run on the chosen plan only (e.g. the WM during a CV run), for diagnostics.
    viz: GIF path for the fixed-camera planning view (one frame per planning step)."""
    import time
    from sim.adapter import add_connector_polygons
    scenario = env.engine.data_manager.current_scenario
    features = add_connector_polygons(scenario["map_features"], scenario["metadata"]["map"])
    ref = reference_from_scenario(scenario)
    ego_size = (float(env.agent.WIDTH), float(env.agent.LENGTH))
    buffer, tracker = HistoryBuffer(), Tracker()
    buffer.push(snapshot_from_env(env))
    steps = scenario["length"] - 1
    poses, speeds, offroad, md_crash, frames, chosen_log, hits, t_plan = [], [], 0, 0, [], [], {}, []
    pred_log, coverage, viz_frames = [], [], []
    plan, plan_t = None, 0.0
    wm = predictor if getattr(predictor, "name", "") == "WM" else shadow
    for step in range(steps):
        a = env.agent
        x, y, yaw, v = float(a.position[0]), float(a.position[1]), float(a.heading_theta), float(a.speed)
        if expert:
            act = [0.0, 0.0]
        else:
            if step % replan_steps == 0:
                t0 = time.perf_counter()
                scene = Scene(buffer, features, (x, y, yaw, v), step)
                sparse, dense, meta = generate_plans(ref, (x, y, yaw, v))
                paths, probs, ids = predictor.predict(scene, sparse)
                total, terms = plan_costs(scene, ref, dense, meta, paths, probs, ids, ego_size, weights)
                k = int(np.argmin(total))
                plan, plan_t = dense[k], 0.0
                t_plan.append(1000 * (time.perf_counter() - t0))
                entry = dict(t=step / 10, speed=meta[k][0], offset=meta[k][1], total=float(total[k]),
                             **{n: float(t_[k]) for n, t_ in terms.items()})
                chosen_log.append(entry)
                # WM predictions for the chosen plan (own, or shadow during a CV run) + coverage of nearby agents
                if wm is not None:
                    if wm is predictor:
                        w_paths, w_probs, w_ids = paths[k], probs[k], ids
                    else:
                        wp, wpr, w_ids = wm.predict(scene, sparse[k:k + 1])
                        w_paths, w_probs = wp[0], wpr[0]
                    now = buffer.now()
                    near = [i for i in w_ids if now[i][5].startswith(AGENT_PREFIXES)
                            and np.hypot(now[i][0] - x, now[i][1] - y) <= nearby_radius]
                    coverage.append((len(near), sum(i in wm.last_wm_ids for i in near)))
                    pred_log.append(dict(step=step, ids=list(w_ids), paths=w_paths, probs=w_probs,
                                         wm_ids=set(wm.last_wm_ids), plan=dense[k].copy()))
                if viz is not None:
                    viz_frames.append(render_planning_frame(scene, ref, dense, k, paths, probs, ids, entry,
                                                            predictor.name, features))
            act = tracker.action(x, y, yaw, v, plan, plan_t)
        _, _, _, _, info = env.step(act)
        plan_t += DT
        buffer.push(snapshot_from_env(env))
        a = env.agent
        pose = (float(a.position[0]), float(a.position[1]), float(a.heading_theta))
        poses.append(pose)
        speeds.append(float(a.speed))
        # off-road = not on any lane surface (lanes + connectors) or on a sidewalk. MetaDrive's own out_of_road
        # is "> 4 m from the navigation route or route_completion < -0.1", i.e. route deviation.
        offroad += bool((not a.on_lane) or a.crash_sidewalk)
        md_crash += bool(info.get("crash"))
        for oid, cat, fault in collisions(pose, ego_size, buffer.now()):
            hits.setdefault(oid, (cat, fault, step + 1))
        if gif is not None and step % 2 == 0:
            frame = env.render(mode="topdown", window=False, screen_size=(700, 700), film_size=(3000, 3000),
                               scaling=4, draw_target_vehicle_trajectory=True)
            frames.append(np.swapaxes(np.asarray(frame), 0, 1))
    poses, speeds = np.asarray(poses), np.asarray(speeds)
    s_end, _ = ref.project(poses[-1, :2])
    p2 = poses[::replan_steps, :2]                              # 2 Hz positions
    v2 = np.linalg.norm(np.diff(p2, axis=0), axis=1) / 0.5
    acc = np.diff(v2) / 0.5 if len(v2) > 1 else np.zeros(1)
    jerk = np.diff(acc) / 0.5 if len(acc) > 1 else np.zeros(1)
    collided = []
    for oid, (cat, fault, t) in hits.items():
        c = dict(object=oid[:8], category=cat, at_fault=fault, t=t / 10)
        if fault and pred_log:
            c.update(wm_prediction_before(pred_log, oid, t, scenario, ego_size))
        collided.append(c)
    metrics = dict(
        scene=scenario["metadata"]["scenario_id"], steps=steps,
        at_fault=sum(f for _, f, _ in hits.values()), not_at_fault=sum(not f for _, f, _ in hits.values()),
        collided=collided, metadrive_crash_steps=md_crash, offroad_steps=offroad,
        progress_ratio=float(min(s_end, ref.length) / ref.length) if ref.length >= 1.0 else float("nan"),
        overshoot_m=float(max(s_end - ref.length, 0.0)), driven_m=float(s_end), logged_m=float(ref.length),
        max_acc=float(np.abs(acc).max()), max_jerk=float(np.abs(jerk).max()),
        plan_ms=float(np.median(t_plan)) if t_plan else 0.0,
        plan_ms_p95=float(np.percentile(t_plan, 95)) if t_plan else 0.0,
        nearby_agents=int(sum(c[0] for c in coverage)), nearby_wm=int(sum(c[1] for c in coverage)))
    if gif is not None and frames:
        save_gif(frames, gif, 100)
    if viz is not None and viz_frames:
        save_gif(viz_frames, viz, 400)
    return metrics, chosen_log


def save_gif(frames, path, ms):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0)


def wm_prediction_before(pred_log, oid, hit_step, scenario, ego_size):
    """What the WM predicted for object oid at the last planning step before the collision."""
    before = [p for p in pred_log if p["step"] < hit_step]
    if not before:
        return {}
    p = before[-1]
    dt = (hit_step - p["step"]) / 10
    out = dict(wm_pred_t=p["step"] / 10, wm_covered=oid in p["wm_ids"], wm_horizon_s=dt)
    if oid not in p["ids"]:
        return {**out, "wm_note": "not within 60 m at that step"}
    a = p["ids"].index(oid)
    paths, probs = p["paths"][a], p["probs"][a]                            # (K, 12, 2), (K,)
    st = scenario["tracks"][oid]["state"]
    actual = np.asarray(st["position"])[hit_step, :2]
    t_in = np.concatenate([[0.0], STEP_TIMES])
    cur = np.asarray(st["position"])[p["step"], :2]
    full = np.concatenate([np.broadcast_to(cur, (len(paths), 1, 2)), paths], 1)
    at_t = np.stack([np.interp(dt, t_in, full[k, :, 0]) for k in range(len(paths))]), \
        np.stack([np.interp(dt, t_in, full[k, :, 1]) for k in range(len(paths))])
    at_t = np.stack(at_t, -1)                                               # (K, 2)
    err = np.linalg.norm(at_t - actual, axis=1)
    ego = p["plan"][min(int(round(dt * 10)), len(p["plan"]) - 1), :2]
    top = int(np.argmax(probs))
    return {**out, "wm_top_prob": float(probs[top]), "wm_err_top_m": float(err[top]),
            "wm_err_best_m": float(err.min()), "wm_prob_best": float(probs[int(err.argmin())]),
            "wm_pred_dist_to_ego_plan_m": float(np.linalg.norm(at_t[top] - ego)),
            "wm_actual_dist_to_ego_plan_m": float(np.linalg.norm(actual - ego))}


def render_planning_frame(scene, ref, dense, k, paths, probs, ids, entry, name, features, radius=40.0):
    """Fixed camera on the scene's route; lanes, reference route (grey), 35 plans (thin), chosen plan (thick
    green), current agents, predicted futures of nearby agents (opacity = prob), text box with the cost terms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    route = ref.xy[ref.s <= max(ref.length, 1.0)]
    lo, hi = route.min(0) - 30, route.max(0) + 30
    span = (hi - lo).max()
    mid = (lo + hi) / 2
    lo, hi = mid - span / 2, mid + span / 2
    fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
    for f in features.values():
        if f["type"].startswith("LANE_SURFACE"):
            pl = np.asarray(f["polyline"])[:, :2]
            if ((pl > lo) & (pl < hi)).all(1).any():
                ax.plot(pl[:, 0], pl[:, 1], color="0.88", lw=0.6, zorder=1)
    ax.plot(route[:, 0], route[:, 1], color="0.55", lw=6, alpha=0.45, solid_capstyle="round", zorder=2)
    for j in range(len(dense)):
        ax.plot(dense[j, :, 0], dense[j, :, 1], color="tab:blue", lw=0.5, alpha=0.35, zorder=3)
    ax.plot(dense[k, :, 0], dense[k, :, 1], color="tab:green", lw=3.5, zorder=6)
    now = scene.buffer.now()
    ex, ey = scene.ego[:2]
    for a, i in enumerate(ids):
        x, y, yaw, w_, l_, cat = now[i]
        veh = cat.startswith("vehicle.")
        color = "tab:red" if veh else ("tab:orange" if cat.startswith("human") else "0.5")
        b = box(x, y, yaw, max(l_, 0.5), max(w_, 0.5))
        ax.fill(*b.exterior.xy, color=color, alpha=0.8 if veh else 0.6, lw=0, zorder=5)
        if np.hypot(x - ex, y - ey) <= radius and cat.startswith(AGENT_PREFIXES):
            for m_ in range(paths.shape[2]):
                pr = float(probs[k, a, m_])
                if pr > 0.02:
                    pth = np.vstack([[x, y], paths[k, a, m_]])
                    ax.plot(pth[:, 0], pth[:, 1], color=color, lw=1.4, alpha=min(1.0, 0.1 + pr), zorder=4)
    eb = box(ex, ey, scene.ego[2], 4.5, 1.85)
    ax.fill(*eb.exterior.xy, color="tab:green", zorder=7)
    txt = (f"{name} planner   t = {entry['t']:.1f} s   ego {scene.ego[3]:.1f} m/s\n"
           f"chosen: speed {entry['speed']:.0f} m/s, offset {entry['offset']:+.2f} m\n"
           f"risk {entry['risk']:.2f}  progress {entry['progress']:.1f} m  comfort {entry['comfort']:.1f}  "
           f"offset² {entry['offset2']:.2f}\ntotal {entry['total']:.1f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"), zorder=10)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
