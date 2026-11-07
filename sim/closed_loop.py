"""Phase D step 4: closed-loop planner with a swappable predictor, plan costs and closed-loop metrics.

Every 0.5 s: 35 candidate plans (sim/planner.py) -> predictor -> cost per plan -> the cheapest plan is tracked
for 5 sim steps. Everyone except the ego replays the log (non-reactive), as in nuPlan's open-loop-agents setting.

Predictor interface:  predict(scene, plans) -> paths (P, A, K, 12, 2) world xy at t + 0.5 ... 6 s,
                                                probs (P, A, K), ids (A,) object ids
    scene = Scene(buffer, map_features, ego_state, step);  plans = (P, 12, 3) world (x, y, yaw)
CVPredictor:          constant speed and heading per agent, K = 1, prob 1, the same for every plan.
WorldModelPredictor:  the v2 model (e.g. run (d) epoch 28) through sim/adapter.py, K = 6, plan-dependent.

Cost per plan (lowest wins), weights FROZEN on 2026-11-06 after the 10-scene mini run (WEIGHTS, VETO):
    col * risk + progress_w * (-progress) + comfort_w * comfort + offset_w * offset^2
    hard veto: a plan whose risk from collisions in the first VETO["horizon_s"] = 2 s exceeds VETO["risk"] = 0.3
    is not allowed (if every plan is vetoed, the one with the lowest early risk is taken).
    risk     = sum_agents sum_k p_k * 1[overlap at some t] / (1 + t_first), every 0.1 s; footprints from each
               object's real size (and the ego's): ceil(length / width) circles that fully cover the box,
               r = sqrt((w/2)^2 + (L/(2n))^2) (circle_footprint);
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

from sim.adapter import EGO_ID, HistoryBuffer, snapshot_from_env, snapshot_from_scenario
from sim.planner import DT, OFFSETS, SPEEDS, Tracker, generate_plans, reference_from_scenario

WEIGHTS = dict(col=500.0, progress=1.0, comfort=0.1, offset=0.5)       # frozen
VETO = dict(risk=0.3, horizon_s=2.0)                                     # frozen
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
        if getattr(self, "_selector_for", None) is not scene.map_features:      # one LaneSelector per scene
            from sim.adapter import LaneSelector
            self.selector, self._selector_for = LaneSelector(scene.map_features), scene.map_features
        sample = build_sample(scene.buffer, scene.map_features, target, ego_plan=plans[0], selector=self.selector)
        item = to_model_inputs(sample, order="ego")                         # slots 2..16 = nearest to the ego
        P = len(plans)
        batch = {k: v.to(self.device)[None].expand(P, *v.shape) for k, v in item.items()}   # one copy, expanded on GPU
        tx, ty, tyaw = scene.buffer.now()[target][:3]
        batch["ego_future"] = torch.as_tensor(np.stack([to_local(p, np.array([tx, ty]), tyaw) for p in plans]),
                                              dtype=torch.float32, device=self.device)
        with torch.no_grad():
            traj, logits = self.model(batch, plan=True)                     # (P, M, 6, 12, 2) target frame
        traj, pr = traj.cpu().numpy(), logits.softmax(-1).cpu().numpy()
        # slot -> object id (target, ego, then neighbours by distance to the ego, as to_model_inputs(order="ego"))
        from sim.adapter import ego_order
        order_ids = sample["agent_instance_tokens"]
        neighbours = [order_ids[j] for j in ego_order(sample)][:62]
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


def circle_footprint(length, width):
    """n = ceil(length / width) circles that fully cover a length x width box: the box is cut into n equal slices
    along its length, each slice gets the circle through its 4 corners, r = sqrt((w / 2)^2 + (L / (2n))^2), centred
    at the slice centre. Returns (offsets (n,) along the length, radius)."""
    width = max(float(width), 0.1)
    length = max(float(length), 0.1)
    n = max(1, int(np.ceil(length / width - 1e-9)))
    r = float(np.hypot(width / 2, length / (2 * n)))
    return -length / 2 + length / (2 * n) + np.arange(n) * length / n, r


def collision_risk(dense_plans, paths, probs, current, yaw, sizes, ego_size, device, chunk=32):
    """Vectorised (torch) collision risk per plan.

    dense_plans (P, 61, 4) ego (x, y, yaw, v) every 0.1 s; paths (P, A, K, 12, 2) at 0.5 ... 6 s; probs (P, A, K);
    current (A, 2) and yaw (A,) now; sizes (A, 2) (width, length).
    Footprints: circle_footprint() of each object's real size and of the ego; agent headings from the predicted
    motion, or the current heading while (nearly) standing still.
    Returns risk (P,) = sum_a sum_k p_k / (1 + t_first), early (P,) = the same counting only t_first <= 2 s,
    and per_agent (P, A) = each agent's share of risk."""
    import torch
    f32 = dict(device=device, dtype=torch.float32)
    P, A, K = probs.shape
    risk, early = torch.zeros(P, **f32), torch.zeros(P, **f32)
    per_agent = torch.zeros(P, A, **f32)
    if A == 0:
        return risk.cpu().numpy(), early.cpu().numpy(), per_agent.cpu().numpy()
    plan = torch.as_tensor(dense_plans, **f32)
    ed = torch.stack([torch.cos(plan[..., 2]), torch.sin(plan[..., 2])], -1)              # (P, 61, 2)
    e_off_np, ego_r = circle_footprint(ego_size[1], ego_size[0])
    e_off = torch.as_tensor(e_off_np, **f32)
    ego_c = plan[..., None, :2] + e_off[:, None] * ed[..., None, :]                         # (P, 61, Ce, 2)
    t_in = torch.arange(13, **f32) * 0.5
    t_out = torch.arange(61, **f32) * DT
    i0 = torch.clamp(torch.searchsorted(t_in, t_out, right=True) - 1, 0, 11)
    w = ((t_out - t_in[i0]) / 0.5)[:, None]
    times = t_out
    for lo in range(0, A, chunk):
        sl = slice(lo, min(A, lo + chunk))
        pa = torch.as_tensor(paths[:, sl], **f32)                                          # (P, a, K, 12, 2)
        cur = torch.as_tensor(current[sl], **f32)
        full = torch.cat([cur[None, :, None, None].expand(P, -1, K, 1, 2), pa], 3)         # (P, a, K, 13, 2)
        dense = full[..., i0, :] * (1 - w) + full[..., i0 + 1, :] * w                      # (P, a, K, 61, 2)
        step = torch.diff(dense, dim=3, append=2 * dense[..., -1:, :] - dense[..., -2:-1, :])
        moving = step.norm(dim=-1) > 0.02                                                  # > 0.2 m/s
        head = torch.where(moving, torch.atan2(step[..., 1], step[..., 0]),
                           torch.as_tensor(yaw[sl], **f32)[None, :, None, None])
        fps = [circle_footprint(l_, w_) for w_, l_ in sizes[sl]]
        C = max(len(o) for o, _ in fps)
        offs = np.zeros((len(fps), C))
        valid = np.zeros((len(fps), C), bool)
        for j, (o, _) in enumerate(fps):
            offs[j, :len(o)], valid[j, :len(o)] = o, True
        a_off = torch.as_tensor(offs, **f32)                                                 # (a, C)
        a_valid = torch.as_tensor(valid, device=device)
        r = torch.as_tensor([rad for _, rad in fps], **f32) + ego_r                          # (a,)
        ad = torch.stack([torch.cos(head), torch.sin(head)], -1)                             # (P, a, K, 61, 2)
        ag_c = dense[..., None, :] + a_off[None, :, None, None, :, None] * ad[..., None, :]  # (P, a, K, 61, C, 2)
        d = (ego_c[:, None, None, :, :, None, :] - ag_c[..., None, :, :]).norm(dim=-1)       # (P, a, K, 61, Ce, C)
        d = d.masked_fill(~a_valid[None, :, None, None, None, :], float("inf"))
        hit = d.amin(dim=(-1, -2)) < r[None, :, None, None]                                  # (P, a, K, 61)
        any_hit = hit.any(-1)
        t_first = torch.where(any_hit, times[hit.float().argmax(-1)], torch.full_like(any_hit, 1e9, dtype=torch.float32))
        pr = torch.as_tensor(probs[:, sl], **f32)
        contrib = pr * torch.where(any_hit, 1.0 / (1.0 + t_first), torch.zeros_like(t_first))
        risk += contrib.sum((1, 2))
        early += (contrib * (t_first <= VETO["horizon_s"])).sum((1, 2))
        per_agent[:, sl] = contrib.sum(2)
    return risk.cpu().numpy(), early.cpu().numpy(), per_agent.cpu().numpy()


def plan_costs(scene, ref, dense_plans, meta, paths, probs, ids, ego_size, weights=WEIGHTS, device="cpu"):
    """Cost terms per plan; dense_plans (P, 61, 4) world (x, y, yaw, v) from generate_plans."""
    P = len(dense_plans)
    by_id = dict(zip(*agent_states(scene)))
    ex, ey, eyaw = scene.ego[:3]
    c, s = np.cos(eyaw), np.sin(eyaw)
    keep = []
    for i in ids:
        x, y, yaw = by_id[i][:3]
        xr, yr = c * (x - ex) + s * (y - ey), -s * (x - ex) + c * (y - ey)
        same_dir = np.cos(yaw - eyaw) > np.cos(np.radians(30))
        keep.append(not (xr < -1.0 and abs(yr) < 1.5 and same_dir))          # behind in the same lane -> ignore
    keep = np.asarray(keep, bool)
    risk, early, agent_risk = np.zeros(P), np.zeros(P), {}
    if keep.any():
        kid = [i for i, k_ in zip(ids, keep) if k_]
        st = [by_id[i] for i in kid]
        risk, early, per_agent = collision_risk(
            dense_plans, paths[:, keep], probs[:, keep], np.array([q[:2] for q in st]), np.array([q[2] for q in st]),
            np.array([q[4:6] for q in st]), ego_size, device)
        agent_risk = {i: per_agent[:, a] for a, i in enumerate(kid)}

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
    vetoed = early > VETO["risk"]
    if vetoed.all():                                  # nothing is allowed: take the least risky in the first 2 s
        total = np.where(early == early.min(), total, np.inf)
    else:
        total = np.where(vetoed, np.inf, total)
    return total, dict(risk=risk, early_risk=early, progress=progress, comfort=comfort, offset2=offset2,
                       vetoed=vetoed.astype(float)), agent_risk


def box(x, y, yaw, length, width):
    from shapely.geometry import Polygon
    c, s = np.cos(yaw), np.sin(yaw)
    pts = [(dx * length / 2, dy * width / 2) for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in pts])


EGO_SIZE_ZOE = (1.73, 4.08)        # (width, length) of the nuScenes ego (Renault Zoe), as in utils/dataset_v2.py
STOPPED_SPEED = 0.5                 # m/s: a collision while the ego is (nearly) stopped is not at fault


def collisions(ego_pose, ego_size, snapshot, ego_speed=None):
    """Objects whose logged box overlaps the ego box now; each tagged at-fault or not.
    Not at fault (as in nuPlan): the other object touches the ego from behind (its centre is behind the ego's
    rear axle region, in roughly the same direction), or the ego is stopped (speed < STOPPED_SPEED)."""
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
        stopped = ego_speed is not None and ego_speed < STOPPED_SPEED
        out.append((i, cat, not (rear or stopped), rear))
    return out


def near_miss(ego_pose, ego_vel, ego_size, snapshot, old_snapshot, radius=30.0, horizon=5.0):
    """Per step: min box-to-box distance to any agent (vehicles / pedestrians / cyclists) and min time to
    collision. TTC: ego and agents move on at their current velocity (agents: from the log 0.5 s apart) with
    the planner's footprints (circle_footprint of the real sizes), first overlap within `horizon` s; agents behind
    the ego in its lane are skipped (rear-end is the follower's job)."""
    ex, ey, eyaw = ego_pose
    eb = box(ex, ey, eyaw, ego_size[1], ego_size[0])
    c, s_ = np.cos(eyaw), np.sin(eyaw)
    min_d, ttc = np.inf, np.inf
    t = np.arange(0, horizon + 1e-9, DT)
    e_dir = np.array([c, s_])
    e_off, e_r = circle_footprint(ego_size[1], ego_size[0])
    ego_c = (np.array([ex, ey]) + ego_vel * t[:, None])[:, None] + e_off[None, :, None] * e_dir   # (T, Ce, 2)
    for i, st in snapshot.items():
        if i == EGO_ID:
            continue
        x, y, yaw, w_, l_, cat = st
        if not (cat.startswith(AGENT_PREFIXES) or cat in ("VEHICLE", "CYCLIST", "PEDESTRIAN")):
            continue
        if np.hypot(x - ex, y - ey) > radius:
            continue
        min_d = min(min_d, eb.distance(box(x, y, yaw, max(l_, 0.3), max(w_, 0.3))))
        xr, yr = c * (x - ex) + s_ * (y - ey), -s_ * (x - ex) + c * (y - ey)
        if xr < -1.0 and abs(yr) < 1.5 and np.cos(yaw - eyaw) > np.cos(np.radians(30)):
            continue
        v = np.zeros(2) if (old_snapshot is None or i not in old_snapshot) else \
            (np.array([x, y]) - np.array(old_snapshot[i][:2])) / 0.5
        a_off, a_r = circle_footprint(l_, w_)
        ag_c = (np.array([x, y]) + v * t[:, None])[:, None] + a_off[None, :, None] * np.array([np.cos(yaw), np.sin(yaw)])
        d = np.linalg.norm(ego_c[:, :, None] - ag_c[:, None], axis=-1).min(axis=(1, 2))           # (T,)
        hit = np.flatnonzero(d < e_r + a_r)
        if len(hit):
            ttc = min(ttc, t[hit[0]])
    return min_d, ttc


AGENT_PREFIXES = ("vehicle.", "human.pedestrian")


CUT_IN_S = 2.0      # reactive mode: hit from behind within 2 s after the ego entered the agent's path = unsafe cut-in


def run_scene(env, predictor, weights=WEIGHTS, gif=None, expert=False, replan_steps=5, shadow=None, viz=None,
              nearby_radius=30.0, device="cuda", reactive=False):
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
    rt = ReactiveTraffic(env) if reactive else None
    buffer.push(snapshot_from_env(env, rt))
    r_err, r_pairs, r_log_pairs, r_standoff = {}, set(), set(), 0
    steps = scenario["length"] - 1
    poses, speeds, offroad, md_crash, frames, chosen_log, hits, t_plan = [], [], 0, 0, [], [], {}, []
    pred_log, coverage, viz_frames, t_parts, near_d, near_ttc, track_err = [], [], [], [], [], [], []
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
                t_pred = time.perf_counter()
                total, terms, agent_risk = plan_costs(scene, ref, dense, meta, paths, probs, ids, ego_size, weights,
                                                      device)
                k = int(np.argmin(total))
                plan, plan_t = dense[k], 0.0
                t_plan.append(1000 * (time.perf_counter() - t0))
                t_parts.append(dict(predict=t_pred - t0, cost=time.perf_counter() - t_pred))
                entry = dict(t=step / 10, speed=meta[k][0], offset=meta[k][1], total=float(total[k]),
                             n_vetoed=int(terms["vetoed"].sum()), min_risk_any=float(terms["risk"].min()),
                             n_zero_risk=int((terms["risk"] < 1e-6).sum()),
                             agent_risk_chosen={i: float(r[k]) for i, r in agent_risk.items() if r[k] > 1e-6},
                             **{n: float(t_[k]) for n, t_ in terms.items() if n != "vetoed"})
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
        if rt is not None:                               # advance the reactive agents to the new step
            pre = snapshot_from_env(env, rt)
            old_r = buffer.snapshots[-5] if len(buffer.snapshots) >= 5 else None
            rt.step(pre, old_r, np.asarray(env.agent.velocity[:2], float))
        if not expert:                                   # tracking error: driven vs planned position at the same time
            j = min(int(round(plan_t / DT)), len(plan) - 1)
            dxy = np.asarray(env.agent.position[:2]) - plan[j, :2]
            ph = plan[j, 2]
            track_err.append((float(np.hypot(*dxy)), float(-np.sin(ph) * dxy[0] + np.cos(ph) * dxy[1])))
        buffer.push(snapshot_from_env(env, rt))
        a = env.agent
        pose = (float(a.position[0]), float(a.position[1]), float(a.heading_theta))
        if rt is not None:                               # validation stats of the reactive agents
            t_now = int(env.episode_step)
            log_now = snapshot_from_scenario(scenario, t_now)
            for ag in rt.agents.values():
                if not ag.active or not (ag.t0 <= t_now < ag.t1):
                    continue
                x_, y_, _ = ag.pose()
                lx, ly = log_now[ag.oid][:2] if ag.oid in log_now else (x_, y_)
                s_log_now, _ = ag.log_state(t_now)
                r_err.setdefault(ag.oid, []).append((round((t_now - ag.t0) * DT, 1), float(np.hypot(x_ - lx, y_ - ly)),
                                                     float(abs(ag.s - s_log_now))))
                bx = box(x_, y_, ag.pose()[2], ag.length, ag.width)
                for oid2, st2 in buffer.now().items():
                    if oid2 in (EGO_ID, ag.oid) or not st2[5].startswith(("vehicle.",)) or \
                            np.hypot(st2[0] - x_, st2[1] - y_) > 15:
                        continue
                    if bx.intersects(box(st2[0], st2[1], st2[2], st2[4], st2[3])):
                        pair = tuple(sorted((ag.oid, oid2)))
                        l1, l2 = log_now.get(pair[0]), log_now.get(pair[1])
                        in_log = l1 is not None and l2 is not None and box(l1[0], l1[1], l1[2], l1[4], l1[3]).intersects(
                            box(l2[0], l2[1], l2[2], l2[4], l2[3]))
                        (r_log_pairs if in_log else r_pairs).add(pair)
            r_standoff = max([r_standoff] + list(rt.deadlock_steps.values()))
        poses.append(pose)
        speeds.append(float(a.speed))
        # off-road = not on any lane surface (lanes + connectors) or on a sidewalk. MetaDrive's own out_of_road
        # is "> 4 m from the navigation route or route_completion < -0.1", i.e. route deviation.
        offroad += bool((not a.on_lane) or a.crash_sidewalk)
        md_crash += bool(info.get("crash"))
        back = poses[max(0, len(poses) - 1 - replan_steps)]
        n_back = min(replan_steps, len(poses) - 1)
        ego_vel = (np.array(pose[:2]) - np.array(back[:2])) / (n_back * DT) if n_back else np.zeros(2)
        for oid, cat, fault, rear in collisions(pose, ego_size, buffer.now(), float(np.linalg.norm(ego_vel))):
            cut_in = False
            if rt is not None and rear and oid in rt.agents and rt.agents[oid].ego_on_path_since is not None:
                cut_in = (int(env.episode_step) - rt.agents[oid].ego_on_path_since) * DT <= CUT_IN_S
            hits.setdefault(oid, (cat, fault or cut_in, step + 1, cut_in))
        old = buffer.snapshots[-6] if len(buffer.snapshots) >= 6 else None
        nm_d, nm_ttc = near_miss(pose, ego_vel, ego_size, buffer.now(), old)
        near_d.append(nm_d)
        near_ttc.append(nm_ttc)
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
    for oid, (cat, fault, t, cut_in) in hits.items():
        kind = classify_collision(t / 10, fault)
        c = dict(object=oid[:8], object_id=oid, category=cat, at_fault=kind == "at_fault", kind=kind, t=t / 10,
                 unsafe_cut_in=bool(cut_in and kind == "at_fault"))
        if kind == "at_fault":
            if pred_log:
                c.update(wm_prediction_before(pred_log, oid, t, scenario, ego_size))
            c.update(diagnose_at_fault(chosen_log, oid, t / 10))
        collided.append(c)
    metrics = dict(
        scene=scenario["metadata"]["scenario_id"], map=scenario["metadata"].get("map", ""), steps=steps,
        at_fault=sum(c["kind"] == "at_fault" for c in collided),
        not_at_fault=sum(c["kind"] == "not_at_fault" for c in collided),
        initial_overlap=sum(c["kind"] == "initial_overlap" for c in collided),
        collided=collided, metadrive_crash_steps=md_crash, offroad_steps=offroad,
        progress_ratio=float(min(s_end, ref.length) / ref.length) if ref.length >= 1.0 else float("nan"),
        overshoot_m=float(max(s_end - ref.length, 0.0)), driven_m=float(s_end), logged_m=float(ref.length),
        max_acc=float(np.abs(acc).max()), max_jerk=float(np.abs(jerk).max()),
        plan_ms=float(np.median(t_plan)) if t_plan else 0.0,
        plan_ms_p95=float(np.percentile(t_plan, 95)) if t_plan else 0.0,
        predict_ms=float(1000 * np.median([q["predict"] for q in t_parts])) if t_parts else 0.0,
        cost_ms=float(1000 * np.median([q["cost"] for q in t_parts])) if t_parts else 0.0,
        nearby_agents=int(sum(c[0] for c in coverage)), nearby_wm=int(sum(c[1] for c in coverage)),
        min_dist_m=float(np.min(near_d)) if near_d else float("inf"),
        min_ttc_s=float(np.min(near_ttc)) if near_ttc else float("inf"),           # inf = none within 5 s
        share_dist_below_1m=float(np.mean(np.asarray(near_d) < 1.0)) if near_d else 0.0,
        track_err_mean_m=float(np.mean([e[0] for e in track_err])) if track_err else 0.0,
        track_err_p95_m=float(np.percentile([e[0] for e in track_err], 95)) if track_err else 0.0,
        track_err_max_m=float(np.max([e[0] for e in track_err])) if track_err else 0.0,
        track_lat_p95_m=float(np.percentile([abs(e[1]) for e in track_err], 95)) if track_err else 0.0,
        reactive=None if rt is None else dict(
            n_agents=len(rt.agents), errors=r_err, new_agent_collisions=len(r_pairs), log_agent_collisions=len(r_log_pairs),
            idm_binding_share=float(rt.binding_steps / max(rt.active_steps, 1)), max_ego_standoff_s=r_standoff * DT),
        ego_size=list(ego_size), trace=np.asarray(poses)[:, :2].round(4).tolist(),
        chosen_index=[(c["speed"], c["offset"]) for c in chosen_log])
    if gif is not None and frames:
        save_gif(frames, gif, 100)
    if viz is not None and viz_frames:
        save_gif(viz_frames, viz, 400)
    return metrics, chosen_log


INITIAL_OVERLAP_S = 0.5


def classify_collision(t, at_fault_rule):
    """initial_overlap if the first contact is within the first 0.5 s (boxes overlapping from the start of the
    log), else at_fault / not_at_fault from the rear / stopped-ego rule."""
    if t <= INITIAL_OVERLAP_S:
        return "initial_overlap"
    return "at_fault" if at_fault_rule else "not_at_fault"


def diagnose_at_fault(chosen_log, oid, t, last=2):
    """From the cost logs of the last `last` planning steps before the collision (in this order):
    cost picked risky - at some step a zero-risk plan existed but the chosen plan had risk > 0;
    zero predicted risk - else, if the risk model gave this agent zero risk for the chosen plan at every step
                        (the predictor and/or the footprint geometry saw no overlap);
    no safe plan      - else (the risk was seen, and no zero-risk plan existed when it was).
    Also: seconds before the collision at which this agent's risk for the chosen plan first appeared."""
    steps = [c for c in chosen_log if c["t"] < t][-last:]
    if not steps:
        return dict(diagnosis="no planning step before")
    agent = [c.get("agent_risk_chosen", {}).get(oid, 0.0) for c in steps]
    out = dict(diag_t=[c["t"] for c in steps], diag_vetoed=[c["n_vetoed"] for c in steps],
               diag_chosen_risk=[c["risk"] for c in steps], diag_agent_risk=agent,
               diag_zero_risk_plans=[c.get("n_zero_risk", -1) for c in steps])
    seen = [c["t"] for c in chosen_log if c["t"] < t and c.get("agent_risk_chosen", {}).get(oid, 0.0) > 1e-6]
    out["first_seen_s_before"] = round(t - seen[0], 1) if seen else None
    if any(c.get("n_zero_risk", 0) > 0 and c["risk"] > 1e-6 for c in steps):
        out["diagnosis"] = "safe plan existed but cost picked a risky one"
    elif all(a_ < 1e-6 for a_ in agent):
        out["diagnosis"] = "zero predicted risk"
    else:
        out["diagnosis"] = "no safe plan existed"
    return out


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


# ============================================================
# Reactive agents, stage 1 (docs/reactive_agents_design.md): log-path IDM vehicles
# ============================================================

# Design-doc values (the overnight tuning, done while a timing bug was present, was reverted on 2026-11-07).
IDM = dict(s0=2.0, T=1.2, a_max=1.5, b=2.0, b_max=6.0, delta=4.0, k_s=0.5, k_v=1.0, v0_factor=1.1, v0_min=3.0,
           lateral_margin=0.3)
DEVIATION_M = 0.3   # a leader "deviates from its log" above this (ego: distance to its logged position; agent: |s - s_log|)


class _HoldPolicy:
    """Placeholder policy for MetaDrive objects driven by ReactiveTraffic: the traffic manager neither replays nor
    removes objects whose policy is not ReplayTrafficParticipantPolicy, so the pose set by ReactiveTraffic persists."""

    def __init__(self, control_object, *args, **kwargs):
        self.control_object = control_object

    def act(self, *args, **kwargs):
        return None

    def destroy(self):
        pass

    def reset(self):
        pass


class LogPathAgent:
    """One reactive vehicle: keeps its logged path exactly; only its speed changes.
    The path is the polyline of its valid logged positions (longest valid run), resampled at 0.5 m; s_log(t) is the
    logged arclength per sim step, v_log(t) the logged speed from the 2 Hz keyframes (0.5 s differences)."""

    def __init__(self, oid, track, sim_dt=DT):
        st = track["state"]
        valid = np.asarray(st["valid"], bool)
        runs, start = [], None
        for t, v in enumerate(list(valid) + [False]):
            if v and start is None:
                start = t
            elif not v and start is not None:
                runs.append((start, t))
                start = None
        self.t0, self.t1 = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)     # [t0, t1)
        pos = np.asarray(st["position"])[self.t0:self.t1, :2].astype(float)
        self.oid, self.width = oid, float(np.ravel(st["width"][self.t0])[0])
        self.length = float(np.ravel(st["length"][self.t0])[0])
        self.category = track["metadata"].get("type", track["type"])
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(0)
        self.s_log = np.concatenate([[0.0], np.cumsum(seg)])                       # per step in [t0, t1)
        self.path_len = float(self.s_log[-1]) if len(self.s_log) else 0.0
        # 2 Hz logged speed (keyframes are every 5th step from the scenario start), interpolated per step
        steps = np.arange(self.t0, self.t1)
        keys = steps[steps % 5 == 0]
        if len(keys) >= 2:
            sk = self.s_log[keys - self.t0]
            vk = np.diff(sk) / 0.5
            mid = (keys[:-1] + keys[1:]) / 2.0
            self.v_log = np.interp(steps, mid, vk)
            # logged acceleration (feed-forward), from the same 2 Hz speeds: 0.5 s differences, at the keyframes
            self.a_log = np.interp(steps, keys[1:-1], np.diff(vk) / 0.5) if len(vk) >= 2 else np.zeros(len(steps))
        else:
            self.v_log = np.zeros(len(steps))
            self.a_log = np.zeros(len(steps))
        self.v_max_log = float(self.v_log.max()) if len(self.v_log) else 0.0
        heading = np.asarray(st["heading"])[self.t0:self.t1]
        self.reversing = False
        if len(pos) > 5:
            d = np.diff(pos, axis=0)
            mv = np.linalg.norm(d, axis=1) > 0.05
            if mv.sum() > 5:
                dirs = np.arctan2(d[mv, 1], d[mv, 0])
                self.reversing = np.mean(np.cos(dirs - heading[1:][mv]) < 0) > 0.5
        disp = float(np.linalg.norm(pos[-1] - pos[0])) if len(pos) else 0.0
        self.eligible = (self.category.startswith("vehicle.") and disp >= 2.0 and self.v_max_log >= 0.5
                         and not self.reversing and (self.t1 - self.t0) >= 2.0 / sim_dt and self.path_len >= 5.0)
        if self.eligible:
            n = max(2, int(np.ceil(self.path_len / 0.5)) + 1)
            grid = np.linspace(0.0, self.path_len, n)
            keep = np.concatenate([[True], np.diff(self.s_log) > 1e-6])
            sx, px = self.s_log[keep], pos[keep]
            self.S = grid
            self.XY = np.stack([np.interp(grid, sx, px[:, 0]), np.interp(grid, sx, px[:, 1])], 1)
            dxy = np.gradient(self.XY, axis=0)
            self.YAW = np.arctan2(dxy[:, 1], dxy[:, 0])
            self.v0 = max(IDM["v0_factor"] * self.v_max_log, IDM["v0_min"])
        self.active, self.done = False, False
        self.s = self.v = 0.0
        self.a_track = self.a_idm = 0.0
        self.leader = None
        self.ego_on_path_since = None
        self.replaying = False          # snapped back to exact replay (hysteresis, see ReactiveTraffic.step)
        self.free_steps = 0             # consecutive steps without a leader within H

    def log_state(self, t):
        i = int(np.clip(t - self.t0, 0, len(self.s_log) - 1))
        return float(self.s_log[i]), float(self.v_log[i]) if t < self.t1 else 0.0

    def log_accel(self, t):
        i = int(np.clip(t - self.t0, 0, len(self.a_log) - 1))
        return float(self.a_log[i]) if t < self.t1 else 0.0

    def pose(self):
        x = np.interp(self.s, self.S, self.XY[:, 0])
        y = np.interp(self.s, self.S, self.XY[:, 1])
        i = int(np.clip(np.searchsorted(self.S, self.s), 0, len(self.S) - 1))
        return float(x), float(y), float(self.YAW[i])


def _corners(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)


SNAP = dict(free_s=1.0, ds=0.3, dv=0.3)     # snap back to replay: no leader within H for 1 s, |ds| < 0.3 m, |dv| < 0.3 m/s


class ReactiveTraffic:
    """Stage-1 reactive traffic: every moving logged vehicle (LogPathAgent.eligible) keeps its logged path and gets
    speed a = clip(min(a_track, a_IDM), -b_max, a_max), a_track = a_log + k_s (s_log - s) + k_v (v_log - v) with the
    logged acceleration a_log as feed-forward (from the 2 Hz positions). Snap-back hysteresis: after SNAP["free_s"]
    without a leader within H and with |s - s_log| < SNAP["ds"], |v - v_log| < SNAP["dv"], the agent replays its log
    exactly until a leader appears again. Reaction rule: a_IDM is applied only if the leader deviates from its log
    (the ego more than DEVIATION_M from its logged position, or a reactive agent with |s - s_log| > DEVIATION_M);
    a leader that is "as logged" (replayed objects, agents and ego on their logs) is ignored and the agent tracks its
    own log, a = clip(a_track, -b_max, max(a_max, a_log)), since its logged gap to that leader is already right; pedestrians, cyclists, static / short / reversing vehicles
    replay the log. Leader = the nearest object physically on the agent's path ahead (any corner within
    width/2 + 0.3 m of the path, in the window [s, s + H], H = max(30 m, v T + 20 m)), including the ego by its
    current state (no anticipation, no plan). All accelerations come from the previous state, then all agents are
    integrated together. This class is the source of truth for the reactive agents' states; MetaDrive objects of
    those vehicles (if spawned) are moved to match (policy swapped to a hold policy)."""

    def __init__(self, env):
        self.env = env
        scen = env.engine.data_manager.current_scenario
        self.sdc = scen["metadata"]["sdc_id"]
        self.scenario = scen
        self.agents = {oid: LogPathAgent(oid, tr) for oid, tr in scen["tracks"].items()
                       if oid != self.sdc and tr["type"] == "VEHICLE"}
        self.agents = {k: a for k, a in self.agents.items() if a.eligible}
        self.t = int(env.episode_step)
        self.binding_steps = self.active_steps = 0
        self.deadlock_steps = {}
        self._activate()

    # -- helpers
    def _tm(self):
        return self.env.engine.managers["traffic_manager"]

    def _sync_object(self, a):
        tm = self._tm()
        obj_id = tm._scenario_id_to_obj_id.get(a.oid) if tm._scenario_id_to_obj_id else None
        if obj_id is None:
            return
        obj = tm.spawned_objects.get(obj_id)
        if obj is None:
            return
        if not isinstance(self.env.engine.get_policy(obj_id), _HoldPolicy):
            self.env.engine.add_policy(obj_id, _HoldPolicy, obj)
        x, y, yaw = a.pose()
        obj.set_position([x, y])
        obj.set_heading_theta(yaw)
        obj.set_velocity([np.cos(yaw), np.sin(yaw)], a.v)

    def _remove_object(self, a):
        tm = self._tm()
        obj_id = tm._scenario_id_to_obj_id.get(a.oid) if tm._scenario_id_to_obj_id else None
        if obj_id is not None and obj_id in tm.spawned_objects:
            tm.clear_objects([obj_id])                  # keep the id mapping so the manager does not respawn it

    def _activate(self):
        ego = self.env.agent
        eb = box(ego.position[0], ego.position[1], ego.heading_theta, ego.LENGTH, ego.WIDTH)
        for a in self.agents.values():
            if not a.active and not a.done and a.t0 <= self.t < a.t1:
                s_log, v_log = a.log_state(self.t)
                a.s, a.v = s_log, v_log
                x, y, yaw = a.pose()
                if eb.intersects(box(x, y, yaw, a.length, a.width)):      # overlaps the ego: replay until clear
                    continue
                a.active = True
                self._sync_object(a)

    def states(self):
        """{oid: (x, y, yaw, width, length, category)} of the active reactive agents (overrides the log)."""
        return {a.oid: (*a.pose(), a.width, a.length, a.category) for a in self.agents.values() if a.active}

    def hidden(self):
        """Ids whose logged state must not be used (reactive agents that are done, or not yet active but valid)."""
        return {a.oid for a in self.agents.values() if a.done}

    def _objects(self, snapshot, old_snapshot, ego_vel):
        """Candidate leaders: (id, corners (4, 2), velocity (2,))."""
        out = []
        for oid, st in snapshot.items():
            x, y, yaw, w, l, cat = st
            if oid == EGO_ID:
                v = ego_vel
            elif oid in self.agents and self.agents[oid].active:
                a = self.agents[oid]
                v = a.v * np.array([np.cos(yaw), np.sin(yaw)])
            else:
                v = np.zeros(2) if (old_snapshot is None or oid not in old_snapshot) else \
                    (np.array([x, y]) - np.array(old_snapshot[oid][:2])) / 0.5
            out.append((oid, _corners(x, y, yaw, max(l, 0.3), max(w, 0.3)), v))
        return out

    def _leader(self, a, objects, ignore=()):
        """(gap, leader speed along the path, id) of the nearest object on the path ahead, or None; and whether the
        ego is on the path ahead."""
        H = max(30.0, a.v * IDM["T"] + 20.0)
        lo, hi = a.s, min(a.s + H, a.path_len + 10.0)
        i0, i1 = np.searchsorted(a.S, lo), np.searchsorted(a.S, min(hi, a.path_len), side="right")
        if i1 - i0 < 1:
            return None, False
        S, XY, YAW = a.S[i0:i1], a.XY[i0:i1], a.YAW[i0:i1]
        ax, ay, _ = a.pose()
        best, ego_on = None, False
        for oid, corners, vel in objects:
            if oid == a.oid or oid in ignore:
                continue
            if np.min(np.linalg.norm(corners - [ax, ay], axis=1)) > H + 10:
                continue
            d = np.linalg.norm(corners[:, None] - XY[None], axis=-1)          # (4, W)
            j = d.argmin(1)
            t = np.stack([np.cos(YAW[j]), np.sin(YAW[j])], 1)
            rel = corners - XY[j]
            lat = np.abs(t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0])
            along = S[j] + (rel * t).sum(1)
            on = (lat < a.width / 2 + IDM["lateral_margin"]) & (along > a.s)
            if not on.any():
                continue
            if oid == EGO_ID:
                ego_on = True
            gap = float(along[on].min() - a.s - a.length / 2)
            k = int(np.argmin(np.where(on, along, np.inf)))
            v_lead = float(vel @ t[k])
            if best is None or gap < best[0]:
                best = (gap, v_lead, oid)
        return best, ego_on

    def step(self, snapshot, old_snapshot, ego_vel):
        """Advance all active agents by one sim step (call after env.step, with the snapshot at the new step built
        from the log + the agents' current states). Returns the agents' new states."""
        self.t = int(self.env.episode_step)
        objects = self._objects(snapshot, old_snapshot, ego_vel)
        act = [a for a in self.agents.values() if a.active]
        leaders = {a.oid: self._leader(a, objects) for a in act}
        # mutual leaders: the agent that reaches the conflict point earlier in its log keeps priority
        for a in act:
            lead = leaders[a.oid][0]
            if lead is None or lead[2] not in leaders:
                continue
            b = self.agents[lead[2]]
            lb = leaders[b.oid][0]
            if lb is None or lb[2] != a.oid:
                continue
            def log_time(x, gap):
                target = x.s + max(gap, 0.0) + x.length / 2
                k = np.searchsorted(x.s_log, target)
                return x.t0 + k
            if log_time(a, lead[0]) <= log_time(b, lb[0]):
                leaders[a.oid] = self._leader(a, objects, ignore=(b.oid,))
        acc = {}
        ego_now = snapshot.get(EGO_ID)
        sdc = self.scenario["tracks"][self.sdc]["state"]
        ego_dev = (ego_now is not None and self.t < len(sdc["valid"]) and bool(sdc["valid"][self.t])
                   and np.hypot(ego_now[0] - sdc["position"][self.t][0], ego_now[1] - sdc["position"][self.t][1])
                   > DEVIATION_M)
        state_dev = {a.oid: abs(a.s - a.log_state(self.t - 1)[0]) > DEVIATION_M for a in act}
        # Timing: this runs after env.step, so episode_step (self.t) is already the new step n, while the agents'
        # states still belong to n - 1: track the log at n - 1 and integrate to n.
        t_state = self.t - 1
        for a in act:
            s_log, v_log = a.log_state(t_state)
            if t_state >= a.t1:
                s_log, v_log = a.path_len, 0.0
            a_track = a.log_accel(t_state) + IDM["k_s"] * (s_log - a.s) + IDM["k_v"] * (v_log - a.v)
            lead, ego_on = leaders[a.oid]
            a.free_steps = 0 if lead is not None else a.free_steps + 1
            if lead is not None:
                a.replaying = False                      # leave replay as soon as a leader appears
            elif (a.free_steps * DT >= SNAP["free_s"] and abs(a.s - s_log) < SNAP["ds"]
                  and abs(a.v - v_log) < SNAP["dv"]):
                a.replaying = True
            free = IDM["a_max"] * (1 - (a.v / a.v0) ** IDM["delta"])
            if lead is None:
                a_idm = free
            else:
                gap, v_l, _ = lead
                dv = a.v - v_l
                s_star = IDM["s0"] + max(0.0, a.v * IDM["T"] + a.v * dv / (2 * np.sqrt(IDM["a_max"] * IDM["b"])))
                a_idm = free - IDM["a_max"] * (s_star / max(gap, 0.1)) ** 2
            a.a_track, a.a_idm, a.leader = a_track, a_idm, (lead[2] if lead else None)
            deviating = lead is not None and ((lead[2] == EGO_ID and ego_dev) or state_dev.get(lead[2], False))
            a.leader_deviating = deviating
            if deviating:
                acc[a.oid] = float(np.clip(min(a_track, a_idm), -IDM["b_max"], IDM["a_max"]))
            else:                                        # leader as logged (or none): track the own log
                acc[a.oid] = float(np.clip(a_track, -IDM["b_max"], max(IDM["a_max"], a.log_accel(t_state))))
            self.active_steps += 1
            self.binding_steps += deviating and a_idm < a_track
            # ego on the path ahead: since when (for the unsafe cut-in rule)
            if ego_on:
                a.ego_on_path_since = a.ego_on_path_since if a.ego_on_path_since is not None else self.t
            else:
                a.ego_on_path_since = None
            if lead is not None and lead[2] == EGO_ID and a.v < 0.1:
                self.deadlock_steps[a.oid] = self.deadlock_steps.get(a.oid, 0) + 1
            else:
                self.deadlock_steps[a.oid] = 0
        for a in act:                                   # integrate all together
            if a.replaying and self.t < a.t1:           # exact replay: take the logged state at step n
                a.s, a.v = a.log_state(self.t)
            else:
                a.v = float(np.clip(a.v + acc[a.oid] * DT, 0.0, a.v0))
                a.s = float(min(a.s + a.v * DT, a.path_len))
            if a.s >= a.path_len - 1e-3 and self.t >= a.t1 - 1:
                a.active, a.done = False, True
                self._remove_object(a)
            else:
                self._sync_object(a)
        self._activate()
        return self.states()
