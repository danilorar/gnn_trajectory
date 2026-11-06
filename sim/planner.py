"""Phase D step 3: reference path, candidate-plan generator and tracker (no predictor yet).

- ReferencePath: the ego's logged route in the scene, resampled at 1 m ("where to go", as in nuPlan).
- generate_plans: target speed x lateral offset along the reference (default 7 x 5 = 35), each rolled out
  for 6 s with a kinematic bicycle model, pure pursuit onto the offset path and |acceleration| <= 3 m/s^2.
  Returns the 12 model points (every 0.5 s, world (x, y, yaw)) and the dense 0.1 s rollout for the tracker.
  plans_to_ego_future() puts the 12 points in a target frame as [x, y, cos, sin], the format of ego_future.
- Tracker: pure pursuit for steering + PID for speed on the chosen plan's dense rollout, giving MetaDrive
  actions [steering, throttle] (steering +1 = 40 deg left; throttle < 0 brakes).
"""

import numpy as np

SPEEDS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)      # m/s
OFFSETS = (-1.5, -0.75, 0.0, 0.75, 1.5)              # m, + = left of the reference
WHEELBASE = 2.47                                     # m, MetaDrive default vehicle (front 1.05 + rear 1.42)
MAX_STEER = np.radians(40.0)                         # MetaDrive max_steering
DT, HORIZON, MODEL_DT = 0.1, 6.0, 0.5


class ReferencePath:
    """Polyline resampled at `spacing` m with headings; extended straight 100 m past its end."""

    def __init__(self, points, spacing=1.0, extend=100.0):
        pts = np.asarray(points, float)[:, :2]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3])
        pts = pts[keep]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(s[-1])
        d_end = (pts[-1] - pts[-2]) / seg[-1]
        grid = np.arange(0.0, s[-1] + extend, spacing)
        inside = grid <= s[-1]
        xy = np.empty((len(grid), 2))
        xy[inside, 0] = np.interp(grid[inside], s, pts[:, 0])
        xy[inside, 1] = np.interp(grid[inside], s, pts[:, 1])
        xy[~inside] = pts[-1] + (grid[~inside] - s[-1])[:, None] * d_end
        d = np.gradient(xy, axis=0)
        self.s, self.xy, self.yaw = grid, xy, np.arctan2(d[:, 1], d[:, 0])
        self.normal = np.stack([-np.sin(self.yaw), np.cos(self.yaw)], -1)     # left

    def project(self, xy):
        """Arc length s and signed lateral offset d (+ = left) of a point (nearest resampled point, refined)."""
        i = int(np.argmin(np.linalg.norm(self.xy - xy, axis=1)))
        rel = np.asarray(xy) - self.xy[i]
        t = np.array([np.cos(self.yaw[i]), np.sin(self.yaw[i])])
        return float(self.s[i] + rel @ t), float(rel @ self.normal[i])

    def point(self, s, d=0.0):
        s = np.clip(s, 0.0, self.s[-1])
        x, y = np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])
        nx, ny = np.interp(s, self.s, self.normal[:, 0]), np.interp(s, self.s, self.normal[:, 1])
        return np.stack([x + d * nx, y + d * ny], -1)


def reference_from_scenario(scenario):
    """The ego's full logged path (valid steps) as a ReferencePath."""
    st = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]
    return ReferencePath(np.asarray(st["position"])[np.asarray(st["valid"], bool)])


def logged_speed(scenario, step):
    """Ego speed in the log at a sim step, from positions 0.5 s apart (ScenarioNet nuScenes velocities are 0)."""
    st = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]
    pos = np.asarray(st["position"])[:, :2]
    a, b = max(0, step - 2), min(len(pos) - 1, step + 3)
    return float(np.linalg.norm(pos[b] - pos[a]) / ((b - a) * DT)) if b > a else 0.0


LOOKAHEAD = (0.5, 2.5, 3.0, 12.0)                    # lookahead = clip(gain * v + base, min, max) [m]


def pure_pursuit(x, y, yaw, v, path_xy, wheelbase=WHEELBASE):
    """Steering angle towards the point on path_xy one lookahead distance ahead of the closest point."""
    gain, base, lo, hi = LOOKAHEAD
    lookahead = float(np.clip(gain * v + base, lo, hi))
    d = np.linalg.norm(path_xy - [x, y], axis=1)
    i = int(np.argmin(d))
    ahead = np.flatnonzero(d[i:] >= lookahead)
    gx, gy = path_xy[i + ahead[0]] if len(ahead) else path_xy[-1]
    alpha = np.arctan2(gy - y, gx - x) - yaw
    return float(np.clip(np.arctan2(2 * wheelbase * np.sin(alpha), lookahead), -MAX_STEER, MAX_STEER))


def generate_plans(ref, state, speeds=SPEEDS, offsets=OFFSETS, a_max=3.0):
    """state = (x, y, yaw, v) world. Returns
    sparse (P, 12, 3) world (x, y, yaw) at t + 0.5 s ... 6 s, dense (P, 61, 4) (x, y, yaw, v) every 0.1 s
    from t, and meta [(speed, offset)] in grid order (speed-major)."""
    x0, y0, yaw0, v0 = state
    s0, _ = ref.project((x0, y0))
    n = int(round(HORIZON / DT))
    path_s = s0 + np.arange(-5.0, max(speeds) * HORIZON + 30.0, 1.0)
    sparse, dense, meta = [], [], []
    for v_target in speeds:
        for off in offsets:
            path = ref.point(path_s, off)
            x, y, yaw, v = x0, y0, yaw0, v0
            roll = [(x, y, yaw, v)]
            for _ in range(n):
                delta = pure_pursuit(x, y, yaw, v, path)
                v = max(0.0, v + float(np.clip((v_target - v) / DT, -a_max, a_max)) * DT)
                x += v * np.cos(yaw) * DT
                y += v * np.sin(yaw) * DT
                yaw += v / WHEELBASE * np.tan(delta) * DT
                roll.append((x, y, yaw, v))
            roll = np.asarray(roll)
            dense.append(roll)
            sparse.append(roll[int(MODEL_DT / DT)::int(MODEL_DT / DT), :3])
            meta.append((v_target, off))
    return np.asarray(sparse), np.asarray(dense), meta


def plans_to_ego_future(sparse, origin, yaw):
    """(P, 12, 3) world plans -> (P, 12, 4) [x, y, cos, sin] in the frame (origin, yaw), as ego_future."""
    from utils.lane_tokens import poses_to_local
    return np.stack([poses_to_local(p, origin, yaw) for p in sparse])


class Tracker:
    """Pure pursuit on the plan's path + PID on the plan's speed profile -> MetaDrive [steering, throttle]."""

    def __init__(self, kp=0.6, ki=0.05, kd=0.0, accel_per_throttle=3.0):
        self.kp, self.ki, self.kd, self.k = kp, ki, kd, accel_per_throttle
        self.integral, self.prev = 0.0, 0.0

    def action(self, x, y, yaw, v, plan_dense, t):
        """plan_dense (61, 4) from generate_plans; t = seconds since the plan started."""
        delta = pure_pursuit(x, y, yaw, v, plan_dense[:, :2])
        i = min(int(round(t / DT)) + 1, len(plan_dense) - 1)
        v_ref = plan_dense[i, 3]
        err = v_ref - v
        self.integral = float(np.clip(self.integral + err * DT, -5, 5))
        a_cmd = self.kp * err + self.ki * self.integral + self.kd * (err - self.prev) / DT
        a_cmd += (plan_dense[i, 3] - plan_dense[i - 1, 3]) / DT            # feed-forward plan acceleration
        self.prev = err
        throttle = a_cmd / self.k if a_cmd >= 0 else a_cmd / 11.0          # MetaDrive full brake ~ -11 m/s^2
        return [float(np.clip(delta / MAX_STEER, -1, 1)), float(np.clip(throttle, -1, 1))]
