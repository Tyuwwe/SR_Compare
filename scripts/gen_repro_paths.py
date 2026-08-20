"""Generate repro camera paths starting at the user's reported Bistro pose."""
import json, math, os

POS = (-11.2, 4.6, 12.0)
FWD = (0.198, -0.015, -0.980)
UP = (0.0, 1.0, 0.0)
FRAMES = 300

def norm(v):
    l = math.sqrt(sum(x * x for x in v))
    return tuple(x / l for x in v)

def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

fwd = norm(FWD)
right = norm(cross(fwd, UP))  # screen-right in world space

def kf(p, f):
    return {"position": list(p), "forward": list(f), "up": list(UP)}

# 1) strafe right (D key): 0.05 m/frame ~ 3 m/s @60fps
path_right = [kf(tuple(POS[i] + right[i] * 0.05 * t for i in range(3)), fwd) for t in range(FRAMES)]

# 2) rotate right (yaw right): turn toward +right, 0.5 deg/frame
path_rotr = []
for t in range(FRAMES):
    th = math.radians(-0.5 * t)  # about +Y; negative turns right (fwd swings toward +right)
    c, s = math.cos(th), math.sin(th)
    f = (fwd[0] * c + fwd[2] * s, fwd[1], -fwd[0] * s + fwd[2] * c)
    path_rotr.append(kf(POS, f))

# 3) strafe left (A key, control: should NOT flicker)
path_left = [kf(tuple(POS[i] - right[i] * 0.05 * t for i in range(3)), fwd) for t in range(FRAMES)]

out = os.path.join(os.path.dirname(__file__), "..", "output", "paths")
os.makedirs(out, exist_ok=True)
for name, p in [("bistro_repro_right", path_right), ("bistro_repro_rotr", path_rotr),
                ("bistro_repro_left", path_left)]:
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(p, f)
    print("wrote", name, len(p))
