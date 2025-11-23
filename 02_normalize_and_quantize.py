import os
import glob
import numpy as np
import trimesh
from pathlib import Path
import csv

IN_DIR = "meshes"
OUT_MINMAX = Path("outputs/normalized/minmax")
OUT_UNIT = Path("outputs/normalized/unitsphere")
Q_MINMAX = Path("outputs/quantized/minmax")
Q_UNIT = Path("outputs/quantized/unitsphere")
OUT_STATS = Path("outputs/task2_summary.csv")

for p in [OUT_MINMAX, OUT_UNIT, Q_MINMAX, Q_UNIT]:
    p.mkdir(parents=True, exist_ok=True)

objs = sorted(glob.glob(os.path.join(IN_DIR, "*.obj")))
if not objs:
    print(f"No .obj files found in '{IN_DIR}'. Add them and run again.")
    raise SystemExit(0)

summary_rows = [("file", "vertices", 
                 "minmax_vmin", "minmax_vmax",
                 "unitsphere_center", "unitsphere_radius")]

def to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return np.rint(x01 * 255.0).astype(np.uint8)

for f in objs:
    name = Path(f).stem
    mesh = trimesh.load(f, force='mesh')
    V = mesh.vertices.copy().astype(np.float64)

    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    span = np.maximum(vmax - vmin, 1e-12)
    V_minmax = (V - vmin) / span

    mesh_minmax = mesh.copy()
    mesh_minmax.vertices = V_minmax
    mesh_minmax.export(OUT_MINMAX / f"{name}_minmax.obj")

    q_minmax = to_uint8(V_minmax)
    np.save(Q_MINMAX / f"{name}_minmax_uint8.npy", q_minmax)

    center = V.mean(axis=0)
    V_centered = V - center
    radius = np.linalg.norm(V_centered, axis=1).max()
    radius = max(radius, 1e-12)
    V_unit = V_centered / radius  # in [-1,1] approximately

    mesh_unit = mesh.copy()
    mesh_unit.vertices = V_unit
    mesh_unit.export(OUT_UNIT / f"{name}_unitsphere.obj")

    V_unit01 = (V_unit + 1.0) * 0.5
    q_unit = to_uint8(V_unit01)
    np.save(Q_UNIT / f"{name}_unitsphere_uint8.npy", q_unit)

    summary_rows.append((
        name, len(V),
        f"{vmin.tolist()} -> {vmax.tolist()}",
        f"{vmax.tolist()}",
        f"{center.tolist()}",
        f"{float(radius):.6f}"
    ))

OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_STATS, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerows(summary_rows)

print("Task 2 complete.")
print(f"- Normalized OBJ (minmax):     {OUT_MINMAX}")
print(f"- Normalized OBJ (unitsphere): {OUT_UNIT}")
print(f"- Quantized NPY (minmax):      {Q_MINMAX}")
print(f"- Quantized NPY (unitsphere):  {Q_UNIT}")
print(f"Summary CSV: {OUT_STATS}")
