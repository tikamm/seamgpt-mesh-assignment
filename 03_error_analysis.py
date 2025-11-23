import os, glob
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt

meshes_dir = "meshes"
bins = 1024
recon_minmax_dir = os.path.join("outputs","reconstructed","minmax")
recon_unitsphere_dir = os.path.join("outputs","reconstructed","unitsphere")
plots_dir = os.path.join("outputs","plots")
os.makedirs(recon_minmax_dir, exist_ok=True)
os.makedirs(recon_unitsphere_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def minmax_norm(V):
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    scale = np.where((vmax - vmin)==0, 1.0, (vmax - vmin))
    Vn = (V - vmin) / scale
    ctx = {"vmin": vmin, "vmax": vmax}
    return Vn, ctx

def minmax_denorm(Vn, ctx):
    vmin = ctx["vmin"]; vmax = ctx["vmax"]
    scale = np.where((vmax - vmin)==0, 1.0, (vmax - vmin))
    return Vn * scale + vmin

def unitsphere_norm(V):
    center = V.mean(axis=0)
    Vc = V - center
    r = np.linalg.norm(Vc, axis=1).max()
    r = 1.0 if r==0 else r
    Vu = Vc / r
    V01 = (Vu + 1.0) * 0.5
    ctx = {"center": center, "r": r}
    return V01, ctx

def unitsphere_denorm(V01, ctx):
    center = ctx["center"]; r = ctx["r"]
    Vu = V01 * 2.0 - 1.0
    return Vu * r + center

def quantize(V01, n_bins):
    V01c = np.clip(V01, 0.0, 1.0)
    Q = np.floor(V01c * (n_bins - 1)).astype(np.int32)
    return Q

def dequantize(Q, n_bins):
    return Q.astype(np.float64) / (n_bins - 1)

def mse_per_axis(a, b):
    diff = a - b
    return (diff**2).mean(axis=0)

def mae_per_axis(a, b):
    diff = np.abs(a - b)
    return diff.mean(axis=0)

rows = []
obj_paths = sorted(glob.glob(os.path.join(meshes_dir, "*.obj")))
if not obj_paths:
    print("No .obj files found in 'meshes/'.")
    raise SystemExit(1)

for p in obj_paths:
    name = os.path.splitext(os.path.basename(p))[0]
    mesh = trimesh.load(p, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces) if hasattr(mesh, "faces") and mesh.faces is not None else None

    Vn_mm, ctx_mm = minmax_norm(V)
    Q_mm = quantize(Vn_mm, bins)
    Vn_mm_dq = dequantize(Q_mm, bins)
    Vrec_mm = minmax_denorm(Vn_mm_dq, ctx_mm)

    Vn_us, ctx_us = unitsphere_norm(V)
    Q_us = quantize(Vn_us, bins)
    Vn_us_dq = dequantize(Q_us, bins)
    Vrec_us = unitsphere_denorm(Vn_us_dq, ctx_us)

    mse_mm = mse_per_axis(V, Vrec_mm)
    mae_mm = mae_per_axis(V, Vrec_mm)
    mse_us = mse_per_axis(V, Vrec_us)
    mae_us = mae_per_axis(V, Vrec_us)

    rows.append({"mesh": name, "method": "minmax", "mse_x": mse_mm[0], "mse_y": mse_mm[1], "mse_z": mse_mm[2], "mae_x": mae_mm[0], "mae_y": mae_mm[1], "mae_z": mae_mm[2]})
    rows.append({"mesh": name, "method": "unitsphere", "mse_x": mse_us[0], "mse_y": mse_us[1], "mse_z": mse_us[2], "mae_x": mae_us[0], "mae_y": mae_us[1], "mae_z": mae_us[2]})

    if F is None or len(F)==0:
        trimesh.Trimesh(vertices=Vrec_mm, process=False).export(os.path.join(recon_minmax_dir, f"{name}.obj"))
        trimesh.Trimesh(vertices=Vrec_us, process=False).export(os.path.join(recon_unitsphere_dir, f"{name}.obj"))
    else:
        trimesh.Trimesh(vertices=Vrec_mm, faces=F, process=False).export(os.path.join(recon_minmax_dir, f"{name}.obj"))
        trimesh.Trimesh(vertices=Vrec_us, faces=F, process=False).export(os.path.join(recon_unitsphere_dir, f"{name}.obj"))

    labels = ["x","y","z"]
    x_pos = np.arange(3)
    width = 0.35
    plt.figure()
    plt.bar(x_pos - width/2, mse_mm, width, label="minmax")
    plt.bar(x_pos + width/2, mse_us, width, label="unitsphere")
    plt.xticks(x_pos, labels)
    plt.ylabel("MSE")
    plt.title(f"{name} reconstruction error per axis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{name}_mse_axes.png"), dpi=200)
    plt.close()

df = pd.DataFrame(rows)
out_csv = os.path.join("outputs","task3_errors.csv")
df.to_csv(out_csv, index=False)
print("Task 3 complete.")
print(f"Reconstructed (minmax): {recon_minmax_dir}")
print(f"Reconstructed (unitsphere): {recon_unitsphere_dir}")
print(f"Per-axis error CSV: {out_csv}")
print(f"Plots: {plots_dir}")
