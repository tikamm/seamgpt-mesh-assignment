import os
import glob
import numpy as np
import trimesh

MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes")

def describe_vertices(verts: np.ndarray):
    stats = {}
    stats["num_vertices"] = int(verts.shape[0])

    axis_names = ["x", "y", "z"]
    for i, ax in enumerate(axis_names):
        col = verts[:, i]
        stats[f"{ax}_min"]  = float(col.min())
        stats[f"{ax}_max"]  = float(col.max())
        stats[f"{ax}_mean"] = float(col.mean())
        stats[f"{ax}_std"]  = float(col.std())
    return stats

def print_stats(mesh_name: str, stats: dict):
    print(f"\n=== {mesh_name} ===")
    print(f"Vertices: {stats['num_vertices']}")
    print(f"x: min={stats['x_min']:.6f}, max={stats['x_max']:.6f}, mean={stats['x_mean']:.6f}, std={stats['x_std']:.6f}")
    print(f"y: min={stats['y_min']:.6f}, max={stats['y_max']:.6f}, mean={stats['y_mean']:.6f}, std={stats['y_std']:.6f}")
    print(f"z: min={stats['z_min']:.6f}, max={stats['z_max']:.6f}, mean={stats['z_mean']:.6f}, std={stats['z_std']:.6f}")

def load_vertices_from_obj(path: str):
    mesh = trimesh.load(path, force='mesh')
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {path}")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ))

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    return verts

def main():
    obj_paths = sorted(glob.glob(os.path.join(MESH_DIR, "*.obj")))
    if not obj_paths:
        print("No .obj files found in 'meshes/'. Add them and run again.")
        return

    all_rows = []
    for p in obj_paths:
        name = os.path.basename(p)
        verts = load_vertices_from_obj(p)
        stats = describe_vertices(verts)
        print_stats(name, stats)

        row = {"mesh": name}
        row.update(stats)
        all_rows.append(row)

    import csv
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "task1_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved Task 1 stats to {csv_path}")

if __name__ == "__main__":
    main()
