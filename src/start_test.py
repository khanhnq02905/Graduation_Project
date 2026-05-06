import torch
import open3d as o3d
import numpy as np

# 1. Check Hardware
print(f"GPU Detection: {'Success' if torch.cuda.is_available() else 'Failed'}")
if torch.cuda.is_available():
    print(f"Running on: {torch.cuda.get_device_name(0)}")

# 2. Create a basic shape to test visualization
print("Generating test point cloud...")
# Create a sphere mesh
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
# Corrected method: Convert the mesh into 5000 points
pcd = mesh.sample_points_uniformly(number_of_points=5000)

# 3. Show the point cloud
print("Opening visualizer (Close the window to finish)...")
o3d.visualization.draw_geometries([pcd])