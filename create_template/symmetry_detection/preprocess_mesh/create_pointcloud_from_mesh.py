import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve
import os

"""メッシュから点群データを作成する
create_pointclouds_from_mesh.py
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PLY file and generate point cloud.')
    parser.add_argument('--ply', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--num_points', type=int, default=20000, help='Number of points in point cloud.')
    parser.add_argument('--output_dir', type=str, default="./points", help='Threshold for observability.')

    args = parser.parse_args()

    if args.ply is not None:
        # 指定されたPLYファイルを読み込み
        mesh = o3d.io.read_triangle_mesh(args.ply)
    else:
        # デフォルトのArmadilloメッシュを読み込み
        armadillo_mesh = o3d.data.ArmailloMesh()
        mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)

    mesh.compute_vertex_normals()

    # 点群を生成
    pcd = mesh.sample_points_poisson_disk(args.num_points)
    voxel_size = 0.0005*1000  # ダウンサンプリングの解像度を適切に指定する
    pcd = pcd.voxel_down_sample(voxel_size)

    
    if args.ply is not None:
        
        # 出力フォルダが存在しない場合は作成する
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        target_obj_filename = os.path.basename(args.ply)
        file_name_without_extension = os.path.splitext(target_obj_filename)[0]
        o3d.io.write_point_cloud(f"{args.output_dir}/{file_name_without_extension}_points.ply", pcd)