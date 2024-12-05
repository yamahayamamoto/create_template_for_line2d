import numpy as np
import open3d as o3d
import argparse
import os


"""
python create_sphere_viewpoints.py　--viewpoint_num 364 --output_dir ../dataset/input/viewpoints --debug
"""

def generate_shpere_surface_points_by_fib(N):
    f = (np.sqrt(5) - 1) / 2
    arr = np.linspace(-N, N, N * 2 + 1)
    theta = np.arcsin(arr / N)
    phi = 2 * np.pi * arr * f
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    points = np.array([x,y,z]).T
    return points

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--viewpoint_num', type=int, default=162*2, help='Path to PLY file.')
    parser.add_argument('--output_dir', type=str, default="./result/viewpoints", help='Path to PLY file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    
    args = parser.parse_args()

    viewpoint_vectors = generate_shpere_surface_points_by_fib(int(args.viewpoint_num/2))
    
    if(args.debug):
        viewpoints = o3d.geometry.PointCloud()
        viewpoints.points = o3d.utility.Vector3dVector(viewpoint_vectors)     
        o3d.visualization.draw_geometries([viewpoints], mesh_show_back_face=True)
    
    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # # JSONファイルに保存するパス
    output_file_path = os.path.join(args.output_dir,f"origin_viewpoints_{args.viewpoint_num}.npy")


    print("the number of viewpoints(before):",len(viewpoint_vectors))

    # NumPy配列を保存
    np.save(output_file_path, viewpoint_vectors)
