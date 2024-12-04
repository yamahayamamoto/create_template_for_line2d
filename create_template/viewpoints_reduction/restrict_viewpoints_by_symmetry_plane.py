import numpy as np
import open3d as o3d
import json
from typing import Optional,List
import os
import argparse
import copy
from utils.object_visualization_helper import CoordinateFrameMesh, Circle, Arrow
from utils.sphere_points_creator import generate_shpere_surface_points_by_fib

"""
python restrcict_view_point_by_symmetry_plane.py
"""
#描写用#################################################################################
def rotation_matrix_to_align_with_z_axis(vec): 
    """ 
    ベクトルvecをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列を計算する。 
    z_axis = z_axis_R_vec × vec
 
    Parameters: 
        vec (numpy.ndarray): 3次元ベクトル (x, y, z) を表すNumPy配列。 
 
    Returns: 
        numpy.ndarray: ベクトルvecをx軸に一致させるための回転行列。vecがx軸と一致している場合は単位行列が返される。 
    """ 
    z_axis = np.array([0, 0, 1])  # x軸の単位ベクトル 
 
    # ベクトルvecとz_axisの外積を計算 
    cross_product = np.cross(vec, z_axis) 
 
    # 回転角度を計算 
    if np.linalg.norm(cross_product) < 1e-12: 
        # 外積がほぼゼロの場合（vecがx軸と一致する場合） 
        return np.identity(3)  # 単位行列を返す 
 
    # 外積を正規化して回転軸を計算 
    rotation_axis = cross_product / np.linalg.norm(cross_product) 
 
    # ドット積を計算 
    dot_product = np.dot(vec, z_axis) 
 
    # 回転角度を計算 
    rotation_angle = np.arccos(np.clip(dot_product / (np.linalg.norm(vec) * np.linalg.norm(z_axis)), -1.0, 1.0)) 
 
    # 回転行列を計算 
    cos_theta = np.cos(rotation_angle) 
    sin_theta = np.sin(rotation_angle) 
    z_axis_R_vec = np.array([ 
        [cos_theta + rotation_axis[0] ** 2 * (1 - cos_theta), rotation_axis[0] * rotation_axis[1] * (1 - cos_theta) - rotation_axis[2] * sin_theta, rotation_axis[0] * rotation_axis[2] * (1 - cos_theta) + rotation_axis[1] * sin_theta], 
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_theta) + rotation_axis[2] * sin_theta, cos_theta + rotation_axis[1] ** 2 * (1 - cos_theta), rotation_axis[1] * rotation_axis[2] * (1 - cos_theta) - rotation_axis[0] * sin_theta], 
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_theta) - rotation_axis[1] * sin_theta, rotation_axis[2] * rotation_axis[1] * (1 - cos_theta) + rotation_axis[0] * sin_theta, cos_theta + rotation_axis[2] ** 2 * (1 - cos_theta)] 
    ]) 
     
    return z_axis_R_vec 

############################################################################################################
def generate_viewpoint_vectors(elevation_angle_range=[0, np.pi], azimuth_angle_range=[0, 2*np.pi], num_elevation=20, num_azimuth=20):
    """
    指定された仰角と方位角の範囲で視点ベクトルを計算し、リストに格納します。

    Args:
        elevation_angle_range (list, optional): 仰角の最小値と最大値を格納したリスト（デフォルトは[0, π]）。
        azimuth_angle_range (list, optional): 方位角の最小値と最大値を格納したリスト（デフォルトは[0, 2π]）。
        num_elevation (int, optional): 仰角の分割数（デフォルトは20）。
        num_azimuth (int, optional): 方位角の分割数（デフォルトは20）。

    Returns:
        list: 計算された視点ベクトルを格納したリスト。

    Example:
        >>> viewpoint_vectors = generate_viewpoint_vectors()
    """
    elevation_angles = np.linspace(elevation_angle_range[0], elevation_angle_range[1], num_elevation)
    azimuth_angles = np.linspace(azimuth_angle_range[0], azimuth_angle_range[1], num_azimuth)

    viewpoint_vectors = []

    # グリッドを繰り返し処理
    for elevation_angle in elevation_angles:
        for azimuth_angle in azimuth_angles:
            # 視点ベクトルを計算
            viewpoint_vector = np.array([np.cos(azimuth_angle) * np.sin(elevation_angle),
                                         np.sin(azimuth_angle) * np.sin(elevation_angle),
                                         np.cos(elevation_angle)])
            viewpoint_vector = viewpoint_vector / np.linalg.norm(viewpoint_vector)
            viewpoint_vectors.append(viewpoint_vector)

    return viewpoint_vectors


def restrict_viewpoints_by_normal(normal_vector, viewpoint_vectors):
    """
    与えられた法線ベクトルに基づいて、制約された視点ベクトルの方位角と仰角を計算します。

    Args:
        normal_vector (numpy.ndarray): 対称平面の法線ベクトル。
        elevation_angle_range (list, optional): 仰角の最小値と最大値を格納したリスト（デフォルトは[0, π]）。
        azimuth_angle_range (list, optional): 方位角の最小値と最大値を格納したリスト（デフォルトは[0, 2π]）。

    Returns:
        tuple: 各方向の最小および最大の角度をradian単位で含むタプル。[最小方位角, 最大方位角], [最小仰角, 最大仰角]

    Example:
        >>> N = np.array([0, 0, -1])
        >>> azimuth_range, elevation_range = restrict_viewpoints_by_normal(N)
    """
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # 有効な方位角と仰角を保存するリスト
    valid_viewpoint_vectors = []

    # グリッドを繰り返し処理
    for viewpoint_vector in viewpoint_vectors:
        viewpoint_vector = viewpoint_vector / np.linalg.norm(viewpoint_vector)

        # 視点ベクトルが制約を満たすか確認
        if np.dot(viewpoint_vector, normal_vector) >= 0:
            valid_viewpoint_vectors.append(viewpoint_vector)

    return valid_viewpoint_vectors

def generate_designated_init2end_arrows(init_points, end_points, color=[1,0,0]):
    """
    点から始点と終点を表す矢印のOpen3Dオブジェクトを生成します。

    Args:
        init_points (List[List[float]]): 矢印の始点を表す点のリスト。各点は3つの浮動小数点数で表されます。
        end_points (List[List[float]]): 矢印の終点を表す点のリスト。各点は3つの浮動小数点数で表されます。
        color (List[float], optional): 矢印の色を示すRGB値のリスト。デフォルトは[1, 0, 0]（赤）です。

    Returns:
        List[Open3D.Geometry]: 矢印オブジェクトのリスト。各矢印はOpen3DのGeometryオブジェクトです。
    """
    arrow_objs = []
    
    # 制限された視点ベクトルを矢印で表現する
    for init_point, end_point in zip(init_points, end_points):
        # 矢印の生成
        arrow = Arrow(init_point, end_point, color)
        arrow_obj = arrow.get_poligon()
        arrow_objs.append(arrow_obj)
    
    return arrow_objs

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Process PLY file and generate point cloud.')
    parser.add_argument('--mesh', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--viewpoint', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    
    args = parser.parse_args()
    target_obj_file = args.mesh
    target_obj_filename=os.path.basename(target_obj_file)
    file_name_without_extension = os.path.splitext(target_obj_filename)[0]

    # 反射平面の法線ベクトル
    # reflection_plane_normals = np.array([[0,0,1], [0,1,0], [1,0,0]])
    reflection_plane_normals = np.array([[1,0,0]])

    file_path = f"./result/reflectional_symmetry/{file_name_without_extension}_reflectional_symmetry.json"
    
    mesh = o3d.io.read_triangle_mesh(target_obj_file)
    mesh.compute_vertex_normals()


    # JSONファイルを読み込む
    with open(file_path, 'r') as file:
        reflection_planes = json.load(file)

    if(args.viewpoint is not None):
        viewpoint_vectors = np.load(args.viewpoint)
    else:
        # # 視点ベクトルを生成
        # viewpoint_vectors = generate_viewpoint_vectors(elevation_angle_range=[0, np.pi], 
        #                                             azimuth_angle_range=[0, 2*np.pi], 
        #                                             num_elevation=20, 
        #                                             num_azimuth=20)
        viewpoint_vectors = generate_shpere_surface_points_by_fib(500)
        
    # obj_id = 1
    # pose_dataset_dir ="./output/mydata"
    # json_file = open(pose_dataset_dir + "/H_cam2w_lefthands_{obj_id:06d}.json".format(obj_id=obj_id), 'r')
    # H_cam2w_lefthands = json.load(json_file)
    # viewpoint_vectors=[np.array(H_cam2w_lefthands[key]).reshape(4,4)[:3,3] for key in H_cam2w_lefthands.keys()]
    # print(np.array(viewpoint_vectors).shape)

    
    # バウンディングボックスの対角線長さを取得(平面描写のために使う)
    bbox_diagonal_length = np.linalg.norm(np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound()))
    filtered_viewpoints = viewpoint_vectors

    # 鏡面反射平面の法線ベクトルごとに視点ベクトルを制限する
    for plane_pos, plane_normal in reflection_planes:
        filtered_viewpoints = restrict_viewpoints_by_normal(plane_normal, 
                                                                     filtered_viewpoints)
    filtered_viewpoints = np.array(filtered_viewpoints)
    # 描写用 ########################################################################
        
    # ワールド座標系
    coordinate_frame     = CoordinateFrameMesh(scale=2)
    coordinate_frame_obj = coordinate_frame.get_poligon()

    origin_pos = [0,0,0]
    plane_objs = []

    # 鏡面反射平面の法線ベクトルを用いて平面の生成
    for reflection_plane_normal in reflection_plane_normals:
        
        ref_plane_normal_H_z_axis = np.eye(4)
        reflection_plane_normal   = reflection_plane_normal / np.linalg.norm(reflection_plane_normal)
        z_axis_R_ref_plane_normal = rotation_matrix_to_align_with_z_axis(reflection_plane_normal)
        ref_plane_normal_R_z_axis = z_axis_R_ref_plane_normal.T
        ref_plane_normal_H_z_axis[:3, :3] = ref_plane_normal_R_z_axis

        # 平面
        circle = Circle(pos=[0,0,0], rpy=[0,0,0], r=bbox_diagonal_length/2)
        circle.create_polygon()
        plane_obj = circle.get_poligon()
        
        # ワールド座標にある平面(法線方向はz)を 鏡面反射平面の法線ベクトルの方向に回転させる
        plane_obj.transform(ref_plane_normal_H_z_axis)

        # ワイヤーフレームに変換
        plane_wire_obj = o3d.geometry.LineSet.create_from_triangle_mesh(plane_obj)
        plane_objs.append(plane_wire_obj)
    
    # 制限された視点ベクトルを矢印で表現する
    vp = copy.deepcopy(filtered_viewpoints)
    restricted_arrow_objs = generate_designated_init2end_arrows(bbox_diagonal_length*np.array(vp.points)+mesh.get_center(),
                                                              0.8*bbox_diagonal_length*np.array(vp.points)+mesh.get_center())

    if(args.debug):
        o3d.visualization.draw_geometries([mesh, *restricted_arrow_objs, *plane_objs, coordinate_frame_obj], mesh_show_back_face=True)

    
    # filtered_viewpointsを保存する
    view_points = filtered_viewpoints.tolist()

    output_dir = "./result/viewpoints"
                    
    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # # JSONファイルに保存するパス
    output_file_path = os.path.join(output_dir,f"{file_name_without_extension}.npy")


    # NumPy配列を保存
    np.save(output_file_path, view_points)