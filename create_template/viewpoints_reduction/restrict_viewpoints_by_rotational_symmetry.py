import numpy as np
import open3d as o3d
import json
import os
import argparse
from typing import List, Optional
from PIL import Image
import copy
from utils.object_visualization_helper import CoordinateFrameMesh, Arrow, Circle, create_rotation_gif
from utils.sphere_points_creator import generate_shpere_surface_points_by_fib

"""回転対称情報を用いて視点を制限
python symmetry\restrict_view_points_by_rotational_symmetry.py --mesh symmetry\tmp_obj_reduce\95D95-06010\95D95-06010.ply 
--viewpoint symmetry\result\viewpoints\restricted_perspected_area\95D95-06010.npy --output_dir symmetry\result\viewpoints\restricted_rotation_symmetry 
--rot_sym_info_path symmetry\result/rotational_symmetry\95D95-06010_rotational_symmetry.json
"""

#描写用#################################################################################
def rotation_matrix_to_align_with_dst_vec(src_vec, dst_vec=np.array([0, 0, 1])): 
    """ 
    ベクトルsrc_vecをz軸ベクトルdst_vecに一致させる回転行列を計算する。 
    dst_vec = dst_R_src × src_vec
 
    Parameters: 
        src_vec (numpy.ndarray): 3次元ベクトル (x, y, z) を表すNumPy配列。 
 
    Returns: 
        numpy.ndarray: ベクトルsrc_vecをdst_vecに一致させるための回転行列。src_vecがdst_vecと一致している場合は単位行列が返される。 
    """ 
 
    # ベクトルsrc_vecとx_axisの外積を計算 
    cross_product = np.cross(src_vec, dst_vec) 
 
    # 回転角度を計算 
    if np.linalg.norm(cross_product) < 1e-12: 
        # 外積がほぼゼロの場合（src_vecがdst_vecと一致する場合） 
        return np.identity(3)  # 単位行列を返す 
 
    # 外積を正規化して回転軸を計算 
    rotation_axis = cross_product / np.linalg.norm(cross_product) 
 
    # ドット積を計算 
    dot_product = np.dot(src_vec, dst_vec) 
 
    # 回転角度を計算 
    rotation_angle = np.arccos(np.clip(dot_product / (np.linalg.norm(src_vec) * np.linalg.norm(dst_vec)), -1.0, 1.0)) 
 
    # 回転行列を計算 
    cos_theta = np.cos(rotation_angle) 
    sin_theta = np.sin(rotation_angle) 
    dst_R_src = np.array([ 
        [cos_theta + rotation_axis[0] ** 2 * (1 - cos_theta), rotation_axis[0] * rotation_axis[1] * (1 - cos_theta) - rotation_axis[2] * sin_theta, rotation_axis[0] * rotation_axis[2] * (1 - cos_theta) + rotation_axis[1] * sin_theta], 
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_theta) + rotation_axis[2] * sin_theta, cos_theta + rotation_axis[1] ** 2 * (1 - cos_theta), rotation_axis[1] * rotation_axis[2] * (1 - cos_theta) - rotation_axis[0] * sin_theta], 
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_theta) - rotation_axis[1] * sin_theta, rotation_axis[2] * rotation_axis[1] * (1 - cos_theta) + rotation_axis[0] * sin_theta, cos_theta + rotation_axis[2] ** 2 * (1 - cos_theta)] 
    ]) 
    return dst_R_src 

############################################################################################################
def generate_shpere_surface_points_by_polarcoordinates(elevation_angle_range=[0, np.pi], azimuth_angle_range=[0, 2*np.pi], num_elevation=20, num_azimuth=20):
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
        >>> viewpoint_vectors = generate_shpere_surface_points_by_polarcoordinates()
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

def generate_restricted_viewpoints_by_polarcoordinates(delta_phi, num_points, rotation_matrix):
    """回転対称性ベクトルによる視点生成を行います

    角度0～Δφの区間、回転対称軸方向周りに極座標系で視点ベクトルを生成する。
    Args:
        delta_phi (float): 球面座標系の回転角度の増分（ラジアン単位）
        num_points (int): 生成する視点の数
        rotation_matrix (numpy.ndarray): 球面座標系の回転を表現する3x3の回転行列

    Returns:
        numpy.ndarray:
            生成された回転対称性ベクトルによる視点の座標を持つ行列
    """
    thetas = np.linspace(0, np.pi, 2*num_points)
    
    if(delta_phi==0):
        phis = 0
    else:
        phis = np.linspace(0, delta_phi, num_points)

    # 2次元グリッドを生成
    theta, phi = np.meshgrid(thetas, phis)

    # 球面座標から直交座標に変換
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)

    # ベクトルを行列にまとめる
    restricted_viewpoints = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    # 回転行列を適用
    restricted_viewpoints = np.dot(rotation_matrix, restricted_viewpoints.T).T

    return restricted_viewpoints

def generate_restricted_viewpoints(delta_angle, num_points=100):
        # 経度を固定して、緯度を上から下まで変化させて半円をカバーする範囲のベクトル生成
        thetas = np.linspace(0, np.pi, num_points)
        phis = np.linspace(0, delta_angle, num_points)

        # 2次元グリッドを生成
        theta, phi = np.meshgrid(thetas, phis)

        # 球面座標から直交座標に変換
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)

        # ベクトルを行列にまとめる
        restricted_viewpoints = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        # Open3D の点群データに変換
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(restricted_viewpoints)

        return point_cloud

def generate_restricted_viewpoints_by_fib(delta_phi, num_points, rotation_matrix):
    """回転対称性ベクトルによる視点生成を行います

    角度0～Δφの区間、回転対称軸方向周りにフィボナッチサンプリングで視点ベクトルを生成する。
    Args:
        delta_phi (float): 球面座標系の回転角度の増分（ラジアン単位）
        num_points (int): 生成する視点の数
        rotation_matrix (numpy.ndarray): 球面座標系の回転を表現する3x3の回転行列

    Returns:
        numpy.ndarray:
            生成された回転対称性ベクトルによる視点の座標を持つ行列
    """

    if(delta_phi==0):
        # 経度を固定して、緯度を上から下まで変化させて半円をカバーする範囲のベクトル生成
        thetas = np.linspace(0, np.pi, 100)
        phis = 0

        # 2次元グリッドを生成
        theta, phi = np.meshgrid(thetas, phis)

        # 球面座標から直交座標に変換
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)

        # ベクトルを行列にまとめる
        restricted_viewpoints = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        # 回転行列を適用
        restricted_viewpoints = np.dot(rotation_matrix, restricted_viewpoints.T).T

        return restricted_viewpoints

    else:

        # generate_shpere_surface_points_by_fib関数を使用して点群を生成
        points = generate_shpere_surface_points_by_fib(num_points)
        
        valid_points = []
        for point in points:
            
            # xy平面の方位角ϕ=arctan2(y,x)
            azimuth = np.arctan2(point[1], point[0])
            if azimuth < 0:
                azimuth += 2.0 * np.pi

            # xy平面とz軸がなす仰角θ=arccos(z)
            elevation = np.arccos(point[2])

            if (0 <= azimuth <= delta_phi and 0 <= elevation <= np.pi):
                valid_points.append(point)

        # 球面座標から直交座標に変換
        spherical_coordinates = np.array(valid_points)

        # 回転行列を適用
        filtered_viewpoints = np.dot(rotation_matrix, spherical_coordinates.T).T
        return filtered_viewpoints

def find_nearest_neighbors(src, dst, threshold=3):
    """
    最近傍法を使用して、dstの各点に対する最も近いsrcの点を見つけます。

    Args:
        src (open3d.geometry.PointCloud): ソース位置の点群データ
        dst (open3d.geometry.PointCloud): 対象位置の点群データ
        threshold (float): 最近傍とみなす距離の閾値

    Returns:
        open3d.geometry.PointCloud:
            各dstの点に対する最近傍のsrcの点の座標行列
    """
    # KD ツリーを構築
    src_tree = o3d.geometry.KDTreeFlann(src)

    indices = set()
    for i in range(len(dst.points)):
        [k, idx, _] = src_tree.search_knn_vector_3d(dst.points[i], 1)  # 最近傍点の検索
        nearest_src_one_point = src.points[idx[0]]  # 最近傍点の座標

        # 閾値内にあるかどうかをチェック
        inner_product = np.dot(dst.points[i], nearest_src_one_point)

        delta_angle = np.arccos(np.clip(inner_product,-1,1))
        if -np.deg2rad(threshold) <= delta_angle <= np.deg2rad(threshold):
            indices.add(idx[0])

    extracted_points = src.select_by_index(list(indices))
    return extracted_points

def generate_designated_init_arrows(points, start_pos=[0,0,0], color=[1,0,0]):
    """
    点からOpen3Dの矢印オブジェクトを生成します。

    Args:
        points (List[List[float]]): 矢印のベクトルを表す点のリスト。各点は3つの浮動小数点数で表されます。
        color (List[float], optional): 矢印の色を示すRGB値のリスト。デフォルトは[1, 0, 0]（赤）です。
        start_pos (List[float], optional): 矢印の始点の位置を示す3つの浮動小数点数のリスト。デフォルトは[0, 0, 0]です。

    Returns:
        List[Open3D.Geometry]: 矢印オブジェクトのリスト。各矢印はOpen3DのGeometryオブジェクトです。
    """
    arrow_objs = []
    
    # 制限された視点ベクトルを矢印で表現する
    for point in points:
        # 矢印の生成
        arrow = Arrow(start_pos, start_pos + point, color)
        arrow_obj = arrow.get_poligon()
        arrow_objs.append(arrow_obj)
    
    return arrow_objs

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

def align_vectors(rotational_axes: List[np.ndarray], reference_vector: Optional[np.ndarray] = np.array([1, 1, 1])) -> List[np.ndarray]:
    """
    回転軸を参照ベクトルに合わせて修正し、その後正規化します。

    Parameters:
        rotational_axes (list of numpy arrays): 回転軸のリスト。
        reference_vector (numpy array, optional): 回転軸を修正するための参照ベクトル。デフォルトは [1, 1, 1]。

    Returns:
        list of numpy arrays: 正規化された回転軸のリスト。
    """
    # 軸対称ベクトルの符号を修正する
    for i in range(len(rotational_axes)):
        dot_product = np.dot(rotational_axes[i], reference_vector)
        if dot_product < 0:
            rotational_axes[i] = -rotational_axes[i]

    # 正規化する
    return [axis / np.linalg.norm(axis) for axis in rotational_axes]


def interpolate_angles_by_linear(angle_list, num_interpolations):
    """
    直線補間を使用して角度を補間し、補間された角度のリストを返します。

    Parameters:
    - angle_list (list): 補間する角度のリスト。0から360までの角度が含まれる必要があります。
    - num_interpolations (int): 各角度間の補間数。

    Returns:
    - list: 補間された角度のリスト。
    """
    interpolated_angles = []
    for i in range(len(angle_list) - 1):
        start_angle = angle_list[i]
        end_angle = angle_list[i + 1]
        interpolated_angles.append(start_angle)  # 開始角度を追加
        for j in range(1, num_interpolations + 1):
            alpha = j / (num_interpolations + 1)  # 補間係数
            # 直線補間を使用して角度を補間
            interpolated_angle = start_angle + (end_angle - start_angle) * alpha
            interpolated_angles.append(interpolated_angle)
        interpolated_angles.append(end_angle)  # 終了角度を追加
    return interpolated_angles

def interpolate_angles_by_expo(angle_list, num_interpolations):
    """
    指数関数を使用して角度を補間し、補間された角度のリストを返します。

    Parameters:
    - angle_list (list): 補間する角度のリスト。0から360までの角度が含まれる必要があります。
    - num_interpolations (int): 各角度間の補間数。

    Returns:
    - list: 補間された角度のリスト。
    """
    interpolated_angles = []
    for i in range(len(angle_list) - 1):
        start_angle = angle_list[i]
        end_angle = angle_list[i + 1]
        interpolated_angles.append(start_angle)  # 開始角度を追加
        for j in range(1, num_interpolations + 1):
            alpha = j / (num_interpolations + 1)  # 補間係数
            # 指数関数を使用して角度を補間
            interpolated_angle = start_angle + (end_angle - start_angle) * (1 - np.cos(np.pi * alpha)) / 2
            interpolated_angles.append(interpolated_angle)
        interpolated_angles.append(end_angle)  # 終了角度を追加
    return interpolated_angles

def generate_circle_on_plane(axis_vector, point_on_axis, radius, angles):
    """
    与えられた軸と始点上の任意の軸ベクトルと直交する平面上に円を描く座標を生成します。

    Parameters:
    - axis_vector: 軸を示すベクトル。
    - point_on_axis: 軸上の始点。
    - radius: 円の半径。
    - angles: 円周上の角度のリスト。

    Returns:
    - np.array: 円を描く座標が格納されたnumpy配列。
    """
    # 与えられた軸に直交する2つのベクトルを見つける
    v1 = np.array([1, 0, 0]) if np.linalg.norm(axis_vector - np.array([1, 0, 0])) > 1e-6 else np.array([0, 1, 0])
    v2 = np.cross(axis_vector, v1)
    v1 = np.cross(axis_vector, v2)

    # 円周上の点を計算
    points = []
    for angle in angles:
        # 円周上の点の座標を計算
        circle_point = point_on_axis + radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
        points.append(circle_point)

    return np.array(points)

def get_view_matrix(upvector, campos_w, tarpos_w):
    """
    ビュー行列を取得する.
    -------------------------------------
    @引数 upvector : アップベクトル
    @引数 campos_w : world座標系上のカメラの位置(x,y,z) [m]
    @引数 tarpos_w : world座標系上のカメラの視点の先(x,y,z) [m]
    -------------------------------------
    @戻り値 world座標系からカメラ座標系への同次座標変換行列cHw
    """

    campos_w = np.array(campos_w)
    tarpos_w = np.array(tarpos_w)
    upvector = np.array(upvector)

    # z 軸 = (l - c) / ||l - c||
    z_cam = tarpos_w - campos_w
    z_cam = z_cam / np.linalg.norm(z_cam)
    

    # x 軸 = (z_cam × u) / ||z_cam × u||
    x_cam = np.cross(z_cam, upvector)

    # upvectorとz_camのベクトルがほぼ同じ方向の場合,x_camの外積の結果が0ベクトルになってしまう
    if np.count_nonzero(x_cam) == 0:
        upvector = np.array([1, 0, 0])
        # x 軸 = (z_cam × u) / ||z_cam × u||
        x_cam = np.cross(z_cam, upvector)

    x_cam = x_cam / np.linalg.norm(x_cam)

    # y 軸 = (z_cam × x_cam) / ||z_cam × x_cam||
    y_cam = np.cross(z_cam, x_cam) # panda3dの場合
    #y_cam = np.cross(x_cam, z_cam)# pyrenderの場合
    y_cam = y_cam / np.linalg.norm(y_cam)
    
    return x_cam,y_cam,z_cam

def create_angle_list(delta_angle):
    """
    0から360までの角度を指定された間隔で分割してリストを作成します。
    """
    if(delta_angle==0):
        # 指定された数値の倍数の角度のみをリストに含める
        original_angles = range(0,360,180)
    elif delta_angle < 0 or delta_angle >= 360:
        raise ValueError("Delta angle must be a positive number less than 360")
    else:
        # divisorを計算
        divisor = int(360 / delta_angle) + 1

        # 指定された数値の倍数の角度のみをリストに含める
        original_angles = [int(i * delta_angle) for i in range(divisor)]

    return original_angles


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process PLY file and generate point cloud.')
    parser.add_argument('--mesh', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--viewpoint', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--rot_sym_info_path', type=str, default=None, help='Path to information of rotation symmetry.')
    parser.add_argument("--output_dir", type=str, default='./result/viewpoints', 
                        help='output data directory')
    parser.add_argument('--gif', action='store_true', help='create GIF.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    
    args = parser.parse_args()
    
    if(args.mesh is None):
        # plyファイルが与えられない場合は例を出す
        rotational_axes = [np.array([-1, 0, 0]),np.array([0, -1, 0]),np.array([0, 0, -1])]
        delta_phis = [np.radians(0),np.radians(0),np.radians(0)]
        # rotational_axes と delta_phis を要素ごとに連結する
        rotational_symmetries = [[rotational_axes[i], [delta_phis[i]]] for i in range(len(rotational_axes))]
        
        # バウンディングボックスの対角線長さを取得(平面描写のために使う)
        bbox_diagonal_length = 1

    else:
        target_obj_file = args.mesh
        target_obj_filename=os.path.basename(target_obj_file)
        file_name_without_extension = os.path.splitext(target_obj_filename)[0]
        # file_path = f"./result/rotational_symmetry/{file_name_without_extension}_rotational_symmetry.json"
        file_path = args.rot_sym_info_path
        
        mesh = o3d.io.read_triangle_mesh(target_obj_file)
        mesh.compute_vertex_normals()

        # JSONファイルを読み込む
        with open(file_path, 'r') as file:
            rotational_symmetries = json.load(file)
        
        # バウンディングボックスの対角線長さを取得(平面描写のために使う)
        bbox_diagonal_length = np.linalg.norm(np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound()))

        rotational_axes=[]
        delta_phis=[]
        for (rotational_axis, delta_phi) in rotational_symmetries:
            rotational_axes.append(rotational_axis)
            delta_phis.append(np.deg2rad(delta_phi))
            
        rotational_axes = np.array(rotational_axes)
        delta_phis = np.array(delta_phis)


    rotational_axes = align_vectors(rotational_axes)

    num_points = 100

    if(args.viewpoint is not None):
        viewpoint_vectors = np.load(args.viewpoint)
    else:
        # 視点ベクトルを生成
        viewpoint_vectors = generate_shpere_surface_points_by_polarcoordinates(elevation_angle_range=[0, np.pi], 
                                                        azimuth_angle_range=[0, 2*np.pi], 
                                                        num_elevation=30, 
                                                        num_azimuth=30)
                                               
    viewpoints = o3d.geometry.PointCloud()
    viewpoints.points = o3d.utility.Vector3dVector(viewpoint_vectors)     
    
    # ワールド座標系
    world_frame     = CoordinateFrameMesh(scale=bbox_diagonal_length/2)
    world_frame_obj = world_frame.get_poligon()
    filtered_viewpoints = viewpoints

    if(args.debug):
        o3d.visualization.draw_geometries([viewpoints], mesh_show_back_face=True)
        
    
    for (rotational_axis, delta_phi) in zip(rotational_axes, delta_phis):
        print(np.rad2deg(delta_phi))
        # 回転行列を使ってz軸を回転軸ベクトルに一致させる
        z_R_n1 = rotation_matrix_to_align_with_dst_vec(np.array([0,0,1]), rotational_axis)
    
        
        restricted_viewpoints = generate_restricted_viewpoints(delta_phi, num_points)
        # restricted_viewpoints.scale(bbox_diagonal_length/2,center=[0,0,0])
        if(args.debug):
            o3d.visualization.draw_geometries([restricted_viewpoints,world_frame_obj], mesh_show_back_face=True)
            # o3d.visualization.draw_geometries([restricted_viewpoints,world_frame_obj,mesh], mesh_show_back_face=True)

        rotational_axis = rotational_axis / np.linalg.norm(rotational_axis)

        H=np.eye(4)
        H[:3,:3]=z_R_n1
        restricted_viewpoints.transform(H)

        if(args.debug):
            o3d.visualization.draw_geometries([restricted_viewpoints,world_frame_obj, viewpoints, mesh], mesh_show_back_face=True)

        filtered_viewpoints = find_nearest_neighbors(filtered_viewpoints, restricted_viewpoints)
        if(args.debug):
            o3d.visualization.draw_geometries([filtered_viewpoints,world_frame_obj], mesh_show_back_face=True)
       
        restricted_viewpoints.transform(np.linalg.inv(H))
        if(args.debug):
            o3d.visualization.draw_geometries([filtered_viewpoints,world_frame_obj], mesh_show_back_face=True)
        # print("元の視点ベクトル数:", len(viewpoints.points))
        # print("条件を満たす視点ベクトル数:", len(filtered_viewpoints.points))

    # H=np.eye(4)
    # H[:3,:3]=z_R_n1
    # filtered_viewpoints.transform(np.linalg.inv(H))

    # 描写用 ########################################################################
    # ワールド座標系
    world_frame     = CoordinateFrameMesh(scale=bbox_diagonal_length*0.5,)
    world_frame_obj = world_frame.get_poligon()
    
    origin_pos = [0,0,0]
    z_axis = np.array([0,0,1])

    plane_objs = []
    # # 鏡面反射平面の法線ベクトルを用いて平面の生成
    for rotational_axis in rotational_axes:
        
        H = np.eye(4)
        rotational_axis = rotational_axis / np.linalg.norm(rotational_axis)
        z_R_rotaxis = rotation_matrix_to_align_with_dst_vec(rotational_axis, z_axis)
        H[:3, :3] = z_R_rotaxis.T

        # 平面
        circle = Circle(pos=[0,0,0], rpy=[0,0,0], r=bbox_diagonal_length/2)
        circle.create_polygon()
        plane_obj = circle.get_poligon()
        plane_obj.transform(H)

        # ワイヤーフレームに変換
        plane_wire_obj = o3d.geometry.LineSet.create_from_triangle_mesh(plane_obj)
        plane_objs.append(plane_wire_obj)

        

    vp = copy.deepcopy(filtered_viewpoints)
    # filtered_arrow_objs = generate_designated_init_arrows(0.5*bbox_diagonal_length*np.array(vp.points)+mesh.get_center(), mesh.get_center())
    filtered_arrow_objs = generate_designated_init2end_arrows(bbox_diagonal_length*np.array(vp.points)+mesh.get_center(),
                                                              0.9*bbox_diagonal_length*np.array(vp.points)+mesh.get_center())
    
    if(args.debug):
        o3d.visualization.draw_geometries([*filtered_arrow_objs,  mesh], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([*filtered_arrow_objs, *plane_objs, mesh], mesh_show_back_face=True)



    if(args.gif):
        num_interpolations = 1  # 各区間で補間する点の数
        from_points = []

        for (rotational_axis, delta_phi) in zip(rotational_axes, delta_phis):
            original_angles = create_angle_list(delta_phi)

            # 視点先位置
            look_at_pos = mesh.get_center()

            # カメラ位置 = 視点先位置 + 大きさ * 方向ベクトル
            cameraPosition = look_at_pos + 0.5 *bbox_diagonal_length * rotational_axis

            x_cam,y_cam,z_cam = get_view_matrix(upvector=np.array([0, 0, 1]),
                                                campos_w=cameraPosition,
                                                tarpos_w=look_at_pos)

            angles = interpolate_angles_by_linear(original_angles, num_interpolations)

            angles = [np.deg2rad(ang) for ang in angles]
            circle_points = generate_circle_on_plane(x_cam, mesh.get_center(), 2*bbox_diagonal_length, angles)
            for point in circle_points:
                from_points.append(point)

            circle_points = generate_circle_on_plane(y_cam, mesh.get_center(), 2*bbox_diagonal_length, angles)
            for point in circle_points:
                from_points.append(point)

            circle_points = generate_circle_on_plane(z_cam, mesh.get_center(),2* bbox_diagonal_length, angles)
            for point in circle_points:
                from_points.append(point)
            

        output_dir = "./result/gif/restricted_view_by_rotation"
                        
        # output ディレクトリが存在しない場合は作成する
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # # JSONファイルに保存するパス
        output_file = os.path.join(output_dir, f"{file_name_without_extension}.gif")

        create_rotation_gif(np.array(from_points), [mesh, *filtered_arrow_objs], output_file, duration=5)

    # filtered_viewpointsを保存する
    view_points = np.array(filtered_viewpoints.points).tolist()

    print("*"*99)
    print("name:",file_name_without_extension)
    print("the number of viewpoints(before):",len(viewpoint_vectors))
    print("the number of viewpoints(after):",len(view_points))
    print("*"*99)


    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # # JSONファイルに保存するパス
    output_file_path = os.path.join(args.output_dir,f"{file_name_without_extension}.npy")


    # NumPy配列を保存
    np.save(output_file_path, view_points)
