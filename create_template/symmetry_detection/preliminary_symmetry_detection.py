import open3d as o3d
import numpy as np
from scipy.optimize import leastsq
import copy
import pprint
import os
import json
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure
import trimesh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from typing import Optional,List

from utils.reflection_plane import ReflectionPlane
from utils.object_visualization_helper import CoordinateFrameMesh, Cylinder, Arrow, Circle, create_rotation_gif

from utils.sphere_points_creator import hinter_sampling


"""対称面の候補を計算する
python symmetry\preliminary_symmetry_detection.py --mesh symmetry\tmp_obj_reduce\95D95-06010\95D95-06010.ply
 --points symmetry\observability\95D95-06010_observable.ply --output_dir  symmetry/result
"""

#描写用#################################################################################
def rotation_matrix_to_align_with_z_axis(vec): 
    """ 
    ベクトルvecをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列を計算する。 
 
    Parameters: 
        vec (numpy.ndarray): 3次元ベクトル (x, y, z) を表すNumPy配列。 
 
    Returns: 
        numpy.ndarray: ベクトルvecをx軸に一致させるための回転行列。vecがx軸と一致している場合は単位行列が返される。 
    """ 
    z_axis = np.array([0, 0, 1])  # x軸の単位ベクトル 
 
    # ベクトルvecとx_axisの外積を計算 
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
    rotation_matrix = np.array([ 
        [cos_theta + rotation_axis[0] ** 2 * (1 - cos_theta), rotation_axis[0] * rotation_axis[1] * (1 - cos_theta) - rotation_axis[2] * sin_theta, rotation_axis[0] * rotation_axis[2] * (1 - cos_theta) + rotation_axis[1] * sin_theta], 
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_theta) + rotation_axis[2] * sin_theta, cos_theta + rotation_axis[1] ** 2 * (1 - cos_theta), rotation_axis[1] * rotation_axis[2] * (1 - cos_theta) - rotation_axis[0] * sin_theta], 
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_theta) - rotation_axis[1] * sin_theta, rotation_axis[2] * rotation_axis[1] * (1 - cos_theta) + rotation_axis[0] * sin_theta, cos_theta + rotation_axis[2] ** 2 * (1 - cos_theta)] 
    ]) 
     
    return rotation_matrix 

def sample_reflection_normal_by_view_sphere(min_n_views,
                                azimuth_range=(0, np.pi),
                                elev_range=(-0.5 * np.pi, 0.5 * np.pi), ):
    """
    球面からの鏡面反射平面の法線ベクトルをサンプリングを行います。

    Parameters:
        min_n_views (int): 全体の球面上にサンプリングする最小のポイント数。
        azimuth_range (tuple): サンプリングする方位角の範囲（ラジアン単位）。デフォルトは (0, 1 * np.pi)。
        elev_range (tuple): サンプリングする仰角の範囲（ラジアン単位）。デフォルトは (-0.5 * np.pi, 0.5 * np.pi)。

    Returns:
        list: サンプリングされた反射法線を表す3Dポイントのリスト。
    """

    # ユニットスフィア上のポイントをサンプリング
    points, points_level = hinter_sampling(min_n_views, radius=1)

    point_list = []
    for point in points:
        # 方位角を計算 [0, 2π]
        azimuth = np.arctan2(point[1], point[0])
        if azimuth < 0:
            azimuth += 2.0 * np.pi

        # 仰角を計算 [-π/2, π/2]
        a = np.linalg.norm(point)
        b = np.linalg.norm([point[0], point[1], 0])
        elevation = np.arccos(b / a)
        if point[2] < 0:
            elevation = -elevation

        # サンプリングされた点が指定された方位角および仰角の範囲内にあるか確認
        if not (azimuth_range[0] <= azimuth < azimuth_range[1] and
                elev_range[0] <= elevation < elev_range[1]):
            continue

        # 球面座標を直交座標に変換
        px = np.cos(azimuth) * np.cos(elevation)
        py = np.sin(azimuth) * np.cos(elevation)
        pz = np.sin(elevation)
        point_list.append((px, py, pz))

    return point_list
    

def draw_planes(pcd, plane_normals, bbox_diagonal_length):
    """
    法線で定義される平面を点群で可視化します。

    Parameters:
        pcd (open3d.geometry.PointCloud): 点群。
        normals (list of numpy.ndarray): 平面を表す法線ベクトルのリスト。

    Returns:
        None
    """

    plane_objs = []
    arrow_objects = []
    plane_normals = np.array(plane_normals)
    for plane_normal in plane_normals:
        # normalをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列
        rotation_matrix = rotation_matrix_to_align_with_z_axis(plane_normal)

        # 回転行列を適用した変換行列を作成
        H = np.eye(4)
        H[:3, :3] = np.linalg.inv(rotation_matrix)

        # 円の生成
        circle = Circle(pos=[0,0,0], rpy=[0,0,0], r=bbox_diagonal_length*0.6)
        circle.create_polygon()
        plane_obj = circle.get_poligon()
        plane_obj.transform(H)

        # ワイヤーフレームに変換
        plane_obj_wire = o3d.geometry.LineSet.create_from_triangle_mesh(plane_obj)
        plane_objs.append(plane_obj_wire)

        # 矢印の生成
        rotated_arrow = Arrow([0, 0, 0], [0,0,0.5*bbox_diagonal_length])
        rotated_arrow_obj = rotated_arrow.get_poligon()
        rotated_arrow_obj.transform(H)
        arrow_objects.append(rotated_arrow_obj)

    # Open3Dを使用して平面と点群を可視化
    o3d.visualization.draw_geometries([pcd, *plane_objs], mesh_show_wireframe=True, mesh_show_back_face=True)
    o3d.visualization.draw_geometries([pcd, *plane_objs, *arrow_objects], mesh_show_wireframe=True, mesh_show_back_face=True)
    o3d.visualization.draw_geometries(arrow_objects, mesh_show_wireframe=True, mesh_show_back_face=True)

###################################################################################
def compute_radius_by_mean_distance(point_cloud, k_neighbors, distance_ratio=2.0):
    """平均距離の割合から最近傍探索半径を計算

    Args:
        point_cloud (_type_): 元点群
        k_neighbors (_type_): K近傍法で使用する近傍点の数
        distance_ratio (float, optional): 平均距離の何倍を半径とするか. Defaults to 2.0.

    Returns:
        radius: 平均距離の割合から計算した最近傍探索半径
    """
    distances = []

    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    for i in range(len(point_cloud.points)):
        # 各点のK近傍点を取得
        [k, idx, _] = kd_tree.search_knn_vector_3d(point_cloud.points[i], k_neighbors)

        if k > 1:  # 自分自身も含まれるので2以上の点がある場合
            # 各点までの距離を計算
            neighbors = np.asarray(point_cloud.points)[idx[1:]]
            distances.extend(np.linalg.norm(neighbors - point_cloud.points[i], axis=1))

    # 平均距離を計算し、指定した割合を掛けて半径を得る
    mean_distance = np.mean(distances)
    radius = mean_distance * distance_ratio

    return radius

def filter_reflection_points(pcd, reflected_pcd):
    """反射点群が入力点群の反射対称になっているかをフィルタリングする

    Parameters:
        pcd (open3d.geometry.PointCloud): 入力点群。
        reflected_pcd (open3d.geometry.PointCloud): 反射点群。

    Returns:
        tuple: フィルタリングされた点群と対応する反射点群の組み合わせ。
    """
    # 法線の差の閾値
    threshold_diff_angle = np.deg2rad(45)

    # 点群の空間分解能
    global threshold_distance

    # 最近傍点の検索
    kd_tree_reflected = o3d.geometry.KDTreeFlann(reflected_pcd)
    reflected_pcd_normals = np.array(reflected_pcd.normals)
    
    # 反射と見なせる点群かどうかを表すフラグのリスト
    inlier_indices = []

    for (pcd_point, pcd_normal) in zip(pcd.points, pcd.normals):
        [k, index, distance] = kd_tree_reflected.search_knn_vector_3d(pcd_point, 1)
        
        # 距離が閾値未満かどうかでフラグを設定
        inliner = distance < threshold_distance

        reflected_normal = reflected_pcd_normals[index][0]

        # 法線ベクトルの成す角を計算
        dot_product = np.dot(pcd_normal, reflected_normal)

        # 内積を[-1, 1]の範囲にクリップ
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # arccosを計算
        angle = np.arccos(dot_product)

        # 鏡面反射前と反射後の対応法線の成す角が閾値より小さいなら、その点は棄却する
        inliner = np.abs(angle) < threshold_diff_angle

        inlier_indices.append(inliner)
    
    # フィルタリングされた点群を選択
    filtered_pcd = pcd.select_by_index(np.where(inlier_indices)[0])
    filtered_reflected_pcd = reflected_pcd.select_by_index(np.where(inlier_indices)[0])

    return filtered_pcd, filtered_reflected_pcd,

def modified_wendland_function(alpha, l):
    alpha_l = alpha * l
    mask = alpha_l <= 2.6
    result = np.zeros_like(alpha_l)
    result[mask] = ((1 - alpha_l[mask] / 2.6)**5) * (8 * (alpha_l[mask] / 2.6)**2 + 5 * alpha_l[mask] / 2.6 + 1)
    return result

def compute_alpha(X):
    centroid = np.mean(X, axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    return 15 / np.mean(distances)


def get_symmetry_score(pcd, reflected_pcd):
    X = np.array(pcd.points)
    Y = np.array(reflected_pcd.points)
    alpha = compute_alpha(X)

    distances = np.linalg.norm(Y[:, np.newaxis] - X, axis=2)
    s = np.sum(modified_wendland_function(alpha, distances))

    return s


def calculate_symmetry_filter(pcd, reflected_pcd,tau_inliner=0.99,tau_fit=0.95):

    # 点群の空間分解能
    global threshold_distance

    threshold_diff_angle = np.deg2rad(45)

    # 最近傍点の検索
    kd_tree_reflected = o3d.geometry.KDTreeFlann(reflected_pcd)
    reflected_pcd_normals = np.array(reflected_pcd.normals)
    
    # 反射と見なせる点群かどうかを表すフラグのリスト
    inliner_indices = []

    fitness_scores=[]

    # 反射と見なせる点群かどうかを表すフラグのリスト
    inliner_indices = []

    for (pcd_point,pcd_normal) in zip(pcd.points, pcd.normals):
        [k, index, distance] = kd_tree_reflected.search_knn_vector_3d(pcd_point, 1)
        if(distance < threshold_distance):
            inliner=True
        else:
            inliner=False
        reflected_normal = reflected_pcd_normals[index][0]
                
        # 法線ベクトルの成す角を計算
        dot_product = np.dot(pcd_normal, reflected_normal)

        # 内積を[-1, 1]の範囲にクリップ
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # arccosを計算
        angle = np.arccos(dot_product)

        # 鏡面反射前と反射後の対応法線の成す角が閾値より小さいなら、その点は棄却する
        if(np.abs(angle) < threshold_diff_angle):
            inliner=True
            fitness_scores.append(1-angle/np.pi)
        else:
            inliner=False
        inliner_indices.append(inliner)
        
    filtered_pcd = pcd.select_by_index(np.where(inliner_indices)[0])

    inlier_score = len(filtered_pcd.points)/len(pcd.points)
    fitness_score = np.sum(fitness_scores)/len(filtered_pcd.points)

    if(inlier_score > tau_inliner and fitness_score > tau_fit):
        return True, inlier_score, fitness_score
    else:
        return False, inlier_score, fitness_score


def orient_normals_to_align_outer_direction(pcd, scale_factor=1.1):
    """法線を外向きに整列させる関数。

    Parameters:
        pcd (open3d.geometry.PointCloud): 法線を整列させる点群。
        scale_factor (float, optional): 対応する点群を拡大する際の倍率。デフォルトは1.1。

    Returns:
        open3d.geometry.PointCloud: 法線が整列された点群。
    """
    # pcdのコピーを作成
    pcd_dst = copy.copy(pcd)

    # 対応する点群をscale_factor倍大きくしたものを生成
    pcd_large = copy.copy(pcd)
    pcd_large = pcd_large.scale(scale_factor, pcd_large.get_center())

    # pcdの各法線を元に、pcd_largeの法線を向きを合わせて更新
    for i, normal in enumerate(np.asarray(pcd.normals)):
        # pcdからpcd_largeへのベクトルを計算
        vector_to_large = np.asarray(pcd_large.points)[i] - np.asarray(pcd.points)[i]

        # 法線を調整
        pcd_dst.normals[i] = vector_to_large / np.linalg.norm(vector_to_large)

        # もし法線が逆向きなら反転させる
        # dot_product = np.dot(normal, vector_to_large)
        # if dot_product < 0:
        #     pcd_dst.normals[i] = -normal
        # else:
        #     pcd_dst.normals[i] = normal

    return pcd_dst


def cost_function(param_reflection_distance, pcd, n):
    """
    コスト関数。

    Parameters:
        param_reflection_distance (numpy.ndarray): 鏡面反射平面の位置ベクトル。
        pcd (open3d.geometry.PointCloud): 元の点群。
        n (numpy.ndarray): 鏡面の法線ベクトル。

    Returns:
        numpy.ndarray: 各点対の距離の中央値。
    """

    # 鏡面反射変換行列4x4の取得
    reflection_matrix = ReflectionPlane(plane_parameter=[*n,param_reflection_distance]).reflection_matrix

    # 点群Pの鏡面反射点群生成
    pcd_transformed = copy.deepcopy(pcd)
    pcd_transformed.transform(reflection_matrix)

    # 鏡面反射点群のフィルタリング
    #pcd,pcd_transformed = filter_reflection_points(pcd,pcd_transformed)
    
    # 点群Piと鏡面反射点群Qiの距離を計算
    each_pointpairs_dist = np.asarray(pcd.points) - np.asarray(pcd_transformed.points)
    
    cost = np.median(each_pointpairs_dist, axis=0)
    # cost = np.sum(each_pointpairs_dist, axis=0)
    return cost


def cost_function2(param_reflection_distance, pcd, n):
    """
    コスト関数

    Parameters:
        param_reflection_distance (numpy.ndarray): 鏡面反射平面の位置ベクトル。
        pcd (open3d.geometry.PointCloud): 元の点群。
        n (numpy.ndarray): 鏡面の法線ベクトル。

    Returns:
        numpy.ndarray: 各点対の距離の中央値。
    """

    # 鏡面反射変換行列4x4の取得
    reflection_matrix = ReflectionPlane(plane_parameter=[*n,param_reflection_distance]).reflection_matrix

    # 点群Pの鏡面反射点群生成
    pcd_transformed = copy.deepcopy(pcd)
    pcd_transformed.transform(reflection_matrix)

    # 鏡面反射点群のフィルタリング
    pcd,pcd_transformed = filter_reflection_points(pcd,pcd_transformed)

    distances = []

    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pcd.points)):
        # 各点のK近傍点を取得
        [k, idx, _] = kd_tree.search_knn_vector_3d(pcd_transformed.points[i], 1)

        # 各点までの距離を計算
        distances.append(list(np.asarray(pcd_transformed.points)[idx[0]] - pcd.points[i]))

    cost = np.mean(distances, axis=0)
    # cost = np.sum(each_pointpairs_dist, axis=0)
    return cost

def align_point_clouds_leastsq(pcd, param_reflection_distance, plane_normal):
    """
    レーベンバーグ・マッカート法を使用して点群を対称に整列させる関数。

    Parameters:
        pcd (open3d.geometry.PointCloud): 整列対象の点群。
        param_reflection_distance (float): 鏡面反射平面への距離ベクトルの初期値。
        plane_normal (numpy.ndarray): 鏡面の法線ベクトル。

    Returns:
        tuple: 反射対称になっているかを表すのフラグ、反射対称点群、最適な反射平面の位置ベクトル。
    """

    if not pcd.has_normals():
        raise ValueError("Error: pointcloud does not have normals.")
    
    # leastsqを使用して最適なreflection_posを求める
    optimal_reflectional_plane_distance, _ = leastsq(cost_function, param_reflection_distance, args=(pcd, plane_normal))

    return optimal_reflectional_plane_distance

def extract_feature_points_by_using_cavature(mesh, mode="gauss"):

    # メッシュモデルの読み込む
    # mesh = trimesh.Trimesh(vertices=pcd_o3d.points)
    # mesh = trimesh.Trimesh(vertices=pcd.vertices,faces=pcd.triangles)

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(mesh.vertices)
    threshold_distance   = compute_radius_by_mean_distance(pcd,3,0.01)
    
    if mode == "gauss":
        cavature = discrete_gaussian_curvature_measure(mesh, mesh.vertices, threshold_distance)
    else:
        cavature = discrete_mean_curvature_measure(mesh, mesh.vertices, threshold_distance)

    mesh_vettices_cavature_low_thresh,mesh_vettices_cavature_high_thresh = np.percentile(cavature, [10,90])

    mesh_vettices_cavature_low = np.array(mesh.vertices)[mesh_vettices_cavature_low_thresh>=cavature]
    mesh_vettices_cavature_high = np.array(mesh.vertices)[mesh_vettices_cavature_high_thresh<=cavature]
    mesh_feature_vetices = np.vstack([mesh_vettices_cavature_low,mesh_vettices_cavature_high])

    feature_pcd = o3d.geometry.PointCloud()
    feature_pcd.points=o3d.utility.Vector3dVector(mesh_feature_vetices)

    return feature_pcd

def calculate_plane_diff_scale_from_normals(normal_vector_p1: np.ndarray, normal_vector_p2: np.ndarray) -> float:
    """
    2つの平面の法線ベクトルから平面差分尺度を計算します。

    Parameters:
        normal_vector_p1 (numpy.ndarray): 平面p1の法線ベクトル。
        normal_vector_p2 (numpy.ndarray): 平面p2の法線ベクトル。

    Returns:
        float: 2つの平面の法線ベクトル間の角度（度単位）に基づいて計算された平面差分尺度。
    """
    normal_vector_p1 = normal_vector_p1 / np.linalg.norm(normal_vector_p1)
    normal_vector_p2 = normal_vector_p2 / np.linalg.norm(normal_vector_p2)
    angle_radians = np.arccos(np.clip(np.dot(normal_vector_p1, normal_vector_p2),-1, 1))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def filter_nearby_planes_normal_vector(normals: np.ndarray, angle_threshold_degrees: float = 5) -> tuple:
    """
    与えられた法線ベクトルのリストから、特定の角度以下の角度を持つ法線ベクトルペアを廃棄します。

    Parameters:
        normals (numpy.ndarray): 平面の法線ベクトルのリスト。
        angle_threshold_degrees (float): 廃棄する法線ベクトルペアの角度の閾値（度単位）。

    Returns:
        tuple: 残された平面の位置と法線ベクトルのリスト、および残ったインデックスのリスト。
    """
    discarded_indices = set()

    for i in range(len(normals)):
        for j in range(i+1, len(normals)):
            angle_degrees = calculate_plane_diff_scale_from_normals(normals[i], normals[j])
            if angle_degrees < angle_threshold_degrees:
                # 角度が閾値以下の場合、片方の法線ベクトルを廃棄する
                if i not in discarded_indices:
                    discarded_indices.add(i)

    # 廃棄されなかったもののみを残す
    remaining_indices = [i for i in range(len(normals)) if i not in discarded_indices]
    remaining_normals = [normals[i] for i in remaining_indices]
    
    return remaining_normals, remaining_indices

def get_feature_points(pcd):
    radius_normal = np.mean(P.compute_nearest_neighbor_distance())
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=10))
    return pcd_fpfh


def render_single_shot(mesh):
    # 可視化ウィンドウを作成
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1800, height=900)
    vis.add_geometry(mesh)

    # カメラの位置を設定
    to_point = np.array(mesh.get_center())  # 注視点はメッシュの中心

    # カメラの設定
    ctr = vis.get_view_control()
    # 完全に真上だと対称平面が見えない可能性があるので少しずらす
    front = np.array([0.01, 0.01, 1.0])
    ctr.set_front(front/np.linalg.norm(front))
    ctr.set_lookat(to_point)
    ctr.set_up([0, -1, 0])
    # ctr.set_zoom(0.6)
    ctr.set_zoom(1)
    # ctr.set_zoom(0.9)

    # ウィンドウの更新とレンダリング
    vis.poll_events()
    vis.update_renderer()

    # 画像をキャプチャして保存
    o3d_screenshot_mat = vis.capture_screen_float_buffer()
    image = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
    image = Image.fromarray(image, "RGB")

    # ウィンドウを閉じる
    vis.destroy_window()

    return np.array(image)

def render_single_shot_from_direction(mesh, direction, mesh_show_back_face=True):
    # 可視化ウィンドウを作成
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)

    # メッシュの両面を表示するために法線を計算して反転する
    mesh.compute_vertex_normals()

    vis.add_geometry(mesh)

    # カメラの位置を設定
    to_point = np.array(mesh.get_center())  # 注視点はメッシュの中心

    # カメラの設定
    front = np.array(direction)
    front /= np.linalg.norm(front)  # 方向ベクトルを正規化
    

    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_lookat(to_point)

    upper=np.array([0, -1, 0])
    
    if 1.0 - np.abs(np.dot(upper,front)) < 1e-03:
         # フロントベクトルがy軸に対してほぼ直行の場合、z軸を下方向とする
        upper=np.array([0, 0, -1])

    ctr.set_up(upper)
    ctr.set_zoom(0.5)

    # メッシュの裏面を表示するかどうかを設定
    vis.get_render_option().mesh_show_back_face = mesh_show_back_face

    # ウィンドウの更新とレンダリング
    vis.poll_events()
    vis.update_renderer()

    # 画像をキャプチャして保存
    o3d_screenshot_mat = vis.capture_screen_float_buffer()
    image = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
    image = Image.fromarray(image, "RGB")

    # ウィンドウを閉じる
    vis.destroy_window()

    return image

def render_from_multiple_directions(mesh):
    delta = 0.001
    directions = [
    [1.0, 1.0, 1.0],  
    [1.0, delta, delta],  # x軸方向
    [-1.0, -delta, -delta], # -x軸方向
    [delta, 1.0, delta],  # y軸方向
    [-delta, -1.0, -delta], # -y軸方向
    [delta, delta, 1.0],  # z軸方向
    [-delta, -delta, -1.0],  # -z軸方向
    ]
    rendered_images = []
    for direction in directions:
        rendered_image = render_single_shot_from_direction(mesh, direction)
        rendered_images.append(rendered_image)
    
    # 画像を縦に並べて1つの画像にする
    max_width = max(image.size[0] for image in rendered_images)
    total_height = sum(image.size[1] for image in rendered_images)
    combined_image = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for image in rendered_images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.size[1]
    
    return combined_image


def plot_images(images):
    num_images = len(images)
    num_cols = min(num_images, 12)  # サブプロットの列数は最大でも12枚分

    # サブプロットの作成
    fig, axs = plt.subplots(1, num_cols, figsize=(3*num_cols, 3))

    # 画像をサブプロットに追加
    for i, image in enumerate(images):
        ax = axs[i % num_cols] if num_cols > 1 else axs
        ax.imshow(image)
        ax.set_title(f"symmetry {i}")  # 画像にインデックス番号で名前を付ける
        ax.axis('off')

    # 余白の調整
    # plt.tight_layout()
    # plt.tight_layout(rect=[0,0,1,0.96])
    plt.subplot_tool()
    # グラフを表示q
    plt.show()


def refine_reflection_plane(P, reflection_matrix, relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=200, debug=False):
    # ICPでPとQを詳細マッチングする

    th = 0.02 *1000

    P = copy.deepcopy(P)
    Q = copy.deepcopy(P)
    Q.transform(reflection_matrix)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = relative_fitness, # fitnessの変化分がこれより小さくなったら収束
                                        relative_rmse = relative_rmse, # RMSEの変化分がこれより小さくなったら収束
                                        max_iteration = max_iteration) # 反復1回だけにする
    est_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    if debug:
        P.paint_uniform_color([1, 0, 0])
        Q.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([P, Q])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        Q, P, th,
        estimation_method=est_method,
        criteria=criteria
    )

    rot=reg_p2p.transformation
    refine_reflection_matrix = rot @ reflection_matrix

    # マッチングした結果を反映
    # Q=Q.transform(rot)
    Q = copy.deepcopy(P)
    Q=Q.transform(refine_reflection_matrix)

    
    # デバッグモードが有効な場合はマッチング結果を描画
    if debug:
        P.paint_uniform_color([1, 0, 0])
        Q.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([P, Q])


    reflection_plane = ReflectionPlane(reflection_matrix=refine_reflection_matrix)
    plane_normal = reflection_plane.plane_normal
    plane_pos = reflection_plane.calculate_plane_pos(np.array(P.points))

    return [plane_pos, plane_normal]

if __name__ == "__main__":
    
    # target_obj_file = "./tmp_obj_reduce/000008/000008.ply"
    
    parser = argparse.ArgumentParser(description='Process PLY file and generate point cloud.')
    parser.add_argument('--mesh', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--points', type=str, default=None, help='Path to PLY file.')
    parser.add_argument("--output_dir", default="result", help="name of output ply file")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

    args = parser.parse_args()
    target_obj_file = args.mesh
    target_obj_filename=os.path.basename(target_obj_file)
    file_name_without_extension = os.path.splitext(target_obj_filename)[0]

    target_observabile = args.points #f"./observability/{file_name_without_extension}_observable.ply"
    # target_obj_file = "teapot.ply"
    # target_obj_file = "monkey.ply"
    
    # 反射対称面の確認用のメッシュモデル読み込み
    mesh = o3d.io.read_triangle_mesh(target_obj_file)
    mesh.compute_vertex_normals()

    # mesh_trimesh = trimesh.load(target_obj_file)
    # P = extract_feature_points_by_using_cavature(mesh_trimesh)

    
    P = o3d.io.read_point_cloud(target_obj_file)
    P = mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5) 
    # ダウンサンプリングを行う
    voxel_size = 0.002*1000  # ダウンサンプリングの解像度を適切に指定する
    P = P.voxel_down_sample(voxel_size)
    # P = o3d.io.read_point_cloud(target_observabile)


    # ダウンサンプリング後の点群数を表示
    print(f"Downsampled Point Cloud: {len(P.points)} points")

    # 3Dビューワーで点群を表示する（オプション）
    # o3d.visualization.draw_geometries([P])

    # バウンディングボックスの対角線長さを取得(平面描写のために使う)
    bbox_diagonal_length = np.linalg.norm(np.asarray(P.get_max_bound()) - np.asarray(P.get_min_bound()))

    P.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    P = orient_normals_to_align_outer_direction(P)

    threshold_distance   = compute_radius_by_mean_distance(P,10,2.0)


    # reflection_normal_vectors = sample_reflection_normal_by_view_sphere(72)
    reflection_normal_vectors = sample_reflection_normal_by_view_sphere(300)

    if(args.debug):
        draw_planes(P,reflection_normal_vectors, bbox_diagonal_length)

    param_reflection_distance = 0.0  # 初期の反射位置の推定値
    candidates=[]
    for plane_normal in reflection_normal_vectors:
        plane_normal = np.array(plane_normal)
        # 反射行列4x4
        init_reflection_matrix = ReflectionPlane(plane_parameter=[*plane_normal,param_reflection_distance]).reflection_matrix
        # 反射点群
        Q_init = copy.deepcopy(P)
        Q_init.transform(init_reflection_matrix)

        # 位置合わせ前
        # o3d.visualization.draw_geometries([P, Q_init])

        # 位置合わせ
        optimal_reflectional_plane_distance = align_point_clouds_leastsq(P, param_reflection_distance, plane_normal)

        # 最適な反射位置での変換行列を取得
        optimal_reflection_matrix = ReflectionPlane(plane_parameter=[*plane_normal,optimal_reflectional_plane_distance]).reflection_matrix

        print("optimal_reflectional_plane_distance", optimal_reflectional_plane_distance)
        # 点群Pの鏡面反射点群生成
        Q_aligned = copy.deepcopy(P)
        Q_aligned.transform(optimal_reflection_matrix)
    

        # 結果の表示
        # o3d.visualization.draw_geometries([P, Q_aligned])
        candidates.append([optimal_reflectional_plane_distance,plane_normal])


    # o3d.visualization.draw_geometries([mesh, P], mesh_show_back_face=True)


    # テスト用の点群PとQ_initを作成
    # P = o3d.io.read_point_cloud(target_obj_file)
    P = o3d.io.read_point_cloud(target_observabile)
    P = mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5) 
    # ダウンサンプリングを行う
    voxel_size = 0.002*1000  # ダウンサンプリングの解像度を適切に指定する
    P = P.voxel_down_sample(voxel_size)

    # ダウンサンプリング前の点群数を表示
    print(f"Original Point Cloud: {len(P.points)} points")
    scores=[]
    good_ratios= []
    # 評価
    for (optimal_reflectional_plane_distance,plane_normal) in candidates:
        plane_normal = np.array(plane_normal)
        # 反射行列4x4
        reflection_matrix = ReflectionPlane(plane_parameter=[*plane_normal,optimal_reflectional_plane_distance]).reflection_matrix
        Q = copy.deepcopy(P)
        Q.transform(reflection_matrix)

        score = get_symmetry_score(P,Q)
        scores.append(score)
        print(score)

    scores = np.array(scores)
    good_ratios = np.array(good_ratios)

    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(scores.reshape(-1, 1))

    # 1. クラスタリングの結果を確認
    print("labels:", kmeans.labels_)

    # 2. クラスタリングされたラベルが正しい数であることを確認
    print("Number of labels:", len(kmeans.labels_))  # ラベルの数を確認

    # クラスタリング結果をプロットする関数
    def plot_clusters(data, labels, centers):
        plt.figure(figsize=(8, 6))
        plt.scatter(data, np.zeros_like(data), c=labels, cmap='viridis', marker='o', s=50, alpha=0.8)
        plt.scatter(centers, np.zeros_like(centers), c='red', marker='x', s=100, label='Cluster Centers')
        plt.xlabel('Data')
        plt.title('Clustering Result')
        plt.legend()
        plt.show()

    if(args.debug):
        # クラスタリング結果をプロット
        plot_clusters(scores, kmeans.labels_, kmeans.cluster_centers_)

    # クラスタリング結果の中心点を取得
    centers = kmeans.cluster_centers_

    # クラスタリング中心位置が大きい方のクラスターのインデックスを抽出
    larger_cluster_index = np.argmax(np.sum(centers, axis=1))

    # クラスタリング中心位置が大きい方のクラスターのインデックスだけを抽出
    larger_cluster_indices = np.where(kmeans.labels_ == larger_cluster_index)[0]
    scores = scores[larger_cluster_indices]

    # 降順でのインデックスを取得
    candidates = [c for i,c in enumerate(candidates) if i in larger_cluster_indices]

    images = []
    plane_objs=[]
    reflection_planes = []
    # 良かった結果のみ表示
    for (optimal_reflectional_plane_distance,plane_normal) in candidates:
        
        plane_normal = np.array(plane_normal)
        # 反射行列4x4
        reflection_plane = ReflectionPlane(plane_parameter=[*plane_normal,optimal_reflectional_plane_distance])
        reflection_matrix = reflection_plane.reflection_matrix
        reflection_plane_pos = reflection_plane.calculate_plane_pos(np.array(P.points))

        Q = copy.deepcopy(P)
        Q.transform(reflection_matrix)

        # normalをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列
        rotation_matrix = rotation_matrix_to_align_with_z_axis(plane_normal)

        # reflection_planes.append(reflection_matrix.tolist())

        H=np.eye(4)
        # 円盤のz軸ベクトルをnormalに一致させる
        H[:3, :3] = np.linalg.inv(rotation_matrix)
        H[:3, 3]  = reflection_plane_pos
        
        # 回転行列をrpyに変換
        # rpy_angles = Rotation.from_matrix(rotation_matrix).as_euler('zyx',degrees=False)
        circle = Circle(pos=[0,0,0], rpy=[0,0,0], r=bbox_diagonal_length*0.6)
        circle.create_polygon()

        plane_obj = circle.get_poligon()
        plane_obj.transform(H)

        # ワイヤーフレームに変換
        plane_obj_wire = o3d.geometry.LineSet.create_from_triangle_mesh(plane_obj)
        plane_objs.append(plane_obj_wire)

        arrow = Arrow(P.get_center(), P.get_center()+0.5*bbox_diagonal_length*plane_normal, color=[1.0,0.0,0.0])
        arrow_obj = arrow.get_poligon()
    
        # 結果の表示
        # o3d.visualization.draw_geometries([arrow_obj, mesh, P, Q, plane_obj_wire], mesh_show_back_face=True)

        # o3d.visualization.draw_geometries([arrow_obj, P, Q, plane_obj_wire], mesh_show_back_face=True)
        # img = render_from_multiple_directions(mesh+plane_obj)
        # images.append(img)

    # plot_images(images)

    # 反射対称面の確認用のメッシュモデル読み込み
    mesh = o3d.io.read_triangle_mesh(target_obj_file)
    mesh.compute_vertex_normals()
    if(args.debug):
        o3d.visualization.draw_geometries([mesh, *plane_objs], mesh_show_back_face=True)


    # refiner
    images = []
    plane_objs = []
    # P = o3d.io.read_point_cloud(target_obj_file)
    P = o3d.io.read_point_cloud(target_observabile)
    # P = mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5) 
    for (optimal_reflectional_plane_distance,plane_normal) in candidates:


        plane_normal = np.array(plane_normal)
        # 反射行列4x4
        reflection_matrix = ReflectionPlane(plane_parameter=[*plane_normal,optimal_reflectional_plane_distance]).reflection_matrix

        Q = copy.deepcopy(P)
        Q.transform(reflection_matrix)

        # 反射平面の精製
        plane_pos_refine, plane_normal_refine = refine_reflection_plane(P, reflection_matrix, debug=args.debug)
        
        # 反射行列4x4
        reflection_matrix_refine = ReflectionPlane(plane_pos=plane_pos_refine, plane_normal=plane_normal_refine).reflection_matrix
        
        Q = copy.deepcopy(P)
        Q.transform(reflection_matrix_refine)

        # normalをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列
        rotation_matrix = rotation_matrix_to_align_with_z_axis(plane_normal_refine)

        reflection_planes.append([plane_pos_refine.tolist(),plane_normal_refine.tolist()])

        # 円盤のz軸ベクトルをnormalに一致させる
        H=np.eye(4)
        H[:3, :3] = rotation_matrix.T
        H[:3, 3]  = plane_pos_refine
        
        # 回転行列をrpyに変換
        circle = Circle(pos=[0,0,0], rpy=[0,0,0], r=bbox_diagonal_length/2)
        circle.create_polygon()

        plane_obj = circle.get_poligon()
        plane_obj.transform(H)

        # ワイヤーフレームに変換
        plane_obj_wire = o3d.geometry.LineSet.create_from_triangle_mesh(plane_obj)
        plane_objs.append(plane_obj_wire)

        arrow = Arrow([0,0,0], [0,0,0.5*bbox_diagonal_length], color=[1.0,0.0,0.0])
        arrow_obj = arrow.get_poligon()
        arrow_obj.transform(H)
    
        # 結果の表示
        # o3d.visualization.draw_geometries([arrow_obj, mesh, P, Q, plane_obj_wire], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([arrow_obj, P, Q, plane_obj_wire], mesh_show_back_face=True)
        # img = render_from_multiple_directions(mesh+plane_obj)
        # images.append(img)

    print("before",len(reflection_planes) )
    # normals=[]
    # for (pos, normal) in reflection_planes:
    #     normals.append(normal)
    normals = np.array([normal for (pos, normal) in reflection_planes])
    # ほぼ同じものは削除
    remain_axes, remaining_indices = filter_nearby_planes_normal_vector(normals, angle_threshold_degrees=5)
    reflection_planes = np.array([reflection_planes[i] for i in remaining_indices])
    print("after",len(reflection_planes) )

    output_dir = f"{args.output_dir}/gif/preliminary/"
                    
    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{file_name_without_extension}.gif")
    create_rotation_gif(np.array(plane_obj.vertices), [mesh, *plane_objs], output_file)

    # 反射対称面の確認用のメッシュモデル読み込み
    mesh = o3d.io.read_triangle_mesh(target_obj_file)
    mesh.compute_vertex_normals()
    if(args.debug):
        o3d.visualization.draw_geometries([mesh, *plane_objs], mesh_show_back_face=True)

    # JSON変換用の関数
    def convert_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # NumPyのarrayをリストに変換
        elif isinstance(obj, tuple):
            return list(obj)  # タプルをリストに変換
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")


    # plot_images(images)

    output_dir = f"{args.output_dir}/preliminary_symmetry"

    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # JSONファイルに保存するパス
    output_file_path = os.path.join(output_dir, f"{file_name_without_extension}.json")

    # データをJSONファイルに保存
    with open(output_file_path, 'w') as json_file:
        json.dump(reflection_planes, json_file, indent=2, default=convert_to_json)



