
import os
import copy
import json

import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional,List
import argparse
import hdbscan
from sklearn.cluster import DBSCAN

from utils.reflection_plane import ReflectionPlane
from utils.object_visualization_helper import CoordinateFrameMesh, Cylinder, Circle, create_rotation_gif

"""
python main_reflectional_symmetry_detection.py
"""

def calculate_obb_corners(obb: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    """Oriented Bounding Box (OBB) の8つの角の座標を計算して返します。

    Args:
        obb: Open3DのOrientedBoundingBoxオブジェクト。

    Returns:
        numpy.ndarray: OBBの8つの角の座標を含む配列。
    """
    obb = copy.deepcopy(obb)
    # OBBの中心座標
    obb_center = np.array(obb.center)
    rotation_matrix= np.array(obb.R)
    # OBBの各辺のベクトルのリスト
    obb_edges = [np.array(edge) for edge in obb.extent]
    
    # OBB辺の半長さを計算
    half_len1 = np.linalg.norm(obb_edges[0]) / 2.0
    half_len2 = np.linalg.norm(obb_edges[1]) / 2.0
    half_len3 = np.linalg.norm(obb_edges[2]) / 2.0
 
    # OBBの8つの角の座標を計算
    p1 = obb_center - half_len1 * rotation_matrix[:, 0] - half_len2 * rotation_matrix[:, 1] - half_len3 * rotation_matrix[:, 2]
    p2 = obb_center + half_len1 * rotation_matrix[:, 0] - half_len2 * rotation_matrix[:, 1] - half_len3 * rotation_matrix[:, 2]
    p3 = obb_center + half_len1 * rotation_matrix[:, 0] + half_len2 * rotation_matrix[:, 1] - half_len3 * rotation_matrix[:, 2]
    p4 = obb_center - half_len1 * rotation_matrix[:, 0] + half_len2 * rotation_matrix[:, 1] - half_len3 * rotation_matrix[:, 2]
    p5 = obb_center - half_len1 * rotation_matrix[:, 0] - half_len2 * rotation_matrix[:, 1] + half_len3 * rotation_matrix[:, 2]
    p6 = obb_center + half_len1 * rotation_matrix[:, 0] - half_len2 * rotation_matrix[:, 1] + half_len3 * rotation_matrix[:, 2]
    p7 = obb_center + half_len1 * rotation_matrix[:, 0] + half_len2 * rotation_matrix[:, 1] + half_len3 * rotation_matrix[:, 2]
    p8 = obb_center - half_len1 * rotation_matrix[:, 0] + half_len2 * rotation_matrix[:, 1] + half_len3 * rotation_matrix[:, 2]
 
    rectangle_pos = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
 
    return rectangle_pos

def calculate_obb_volume(obb: o3d.geometry.OrientedBoundingBox) -> float:
    """Oriented Bounding Box (OBB) の体積を計算します。

    Args:
        obb: Open3DのOrientedBoundingBoxオブジェクト。

    Returns:
        float: バウンディングボックスの体積。
    """
    # OBBの各辺の長さを取得
    extent = obb.extent

    # 体積を計算
    volume = np.prod(extent)
    
    return volume

def get_3D_rectangle_polygon(p1: List[float],
                             p2: List[float],
                             p3: List[float],
                             p4: List[float],
                             p5: List[float],
                             p6: List[float],
                             p7: List[float],
                             p8: List[float]) -> List[Cylinder]:
    """OBBの8つの角の座標を使用して、3D直方体（直方体）のポリゴンデータを生成します。"""
    rectangle = []
    rectangle.append(Cylinder(p1, p2).get_poligon())  # | (上)
    rectangle.append(Cylinder(p2, p3).get_poligon())  # -->
    rectangle.append(Cylinder(p3, p4).get_poligon())  # | (下)
    rectangle.append(Cylinder(p4, p1).get_poligon())  # <--

    rectangle.append(Cylinder(p5, p6).get_poligon())  # | (上)
    rectangle.append(Cylinder(p6, p7).get_poligon())  # -->
    rectangle.append(Cylinder(p7, p8).get_poligon())  # | (下)
    rectangle.append(Cylinder(p8, p5).get_poligon())  # <--

    rectangle.append(Cylinder(p1, p5).get_poligon())  # | (上)
    rectangle.append(Cylinder(p3, p7).get_poligon())  # | (上)
    rectangle.append(Cylinder(p4, p8).get_poligon())  # | (上)
    rectangle.append(Cylinder(p2, p6).get_poligon())  # | (上)

    return rectangle


def find_point_cloud_diff(point_cloud1: o3d.geometry.PointCloud, point_cloud2: o3d.geometry.PointCloud, threshold: float = 3) -> tuple:
    """
    2つの点群の違いを見つけます。

    Parameters:
        point_cloud1 (open3d.geometry.PointCloud): 最初の点群。
        point_cloud2 (open3d.geometry.PointCloud): 2番目の点群。
        threshold (float): 点を異なると見なすための閾値距離（デフォルトは3）。

    Returns:
        tuple: 点群の差異（point_cloud1 - point_cloud2、point_cloud2 - point_cloud1）。
    """
    # point_cloud1とpoint_cloud2の間の点の距離を計算します
    distances1 = np.array(point_cloud1.compute_point_cloud_distance(point_cloud2))
    distances2 = np.array(point_cloud2.compute_point_cloud_distance(point_cloud1))
  
    # 距離が閾値を超える点のインデックスを見つけます（point_cloud1内）
    diff_indices1 = np.where(distances1 > threshold)[0]
   
    # 距離が閾値を超える点のインデックスを見つけます（point_cloud2内）
    diff_indices2 = np.where(distances2 > threshold)[0]
    
    # point_cloud1内に存在せず、point_cloud2に存在する点を抽出します
    diff_points1 = point_cloud1.select_by_index(diff_indices1)
    # point_cloud2内に存在せず、point_cloud1に存在する点を抽出します
    diff_points2 = point_cloud2.select_by_index(diff_indices2)
    
    return diff_points1, diff_points2


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


# def pc_asim(A, B, POOLING_TYPE='Mean'):
#     """
#     点群Aと点群Bの間の角度類似度。
#     reference:
#        paper:https://infoscience.epfl.ch/record/254987
#        code:https://github.com/mmspg/point-cloud-angular-similarity-metric/tree/master

#     入力
#         A, B: numpy配列で表される点群。各配列は形状が(N, 3)である必要があります。
#               ここで、Nは点の数です。A[i]はAのi番目の点の座標を示し、B[i]はBのi番目の点の座標を示します。
#         POOLING_TYPE: 1つの角度類似度値を得るために、対応する点のペアから得られた個々の角度類似度値に適用されるプーリング方法を定義します。
#                       次のオプションが使用できます： {'Mean', 'Min', 'Max', 'MS', 'RMS'}。

#     出力
#         ※0～1の値をとる。1に近いほど似ている
#         asimBA: 点群Aを参照として点群Bの角度類似度スコア。計算されるスコアはPOOLING_TYPEに依存します。
#         asimAB: 点群Bを参照として点群Aの角度類似度スコア。計算されるスコアはPOOLING_TYPEに依存します。
#         asimSym: 両方の点群AとBを参照として対称な角度類似度スコア。計算されるスコアはPOOLING_TYPEに依存します。
#     """
#     if POOLING_TYPE not in ['Mean', 'Min', 'Max', 'MS', 'RMS']:
#         raise ValueError('POOLING_TYPEはサポートされていません。')

#     if A.shape[0] == 0 or B.shape[0] == 0:
#         raise ValueError('入力された点群に座標が見つかりません。')

#     # 点群Aと点群Bの間の点の関連付け
#     tree_A = cKDTree(A)
#     idBA = tree_A.query(B)[1]
#     tree_B = cKDTree(B)
#     idAB = tree_B.query(A)[1]

#     # 点群Bの角度類似度スコア（点群Aを参照）
#     asBA = 1 - 2 * np.arccos(np.abs(np.sum(A[idBA] * B, axis=1) /
#                                    (np.linalg.norm(A[idBA], axis=1) * np.linalg.norm(B, axis=1)))) / np.pi

#     # プーリング
#     if POOLING_TYPE == 'Mean':
#         asimBA = np.nanmean(asBA)
#     elif POOLING_TYPE == 'Min':
#         asimBA = np.nanmin(asBA)
#     elif POOLING_TYPE == 'Max':
#         asimBA = np.nanmax(asBA)
#     elif POOLING_TYPE == 'MS':
#         asimBA = np.nanmean(asBA ** 2)
#     elif POOLING_TYPE == 'RMS':
#         asimBA = np.sqrt(np.nanmean(asBA ** 2))

#     # 点群Aの角度類似度スコア（点群Bを参照）
#     asAB = 1 - 2 * np.arccos(np.abs(np.sum(A * B[idAB], axis=1) /
#                                    (np.linalg.norm(A, axis=1) * np.linalg.norm(B[idAB], axis=1)))) / np.pi

#     # プーリング
#     if POOLING_TYPE == 'Mean':
#         asimAB = np.nanmean(asAB)
#     elif POOLING_TYPE == 'Min':
#         asimAB = np.nanmin(asAB)
#     elif POOLING_TYPE == 'Max':
#         asimAB = np.nanmax(asAB)
#     elif POOLING_TYPE == 'MS':
#         asimAB = np.nanmean(asAB ** 2)
#     elif POOLING_TYPE == 'RMS':
#         asimAB = np.sqrt(np.nanmean(asAB ** 2))

#     # 対称な角度類似度スコア
#     asimSym = np.nanmin([asimBA, asimAB])

#     return asimBA, asimAB, asimSym


# def get_metrics_btw_two_pointclouds(pcd1: np.ndarray, pcd2: np.ndarray, metrics: str = "chamfer") -> float:
#     """2つの点群間の距離メトリックスを計算します。

#     Args:
#         pcd1 (np.ndarray): 点群1。
#         pcd2 (np.ndarray): 点群2。
#         metrics (str, optional): 使用するメトリックス。"chamfer", "Hausdorff", "One side Hausdorff", "Sinkhorn", "angle" のいずれか。Defaults to "chamfer".

#     Raises:
#         ValueError: サポートされていないメトリックスが指定された場合。

#     Returns:
#         float: 点群間の距離メトリックス。
#     """
#     if metrics == "chamfer":
#         # 小さいほど良い, 完全一致で0
#         # p1とp2のChamfer距離を計算
#         distance = pcu.chamfer_distance(pcd1, pcd2)
#     elif metrics == "Hausdorff":
#         # 小さいほど良い, 完全一致で0
#         # p1とp2のHausdorff距離を計算
#         distance = pcu.hausdorff_distance(pcd1, pcd2)
#     elif metrics == "One side Hausdorff":
#         # 小さいほど良い, 完全一致で0
#         # p1とp2の片側Hausdorff距離を計算
#         hd_p1_to_p2 = pcu.one_sided_hausdorff_distance(pcd1, pcd2)
#         distance = hd_p1_to_p2[0]
#     elif metrics == "Sinkhorn":
#         # 小さいほど良い, 完全一致で0
#         # p1とp2のEarth-Mover's (Sinkhorn)距離を計算
#         emd, _ = pcu.earth_movers_distance(pcd1, pcd2)
#         distance = emd
#     elif metrics == "angle":
#         # p1とp2の角度距離を計算
#         # 0～1の値をとる。1に近いほど似ている
#         asimBA, _, asimSym = pc_asim(pcd1, pcd2)
#         #小さいほど良い, 完全一致で0
#         distance = 1-asimSym
#     else:
#         raise ValueError('そのmetricsはサポートされていません')

#     return distance


# def get_all_metrics_btw_two_pointclouds(pcd1: np.ndarray, pcd2: np.ndarray) -> dict:
#     """2つの点群間のすべての距離メトリックスを計算します。

#     Args:
#         pcd1 (np.ndarray): 点群1。
#         pcd2 (np.ndarray): 点群2。

#     Returns:
#         dict: 各メトリックスに対する点群間の距離。
#     """
#     metrics = ["chamfer", "Hausdorff", "One side Hausdorff", "Sinkhorn", "angle"]

#     results = {}
#     for metric in metrics:
#         similarity = get_metrics_btw_two_pointclouds(pcd1, pcd2, metric)
#         results[metric] = similarity 
#     return results

def render_single_shot(mesh):
    # 可視化ウィンドウを作成
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1800, height=900)
    vis.add_geometry(mesh)

    # カメラの位置を設定
    to_point = np.array(mesh.get_center())  # 注視点はメッシュの中心

    # カメラの設定
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])
    ctr.set_lookat(to_point)
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.6)

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
    plt.tight_layout()

    # グラフを表示
    plt.show()

def plot_diff_pcd(diff_pcd):
    fig, axs = plt.subplots(5, 1, figsize=(8, 12))
    metrics_type = list(diff_pcd.values())[0].keys()  # 結果からメトリックスの種類を取得
    symmetry_numbers = list(diff_pcd.keys())  # 対称性の番号を取得

    for i, key in enumerate(metrics_type):
        values = [d[key] for d in diff_pcd.values()]
        axs[i].plot(symmetry_numbers, values, label=key)
        axs[i].set_ylabel(key)
        
        # 各数値をプロット
        for symmetry_number, value in zip(symmetry_numbers, values):
            axs[i].text(symmetry_number, value, f'{value:.2f}', fontsize=8, ha='center', va='bottom')
        
        # 赤い丸でデータ点をプロット
        axs[i].scatter(symmetry_numbers, values, color='red')
        
    axs[-1].set_xlabel('symmetry Number')
    plt.suptitle('symmetry metrics')

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    
def calculate_difference_volume(pcd: o3d.geometry.PointCloud, 
                                diff_pcd: o3d.geometry.PointCloud,
                                min_samples: int = 3,
                                cluster_method: str = "DBSCAN",
                                debug: bool = False) -> float:
    """
    2つのポイントクラウドの差分の体積を計算します。

    Args:
    - pcd: Open3DのPointCloud
    - diff_pcd: Open3DのPointCloud
    - min_samples: HDBSCAN or DBSCANの最小サンプル数パラメータ。デフォルト値は3です。
    - cluster_method: 使用するクラスタリング手法。デフォルトは"DBSCAN"です。
    - debug: デバッグモードを有効化または無効化します。デフォルト値はFalseです。

    Returns:
    - float: 差分点群のOBBの体積を返します。
    """
    diff_pcd =  copy.deepcopy(diff_pcd)

    # Open3Dデータをnumpy配列に変換
    points = np.asarray(diff_pcd.points)

    if(cluster_method == "DBSCAN"):
        distance_near = np.max(pcd.compute_nearest_neighbor_distance())
        clusterer = DBSCAN(eps=distance_near, min_samples=min_samples)
    elif(cluster_method == "HDBSCAN"):
        # HDBSCANを使用してクラスタリング
        clusterer = hdbscan.HDBSCAN(min_samples=min_samples)
        # clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)

    labels = clusterer.fit_predict(points)

    # 結果の表示
    print("the number of cluster:", len(np.unique(labels)))
    
    # Matplotlibのカラーマップを使用して色を生成
    num_clusters = len(np.unique(labels))
    colors = plt.cm.jet(np.linspace(0, 1, num_clusters))[:, :3]  # カラーマップから色を取得

    sum_volume=0
    bounding_boxes=[]
    pcd_clusters = []
    for label, color in zip(np.unique(labels), colors):
        
        # ラベル-1は外れ値。ノイズは考慮しない
        if(label==-1):
            continue
        
        cluster_points = points[labels == label]

        if len(cluster_points) < 4:  # バウンディングボックスを計算するための点の数が不足している場合
            continue
        
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)
        pcd_cluster.paint_uniform_color(color)

        # バウンディングボックスの作成
        obb = pcd_cluster.get_oriented_bounding_box(robust=True)

        if debug:
            # OBBの各角の座標を計算
            obb_corners = calculate_obb_corners(obb)

            # OBBの各面に対応するポリゴンデータを生成
            obb_meshes = get_3D_rectangle_polygon(*obb_corners)
            for obb_mesh in obb_meshes:
                bounding_boxes.append(obb_mesh)
            pcd_clusters.append(pcd_cluster)
            
        sum_volume += calculate_obb_volume(obb)

    if debug:
        P.paint_uniform_color([0.5,0.5,0.5])
        o3d.visualization.draw_geometries([diff_pcd, P, *bounding_boxes,*pcd_clusters])
    return sum_volume
    
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

def is_reflection_matrix(P: o3d.geometry.PointCloud,
                         reflection_matrix: np.ndarray,
                         threshold_points: int = 10,
                         thresh_diff_volume_ratio: float = 1.0,
                         debug: bool = False) -> bool:
    """
    ポイントクラウドPと反射行列を使用して、鏡面反射の面を検出します。

    Parameters:
    - P: Open3DのPointCloud。基準となるポイントクラウドです。
    - reflection_matrix: np.ndarray。反射行列です。
    - threshold_points: int。バウンディングボックスを計算するための最小の点の数のしきい値です。デフォルト値は10です。
    - thresh_diff_volume_ratio: float。差異の体積比のしきい値です。デフォルト値は0.3です。
    - debug: bool。デバッグモードの有効化または無効化を指定します。デフォルト値はFalseです。

    Returns:
    - bool: 鏡面反射の面であればTrue、それ以外はFalseを返します。
    """
    
    entire_obb_volume = calculate_obb_volume(P.get_oriented_bounding_box())
    Q = copy.deepcopy(P)
    Q.transform(reflection_matrix)

    thresh = np.mean(P.compute_nearest_neighbor_distance())
    diff_points1, diff_points2 = find_point_cloud_diff(P, Q, thresh * 1.5)

    
    # 差異の可視化
    if debug:
        o3d.visualization.draw_geometries([diff_points1, diff_points2])

    if len(diff_points1.points) <= threshold_points and len(diff_points2.points) <= threshold_points:
        return True

    diff1_volume = calculate_difference_volume(P, diff_points1, debug=debug)
    diff2_volume = calculate_difference_volume(P, diff_points2, debug=debug)

    sum_diff_volume_ratio = 100 * (diff1_volume + diff2_volume) / entire_obb_volume
    print("sum_diff_volume_ratio", sum_diff_volume_ratio)
    if sum_diff_volume_ratio < thresh_diff_volume_ratio:
        return True
    
    return False

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

def filter_nearby_planes_normal_vector(scores: np.ndarray, normals: np.ndarray, angle_threshold_degrees: float = 5) -> tuple:
    """
    与えられた法線ベクトルのリストから、特定の角度以下の角度を持つ法線ベクトルペアを廃棄します。

    Parameters:
        scores (numpy.ndarray): スコア(高いほどいい)
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
                    if(scores[i]<=scores[j]):
                        discarded_indices.add(j)
                    else:
                        discarded_indices.add(i)

    # 廃棄されなかったもののみを残す
    remaining_indices = [i for i in range(len(scores)) if i not in discarded_indices]
    remaining_normals = [normals[i] for i in remaining_indices]
    
    return remaining_normals, remaining_indices


# def align_vectors(vectors):
#     # 基本ベクトルの定義
#     axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

#     # 各ベクトルに対して最も内積が大きい基本ベクトルを特定し、ベクトルの方向を揃える
#     aligned_vectors = []
#     for vector in vectors:
#         max_unsigned_dot_product_axis_index = np.argmax(np.abs(np.dot(vector, axes))) 
#         max_unsigned_dot_product_axis = axes[max_unsigned_dot_product_axis_index]
        
#         aligned_vector = vector if np.dot(vector, max_unsigned_dot_product_axis) > 0 else -vector
#         # print(np.dot(vector, aligned_vector))
#         aligned_vectors.append(aligned_vector)

#     return aligned_vectors

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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process PLY file and generate point cloud.')
    parser.add_argument('--mesh', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--points', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    
    args = parser.parse_args()
    target_obj_file = args.mesh

    target_obj_filename=os.path.basename(target_obj_file)
    file_name_without_extension = os.path.splitext(target_obj_filename)[0]
    file_path = f"./result/preliminary_symmetry/{file_name_without_extension}.json"
    target_observabile = args.points

    P = o3d.io.read_point_cloud(target_observabile)
    # voxel_size = 0.002*1000  # ダウンサンプリングの解像度を適切に指定する
    # P = P.voxel_down_sample(voxel_size)
    
    # JSONファイルを読み込む
    with open(file_path, 'r') as file:
        reflection_planes = json.load(file)

    # バウンディングボックスの対角線長さを取得(平面描写のために使う)
    bbox_diagonal_length = np.linalg.norm(np.asarray(P.get_max_bound()) - np.asarray(P.get_min_bound()))

    # 反射対称面の確認用のメッシュモデル読み込み
    mesh = o3d.io.read_triangle_mesh(target_obj_file)
    mesh.compute_vertex_normals()

    results = {}
    good_reflection_planes = []
    scores = []
    images=[]

    for i,(plane_pos, plane_normal) in enumerate(reflection_planes):
        reflection_matrix = ReflectionPlane(plane_pos=plane_pos, plane_normal=plane_normal).reflection_matrix

        # normalをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列
        rotation_matrix = rotation_matrix_to_align_with_z_axis(plane_normal)
        # 円盤のz軸ベクトルをnormalに一致させる
        H=np.eye(4)
        H[:3, :3] = np.linalg.inv(rotation_matrix)
        H[:3, 3]  = plane_pos
        
        circle = Circle(pos=[0,0,0], rpy=[0,0,0], r=bbox_diagonal_length/2)
        circle.create_polygon()

        plane_obj = circle.get_poligon()
        plane_obj.transform(H)

        # ワイヤーフレームに変換
        plane_obj_wire = o3d.geometry.LineSet.create_from_triangle_mesh(plane_obj)

        Q = copy.deepcopy(P)
        Q.transform(reflection_matrix)

        # Open3Dを使用して平面と点群を可視化
        if(args.debug):
            o3d.visualization.draw_geometries([mesh, plane_obj_wire], mesh_show_back_face=True)
        
            o3d.visualization.draw_geometries([P, Q], mesh_show_back_face=True)
        
        img = render_single_shot(P+Q)
        images.append(img)
        

        # 関数を使用してgood_reflection_planesに追加するかどうかを決定する
        if is_reflection_matrix(P, reflection_matrix,debug=args.debug):
            good_reflection_planes.append([plane_pos, plane_normal])
            scores.append(np.mean(P.compute_point_cloud_distance(Q)))
                
                
    output_dir = "./result/gif/reflectional_symmetry"
    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{file_name_without_extension}.gif")

    if(len(good_reflection_planes) >= 1):
        print("before", len(good_reflection_planes))
        good_reflection_planes = np.array(good_reflection_planes)
        # 反射対称平面がほぼ重なっていたら削除する
        remain_axes, remaining_indices = filter_nearby_planes_normal_vector(scores, good_reflection_planes[:,1,:], angle_threshold_degrees=5)
        good_reflection_planes = np.array([good_reflection_planes[i] for i in remaining_indices])
        # ベクトルの方向を揃える
        print()
        good_reflection_planes[:,1,:] = align_vectors(good_reflection_planes[:,1,:])
        print("after", len(good_reflection_planes))
        
        plane_objs=[]
        plane_obj_wires=[]
        for i,(plane_pos, plane_normal) in enumerate(good_reflection_planes):

            # normalをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列
            rotation_matrix = rotation_matrix_to_align_with_z_axis(plane_normal)
            # 円盤のz軸ベクトルをnormalに一致させる
            H=np.eye(4)
            H[:3, :3] = np.linalg.inv(rotation_matrix)
            H[:3, 3]  = plane_pos
            
            circle = Circle(pos=[0,0,0], rpy=[0,0,0], r=bbox_diagonal_length/2)
            circle.create_polygon()

            plane_obj = circle.get_poligon()
            plane_obj.transform(H)

            # ワイヤーフレームに変換
            plane_obj_wire = o3d.geometry.LineSet.create_from_triangle_mesh(plane_obj)
            plane_objs.append(plane_obj)
            plane_obj_wires.append(plane_obj_wire)


        # Open3Dを使用して平面と点群を可視化
        if(args.debug):
            o3d.visualization.draw_geometries([mesh, *plane_objs], mesh_show_back_face=True)
        
        output_dir = "./result/gif/reflectional_symmetry"
        # output ディレクトリが存在しない場合は作成する
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        from_points= []
        for plane_obj in plane_objs:
            for v in np.array(plane_obj.vertices):
                from_points.append(v)
                
        create_rotation_gif(np.array(from_points), [mesh, *plane_obj_wires], output_file)
    else:
        if(args.debug):
            o3d.visualization.draw_geometries([mesh], q=True)

            
        # 回転の角度を定義する
        angles = np.linspace(0, 2*np.pi, num=20)  # 0から2πまでの角度を等間隔で生成

        # 回転の中心点を設定する
        center = mesh.get_center()  # y軸を中心に回転するため、中心点は原点(0, 0, 0)

        # リストfrom_pointsを作成する
        from_points = []

        for angle in angles:
            # 回転行列を計算する
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
            
            # y平面上の点を定義する
            y_plane_point = np.array([0, 1, 0])  # y軸上の点
                    
            # 回転を適用する
            rotated_point = np.dot(rotation_matrix, y_plane_point - center) + center
            
            # リストfrom_pointsに追加する
            from_points.append(rotated_point.tolist())

        for angle in angles:
            # 回転行列を計算する
            rotation_matrix = np.array([[1, 0, 0],
                                   [0, np.cos(angle), -np.sin(angle)],
                                   [0, np.sin(angle), np.cos(angle)]])
            
            # z平面上の点を定義する
            z_plane_point = np.array([0, 0, 1])  # y軸上の点
                    
            # 回転を適用する
            rotated_point = np.dot(rotation_matrix, z_plane_point - center) + center
            
            # リストfrom_pointsに追加する
            from_points.append(rotated_point.tolist())


        # from_pointsをnumpy配列に変換する
        from_points = np.array(from_points)

        create_rotation_gif(from_points, [mesh], output_file)


    # 画像をプロット
    if(args.debug):
        plot_images(images)
    output_dir = "./result/reflectional_symmetry"

    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # # JSONファイルに保存するパス
    output_file_path = os.path.join(output_dir,f"{file_name_without_extension}_reflectional_symmetry.json")


    def convert_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # NumPyのarrayをリストに変換
        elif isinstance(obj, tuple):
            return list(obj)  # タプルをリストに変換
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")
        
    # データをJSONファイルに保存
    with open(output_file_path, 'w') as json_file:
        json.dump(good_reflection_planes, json_file, indent=2, default=convert_to_json)

