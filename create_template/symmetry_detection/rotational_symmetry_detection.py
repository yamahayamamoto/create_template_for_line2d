import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation
import os
import sys
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Optional,List,Union, Tuple
import argparse
import hdbscan
from sklearn.cluster import DBSCAN
from PIL import Image

from utils.object_visualization_helper import (interpolate_angles_by_expo, generate_circle_on_plane,
                                               interpolate_angles_by_linear, 
                                               Arrow, Circle, create_rotation_gif, 
                                               text_3d, get_3D_rectangle_polygon)


"""回転対称軸とその回転角度を求める
python symmetry\rotational_symmetry_detection.py --mesh symmetry\tmp_obj_reduce\95D95-06010\95D95-06010.ply 
--points symmetry\observability\95D95-06010_observable.ply  --output_dir  symmetry/result

"""

def rotation_matrix_to_align_with_dst_axis(vec, dst_axis= np.array([0, 0, 1])): 
    """ 
    ベクトルvecをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列を計算する。 
 
    Parameters: 
        vec (numpy.ndarray): 3次元ベクトル (x, y, z) を表すNumPy配列。 
 
    Returns: 
        numpy.ndarray: ベクトルvecをx軸に一致させるための回転行列。vecがx軸と一致している場合は単位行列が返される。 
    """ 
 
    # ベクトルvecとx_axisの外積を計算 
    cross_product = np.cross(vec, dst_axis) 
 
    # 回転角度を計算 
    if np.linalg.norm(cross_product) < 1e-12: 
        # 外積がほぼゼロの場合（vecがx軸と一致する場合） 
        return np.identity(3)  # 単位行列を返す 
 
    # 外積を正規化して回転軸を計算 
    rotation_axis = cross_product / np.linalg.norm(cross_product) 
 
    # ドット積を計算 
    dot_product = np.dot(vec, dst_axis) 
 
    # 回転角度を計算 
    rotation_angle = np.arccos(np.clip(dot_product / (np.linalg.norm(vec) * np.linalg.norm(dst_axis)), -1.0, 1.0)) 
 
    # 回転行列を計算 
    cos_theta = np.cos(rotation_angle) 
    sin_theta = np.sin(rotation_angle) 
    rotation_matrix = np.array([ 
        [cos_theta + rotation_axis[0] ** 2 * (1 - cos_theta), rotation_axis[0] * rotation_axis[1] * (1 - cos_theta) - rotation_axis[2] * sin_theta, rotation_axis[0] * rotation_axis[2] * (1 - cos_theta) + rotation_axis[1] * sin_theta], 
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_theta) + rotation_axis[2] * sin_theta, cos_theta + rotation_axis[1] ** 2 * (1 - cos_theta), rotation_axis[1] * rotation_axis[2] * (1 - cos_theta) - rotation_axis[0] * sin_theta], 
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_theta) - rotation_axis[1] * sin_theta, rotation_axis[2] * rotation_axis[1] * (1 - cos_theta) + rotation_axis[0] * sin_theta, cos_theta + rotation_axis[2] ** 2 * (1 - cos_theta)] 
    ]) 
     
    return rotation_matrix 
 
    

def find_intersection_axis(normals):
    num_normals = len(normals)
    
    # 交線の軸ベクトルを初期化
    intersection_axes = []

    # 全ての法線ベクトルの組み合わせで計算
    for i in range(num_normals):
        for j in range(i+1, num_normals):
            
            # 法線ベクトルの内積を求める
            dot_product = np.dot(normals[i], normals[j])

            # 負の向きなら正の向きにする
            if(dot_product<0):
                dot_product -= dot_product

            # 法線ベクトルのペアがほぼ同じなら外積計算を行わない
            if(np.abs(1-dot_product)<1e-6):
                continue

            # 法線ベクトルの外積を計算し、軸ベクトルに加算
            axis = np.cross(normals[i], normals[j])
            
            # 外積が十分に大きい場合は正規化して軸ベクトルを求める
            axis /= np.linalg.norm(axis)
            intersection_axes.append(axis)

    # 向きを揃える
    intersection_axes = align_vectors(intersection_axes)
    return intersection_axes


def pick_peak(data):
    #ピークピッキング
	peaks_val = []
	peaks_indices = []
	for i in range(2, len(data)):
		if data[i-1] - data[i-2] >= 0 and data[i] - data[i-1] < 0:
			peaks_val.append(data[i-1])
			peaks_indices.append(i-1)
	# max_index = peaks_val.index(max(peaks_val))
	return peaks_indices #[max_index]


def get_point_cloud_didstance(point_cloud1: o3d.geometry.PointCloud, point_cloud2: o3d.geometry.PointCloud, threshold: float = 3) -> tuple:
    """
    2つの点群の違いを見つけます。

    Parameters:
        point_cloud1 (open3d.geometry.PointCloud): 最初の点群。
        point_cloud2 (open3d.geometry.PointCloud): 2番目の点群。
        threshold (float): 点を異なると見なすための閾値距離（デフォルトは3）。

    """
    # point_cloud1とpoint_cloud2の間の点の距離を計算します
    distances1 = np.array(point_cloud1.compute_point_cloud_distance(point_cloud2))
  
    # 距離が閾値を超える点のインデックスを見つけます（point_cloud1内）
    diff_indices1 = np.where(distances1 > threshold)[0]
    
    return diff_indices1


def extract_variable_parts_wrt_axis(pcd, axis_vector, angle_resolution_deg=5, debug=False):
    angle_resolution_rad = np.deg2rad(angle_resolution_deg)
    # 角度の範囲を設定 (0からπ)
    angles = np.arange(0, 2*np.pi + angle_resolution_rad, angle_resolution_rad)
    thresh = np.mean(pcd.compute_nearest_neighbor_distance())
    variable_pcd_indices = set()
    
    # 各角度に対してチェック
    for angle in angles:
        rotated_pcd = copy.deepcopy(pcd)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        # 点群データを回転して軸対称性を判定
        R = np.array(Rotation.from_rotvec(angle * axis_vector).as_matrix())
        # print(R)
        
        M = np.eye(4)
        M[:3,3] = pcd.get_center()
        M_inv   = np.linalg.inv(M)

        H = np.eye(4)
        H[:3,:3]=R
        rotated_pcd = rotated_pcd.transform(M@H@M_inv)

        # 点群間の差異を計算
        diff_indices = get_point_cloud_didstance(pcd, rotated_pcd, thresh*2)

        # 変化がある部分のインデックスを保存
        if len(diff_indices) > 0:
            variable_pcd_indices.update(diff_indices)

    # 変化がある部分の点群を抽出
    variable_pcd = pcd.select_by_index(list(variable_pcd_indices))

    if(debug):
        o3d.visualization.draw_geometries([variable_pcd], mesh_show_back_face=True)
    return variable_pcd

def get_diff_distance_per_resolution_angle(pcd, pcd_partial, axis_vector, angle_resolution_deg=5, debug=False):
    """
    制約条件を満たす軸対称性が存在するかを判定します。

    Parameters:
        normals (list of numpy.ndarray): 法線ベクトルのリスト。
        angle_resolution_deg (float): 角度の刻み幅。デフォルトは0.01 deg

    Returns:
        bool: 軸対称性が存在する場合はTrue、それ以外はFalse。
        list: 軸対称性が存在する場合の角度のリスト。
    """

    axis_vector = copy.deepcopy(axis_vector)
    angle_resolution_rad = np.deg2rad(angle_resolution_deg)
    # 角度の範囲を設定 (0からπ)
    angles = np.arange(0, 2*np.pi + angle_resolution_rad, angle_resolution_rad)

    distances=[]
    # 各角度に対してチェック
    for angle in angles:
        rotated_pcd = copy.deepcopy(pcd_partial)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        # 点群データを回転して軸対称性を判定
        R=np.array(Rotation.from_rotvec(angle * axis_vector).as_matrix())
        # print(R)
        
        M = np.eye(4)
        M[:3,3] = pcd.get_center()
        M_inv   = np.linalg.inv(M)

        H = np.eye(4)
        H[:3,:3]=R
        rotated_pcd = rotated_pcd.transform(M@H@M_inv)

        # pcd.paint_uniform_color([1, 0, 0])
        # rotated_pcd.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([ pcd, rotated_pcd], mesh_show_back_face=True)
        # 点群データと回転後の点群データの距離を計算
        # distance = np.sum(np.linalg.norm(np.asarray(pcd.points) - np.asarray(rotated_pcd.points), axis=1))
        distances1 = np.array(pcd.compute_point_cloud_distance(rotated_pcd))
        distance = np.mean(distances1)

        distances.append(distance)
        

    # from scipy.signal import savgol_filter
    num=10#移動平均の個数
    b=np.ones(num)/num
    distances=np.convolve(distances, b, mode='same')#移動平均

    angles = np.array([np.rad2deg(angle) for angle in angles])
    # distances = savgol_filter(distances, 11, 4)
    # distances = distances - (distances - savgol_filter(distances, 11, 4))
    
    
    from scipy.fft import fft, fftfreq
    
    from scipy.optimize import curve_fit

    def remove_offset(signal):
        """
        信号から多項式オフセットを取り除きます。

        Parameters:
            signal (numpy.ndarray): 入力信号。

        Returns:
            numpy.ndarray: オフセットを取り除いた信号。
        """
        # 多項式オフセットを除去
        p = np.polyfit(np.arange(len(signal)), signal, 1)
        offset = np.polyval(p, np.arange(len(signal)))
        return signal - offset

    def fit_sin_wave(t, signal):
        """
        与えられた信号に対してsin波形をフィットします。

        Parameters:
            t (numpy.ndarray): 時間（または位置）の配列。
            signal (numpy.ndarray): 入力信号。

        Returns:
            tuple: フィットしたsin波形のパラメータ（振幅、周波数、位相）。
        """
        # フーリエ変換を適用
        spectrum = fft(signal)
        freqs = fftfreq(len(signal))

        # フーリエ変換の結果から主要な周波数成分を抽出
        main_freq_index = np.argmax(np.abs(spectrum))
        main_freq = freqs[main_freq_index]

        # sin関数のフィット
        initial_guess = [np.max(signal), main_freq, 0]  # 初期推定値：振幅、周波数、位相
        popt, _ = curve_fit(sin_function, t, signal, p0=initial_guess)

        return popt

    def sin_function(t, amplitude, frequency, phase):
        """
        sin関数の定義。

        Parameters:
            t (numpy.ndarray): 時間（または位置）の配列。
            amplitude (float): 振幅。
            frequency (float): 周波数。
            phase (float): 位相。

        Returns:
            numpy.ndarray: sin関数の値。
        """
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
 
    # distances=remove_offset(distances)
    distances = distances[num:-num]
    angles = angles[num:-num]

    if debug:
        plt.plot(angles, distances, color='b', linestyle='-')
        plt.xlabel('angles[deg]')
        plt.ylabel('similarity(distance)')
        plt.title('rotational similarity')
        plt.show()

    return angles, distances

def check_rotation_symmetry(angles, distances, debug=False):
    negative_peaks_indices = pick_peak(-distances)

    if debug:
        plt.plot(angles, distances, color='b', linestyle='-')
        plt.xlabel('angles[deg]')
        plt.ylabel('similarity(distance)')
        plt.title('rotational similarity')

        if(len(negative_peaks_indices)>0):
            # ピークの位置を垂直線で示す
            for negative_peak_index in negative_peaks_indices:
                plt.axvline(x=angles[negative_peak_index], color='r', linestyle='--')
                plt.text(angles[negative_peak_index], distances[negative_peak_index], str(round(angles[negative_peak_index], 1)), color='r')

        plt.show()
    if(len(negative_peaks_indices)>0):
        return True, angles[negative_peaks_indices], distances[negative_peaks_indices]
    else:
        return False, None, None


def extract_reflection_normal_from_matrix(reflection_matrix):
    """
    反射行列から反射平面の法線ベクトルを抽出します。

    Parameters:
        reflection_matrix (numpy.ndarray): 反射行列。

    Returns:
        reflection_normal_vector (numpy.ndarray): 平面の法線ベクトル。
    """
    reflection_matrix = np.array(reflection_matrix)
    
    # 平行移動を除く反射行列のみを抽出します
    reflection_matrix_only_rot = reflection_matrix[:3, :3]
    
    # 回転行列の固有値と固有ベクトルを計算します
    eigenvalues, eigenvectors = np.linalg.eig(reflection_matrix_only_rot)
    
    # 固有値が-1に最も近い固有ベクトルのインデックスを見つけます
    symmetry_idx = np.argmin(np.abs(eigenvalues + 1))
    
    if np.abs(eigenvalues[symmetry_idx] + 1) >= 1e-6:
        raise ValueError('no -1 eigenvalue')

    # 対応する固有ベクトルを取得します
    reflection_normal_vector = eigenvectors[:, symmetry_idx].real
    
    return reflection_normal_vector



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

def filter_nearby_planes_normal_vector_with_score(scores: np.ndarray, normals: np.ndarray, angle_threshold_degrees: float = 5) -> tuple:
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

    Q = copy.deepcopy(P)
    Q.transform(reflection_matrix)

    entire_obb_volume = calculate_obb_volume(P.get_oriented_bounding_box())

    thresh = np.mean(P.compute_nearest_neighbor_distance())
    diff_points1, diff_points2 = find_point_cloud_diff(P, Q, thresh * 1.5)
    
    # 差異の可視化
    if False:
        o3d.visualization.draw_geometries([diff_points1, diff_points2])

    if len(diff_points1.points) <= threshold_points and len(diff_points2.points) <= threshold_points:
        return True


    diff1_obb_volume = calculate_difference_volume(P, diff_points1, debug=False)
    diff2_obb_volume = calculate_difference_volume(P, diff_points2, debug=debug)

    sum_diff_obb_volume_ratio = 100 * (diff1_obb_volume + diff2_obb_volume) / entire_obb_volume
    print("sum_diff_obb_volume_ratio", sum_diff_obb_volume_ratio)
    if sum_diff_obb_volume_ratio < thresh_diff_volume_ratio:
        print("True!!!!!!!!!!!!")
        return True
    
    print("False!!!!!!!!!!!!")
    return False

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


def refine_rotational_axis(P, axis_vector, relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=2000, debug=False):
    # ICPでPとQを詳細マッチングする

    th = 0.04 *1000

    P = copy.deepcopy(P)
    Q = copy.deepcopy(P)

    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    angle = np.pi
    # 点群データを回転して軸対称性を判定
    R = np.array(Rotation.from_rotvec(angle * axis_vector).as_matrix())
    
    M = np.eye(4)
    M[:3,3] = P.get_center()
    M_inv   = np.linalg.inv(M)

    H = np.eye(4)
    H[:3,:3]= R
    Q.transform(M@H@M_inv)
    

    if debug:
        P.paint_uniform_color([1, 0, 0])
        Q.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([P, Q])

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = relative_fitness, # fitnessの変化分がこれより小さくなったら収束
                                        relative_rmse = relative_rmse, # RMSEの変化分がこれより小さくなったら収束
                                        max_iteration = max_iteration) 
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
    print(reg_p2p)
    inlier_rmse = reg_p2p.inlier_rmse
    rot=reg_p2p.transformation[:3,:3]
    refine_axis = np.dot(rot, axis_vector)
    refine_axis = refine_axis / np.linalg.norm(refine_axis)

    # # マッチングした結果を反映
    # # Q=Q.transform(rot)
    # Q = copy.deepcopy(P)
    # Q=Q.transform(refine_reflection_matrix)

    
    # デバッグモードが有効な場合はマッチング結果を描画
    if debug:
        P.paint_uniform_color([1, 0, 0])
        Q.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([P, Q])

    return refine_axis, inlier_rmse


def round_first_digit(values, threshold=5):
    """
    配列の各要素の1桁目の値を10の位に丸める。

    Parameters:
        values (ndarray): 処理対象の配列
        threshold (int): 切り上げの閾値。この値以上の場合は切り上げ、以下の場合は切り捨てる。

    Returns:
        ndarray: 処理後の配列
    """
    result = np.empty_like(values)
    for i, val in enumerate(values):
        first_digit = val // 10  # 数値の1桁目を取得
        if val % 10 <= threshold:  # 1桁目の値が閾値以下の場合
            result[i] = first_digit * 10  # 切り捨て
        else:  # 1桁目の値が閾値より大きい場合
            result[i] = (first_digit + 1) * 10  # 切り上げ
    return result
    
def get_multiples_with_variance(lst):
    if(len(lst)<=2):
        return False, None

    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            ratio = max(lst[i], lst[j]) / min(lst[i], lst[j])
            if ratio.is_integer() and abs(ratio - round(ratio)) <= 2:
                return True, i if lst[i] < lst[j] else j
    return False, None


if __name__ =="__main__":
    
    parser = argparse.ArgumentParser(description='Process PLY file and generate point cloud.')
    parser.add_argument('--mesh', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--points', type=str, default=None, help='Path to PLY file.')
    parser.add_argument("--output_dir", default="result", help="name of output ply file")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    
    args = parser.parse_args()
    target_obj_file = args.mesh
    target_obj_filename=os.path.basename(target_obj_file)
    file_name_without_extension = os.path.splitext(target_obj_filename)[0]
    file_path = f"{args.output_dir}/preliminary_symmetry/{file_name_without_extension}.json"

    target_observabile = args.points
    
    mesh = o3d.io.read_triangle_mesh(target_obj_file)
    mesh.compute_vertex_normals()

    
    # テスト用の点群PとQ_initを作成
    P = o3d.io.read_point_cloud(target_observabile)
    
    # voxel_size = 0.002*1000  # ダウンサンプリングの解像度を適切に指定する
    # P_down = P.voxel_down_sample(voxel_size)
    P_down = o3d.io.read_point_cloud(target_observabile)
    
    # ダウンサンプリング前の点群数を表示
    print(f"Original Point Cloud: {len(P.points)} points")

    # ダウンサンプリングを行う
    # voxel_size = 0.0001*1000  # ダウンサンプリングの解像度を適切に指定する
    # P = P.voxel_down_sample(voxel_size)

    # ダウンサンプリング後の点群数を表示
    print(f"Downsampled Point Cloud: {len(P.points)} points")

    # 3Dビューワーで点群を表示する（オプション）
    # if(args.debug):
    #     o3d.visualization.draw_geometries([P])

    # バウンディングボックスの対角線長さを取得(平面描写のために使う)
    bbox_diagonal_length = np.linalg.norm(np.asarray(P.get_max_bound()) - np.asarray(P.get_min_bound()))


    # JSONファイルを読み込む
    with open(file_path, 'r') as file:
        reflection_planes = json.load(file)
    
    print("the number of the symmetry planes:",len(reflection_planes))
    rotational_symmetries = []
    plane_normals = []
    plane_objs=[]
    for i,(plane_pos, plane_normal) in enumerate(reflection_planes):
        plane_normals.append(plane_normal)

        # normalをz軸ベクトルz_axis=[1,0,0]に一致させる回転行列
        rotation_matrix = rotation_matrix_to_align_with_dst_axis(plane_normal, np.array([0,0,1]))
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
        plane_objs.append(plane_obj_wire)

    
    # 対称面が1つ以上の時
    if(len(plane_normals) > 1):
        # 2つの平面の法線ベクトルの組み合わせから交差する軸ベクトルを計算する
        axis_vectors = find_intersection_axis(plane_normals)
        print("the number of the symmetry axes:",len(axis_vectors))
        arrow_objs=[]
        for axis_vector in axis_vectors:
            # 矢印の生成
            arrow = Arrow(P.get_center(), P.get_center()+0.5*bbox_diagonal_length*axis_vector, color=[1.0,0.0,0.0])
            arrow_obj = arrow.get_poligon()
            arrow_objs.append(arrow_obj)
            # o3d.visualization.draw_geometries([ mesh, arrow_obj, *plane_objs], mesh_show_back_face=True)
        
        if(args.debug):
            o3d.visualization.draw_geometries([ mesh, *arrow_objs, *plane_objs], mesh_show_back_face=True)

    scores = []
    if(len(plane_normals) > 1):
        arrow_objs=[]
        for i,axis_vector in enumerate(axis_vectors):
            # ICPで回転対称軸の精度を高める
            axis_vectors[i],inlier_rmse = refine_rotational_axis(P, axis_vector)
            scores.append(1/inlier_rmse)
            
        print("before",len(axis_vectors) )
        remain_axes, remaining_indices = filter_nearby_planes_normal_vector_with_score(scores, axis_vectors, angle_threshold_degrees=5)
        axis_vectors = np.array([axis_vectors[i] for i in remaining_indices])
        print("after",len(axis_vectors) )

        for axis_vector in axis_vectors:
            # 矢印の生成
            arrow = Arrow(P.get_center(), P.get_center()+0.5*bbox_diagonal_length*axis_vector, color=[1.0,0.0,0.0])
            arrow_obj = arrow.get_poligon()
            arrow_objs.append(arrow_obj)
        if(args.debug):
            o3d.visualization.draw_geometries([ mesh, *arrow_objs, *plane_objs], mesh_show_back_face=True)

    rotational_axes=[]
    scores=[]
    symmetry_angles=[]
    peak_nums = []

    angle_resolution_deg=2
    
    max_min_distance_threshold = 0.08

    if(len(plane_normals) > 1):
        for axis_vector in axis_vectors:
            
            pcd_partial = extract_variable_parts_wrt_axis(P, axis_vector, angle_resolution_deg, debug=args.debug)
            print(len(pcd_partial.points))

            # # 無限刻み幅で回転対称
            # if(len(pcd_partial.points)==0):
            #     rotational_symmetries.append([axis_vector,0])
            #     continue

            # 軸対称性の判定と成り立つ角度のリストを取得
            angles, distances = get_diff_distance_per_resolution_angle(P, pcd_partial, axis_vector, angle_resolution_deg,debug=args.debug)


            is_symmetric,symmetric_angles,symmetric_scores = check_rotation_symmetry(angles, distances,debug=args.debug)
            print("is_symmetric?",is_symmetric)

            
            if(not is_symmetric):
                # 無限刻み幅で回転対称
                if(np.abs(np.max(distances)-np.min(distances)) < max_min_distance_threshold ):
                    # rotational_symmetries.append([axis_vector,0])
                    print("infinity revolution!!!!!!!",np.abs(np.max(distances)-np.min(distances)))
                    rotational_axes.append(axis_vector)
                    scores.append(1)
                    symmetry_angles.append(0)
                    peak_nums.append(np.inf)
                continue

            print("check distance range:",np.abs(np.max(distances)-np.min(distances)))
            
            if(not is_symmetric):
                continue

                        
            print("symmetric_angles",symmetric_angles)

            symmetric_angles = round_first_digit(symmetric_angles, threshold=5)
            # delta_angleを計算
            symmetric_angles_div = np.array(360 / symmetric_angles) 
            symmetric_angles_div = np.round([round(div) for div in symmetric_angles_div])
            
            print("symmetric_angles(round)",symmetric_angles)
            print("symmetric_angles_div",symmetric_angles_div)

            diff_angles_bw_ideal = np.array([(360-ang*(div))/div for ang,div in zip(symmetric_angles,symmetric_angles_div)])
            diff_angles_bw_ideal = diff_angles_bw_ideal
            print("diff_angles_bw_ideal",diff_angles_bw_ideal)
            diff_angles_bw_ideal_abs = np.array([np.abs(diff) for diff in diff_angles_bw_ideal])
          

            print("diff_angles_bw_ideal_abs",diff_angles_bw_ideal_abs)
            # delta_angleが3以下のインデックスを取得
            indices = np.where(diff_angles_bw_ideal_abs <= 5)[0]
            print("indices",indices)

            if(len(indices)==1):
                # 180degのとき
                index = indices[0]
                diff_symmetric_angle = symmetric_angles[index]
                symmetric_score = symmetric_scores[index]
                
            elif(len(indices)>=2):
                print("indices",indices)

                # 最小値のインデックスを取得
                # min_index = np.argmin(symmetric_angles[indices])
                ideal_angles = np.array(symmetric_angles[indices]+diff_angles_bw_ideal[indices])
                
                print("ideal_angles:",ideal_angles)

                # ユニークな要素とそのインデックスを取得
                ideal_angles, indices = np.unique(ideal_angles, return_index=True)

                is_multiple_relationship,index_multiple_relationship_smaller = get_multiples_with_variance(ideal_angles)
                
                print("is_multiple_relationship?",is_multiple_relationship)
                
                # 倍数関係にある場合
                if(is_multiple_relationship):
                    def check_nearby_values(target_values, reference_values, tolerance=4):
                        """
                        与えられた配列target_valuesの各要素について、配列reference_valuesの各要素の周囲にある値が全て存在するかどうかをチェックする。

                        Parameters:
                            target_values (ndarray): チェック対象の配列
                            reference_values (ndarray): 比較対象の配列
                            tolerance (int, optional): 周囲の許容範囲。デフォルトは4。

                        Returns:
                            bool: 全ての要素が周囲に存在する場合はTrue、そうでない場合はFalse。
                        """
                        result = np.full_like(reference_values, False, dtype=bool)  # 初期化：すべての要素をFalseにする

                        for i, val in enumerate(reference_values):
                            nearby_values = np.logical_and(target_values >= val - tolerance, target_values <= val + tolerance)
                            result[i] = np.any(nearby_values)

                        return np.all(result)

                    check_multiple_relationship_angles = np.array([ ideal_angles[index_multiple_relationship_smaller]*(i+1) for i in np.arange(360 / float(ideal_angles[index_multiple_relationship_smaller]))])
                    print("check_multiple_relationship_angles",check_multiple_relationship_angles)
                    print("check_nearby_values?",check_nearby_values(symmetric_angles,check_multiple_relationship_angles[:-1]))
                    if(check_nearby_values(symmetric_angles,ideal_angles)):
                        # 倍数関係にある場合はその小さい方の値を採用。　例 [60,70,180]なら60のインデックス0が得られる
                        index = indices[index_multiple_relationship_smaller]
                        print("index",index)
                        diff_symmetric_angle = symmetric_angles[index]
                        symmetric_score = symmetric_scores[index]

                else:
                    # 無限刻み幅で回転対称
                    if(np.abs(np.max(distances)-np.min(distances)) < max_min_distance_threshold ):
                        # rotational_symmetries.append([axis_vector,0])
                        print("infinity revolution!!!!!!!",np.abs(np.max(distances)-np.min(distances)))
                        rotational_axes.append(axis_vector)
                        scores.append(1)
                        symmetry_angles.append(0)
                        peak_nums.append(np.inf)
                    continue

            else:
                # 無限刻み幅で回転対称
                if(np.abs(np.max(distances)-np.min(distances)) < max_min_distance_threshold ):
                    # rotational_symmetries.append([axis_vector,0])
                    print("infinity revolution!!!!!!!",np.abs(np.max(distances)-np.min(distances)))
                    rotational_axes.append(axis_vector)
                    scores.append(1)
                    symmetry_angles.append(0)
                    peak_nums.append(np.inf)
                continue
            
            # 対応するSymmetric_Anglesの値を表示
            print("****Symmetric_Angles:", symmetric_angles[index])

            
            print("Is Symmetric:", is_symmetric)
            print("Symmetric Angles:", symmetric_angles)
            print("Symmetric Scores:", symmetric_scores)
            print("Symmetric Score:", symmetric_score)
            print("difference Symmetric value:", diff_symmetric_angle)
            print("360?",(len(symmetric_angles)+1)*diff_symmetric_angle)

            if(is_symmetric):
                rotational_axes.append(axis_vector)
                scores.append(symmetric_score)
                symmetry_angles.append(diff_symmetric_angle)
                
                peak_nums.append(symmetric_angles_div[index])

        # ほぼ同じものは削除
        remain_axes, remaining_indices = filter_nearby_planes_normal_vector_with_score(scores, rotational_axes, angle_threshold_degrees=5)
        print("remaining_indices",remaining_indices)
   
        print("----------------strict check----------------------")
        for remaining_index in remaining_indices:
            rotational_axis = rotational_axes[remaining_index]
            symmetry_angle = symmetry_angles[remaining_index]
            print(remaining_index)
            print(scores[remaining_index])
            print(rotational_axis)
            print(symmetry_angle)
    
            if(peak_nums[remaining_index]==np.inf):
                print("infinity revolution!!!!!!!")
                rotational_symmetries.append([rotational_axis,symmetry_angles[remaining_index]])
            
            else:
                # 理想の値との誤差を計算
                diff_angle_bw_ideal = (360-symmetry_angle*(peak_nums[remaining_index]))/float(peak_nums[remaining_index])
                print("diff_angle_bw_ideal:",diff_angle_bw_ideal)

                # 誤差が5度未満であれば修正された平均値を採用
                if np.abs(diff_angle_bw_ideal) < 5:
                    # if(symmetry_angle - 360 < 0):
                    #     modified_symmetry_angle = (symmetry_angle - diff_angle_bw_ideal)
                    # else:
                    #     modified_symmetry_angle = (symmetry_angle + diff_angle_bw_ideal)

                    modified_symmetry_angle = round((symmetry_angle + diff_angle_bw_ideal),1)

                    # print("修正された平均値:", adjusted_mean)
                    print("modified angle[deg]:", modified_symmetry_angle)
                    

                    # 厳密チェック, 求めた角度ごとに回転させて差分をとり、差がほとんどないか確認する
                    axis_vector = rotational_axis / np.linalg.norm(rotational_axis)

                    sum_angle=0
                    is_valid_symmetry = True
                    print("peak_nums",peak_nums,peak_nums[remaining_index])
                    for i in range(peak_nums[remaining_index]):
                        if(is_valid_symmetry):
                            sum_angle = i * modified_symmetry_angle
                            print("trial[",i,"]:angle=",sum_angle,"deg")
                            # 点群データを回転して軸対称性を判定
                            R=np.array(Rotation.from_rotvec(np.deg2rad(sum_angle) * axis_vector).as_matrix())
                            
                            M        = np.eye(4)
                            M[:3,3]  = P_down.get_center()
                            M_inv    = np.linalg.inv(M)
                            H        = np.eye(4)
                            H[:3,:3] = R

                            reflection_matrix = M @ H @ M_inv
                            if is_reflection_matrix(P_down, reflection_matrix, debug=args.debug):
                                is_valid_symmetry = True
                            else:
                                is_valid_symmetry = False

                    if(is_valid_symmetry):
                        print("rotational_symmetry!", modified_symmetry_angle)
                        rotational_symmetries.append([axis_vector,modified_symmetry_angle])


                else:
                    print("誤差が大きすぎます。修正された平均値を採用できません。")
    print(rotational_symmetries)

    # 最終結果
    arrow_objs = []
    for i, (axis_vector, angle) in enumerate(rotational_symmetries):
        axis_vector = copy.deepcopy(axis_vector)
        # 矢印の生成
        arrow = Arrow(P.get_center(), P.get_center()+0.7*bbox_diagonal_length*axis_vector, color=[1.0,0.0,0.0])
        arrow_obj = arrow.get_poligon()
        arrow_objs.append(arrow_obj)

    if(args.debug):
        o3d.visualization.draw_geometries([ mesh, *arrow_objs ], mesh_show_back_face=True)


    if(len(rotational_symmetries)>0):
        
        output_dir = f"{args.output_dir}/gif/rotational_symmetry"
        # output ディレクトリが存在しない場合は作成する
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{file_name_without_extension}.gif")
        

        num_interpolations = 3  # 各区間で補間する点の数
        from_points = []
        text_messages = []
        for i, (axis_vector, angle) in enumerate(rotational_symmetries):
            axis_vector = copy.deepcopy(axis_vector)
            
            original_angles = create_angle_list(angle)

            # 視点先位置
            look_at_pos = mesh.get_center()

            # カメラ位置 = 視点先位置 + 大きさ * 方向ベクトル
            cameraPosition = look_at_pos + 0.8 *bbox_diagonal_length * axis_vector

            x_cam,y_cam,z_cam = get_view_matrix(upvector=np.array([0, 0, 1]),
                                                campos_w=cameraPosition,
                                                tarpos_w=look_at_pos)

            angles = interpolate_angles_by_linear(original_angles, num_interpolations)
            angles = [np.deg2rad(ang) for ang in angles]
            circle_points = generate_circle_on_plane(x_cam, mesh.get_center(), bbox_diagonal_length, angles)
            for point in circle_points:
                from_points.append(point)

            circle_points = generate_circle_on_plane(y_cam, mesh.get_center(), bbox_diagonal_length, angles)
            for point in circle_points:
                from_points.append(point)

            angles = interpolate_angles_by_expo(original_angles, num_interpolations)
            angles = [np.deg2rad(ang) for ang in angles]
            circle_points = generate_circle_on_plane(z_cam, mesh.get_center(), bbox_diagonal_length, angles)
            for point in circle_points:
                from_points.append(point)

                
            rotation_matrix = rotation_matrix_to_align_with_dst_axis(axis_vector, np.array([-1,0,0]))
       
            H=np.eye(4)
            H[:3, :3] = np.linalg.inv(rotation_matrix)

            M = np.eye(4)
            M[:3,3] = mesh.get_center()
            M_inv   = np.linalg.inv(M)
            
            if sys.platform.startswith('win32'):
                # Windowsの場合の処理
                text_message = text_3d(f"{angle}", arrow.end_pos, matrix=M@H@M_inv, font_size=900, density=0.1)
                text_messages.append(text_message)

        if(args.debug):
            # Open3Dの点群データに変換
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(from_points)
            o3d.visualization.draw_geometries([point_cloud, mesh, *arrow_objs])

        if sys.platform.startswith('win32'):
            create_rotation_gif(np.array(from_points), [mesh, *arrow_objs], output_file)
        elif sys.platform.startswith('linux'):
            create_rotation_gif(np.array(from_points), [mesh, *arrow_objs, *text_messages], output_file)


    output_dir = f"{args.output_dir}/rotational_symmetry"
                    
    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # # JSONファイルに保存するパス
    output_file_path = os.path.join(output_dir,f"{file_name_without_extension}_rotational_symmetry.json")


    def convert_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # NumPyのarrayをリストに変換
        elif isinstance(obj, tuple):
            return list(obj)  # タプルをリストに変換
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")
        
    # データをJSONファイルに保存
    with open(output_file_path, 'w') as json_file:
        json.dump(rotational_symmetries, json_file, indent=2, default=convert_to_json)
