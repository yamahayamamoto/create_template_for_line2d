import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve
import os


"""球面上での視点から物体の観測性を評価し、自動分離（閾値はヒストグラムの最初の谷から判定する）
python symmetry\observability_map.py --ply symmetry\tmp_obj_reduce\95D95-06010\95D95-06010.ply  
--output_dir symmetry\observability
"""

def hinter_sampling(min_n_points, radius=1.0):
    """Samples 3D points on a sphere surface by refining an icosahedron, as in:
    Hinterstoisser et al., Simultaneous Recognition and Homography Extraction of
    Local Patches with a Simple Linear Classifier, BMVC 2008

    :param min_n_points: The minimum number of points to sample on the whole sphere.
    :param radius: Radius of the sphere.
    :return: 3D points on the sphere surface and a list with indices of refinement
        levels on which the points were created.
    """
    # Vertices and faces of an icosahedron.
    a, b, c = 0.0, 1.0, (1.0 + np.sqrt(5.0)) / 2.0
    points = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
            (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b),
            (-c, a, b)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
            (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
            (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
            (8, 6, 7), (9, 8, 1)]

    # Refinement levels on which the points were created.
    points_level = [0 for _ in range(len(points))]

    ref_level = 0
    while len(points) < min_n_points:
        ref_level += 1
        edge_pt_map = {}  # Mapping from an edge to a newly added point on the edge.
        faces_new = []  # New set of faces.

        # Each face is replaced by four new smaller faces.
        for face in faces:
            pt_inds = list(face)  # List of point ID's involved in the new faces.
            for i in range(3):

                # Add a new point if this edge has not been processed yet, or get ID of
                # the already added point.
                edge = (face[i], face[(i + 1) % 3])
                edge = (min(edge), max(edge))
                if edge not in edge_pt_map.keys():
                    pt_new_id = len(points)
                    edge_pt_map[edge] = pt_new_id
                    pt_inds.append(pt_new_id)

                    pt_new = 0.5 * (np.array(points[edge[0]]) + np.array(points[edge[1]]))
                    points.append(pt_new.tolist())
                    points_level.append(ref_level)
                else:
                    pt_inds.append(edge_pt_map[edge])

            # Replace the current face with four new faces.
            faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
                            (pt_inds[3], pt_inds[1], pt_inds[4]),
                            (pt_inds[3], pt_inds[4], pt_inds[5]),
                            (pt_inds[5], pt_inds[4], pt_inds[2])]
        faces = faces_new

    # Project the points to a sphere.
    points = np.array(points)
    points *= np.reshape(radius / np.linalg.norm(points, axis=1), (points.shape[0], 1))

    # Collect point connections.
    pt_conns = {}
    for face in faces:
        for i in range(len(face)):
            pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
            pt_conns[face[i]].add(face[(i + 2) % len(face)])

    # Order the points - starting from the top one and adding the connected points
    # sorted by azimuth.
    top_pt_id = np.argmax(points[:, 2])
    points_ordered = []
    points_todo = [top_pt_id]
    points_done = [False for _ in range(points.shape[0])]

    def calc_azimuth(x, y):
        two_pi = 2.0 * np.pi
        return (np.arctan2(y, x) + two_pi) % two_pi

    while len(points_ordered) != points.shape[0]:
        # Sort by azimuth.
        points_todo = sorted(points_todo, key=lambda i: calc_azimuth(points[i][0], points[i][1]))
        points_todo_new = []
        for pt_id in points_todo:
            points_ordered.append(pt_id)
            points_done[pt_id] = True
            points_todo_new += [i for i in pt_conns[pt_id]]  # Find the connected points.

        # Points to be processed in the next iteration.
        points_todo = [i for i in set(points_todo_new) if not points_done[i]]

    # Re-order the points and faces.
    points = points[np.array(points_ordered), :]
    points_level = [points_level[i] for i in points_ordered]
    points_order = np.zeros((points.shape[0],))
    points_order[np.array(points_ordered)] = np.arange(points.shape[0])
    for face_id in range(len(faces)):
        faces[face_id] = [points_order[i] for i in faces[face_id]]
    # import open3d as o3d
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(points)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # o3d.io.write_triangle_mesh("./output/_hinter_sampling.ply", mesh, write_ascii=True)

    return points, points_level


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
    

def compute_observability_map(pcd, viewpoints, debug=False):
    """
    観測可能性を計算する関数。

    Parameters:
        pcd (open3d.geometry.PointCloud): 元の点群データ。
        viewpoints (list): 視点のリスト。
        debug (bool, optional): デバッグモードの有無を指定するフラグ。デフォルトはFalse。

    Returns:
        numpy.ndarray: 観測可能性を表す配列。

    References:
        - "3D Object Detection and Pose Estimation of Unseen Objects in Color Images with Local Surface Embeddings"
          Link: https://arxiv.org/abs/2010.04075
        - "Position and pose recognition of randomly stacked objects using highly observable 3D vector pairs"
          Link: http://lars.mec.ua.pt/public/LAR%20Projects/BinPicking/2016_RodrigoSalgueiro/LIB/2014_Position_and_Pose_Recognition_of_Randomly_Stacked_Objects_using_Highly_Observable_3D_Vector_Pairs.pdf
    """
    observability_map = []

    pxyz_list = [] # px, py, pzの値を保存するリスト
    for viewpoint in viewpoints:
        # 点群データの最大最小境界を取得
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

        # 方位角を計算 [0, 2π]
        azimuth = np.arctan2(viewpoint[1], viewpoint[0])
        if azimuth < 0:
            azimuth += 2.0 * np.pi

        # 仰角を計算 [-π/2, π/2]
        a = np.linalg.norm(viewpoint)
        b = np.linalg.norm([viewpoint[0], viewpoint[1], 0])

        elevation = np.arccos(b / a)
        
        if viewpoint[2] < 0:
            elevation = -elevation

        px = diameter*np.cos(azimuth) * np.cos(elevation)
        py = diameter*np.sin(azimuth) * np.cos(elevation)
        pz = diameter*np.sin(elevation)
        
        # px, py, pzの値を保存
        pxyz_list.append([px, py, pz])

        # カメラと半径のパラメータを設定
        camera = [px, py, pz]
        
        radius = diameter * 100
        # 隠れた点を削除し、残った点群のメッシュを取得
        mesh_remaining, observability_indices = pcd.hidden_point_removal(camera, radius)  # 適切な半径を設定してください
        
        if(debug):
            mesh_remaining.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh_remaining])
            pcd_remaining = pcd.select_by_index(observability_indices)
            o3d.visualization.draw_geometries([pcd_remaining])

        # メッシュから見える点のインデックスを取得
        observability_map.append(observability_indices)
    
  
    if(debug):
        pxyz_list = np.array(pxyz_list)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pxyz_list[:, 0], pxyz_list[:, 1], pxyz_list[:, 2])
        ax.set_xlabel('px')
        ax.set_ylabel('py')
        ax.set_zlabel('pz')
        plt.show()

    # 観測可能性を計算
    num_viewpoints = len(viewpoints)
    # 観測可能性を保持する配列
    observability = np.zeros(len(pcd.points))

    for indices in observability_map:
        # インデックスに対応する点の観測可能性を増やす
        observability[list(indices)] += 1
    
    # 観測可能性を視点の数で割る
    observability /= num_viewpoints

    return observability

def filter_points_by_observability(pcd, observability, threshold=0.1):
    """
    観測可能性に基づいて点群をフィルタリングする関数。

    Parameters:
        pcd (open3d.geometry.PointCloud): 元の点群データ。
        observability (numpy.ndarray): 点群の観測可能性。
        threshold (float, optional): 観測可能性の閾値。この閾値以上の観測可能性を持つ点が残る。デフォルトは0.1。

    Returns:
        open3d.geometry.PointCloud: フィルタリングされた点群データ。
    """
    filtered_indices = [i for i, obs_value in enumerate(observability) if obs_value >= threshold]
    filtered_pcd = pcd.select_by_index(filtered_indices)
    return filtered_pcd

def visualize_observability(pcd, observability, threshold=None, hist=None, smoothed_hist=None):
    """
    点群の観測可能性に基づいて色分けして表示する関数。

    Parameters:
        pcd (open3d.geometry.PointCloud): 色分けする点群データ。
        observability (numpy.ndarray): 点群の観測可能性。
        threshold (float, optional): 観測可能性の閾値。デフォルトはNone。
        hist (numpy.ndarray, optional): ヒストグラムの値。デフォルトはNone。
        smoothed_hist (numpy.ndarray, optional): スムージングされたヒストグラムの値。デフォルトはNone。

    Returns:
        None
    """
    colors = []
    def sigmoid(x,a=5):
        return 1 / (1 + np.exp(-a*x))
    
    observability_enhanced = sigmoid(observability)
    # 観測可能性に応じた色を設定
    cmap = plt.cm.jet  # カラーマップを選択
    norm = plt.Normalize(vmin=min(observability_enhanced), vmax=max(observability_enhanced))  # 観測可能性の範囲を正規化

    for obs_value in observability_enhanced:
        # 観測可能性に基づいてカラーマップから色を取得
        color = cmap(norm(obs_value))[:3]  # RGBの値のみを使用
        colors.append(color)

    # 点群の色を設定して表示
    pcd.colors = o3d.utility.Vector3dVector(colors)

    fig, ax = plt.subplots(figsize=(10, 5))

    # ヒストグラムをプロット
    if hist is not None:
        ax.hist(observability, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    if smoothed_hist is not None:
        x_values = np.linspace(np.min(observability), np.max(observability), len(smoothed_hist))
        ax.plot(x_values, smoothed_hist, color='blue', label='Smoothed Histogram')
    ax.set_xlabel('Observability')
    ax.set_ylabel('Frequency')
    ax.set_title('Observability Histogram')
    ax.grid(True)
    
    # 閾値が指定されている場合はヒストグラムに縦線を描画する
    if threshold is not None:
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {round(threshold,3)}')
        ax.legend()

    # カラーバーをプロット
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=min(observability), vmax=max(observability))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ダミーの空配列を設定
    cbar = plt.colorbar(sm)
    cbar.set_label('Observability')  # カラーバーのラベルを設定

    plt.show()
    o3d.visualization.draw_geometries([pcd])

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

def filter_and_visualize_observability(pcd, min_n_views=360, threshold=None, debug=False):
    """
    観測可能性に基づいて点群をフィルタリングし、観測可能性に応じた色付けを行い、可視化します。

    Parameters:
        pcd (open3d.geometry.PointCloud): 元の点群データ。
        min_n_views (int): サンプリングする視点の最小数。デフォルトは360です。
        threshold (float): 観測可能性の閾値。この閾値以下の観測可能性を持つ点はフィルタリングされます。
        debug (bool): デバッグモード。Trueの場合は観測可能性を色分けして表示します。デフォルトはTrueです。

    Returns:
        open3d.geometry.PointCloud: フィルタリングされた点群データ。
    """
    # サンプリングする視点のリストを作成
    viewpoints = sample_reflection_normal_by_view_sphere(min_n_views=min_n_views,
                                                         azimuth_range=(0,2*np.pi),
                                                         elev_range=(-0.5*np.pi,0.5*np.pi),)

    # 観測可能性を計算
    observability_map = compute_observability_map(pcd, viewpoints)


    # 閾値を動的に決定する.スムージングされたヒストグラムの最初の谷を閾値にする
    if(threshold is None):
        hist, bins = np.histogram(observability_map, bins=50)
        
        # ガウシアンカーネルを定義
        kernel_size = 5  # カーネルサイズ
        sigma = 1  # ガウシアンの標準偏差
        kernel = np.exp(-(np.arange(kernel_size) - kernel_size // 2) ** 2 / (2 * sigma ** 2))
        kernel /= np.sum(kernel)  # 正規化

        # ヒストグラムをスムージング
        smoothed_hist = convolve(hist, kernel, mode='same')

        # 谷検知
        valley_indices = pick_peak(-smoothed_hist)
        # 最初の谷
        threshold = bins[valley_indices[0]]
    
    if debug:
        # 観測可能性を色分けして表示
        visualize_observability(pcd, observability_map, threshold, hist, smoothed_hist)

    # 指定された閾値以下の観測可能性を持つ点を排除
    filtered_pcd = filter_points_by_observability(pcd, observability_map, threshold)

    # 統計的外れ値を除去
    filtered_pcd, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return filtered_pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PLY file and generate point cloud.')
    parser.add_argument('--ply', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--num_points', type=int, default=20000, help='Number of points in point cloud.')
    parser.add_argument('--min_n_views', type=int, default=120, help='Minimum number of views.')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for observability.')
    parser.add_argument('--output_dir', type=str, default="./observability", help='Threshold for observability.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

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

    filtered_pcd = filter_and_visualize_observability(pcd, min_n_views=args.min_n_views, 
                                                      threshold=args.threshold,
                                                      debug=args.debug)
    
    if(args.debug):
        o3d.visualization.draw_geometries([filtered_pcd])

    
    if args.ply is not None:
        
        # 出力フォルダが存在しない場合は作成する
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        target_obj_filename = os.path.basename(args.ply)
        file_name_without_extension = os.path.splitext(target_obj_filename)[0]
        o3d.io.write_point_cloud(f"{args.output_dir}/{file_name_without_extension}_observable.ply", filtered_pcd)