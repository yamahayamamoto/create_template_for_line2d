import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import open3d as o3d
from tqdm import tqdm
from collections import defaultdict
import json
import configargparse
import copy 
import cv2
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation
# cv2.imshow('Watershed Segmentation', np.zeros((500,500)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""物体の形状のシャープさに応じて視点を制限
python pyrender\restrict_view_point_by_perspective_area.py --mesh symmetry\tmp_obj_reduce\95D95-06010\95D95-06010.ply 
--viewpoint symmetry\result\viewpoints\origin_viewpoints.npy --output_dir symmetry\result\viewpoints\restricted_perspected_area --gif

"""


def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default='./dataset/mydata', 
                        help='input data directory')
    parser.add_argument('--variance_threshold', type=float, default=0.35, help='Variance threshold value')
    parser.add_argument("--output_dir", type=str, default='./result/viewpoints', 
                        help='output data directory')
    parser.add_argument('--viewpoint', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--gif', action='store_true', help='create GIF.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    

    # -- Data options
    parser.add_argument("--min_n_views", type=int, default=132, #default=162, 
                        help='the number of view points on sphere')
    return parser


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
    print("rotation_matrix",dst_R_src)
    return dst_R_src 


def rpy_pos2matrix(pos:float, rpy:float) -> np.ndarray:
    """roll-pitch-yawから同次座標変換行列を計算する
    
    Args:
    pos (float): 位置(x,y,z)[m]
    rpy (float): 姿勢(roll,pitch,yaw)
    
    Returns:
    np.ndarray: _description_
    """
    px,py,pz = pos
    tx,ty,tz = rpy
    
    c = np.cos
    s = np.sin
    
    # 並進部分
    T = np.array([[1, 0, 0, px],
    [0, 1, 0, py],
    [0, 0, 1, pz],
    [0, 0, 0,  1]])
    
    # 回転部分
    R = np.array([[c(tz)*c(ty), c(tz)*s(ty)*s(tx)-c(tx)*s(tz), s(tz)*s(tx)+c(tz)*c(tx)*s(ty), 0],
    [c(ty)*s(tz), c(tz)*c(tx)+s(tz)*s(ty)*s(tx), c(tx)*s(tz)*s(ty)-c(tz)*s(tx), 0],
    [     -s(ty),                   c(ty)*s(tx), c(ty)*c(tx)                  , 0],
    [          0,                             0,                             0, 1]])
    
    return T.dot(R)
 
 
class CoordinateFrameMesh():
    def __init__(self, pos=[0,0,0], *, scale=1.0, rpy=None, qwxyz=None):
 
        self._pos   = pos
        self._rpy   = rpy
        self._qwxyz = qwxyz
 
        self._obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
 
        if(self._rpy is not None):
            # 姿勢(roll,pitch,yaw)から同次座標変換行列を計算する
            self._matrix = rpy_pos2matrix(self._pos, self._rpy)
        elif(self._qwxyz is not None):
            # クォータニオンから同次座標変換行列を計算する
            self._matrix = self.quaternion_pos2matrix(self._pos, self._qwxyz)
        else:
            # 回転させない
            self._matrix = rpy_pos2matrix(self._pos, [0,0,0])
 
        # 同次座標変換行列を用いて座標系を変換する
        self._obj.transform(self._matrix)
 
    @property
    def matrix(self):
        return self._matrix
 
    def get_poligon(self):
        return self._obj
 
    def get_center(self):
        return self._obj.get_center()
    
def move2origin_orientation(pcd):
    """
    点群を原点に移動させ、回転もワールド座標に一致させる関数

    Args:
        pcd (open3d.geometry.PointCloud): 処理対象の点群

    Returns:
        open3d.geometry.PointCloud: 処理後の点群
        numpy.ndarray: ワールド座標とオブジェクト座標の間のホモグラフィ行列
    """
    # 点群の座標をNumPy配列に変換
    points = np.array(pcd.points)

    # ホモグラフィ行列を計算してオブジェクト座標をワールド座標に一致させる
    objHworld = calculate_homography_matrix_by_PCA(points)

    # ホモグラフィ行列の逆行列を取得
    worldHobj = np.linalg.inv(objHworld)

    # 点群をワールド座標に変換
    pcd.transform(worldHobj)

    return pcd, worldHobj

def calculate_homography_matrix_by_PCA(xyz):
    """PCAを使用してホモグラフィ行列を計算する

    Args:
        xyz (numpy.ndarray): 3Dポイントクラウドの座標情報 (N x 3)

    Returns:
        numpy.ndarray: ホモグラフィ行列 (4x4)
    """

    # 引数xyzがNumPy配列でない場合、NumPy配列に変換
    if not isinstance(xyz, np.ndarray):
        xyz = np.array(xyz)
 
    # ポイントクラウドの中心座標を計算
    center = xyz.mean(axis=0)
    tx, ty, tz = center
 
    # ポイントクラウドを中心座標に対して平行移動
    normalized_pts = xyz - center
 
    # 共分散行列を計算
    pp = normalized_pts.T.dot(normalized_pts)
 
    # 特異値分解（SVD）を実行
    u, w, vt = np.linalg.svd(pp)
 
    # 主成分ベクトルを取得
    v1 = u[:, 0]
    v2 = u[:, 1]
    v3 = u[:, 2]
    
    # ホモグラフィ行列を作成
    H = np.array([[v1[0], v2[0], v3[0], tx],
                  [v1[1], v2[1], v3[1], ty],
                  [v1[2], v2[2], v3[2], tz],
                  [0, 0, 0, 1]])
 
    return H
 
def rotation_matrix_to_align_with_x_axis(vec): 
    """ 
    ベクトルvecをx軸ベクトルx_axis=[1,0,0]に一致させる回転行列を計算する。 
 
    Parameters: 
        vec (numpy.ndarray): 3次元ベクトル (x, y, z) を表すNumPy配列。 
 
    Returns: 
        numpy.ndarray: ベクトルvecをx軸に一致させるための回転行列。vecがx軸と一致している場合は単位行列が返される。 
    """ 
    x_axis = np.array([1, 0, 0])  # x軸の単位ベクトル 
 
    # ベクトルvecとx_axisの外積を計算 
    cross_product = np.cross(vec, x_axis) 
 
    # 回転角度を計算 
    if np.linalg.norm(cross_product) < 1e-12: 
        # 外積がほぼゼロの場合（vecがx軸と一致する場合） 
        return np.identity(3)  # 単位行列を返す 
 
    # 外積を正規化して回転軸を計算 
    rotation_axis = cross_product / np.linalg.norm(cross_product) 
 
    # ドット積を計算 
    dot_product = np.dot(vec, x_axis) 
 
    # 回転角度を計算 
    rotation_angle = np.arccos(np.clip(dot_product / (np.linalg.norm(vec) * np.linalg.norm(x_axis)), -1.0, 1.0)) 
 
    # 回転行列を計算 
    cos_theta = np.cos(rotation_angle) 
    sin_theta = np.sin(rotation_angle) 
    rotation_matrix = np.array([ 
        [cos_theta + rotation_axis[0] ** 2 * (1 - cos_theta), rotation_axis[0] * rotation_axis[1] * (1 - cos_theta) - rotation_axis[2] * sin_theta, rotation_axis[0] * rotation_axis[2] * (1 - cos_theta) + rotation_axis[1] * sin_theta], 
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_theta) + rotation_axis[2] * sin_theta, cos_theta + rotation_axis[1] ** 2 * (1 - cos_theta), rotation_axis[1] * rotation_axis[2] * (1 - cos_theta) - rotation_axis[0] * sin_theta], 
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_theta) - rotation_axis[1] * sin_theta, rotation_axis[2] * rotation_axis[1] * (1 - cos_theta) + rotation_axis[0] * sin_theta, cos_theta + rotation_axis[2] ** 2 * (1 - cos_theta)] 
    ]) 
     
    return rotation_matrix 

def plot_figure(points, variance_heatmap,save_path=None, debug=False):
    
    # データを取得
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # PCAを実行
    pca = PCA(n_components=3)
    pca.fit(points)

    # 主成分ベクトルを取得
    components = pca.components_

    # プロット
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection='3d')

    colors = ["red", "green", "blue"]
    # 主成分ベクトルをプロットに追加
    for i in range(3):
        ax1.quiver(0, 0, 0, 5 * components[i, 0], 5 * components[i, 1], 5 * components[i, 2], color=colors[i], label=f'PC{i+1}')

    # データ点をプロット
    ax1.scatter(x, y, z, c='blue', label='Data Points')

    # グラフの設定
    ax1.set_title('Correlated 3D Data Points')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')
    # アスペクト比を等しくする
    ax1.set_box_aspect([1, 1, 1])  # x, y, zのアスペクト比を1に設定
    ax1.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(sphere_edge_x, sphere_edge_y, sphere_edge_z, cstride=1,
                      rstride=1, facecolors=plt.cm.jet(variance_heatmap), shade=False)
    ax2.set_box_aspect([1, 1, 1])  # x, y, zのアスペクト比を1に設定

    # 主成分ベクトルをプロットに追加
    for i in range(3):
        ax2.quiver(0, 0, 0, 2 * components[i, 0], 2 * components[i, 1], 2 * components[i, 2], color=colors[i], label=f'PC{i+1}')

    # カラーマップと色の対応を取得
    norm = Normalize(vmin=np.min(variance_heatmap), vmax=np.max(variance_heatmap))
    sm = ScalarMappable(cmap=plt.cm.jet, norm=norm)

    # カラーバーを追加
    cbar = plt.colorbar(sm, ax=ax2, orientation='vertical')
    cbar.set_label('Variance')

    # プロット用の関数
    def update(frame):
        # ax1.view_init(elev=10, azim=frame)
        # ax2.view_init(elev=10, azim=frame)
        ax1.view_init(elev=frame, azim=0)
        ax2.view_init(elev=frame, azim=0)

    # アニメーションを設定
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 30), interval=30)

    if(save_path is not None):
        # GIFを保存
        ani.save(save_path, writer='pillow',fps=5)
        print("saved",save_path)
    else:
        pass

    if(debug):
        plt.show()

def normalize_point_cloud(pcd, voxel_size=0.05):
    import copy
    sample_pcd = copy.deepcopy(pcd)
    # 1. 3Dバウンディングボックスを計算
    aabb = pcd.get_oriented_bounding_box()
    corners = np.asarray(aabb.get_box_points())

    # 2. バウンディングボックスの対角線ベクトルを計算
    diagonal_vector = np.max(corners, axis=0) - np.min(corners, axis=0)

    # 3. バウンディングボックスの対角線ベクトルの最大値を取得
    max_diagonal = np.max(diagonal_vector)

    # 4. スケーリングファクターを計算
    scale_factor = 1.0 / max_diagonal

    # 5. 点群データを中心を基準にスケーリング
    sample_pcd.scale(scale_factor, center=aabb.center)

    # Down-sampling by average voxel grid
    sample_pcd = sample_pcd.voxel_down_sample(voxel_size=voxel_size)

    return sample_pcd


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


def create_cone_along_direction(position, target, radius=0.1, height=0.2):
    # コーンの方向ベクトルを計算
    direction = target - position
    direction /= np.linalg.norm(direction)  # 単位ベクトルに正規化

    print(direction)
    # コーンを生成
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height)
    mesh.transform([[1, 0, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi), 0],
                    [0, np.sin(np.pi), np.cos(np.pi),0], [0, 0, 0, 1]])

    # コーンの向きを調整するための回転行列を計算
    z_axis = np.array([0, 0, 1])  # コーンの元の向きはZ軸に沿っている
    rotation_axis = np.cross(z_axis, direction)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction),-1,1))

    print(rotation_axis,rotation_angle)
    
    # 回転角度を計算 
    if np.linalg.norm(rotation_axis) < 1e-12: 
        # 外積がほぼゼロの場合（src_vecがdst_vecと一致する場合） 
        rotation_matrix = np.identity(3)  # 単位行列を返す 
    else:
        rotation_matrix= np.array(Rotation.from_rotvec(rotation_axis*rotation_angle).as_matrix())

    # コーンの位置を指定
    mesh.translate(position)

    # コーンの向きを調整
    mesh.rotate(rotation_matrix, center=position)

    return mesh

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    target_obj_file = args.mesh
    target_obj_filename = os.path.basename(target_obj_file)
    file_name_without_extension = os.path.splitext(target_obj_filename)[0]        

    mesh = o3d.io.read_triangle_mesh(target_obj_file)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(number_of_points=7000, init_factor=5)


    if(args.viewpoint is not None):
        viewpoint_vectors = np.load(args.viewpoint)
    else:
        raise Exception()
    
    viewpoints = o3d.geometry.PointCloud()
    viewpoints.points = o3d.utility.Vector3dVector(viewpoint_vectors) 


    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ワールド座標系
    world_frame     = CoordinateFrameMesh(scale=1)
    world_frame_obj = world_frame.get_poligon()

    # データ点の数
    num_points = 200
    thetas = np.linspace(0,  2*np.pi, num_points)  # [0, 2π]の範囲で角度θを生成
    phis = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # [-π/2, π/2]の範囲で角度φを生成

    # データ点を球面座標から直交座標に変換
    sphere_edge_x = np.outer(np.cos(thetas), np.cos(phis))
    sphere_edge_y = np.outer(np.sin(thetas), np.cos(phis))
    sphere_edge_z = np.outer(np.ones(num_points), np.sin(phis))

    # 単位円にmeshの分散を投影するために標準化する
    sample_pcd = normalize_point_cloud(pcd)

    # 重心を原点に移動
    sample_pcd.translate(-sample_pcd.get_center())

    sampling_point = np.array(sample_pcd.points)

    variances = np.zeros((len(thetas), len(phis)))

    sphere_viewpoints = np.zeros((len(thetas), len(phis),3))

    for i, theta in tqdm(enumerate(thetas)):
        for j, phi in enumerate(phis):
            rotated_pcd = copy.deepcopy(sample_pcd)
            vertex_point = np.array([sphere_edge_x[i,j],sphere_edge_y[i,j],sphere_edge_z[i,j]])
            # オイラー角に基づいて回転行列を計算
            # zyxオイラー角の逆回転を行うのでxyzオイラー角で行う。
            rotation_matrix = rotation_matrix_to_align_with_x_axis(vertex_point)
            
            H=np.eye(4)
            H[:3,:3]=rotation_matrix
            rotated_pcd.transform(H) 
            
            if(args.debug):
                
                # コーンを生成
                cone = create_cone_along_direction(vertex_point, sample_pcd.get_center())
                rotated_pcd.paint_uniform_color([1,0,0])
                o3d.visualization.draw_geometries([sample_pcd, world_frame_obj, rotated_pcd,cone], mesh_show_back_face=True)

            rotate_data = np.array(rotated_pcd.points)
            rotated_x = rotate_data[:,0]
            rotated_y = rotate_data[:,1]
            rotated_z = rotate_data[:,2]

            y_min = np.min(rotated_y[rotated_x<0]) 
            y_max = np.max(rotated_y[rotated_x<0]) 
            z_min = np.min(rotated_z[rotated_x<0]) 
            z_max = np.max(rotated_z[rotated_x<0]) 

            # y_min = np.min(rotated_y) 
            # y_max = np.max(rotated_y) 
            # z_min = np.min(rotated_z) 
            # z_max = np.max(rotated_z) 
        
            len_y = np.fabs(y_max - y_min)
            len_z =  np.fabs(z_max - z_min)
            area = len_y * len_z

            # x軸における分散を計算
            variance_x = np.var(rotated_x)
            # variance_y = np.max(np.fabs(np.var(rotated_y[rotated_x<0])))
            # variance_z = np.max(np.fabs(np.var(rotated_z[rotated_x<0])))
            variance_y = np.fabs(np.var(rotated_y))
            variance_z = np.fabs(np.var(rotated_z))
        
            # variances[i, j] = np.sqrt(2)*np.max([variance_y,variance_z])#np.log(frobenius_norm) #variance_y+variance_z
            # variances[i, j] = np.min([variance_y,variance_z])#np.log(frobenius_norm) #variance_y+variance_z
            variances[i, j] = variance_y+variance_z #np.log(frobenius_norm) #variance_y+variance_z
            # variances[i, j] = area
            sphere_viewpoints[i, j] = vertex_point

    variance_heatmap = variances / np.amax(variances)

    if(args.gif):
        # gifの保存
        parent_dir = os.path.join(os.path.dirname(os.path.dirname(args.output_dir)), "gif",  "restricted_view_by_perspective_area")
        
        # output ディレクトリが存在しない場合は作成する
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        path_save_gif = os.path.join(parent_dir, f"{file_name_without_extension}.gif")
        print(path_save_gif)
        plot_figure(sampling_point, variance_heatmap, path_save_gif, args.debug)


    normalized_variances = variances / np.amax(variances)
    valid_indices = np.where(normalized_variances > args.variance_threshold)

    restricted_viewpoints = np.array(sphere_viewpoints[valid_indices])

    restricted_viewpoints_pcd = o3d.geometry.PointCloud()
    restricted_viewpoints_pcd.points = o3d.utility.Vector3dVector(restricted_viewpoints.reshape((-1,3)))   

    if(args.debug):
        o3d.visualization.draw_geometries([sample_pcd, world_frame_obj, restricted_viewpoints_pcd], mesh_show_back_face=True)

    filtered_viewpoints_pcd = find_nearest_neighbors(viewpoints, restricted_viewpoints_pcd)


    if(args.debug):
        filtered_viewpoints_pcd.paint_uniform_color([0,0,1])
        restricted_viewpoints_pcd.paint_uniform_color([1,0,0])
        o3d.visualization.draw_geometries([sample_pcd, world_frame_obj, filtered_viewpoints_pcd,restricted_viewpoints_pcd], mesh_show_back_face=True)


    # # JSONファイルに保存するパス
    output_file_path = os.path.join(args.output_dir,f"{file_name_without_extension}.npy")

    filtered_viewpoints = np.array(filtered_viewpoints_pcd.points)

    print("*"*99)
    print("name:",file_name_without_extension)
    print("the number of viewpoints(before):",len(viewpoint_vectors))
    print("the number of viewpoints(after):",len(filtered_viewpoints))
    print("*"*99)

    # NumPy配列を保存
    np.save(output_file_path, filtered_viewpoints)