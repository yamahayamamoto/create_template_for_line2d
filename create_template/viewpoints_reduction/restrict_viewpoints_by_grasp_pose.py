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
import yaml
# cv2.imshow('Watershed Segmentation', np.zeros((500,500)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""物体の形状のシャープさに応じて視点を制限
python pyrender\restrict_view_point_by_grasp_pose.py --mesh symmetry\tmp_obj_reduce\95D95-06010\95D95-06010.ply 
--viewpoint symmetry\result\viewpoints\origin_viewpoints.npy --output_dir symmetry\result\viewpoints\restricted_perspected_area

"""


def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default='./dataset/mydata', 
                        help='input data directory')
    parser.add_argument('--grasp_dataset_dir', type=str)
    parser.add_argument("--output_dir", type=str, default='./result/viewpoints', 
                        help='output data directory')
    parser.add_argument('--viewpoint', type=str, default=None, help='Path to PLY file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    
    parser.add_argument("--sphere_radius", type=float, default=0.5, 
                        help='Radius of the sphere')
    parser.add_argument("--theta_deg", type=float, default=15, 
                        help='Angle in degrees')

    # -- Data options
    parser.add_argument("--min_n_views", type=int, default=132, #default=162, 
                        help='the number of view points on sphere')
    parser.add_argument("--valid_ratio", type=float, default=80, help="A valid ratio between 0 and 100. Default is 80.")

    return parser


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

class Corn():
    def __init__(self, pos=[0,0,0], rpy=None, qwxyz=None, CORNER_NUM=12, radius=2, length=2.0):
        """多角錐を作成するためのクラス

        角数が多いほど、円錐に近づく

        Args:
            pos (float)                       : 位置(x,y,z)[m]
            rpy (float)                       : 姿勢(roll,pitch,yaw)
            qwxyz (float)                     : クォータニオン(qw,qx,qy,qz)
            self.CORNER_NUM (int, optional)   : 角数
            R (int, optional)                 : 図形の大きさ
            L (float, optional)               : 図形の高さ
        """

        self.CORNER_NUM  = CORNER_NUM
        self.length      = length
        self.radius      = radius
        self._pos        = pos
        self._rpy        = rpy
        self._qwxyz      = qwxyz
        
        # オブジェクトの姿勢・位置を表す行列の計算
        if(self._rpy is not None):
            # 姿勢(roll,pitch,yaw)から同次座標変換行列を計算する
            self._matrix = self.rpy_pos2matrix(self._pos, self._rpy)
        elif(self._qwxyz is not None):
            # クォータニオンから同次座標変換行列を計算する
            self._matrix = self.quaternion_pos2matrix(self._pos, self._qwxyz)
        else:
            # 回転させない
            self._matrix = self.rpy_pos2matrix(self._pos, [0,0,0])

    @property
    def matrix(self):
        return self._matrix

    def get_polygon(self):
        return self._obj

    def get_center(self):
        return self._obj.get_center()

    def rpy_pos2matrix(self, pos:float, rpy:float) -> np.ndarray:
        """roll-pitch-yawから同次座標変換行列を計算する

        Args:
            pos (float): 位置(x,y,z)[m]
            rpy (float): 姿勢(roll,pitch,yaw)

        Returns:
            np.ndarray: _description_
        """
        px = pos[0] # x
        py = pos[1] # y
        pz = pos[2] # z
        tx = rpy[0] # θx
        ty = rpy[1] # θy
        tz = rpy[2] # θz

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

    def quaternion_pos2matrix(self, pos:float, qwxyz:float) -> np.ndarray:
        """四元数から同次座標変換行列を計算する

        Args:
            pos (float): 位置(x,y,z)[m]
            qwxyz (float): クォータニオン(qw,qx,qy,qz)

        Returns:
            np.ndarray: 同次座標変換行列
        """
        x  = pos[0]
        y  = pos[1]
        z  = pos[2]
        qw = qwxyz[0]
        qx = qwxyz[1]
        qy = qwxyz[2]
        qz = qwxyz[3]

        # 回転クォータニオンは単位クォータニオンである必要がある
        q_norm = np.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)
        qw = qw / q_norm
        qx = qx / q_norm
        qy = qy / q_norm
        qz = qz / q_norm

        # 並進部分 
        T = np.array([[1, 0, 0, x],
                      [0, 1, 0, y],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]])

        R = np.array([[ 1-2*(qy*qy+qz*qz),  2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy),   0],
                      [ 2*(qx*qy+qw*qz),    1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx),   0], 
                      [ 2*(qx*qz-qw*qy),    2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy), 0],
                      [ 0,                  0,                 0,                 1] ])

        return T.dot(R)

    def create_vert(self):
        """多角錐の頂点作成
        """

        # 角度分解能
        self.fAngleDelta = 2.0 * np.pi / self.CORNER_NUM
        
        # 角度
        fAngle1 = 0.0
        
        # 頂点の定義
        self.vert = []
        self.vert.append([ 0.0, 0.0, 0.0])
        
        # 側面
        for i in range(self.CORNER_NUM):
            p1 = [self.radius * np.cos(fAngle1), self.radius * np.sin(fAngle1), self.length]
            fAngle1 += self.fAngleDelta
            self.vert.append(p1)
            
        # 底面
        self.vert.append([0.0, 0.0, self.length])
        for i in range(self.CORNER_NUM):
            p1 = [self.radius * np.cos(fAngle1), self.radius * np.sin(fAngle1), self.length]
            fAngle1 += self.fAngleDelta
            self.vert.append(p1)
        
    def create_face(self):
        """多角錐の面
        """
        # インデックスデータ作成
        self.faces = []
        
        # 側面
        for i in range(1,self.CORNER_NUM+1):
            self.faces.append([0,(i%self.CORNER_NUM)+1, i])
        
        # 底面
        for i in range(1,self.CORNER_NUM+1):
            self.faces.append([(self.CORNER_NUM+1),i+(self.CORNER_NUM+1),(i%self.CORNER_NUM)+1+(self.CORNER_NUM+1)])
    
    def create_line(self):
        """多角錐の線
        """
        # 線作成
        self.lines = []
        
        # 側面
        for i in range(1,self.CORNER_NUM+1):
            self.lines.append([0,(i%self.CORNER_NUM)+1])
            self.lines.append([0,i])
            self.lines.append([(i%self.CORNER_NUM)+1, i])
        
        # 底面
        for i in range(1,self.CORNER_NUM+1):
            self.lines.append([(self.CORNER_NUM+1),(i%self.CORNER_NUM)+1+(self.CORNER_NUM+1)])
            self.lines.append([(self.CORNER_NUM+1),i+(self.CORNER_NUM+1)])
            self.lines.append([i+(self.CORNER_NUM+1),(i%self.CORNER_NUM)+1+(self.CORNER_NUM+1)])

    def get_polygon(self):
        """_メッシュデータの作成

        Returns:
            _type_: _メッシュデータ
        """
        self.create_vert()
        self.create_face()
        self.object = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vert),
                                                o3d.utility.Vector3iVector(self.faces))
                                                
        # 同次座標変換行列を用いて座標系を変換する
        self.object.transform(self._matrix)
        self.object.compute_vertex_normals()
        return self.object

    def get_wireframe(self, color=[1,0,0]):
        """ワイヤーフレームデータの作成

        Args:
            color (list, optional): RGBデータの各々の輝度を0～1で表したもの

        Returns:
            _type_: ワイヤーフレームデータ
        """
        self.create_vert()
        self.create_line()
        self.wireframe = o3d.geometry.LineSet(o3d.utility.Vector3dVector(self.vert),
                                                o3d.utility.Vector2iVector(self.lines))
                                                
        # 同次座標変換行列を用いて座標系を変換する
        self.wireframe.transform(self._matrix)

        colors = [color for i in range(len(self.lines))]
        self.wireframe.colors = o3d.utility.Vector3dVector(colors)
        return self.wireframe
    
def generate_sphere_pointcloud(radius=0.5, resolution=50):
    # 球のメッシュを生成する
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
    mesh_sphere.compute_vertex_normals()
    
    # メッシュの頂点をポイントクラウドに変換する
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = mesh_sphere.vertices
    
    pointcloud.paint_uniform_color([0, 0, 0])
    return pointcloud

def extract_points_in_conical_region(sphere, theta_deg, direction):

    # 球の頂点を取得する
    points = np.asarray(sphere.points)
    # θをラジアンに変換する
    theta_rad = np.radians(theta_deg)
    # 方向ベクトルを正規化する
    direction = direction / np.linalg.norm(direction)
    # 各点と方向ベクトルの間の内積を計算する
    dot_product = np.dot(points, direction)
    # 各点と方向ベクトルの間の角度を計算する
    angles_rad = np.arccos(dot_product / np.linalg.norm(points, axis=1))

    points_in_cone_indices = np.where(angles_rad <= theta_rad)[0]
    
    # 角度θ以内の点を抽出する
    points_in_cone = sphere.select_by_index(points_in_cone_indices)
    points_in_cone.paint_uniform_color([0.0, 0.0, 1.0])
    return points_in_cone_indices, points_in_cone

def extract_points_in_multiple_conical_region(sphere, theta_deg, directions):
        
    # 指定した方向と角度内の点を抽出する
    multiple_points_in_cone_indices = []
    for direction in directions:
        points_in_cone_indices, _ = extract_points_in_conical_region(sphere, theta_deg, direction)
        multiple_points_in_cone_indices.extend(points_in_cone_indices)
        
    # 全体の重複を排除
    multiple_points_in_cone_indices = list(set(multiple_points_in_cone_indices))
    
    multiple_points_in_cone = sphere.select_by_index(multiple_points_in_cone_indices)
    return multiple_points_in_cone_indices, multiple_points_in_cone

def multiple_cones(viewpoints, directions):
    sphere_radius = (np.max(viewpoints.points)-np.min(viewpoints.points))/2.0

    z_axis = np.array([0, 0, 1])

    cones = []
    for direction in directions:
        # 円錐を正しい方向に回転させる
        # 方向ベクトルに基づいて回転行列を計算する
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))

        if np.linalg.norm(rotation_axis) != 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
        rotation_matrix,_ = cv2.Rodrigues(rotation_axis * rotation_angle)

        # 円錐を生成する
        cone = Corn(pos=viewpoints.get_center(), radius=sphere_radius*(2*np.deg2rad(theta_deg))/2.0, length=sphere_radius*1.0).get_wireframe()
        cone.rotate(rotation_matrix, center=viewpoints.get_center())
        cones.append(cone)
    return cones


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    path_mesh = args.mesh
    filename_mesh = os.path.basename(path_mesh)
    filename_mesh_wo_ex = os.path.splitext(filename_mesh)[0]        

    mesh = o3d.io.read_triangle_mesh(path_mesh)
    mesh.compute_vertex_normals()


    with open(os.path.join(args.grasp_dataset_dir, filename_mesh_wo_ex, "good_grips.yaml"),"r") as f:
        dataset = yaml.safe_load(f)

    
    # top_kを計算
    top_k = int(len(dataset) * (args.valid_ratio / 100.0))
    dataset = dataset[:top_k]
    
    directions=[]
    for data in dataset:
        directions.append(data['grasp_approach_vector'])

    if(args.viewpoint is not None):
        viewpoint_vectors = np.load(args.viewpoint)
    else:
        viewpoint_vectors = generate_shpere_surface_points_by_fib(int(args.min_n_views/2))
    

    sphere_radius = args.sphere_radius
    theta_deg = args.theta_deg


    viewpoints = o3d.geometry.PointCloud()
    viewpoints.points = o3d.utility.Vector3dVector(viewpoint_vectors) 

    if(args.debug):
        o3d.visualization.draw_geometries([viewpoints])

    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 指定した方向と角度内の点を抽出する
    _, filtered_viewpoints = extract_points_in_multiple_conical_region(viewpoints, theta_deg, directions)
    cones = multiple_cones(viewpoints, directions)

    if(args.debug):
        # 球、円錐領域内の点、およびワイヤーフレーム円錐を可視化する
        o3d.visualization.draw_geometries([filtered_viewpoints, *cones])
        o3d.visualization.draw_geometries([filtered_viewpoints, viewpoints, *cones])
        o3d.visualization.draw_geometries([viewpoints, *cones])
        mesh.scale(0.01,mesh.get_center())

        o3d.visualization.draw_geometries([filtered_viewpoints, mesh])

    filtered_viewpoints_array = np.array(filtered_viewpoints.points)

    # # JSONファイルに保存するパス
    output_file_path = os.path.join(args.output_dir,f"{filename_mesh_wo_ex}.npy")

    print("*"*99)
    print("name:",filename_mesh_wo_ex)
    print("the number of viewpoints(before):",len(viewpoint_vectors))
    print("the number of viewpoints(after):",len(filtered_viewpoints_array))
    print("*"*99)

    # # NumPy配列を保存
    np.save(output_file_path, filtered_viewpoints_array)
