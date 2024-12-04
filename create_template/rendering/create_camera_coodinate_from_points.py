import numpy as np
import open3d as o3d
import json
from typing import Optional,List
import os
import configargparse

import numpy as np
import configargparse
import open3d as o3d
import cv2
import copy


"""視点からカメラの姿勢を作成する
python pyrender\create_camera_coodinate_from_points.py --viewpoints_file symmetry\result\viewpoints\restricted_rotation_symmetry\95D95-06010.npy --ply pyrender\dataset\mydata\models\95D95-06010\95D95-06010.ply
--axial_angle_division_number 1 --output_dir pyrender\result\camera_coordinates
"""

def config_parser():
    parser = configargparse.ArgumentParser()
    # -- Data options
    parser.add_argument("--viewpoints_file", type=str, help="Path to viewpoints file (.npy)")
    parser.add_argument("--ply", type=str, help="Path to input PLY file")
    parser.add_argument("--output_dir", type=str, default="./result/camera_coordinates", help="Path to output numpy file")
    parser.add_argument("--initial_distance_mm", type=float, default=330, help='camera initial distance[mm]')
    parser.add_argument("--end_distance_mm", type=float, default=220, help='camera final distance[mm]')
    parser.add_argument("--distance_division_number", type=int, default=6, help='division number of the distance between initial_distance_mm and final_distance_mm')
    parser.add_argument("--axial_angle_division_number", type=int, default=1, help='division number of axial angle[0,2*pi]')
    parser.add_argument("--azimuth_range", type=tuple, default=(0, 2*np.pi), help="Azimuth range (default: (0, 2*pi))")
    parser.add_argument("--elev_range", type=tuple, default=(-0.5*np.pi, 0.5*np.pi), help="Elevation range (default: (-0.5*pi, 0.5*pi))")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    return parser


def rpy_pos2matrix(pos:float, rpy:float) -> np.ndarray:
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

def quaternion_pos2matrix(pos:float, qwxyz:float) -> np.ndarray:
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

class CoordinateFrameMesh():
    def __init__(self, pos=[0,0,0], rpy=[0,0,0], scale=0.5, qwxyz=None):

        self._pos   = pos
        self._rpy   = rpy
        self._qwxyz = qwxyz

        self._obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)

        if(self._rpy is not None):
            # 姿勢(roll,pitch,yaw)から同次座標変換行列を計算する
            self._matrix = rpy_pos2matrix(self._pos, self._rpy)
        elif(self._qwxyz is not None):
            # クォータニオンから同次座標変換行列を計算する
            self._matrix = quaternion_pos2matrix(self._pos, self._qwxyz)
        else:
            # 回転させない
            self._matrix = rpy_pos2matrix(self._pos, [0,0,0])

        # 同次座標変換行列を用いて座標系を変換する
        self._obj.transform(self._matrix)

    @property
    def matrix(self):
        return self._matrix

    def transform(self, H):
        self._matrix = np.matmul(H, self._matrix)

    def get_poligon(self):
        return self._obj

    def get_center(self):
        return self._obj.get_center()

def load_viewpoints(viewpoints_file):
    viewpoints = np.load(viewpoints_file)
    return viewpoints


def get_direction(azimuth, elevation):
    """極座標系で方向ベクトルを求める
    """
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)

    direction_vector =  np.array([x, y, z])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    return direction_vector
 

def get_view_matrix(campos_w:list, tarpos_w:list, upvector:list): 
    """視野変換行列を求める 

    Args: 
        campos_w (list): world座標系上のカメラの位置(x,y,z) [m] 
        tarpos_w (list): world座標系上のカメラの視点の先(x,y,z) [m] 
        upvector (list): カメラの上方向ベクトル 
    """ 

    campos_w = np.array(campos_w) 
    tarpos_w = np.array(tarpos_w) 
    upvector = np.array(upvector) 

    # z 軸 = (l - c) / ||l - c|| 
    # z_cam = tarpos_w - campos_w 
    z_cam = -(tarpos_w - campos_w) 
    z_cam = z_cam / np.linalg.norm(z_cam) 


    # x 軸 = (z_cam × u) / ||z_cam × u|| 
    x_cam = np.cross(z_cam, upvector) 

    # upvectorとz_camがほぼ同じ向きの場合、x_camの外積の結果が0ベクトルになってしまう 
    if np.count_nonzero(x_cam) == 0: 
        upvector = np.array([1, 0, 0]) 
        x_cam = np.cross(z_cam, upvector) 

    x_cam = x_cam / np.linalg.norm(x_cam) 

    # y 軸 = (z_cam × x_cam) / ||z_cam × x_cam|| 
    y_cam = np.cross(z_cam, x_cam) 
    # y_cam = np.cross(x_cam, z_cam) 
    y_cam = y_cam / np.linalg.norm(y_cam) 


    tx = -np.dot(campos_w , x_cam) 
    ty = -np.dot(campos_w , y_cam) 
    tz = -np.dot(campos_w , z_cam) 
    
    cHw = np.array([[x_cam[0], x_cam[1], x_cam[2], tx], 
                   [ y_cam[0], y_cam[1], y_cam[2], ty], 
                   [ z_cam[0], z_cam[1], z_cam[2], tz], 
                   [        0,        0,        0,  1]]) 

    return cHw 



def fix_camera_pose_generate(look_at_pos, azimuth, elevation, axial_angle, distance):

    # 視点先位置
    look_at_pos = look_at_pos

    direction = get_direction(azimuth, elevation)

    # カメラ位置 = 視点先位置 + 大きさ * 方向ベクトル
    cameraPosition = look_at_pos + distance * direction

    H_w2cam = get_view_matrix(
        upvector=np.array([0, 0, 1]),
        campos_w=cameraPosition,
        tarpos_w=look_at_pos)

    # 視点軸周りに回転させる
    # 回転ベクトル
    rotation_vector = axial_angle * copy.deepcopy(direction)

    # ロドリゲスの回転公式で回転ベクトルを回転行列に変換
    R,_       = cv2.Rodrigues(rotation_vector)
    HR        = np.zeros((4,4))
    HR[:3,:3] = R
    HR[3,3]   = 1
    H_w2cam   = H_w2cam @ HR
    H_cam2w   = np.linalg.inv(H_w2cam)

    return H_cam2w


def generate_camera_poses(viewpoints,
                          look_at_pos,
                          axial_angles,
                          initial_distance_mm,
                          end_distance_mm,
                          distance_division_number,
                          azimuth_range=(0, 2 * np.pi),
                          elev_range=(-0.5 * np.pi, 0.5 * np.pi)):
    H_cam2w_lefthands = []

    distances  = np.linspace(initial_distance_mm,
                             end_distance_mm,
                             distance_division_number)

    for distance in distances:
        for viewpoint in viewpoints:

            # xy平面の角度
            azimuth = np.arctan2(viewpoint[1], viewpoint[0])
            if azimuth < 0:
                azimuth += 2.0 * np.pi

            # xy平面とz軸がなす角度
            a = np.linalg.norm(viewpoint)
            b = np.linalg.norm([viewpoint[0], viewpoint[1], 0])
            elevation = np.arccos(b / a)
            if viewpoint[2] < 0:
                elevation = -elevation

            if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
                    elev_range[0] <= elevation <= elev_range[1]):
                continue

            for axial_angle in axial_angles:
                H_cam2w_lefthand = fix_camera_pose_generate(look_at_pos, azimuth, elevation, axial_angle, distance)
                H_cam2w_lefthands.append(H_cam2w_lefthand)

    return H_cam2w_lefthands

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    viewpoints_file = args.viewpoints_file
    target_obj_file = args.ply

    target_obj_filename=os.path.basename(target_obj_file)
    file_name_without_extension = os.path.splitext(target_obj_filename)[0]


    output_dir = args.output_dir

    # output ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    initial_distance_mm = args.initial_distance_mm
    end_distance_mm = args.end_distance_mm
    distance_division_number = args.distance_division_number
    azimuth_range = args.azimuth_range
    elev_range = args.elev_range
    axial_angle_division_number = args.axial_angle_division_number

    print("axial_angle_division_number",axial_angle_division_number)
    axial_angles = np.linspace(0, 2.0*np.pi, axial_angle_division_number)
    # Load the viewpoints
    viewpoints = load_viewpoints(viewpoints_file)
    print("length of viewpoints:",len(viewpoints))

    if(args.debug):
        # Convert viewpoints to Open3D point cloud
        viewpoints_pc = o3d.geometry.PointCloud()
        viewpoints_pc.points = o3d.utility.Vector3dVector(viewpoints)
        # Visualize the viewpoints
        o3d.visualization.draw_geometries([viewpoints_pc])

    # Load the PLY file
    mesh = o3d.io.read_triangle_mesh(target_obj_file)

    bbox_diagonal_length = np.linalg.norm(np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound()))

    world_coordinate = CoordinateFrameMesh([0, 0, 0], rpy=[0, 0, 0], scale=bbox_diagonal_length).get_poligon()

    # Move the viewpoints to the center of the mesh
    look_at_pos = mesh.get_center()



    # Generate camera poses
    camera_poses = generate_camera_poses(viewpoints,
                                         look_at_pos,
                                         axial_angles,
                                         initial_distance_mm,
                                         end_distance_mm,
                                         distance_division_number,
                                         azimuth_range,
                                         elev_range)


    # Save the camera poses
    np.save(os.path.join(output_dir,f"{file_name_without_extension}.npy"), camera_poses)

    print("*"*99)
    print("name:",file_name_without_extension)
    print("the number of viewpoints:",len(viewpoints))
    print("the number of camerapose:",len(camera_poses))
    print("*"*99)

    if(args.debug):

        # Visualize the camera coordinates
        camera_coordinates = []
        for camera_pose in camera_poses:
            camera_coordinate = CoordinateFrameMesh([0, 0, 0], rpy=[0, 0, 0], scale=bbox_diagonal_length/2).get_poligon()
            camTworld = np.linalg.inv(camera_pose)
            camera_coordinate.transform(camera_pose)
            camera_coordinates.append(camera_coordinate)

        # o3d.visualization.draw_geometries([mesh, *camera_coordinates])
        o3d.visualization.draw_geometries([mesh, *camera_coordinates, world_coordinate])
