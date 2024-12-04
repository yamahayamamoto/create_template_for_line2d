import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List


def text_3d(text, pos, rpy=None, matrix=None, font="arial.ttf", font_size=16, density=100):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """

    if(matrix is None):
        if(rpy is not None):
            # 姿勢(roll,pitch,yaw)から同次座標変換行列を計算する
            matrix = rpy_pos2matrix(pos, rpy)
        else:
            # 回転させない
            matrix = rpy_pos2matrix(pos, [0, 0, 0])
    else:
        T  = rpy_pos2matrix(pos, [0, 0, 0])
        matrix = T @ matrix


    # from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, int(font_size * density))
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(
        img[img_mask, :].astype(float) / 255.0)

    pcd.points = o3d.utility.Vector3dVector(indices / 100 / density)

    pcd.transform(matrix)
    return pcd

def rpy_pos2matrix(pos, rpy):
    """roll-pitch-yawから同次座標変換行列を計算する"""
    px, py, pz = pos
    tx, ty, tz = rpy

    c = np.cos
    s = np.sin

    # 並進部分 
    T = np.array([[1, 0, 0, px],
                  [0, 1, 0, py],
                  [0, 0, 1, pz],
                  [0, 0, 0, 1]])

    # 回転部分
    R = np.array([[c(tz)*c(ty), c(tz)*s(ty)*s(tx)-c(tx)*s(tz), s(tz)*s(tx)+c(tz)*c(tx)*s(ty), 0],
                  [c(ty)*s(tz), c(tz)*c(tx)+s(tz)*s(ty)*s(tx), c(tx)*s(tz)*s(ty)-c(tz)*s(tx), 0],
                  [-s(ty), c(ty)*s(tx), c(ty)*c(tx), 0],
                  [0, 0, 0, 1]])

    return T.dot(R)

class CoordinateFrameMesh():
    def __init__(self, pos=[0,0,0], scale=1.0, rpy=None, qwxyz=None):
        self._pos = pos
        self._rpy = rpy
        self._qwxyz = qwxyz
        self._obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)

        if self._rpy is not None:
            # 姿勢(roll,pitch,yaw)から同次座標変換行列を計算する
            self._matrix = rpy_pos2matrix(self._pos, self._rpy)
        elif self._qwxyz is not None:
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

class Cylinder():
    def __init__(self, start_pos, end_pos, radius=0.1, color=[0.0, 0.0, 1.0]):
        self._pos = start_pos
        delta_vector = np.array(end_pos)-np.array(start_pos)
        length = np.linalg.norm(delta_vector)
        self._obj = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        self._obj.paint_uniform_color(color)

        # 矢印がまずx軸を向くように変換する
        self._obj.transform(rpy_pos2matrix([length/2.0,0,0], [0,np.deg2rad(90),0]))

        # 矢印の姿勢を求める
        self._rpy = [0,
                     -np.arctan2(delta_vector[2], np.sqrt(delta_vector[0]**2+delta_vector[1]**2)),
                     np.arctan2(delta_vector[1], delta_vector[0])]

        # 同次座標変換行列計算
        self._matrix = rpy_pos2matrix(self._pos, self._rpy)

        # 座標変換
        self._obj.transform(self._matrix)

        # 法線ベクトルの計算
        self._obj.compute_vertex_normals()

    def get_poligon(self):
        return self._obj

class Circle:
    def __init__(self, pos=[0.0,0.0,0.0], rpy=[0.0,0.0,0.0], r=2.0, resolution=20):
        
        self.pos = np.array(pos)
        self.rpy = np.array(rpy)
        self.r = r
        self.H = rpy_pos2matrix(self.pos, self.rpy)
        
        # 角数
        self.resolution = resolution
        
        # 頂点データ
        self.vert = []
        
        # 面
        self.faces = []
 
    def create_vert(self):
    
        # 角度の分解能
        fAngleDelta = 2.0 * np.pi / self.resolution
        
        # 角度
        fTheta = 0.0
        
        self.vert.append([ 0.0,  0.0, 0.0]) # 底面の頂点
        self.top_center = len(self.vert)-1
        
        for i in range(self.resolution+1):
            p1 = [self.r * np.cos( fTheta ),  self.r * np.sin( fTheta), 0]
            fTheta += fAngleDelta
            self.vert.append(p1)
        
    def create_face(self):
        
        # インデックスデータ作成
        for i in range(1,self.resolution+1):
            self.faces.append([self.top_center,
            i+self.top_center,
            i+self.top_center + 1])
    
    def create_polygon(self):
        self.create_vert()
        self.create_face()
        self.object=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vert),
        o3d.utility.Vector3iVector(self.faces))
        
        self.object.compute_vertex_normals()
        
        self.object.transform(self.H)
    
    def change_color(self, color):
        self.object.paint_uniform_color(color)
    
    def get_poligon(self):
        """_メッシュデータの作成
        Returns:
        _type_: _メッシュデータ
        """
        return self.object

class Arrow():
    def __init__(self, start_pos, end_pos, color=[0.7,0.6,0.2]):
 
        
        self._pos    = start_pos
        delta_vector = np.array(end_pos)-np.array(start_pos)
        self._end_pos    = start_pos + delta_vector
        length       = np.linalg.norm(delta_vector)
        scale        = 0.03*length
        
        self._obj = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.8*scale,
                                                           cone_radius=1.5*scale,
                                                           cylinder_height=length*0.8,
                                                           cone_height=length*0.2,
                                                           resolution=4,
                                                           cylinder_split=4,
                                                           cone_split=1)
 
        self._obj.paint_uniform_color(color)
 
        # 矢印がまずx軸を向くように変換する
        self._obj.transform(rpy_pos2matrix([0,0,0], [0,np.deg2rad(90),0]))
 
        # 矢印の姿勢を求める
        self._rpy = [0,
                     -np.arctan2(delta_vector[2], np.sqrt(delta_vector[0]**2+delta_vector[1]**2)),
                     np.arctan2(delta_vector[1], delta_vector[0])]
 
        # 同次座標変換行列計算
        self._matrix = rpy_pos2matrix(self._pos, self._rpy)
 
        # 座標変換
        self._obj.transform(self._matrix)
 
        # 法線ベクトルの計算
        self._obj.compute_vertex_normals()
 
    def get_poligon(self):
        return self._obj
    @property
    def end_pos(self):
        return self._end_pos
 

def create_rotation_gif(from_points: np.ndarray, obj_meshs: o3d.geometry.TriangleMesh, output_file: str, duration: int = 125, loop: int = 0):
    """Create GIF animation by rotating camera position.

    Args:
        vis (o3d.visualization.Visualizer): Open3D visualizer object.
        from_points (np.ndarray): Array of camera locations in 3D space.
        obj_mesh (o3d.geometry.TriangleMesh): Object mesh to rotate.
        output_file (str): Output GIF file path.
        duration (int, optional): Duration of each frame in milliseconds. Defaults to 125.
        loop (int, optional): Number of loops. Defaults to 0 (infinite loop).

    Returns:
        None
    """
    # 使用例
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)
    def view_from(from_point: np.ndarray, to_point: np.ndarray = np.array([0, 0, 0])):
        """Helper to setup view direction for Open3D Visualiser."""
        ctr    = vis.get_view_control()
        dir_v  = to_point - from_point
        dir_v /= np.linalg.norm(dir_v)  # 方向ベクトルを正規化
        left_v = np.cross([0, 0, 1], dir_v)
        left_v /= np.linalg.norm(left_v)  # 左方向ベクトルを正規化
        up     = np.cross(dir_v, left_v)
        ctr.set_lookat(to_point)
        ctr.set_front(-dir_v)
        ctr.set_up(up)

    # カメラの設定
    view_from(from_points[0])

    for obj_mesh in obj_meshs:
        vis.add_geometry(obj_mesh)
    images = []
    for from_point in from_points:
        # カメラの位置を設定
        view_from(from_point)

        vis.poll_events()
        vis.update_renderer()
        
        # get the image
        o3d_screenshot_mat = vis.capture_screen_float_buffer()
        # scale and convert to uint8 type
        image = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
        image = Image.fromarray(image , "RGB")
        images.append(image)

    # GIF を保存
    images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=loop)

    
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
        rotation_matrix = rotation_matrix_to_align_with_dst_axis(plane_normal)

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
        # rotated_arrow = Arrow([0, 0, 0], [0, 0, 1])
        rotated_arrow = Arrow([0, 0, 0], [0,0,0.5*bbox_diagonal_length])
        rotated_arrow_obj = rotated_arrow.get_poligon()
        rotated_arrow_obj.transform(H)
        arrow_objects.append(rotated_arrow_obj)

    # Open3Dを使用して平面と点群を可視化
    o3d.visualization.draw_geometries([pcd, *plane_objs], mesh_show_wireframe=True, mesh_show_back_face=True)
    o3d.visualization.draw_geometries([pcd, *plane_objs, *arrow_objects], mesh_show_wireframe=True, mesh_show_back_face=True)
    o3d.visualization.draw_geometries(arrow_objects, mesh_show_wireframe=True, mesh_show_back_face=True)


    
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