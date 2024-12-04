import numpy as np
import open3d as o3d
import argparse
from PIL import Image
import os
import json

"""STLもしくはobjファイルをplyデータに変換
python symmetry\obj2ply.py --mesh mm.STL  --output_dir symmetry\tmp_obj_reduce
"""

def loadOBJ(filePath):
    """objファイルを読み込み、open3dのメッシュデータに必要な情報を返す

    Args:
        filePath (_type_): objファイルのパス

    Returns:
        頂点座標データ: vertices (numVertices x 3)
        テクスチャ座標データ: uv (numUVs x 3)
        法線データ: normals (numNormals x 3)
        面を構成する頂点のID: faceVertIDs (numFaces x 3 or 4)
        面頂点と対応したテクスチャ座標ID: uvIDs (numFaces x 3 or 4)
        面を構成する法線のID: normalIDs (numFaces x 3 or 4)
        頂点カラーデータ: veruv_mapColors (numVertices x 3)
        mtlファイルの名前: mtlFileName
    """
    
    numVertices = 0
    numUVs = 0
    numNormals = 0
    numFaces = 0
    vertices = []
    uv_maps = []
    normals = []
    veruv_mapColors = []
    faceVertIDs = []
    uvIDs = []
    normalIDs = []
    mtlFileName = None

    with open(filePath, "r") as fp:
        for line in fp:
            vals = line.split()
            if len(vals) == 0:
                continue
            if vals[0] == "mtllib":
                mtlFileName = vals[1]
            elif vals[0] == "v":
                v = [float(n) for n in vals[1:4]]
                vertices.append(v)
                if len(vals) == 7:
                    vc = [float(n) for n in vals[4:7]]
                    veruv_mapColors.append(vc)
                numVertices += 1
            elif vals[0] == "vt":
                vt = [float(n) for n in vals[1:3]]
                uv_maps.append(vt)
                numUVs += 1
            elif vals[0] == "vn":
                vn = [float(n) for n in vals[1:4]]
                normals.append(vn)
                numNormals += 1
            elif vals[0] == "f":
                fvID = []
                uvID = []
                nvID = []
                for f in vals[1:]:
                    w = f.split("/")
                    if numVertices > 0:
                        fvID.append(int(w[0])-1)
                    if numUVs > 0 and len(w) > 1 and w[1]:
                        uvID.append(int(w[1])-1)
                    if numNormals > 0 and len(w) > 2 and w[2]:
                        nvID.append(int(w[2])-1)
                if len(fvID) > 0:
                    faceVertIDs.append(fvID)
                if len(uvID) > 0:
                    uvIDs.append(uvID)
                if len(nvID) > 0:
                    normalIDs.append(nvID)
                numFaces += 1

    if len(uvIDs) != 0:
        uvs = []
        for uv_id in uvIDs:
            uvs.append([uv_maps[uv_id[0]][0], 1 - uv_maps[uv_id[0]][1]])
            uvs.append([uv_maps[uv_id[1]][0], 1 - uv_maps[uv_id[1]][1]])
            uvs.append([uv_maps[uv_id[2]][0], 1 - uv_maps[uv_id[2]][1]])
    else:
        uvs = None

    if len(normals) != 0:
        normals_ = []
        for normal_id in normalIDs:
            normals_.append(normals[normal_id[0]])
            normals_.append(normals[normal_id[1]])
            normals_.append(normals[normal_id[2]])
        normals = normals_

    print("numVertices: ", numVertices)
    print("numUVs: ", numUVs)
    print("numNormals: ", numNormals)
    print("numFaces: ", numFaces)

    return vertices, uvs, normals, faceVertIDs, normalIDs, veruv_mapColors, mtlFileName



def read_material_file(mtl_file_path):
    """マテリアルファイル(.mtl)からマテリアル情報を取得する"""
    specular = None
    diffuse = None
    ambient = None
    glossiness = 0
    texture_map = None

    # OBJファイル読込
    with open(mtl_file_path, 'r') as fp:
        for line in fp:
            ary = line.strip().split(' ')
            if ary[0] == 'Ka':
                # 環境光反射成分
                ambient = [float(ary[1]), float(ary[2]), float(ary[3])]
            elif ary[0] == 'Kd':
                # 拡散反射成分
                diffuse = [float(ary[1]), float(ary[2]), float(ary[3])]
            elif ary[0] == 'Ks':
                # 鏡面反射成分
                specular = [float(ary[1]), float(ary[2]), float(ary[3])]
                if np.all(np.array(specular) == 0):
                    specular = None
            elif ary[0] == 'Ns':
                # 鏡面反射角度
                glossiness = float(ary[1])
            elif ary[0] == 'map_Kd':
                # 画像データ名
                texture_map = ary[1]

    return specular, diffuse, ambient, glossiness, texture_map
    
def load_open3d_mesh_data_from_obj(path_obj:str=None, path_uv_map:str=None) -> o3d.geometry.TriangleMesh:
    """objファイルを読み込んで、open3dのメッシュデータにする

    Args:
        path_obj (str, optional): objファイルのパス. Defaults to None.
        path_uv_map (str, optional): uvマップ. Defaults to None.

    Returns:
        o3d.geometry.TriangleMesh: メッシュ
    """

    vertices, uvs, normals, faceVertIDs, normalIDs, veruv_mapColors, mtlFileName = loadOBJ(path_obj)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                   o3d.utility.Vector3iVector(faceVertIDs))
    

    vertices, uvs, normals, faceVertIDs, normalIDs, veruv_mapColors, mtlFileName = loadOBJ(path_obj)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(faceVertIDs))

    texture_map = None

    if mtlFileName is not None:
        mtl_file_path = os.path.join(os.path.dirname(path_obj), mtlFileName)
        _, _, _, _, texture_map = read_material_file(mtl_file_path)

    if texture_map is not None:
        texture_path = os.path.join(os.path.dirname(path_obj), texture_map)
        mesh.textures = [o3d.io.read_image(texture_path)]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(mesh.triangles)), dtype=np.int32))
    elif path_uv_map is not None:
        mesh.textures = [o3d.io.read_image(path_uv_map)]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(mesh.triangles)), dtype=np.int32))

    if len(normals) == 0:
        #mesh.compute_vertex_normals()
        pass
    else:
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    #mesh.compute_vertex_normals()
    return mesh
    

def open3d_image_to_pil_image(o3d_image):
    """Convert Open3D image to PIL image"""
    o3d_image_np = np.asarray(o3d_image)
    pil_image = Image.fromarray(o3d_image_np)
    return pil_image

def move2origin_position(mesh:o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """重心を原点に移動させる

    Args:
        mesh (o3d.geometry.TriangleMesh): メッシュ

    Returns:
        o3d.geometry.TriangleMesh: メッシュ
    """

    mesh.translate(-mesh.get_center())
    return mesh

def move2origin_orientation(mesh:o3d.geometry.TriangleMesh):
    
    points = np.array(mesh.vertices)
    # 点群を原点に移動させ、回転もワールド座標に一致させる ####
    H      = calculate_homography_matrix_by_PCA(points)
    inv_H  = np.linalg.inv(H)
    mesh.transform(inv_H)
    return mesh

def calculate_homography_matrix_by_PCA(xyz):
 
    if not isinstance(xyz, np.ndarray):
        xyz = np.array(xyz)
 
    center = xyz.mean(axis=0)
    tx,ty,tz = center
 
    normalized_pts = xyz - center
 
    pp = normalized_pts.T.dot(normalized_pts)
 
    u, w, vt = np.linalg.svd(pp)
 
    norm_vec = u[:,2]
 
    v1 = u[:,0]
    v2 = u[:,1]
    v3 = u[:,2]
    
    H = np.array([[v1[0], v2[0], v3[0], tx],
                  [v1[1], v2[1], v3[1], ty],
                  [v1[2], v2[2], v3[2], tz],
                  [    0,     0,     0,  1]])
 
    return H

def calculate_obb(mesh):
    
    new_points = np.array(mesh.vertices).T
 
    x = new_points[0, :]
    y = new_points[1, :]
    z = new_points[2, :]
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
 
    p1 = [xmin, ymin, zmin]; p2 = [xmax, ymin, zmin]
    p3 = [xmax, ymax, zmin]; p4 = [xmin, ymax, zmin]
    p5 = [xmin, ymin, zmax]; p6 = [xmax, ymin, zmax]
    p7 = [xmax, ymax, zmax]; p8 = [xmin, ymax, zmax]
    
    bbox = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
    return bbox

def change_scale(mesh:o3d.geometry.TriangleMesh, scale:float) -> o3d.geometry.TriangleMesh:
    """スケールを変更する

    Args:
        mesh (o3d.geometry.TriangleMesh): メッシュ
        scale (float): スケール

    Returns:
        o3d.geometry.TriangleMesh: スケール
    """

    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices)*scale)
    return mesh

def extract_vertex_color_from_uvmap(mesh:o3d.geometry.TriangleMesh, image:np.ndarray) -> o3d.geometry.TriangleMesh:
    """画像から色を抜き出し、頂点色にする

    Args:
        mesh (o3d.geometry.TriangleMesh): メッシュ
        image (np.ndarry): 画像

    Returns:
        o3d.geometry.TriangleMesh: 頂点色情報を与えたメッシュ
    """

    h,w = image.shape[:2]
    triangles = np.array(mesh.triangles)
    triangle_uvs = np.array(mesh.triangle_uvs)

    vertex_colors = np.zeros((len(mesh.vertices),3))

    for i, indice in enumerate(triangles):
        for j, index in enumerate(indice): 
            u = int(triangle_uvs[3*i+j][0]*w)
            v = int(triangle_uvs[3*i+j][1]*h)
            vertex_colors[index] = image[v,u][:3] / 255
        
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
       
    return mesh    

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

class Cylinder():
    def __init__(self, start_pos, end_pos, radius=0.01, color=[0.0, 0.0, 1.0]):
 
        
        self._pos    = start_pos
        delta_vector = np.array(end_pos)-np.array(start_pos)
        length       = np.linalg.norm(delta_vector)
        self._obj = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                              height=length)
 
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
    
def get_3D_rectangle_polygon(p1, p2, p3, p4, p5, p6, p7, p8):
    # 直方体の反対側の対角点からデータの 2 セットの座標を変換します
 
    rectangle = []
    rectangle.append(Cylinder(p1, p2).get_poligon()) # | (up)
    rectangle.append(Cylinder(p2, p3).get_poligon())# -->
    rectangle.append(Cylinder(p3, p4).get_poligon())# | (down)
    rectangle.append(Cylinder(p4, p1).get_poligon())# <--
 
    rectangle.append(Cylinder(p5, p6).get_poligon())# | (up)
    rectangle.append(Cylinder(p6, p7).get_poligon())# -->
    rectangle.append(Cylinder(p7, p8).get_poligon())# | (down)
    rectangle.append(Cylinder(p8, p5).get_poligon())# <--
 
    rectangle.append(Cylinder(p1, p5).get_poligon())# | (up)
    rectangle.append(Cylinder(p3, p7).get_poligon())# | (up)
    rectangle.append(Cylinder(p4, p8).get_poligon())# | (up)
    rectangle.append(Cylinder(p2, p6).get_poligon())# | (up)
 
    return rectangle
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--mesh", help="path of cad model ")
    parser.add_argument("--tex", default=None, help="path of image ")
    parser.add_argument("--name", default=None, help="name of output ply file")
    parser.add_argument("--output_dir", default="tmp_obj_reduce", help="name of output ply file")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    args = parser.parse_args()

    input_file_path = args.mesh
    path_tex = args.tex

    # CADファイル読み込み
    if input_file_path.lower().endswith((".stl", ".STL", ".ply")):
        mesh = o3d.io.read_triangle_mesh(input_file_path)
    elif input_file_path.lower().endswith(".obj"):
        mesh = load_open3d_mesh_data_from_obj(input_file_path, path_tex)
    else:
        print("Unsupported file format. Please provide an STL or OBJ file.")

    output_dir = args.output_dir

    if(args.name is None):
        # ファイル名を取得して拡張子を除外
        save_name = os.path.splitext(os.path.basename(input_file_path))[0]

    else:
        save_name = args.name

    # 重心を原点に移動
    mesh = move2origin_position(mesh)
    mesh = move2origin_orientation(mesh)

    # 単位を[m]→[mm]
    mesh = change_scale(mesh, scale=1000)
    # mesh = change_scale(mesh, scale=100)

    if(args.debug):
        o3d.visualization.draw_geometries([mesh])
        
    os.makedirs(f"{output_dir}/{save_name}", exist_ok=True)
    o3d.io.write_triangle_mesh(f"{output_dir}/{save_name}/{save_name}.obj", mesh, write_ascii=True)
  
    # バウンディングボックスの計算
    bbox = calculate_obb(mesh)
    
    # バウンディングボックス可視化
    if(args.debug):
        rectangle_obj = get_3D_rectangle_polygon(*bbox)
        o3d.visualization.draw_geometries([mesh, *rectangle_obj])

    bbox_dict = {}
    for i, point in enumerate(bbox):
        bbox_dict[f"p{i}"] = [p for p in point]

    # バウンディングボックス保存
    with open(f"{output_dir}/{save_name}/{save_name}_bbox.json", 'w') as f:
        json.dump(bbox_dict, f)
    
    # 画像から色を抜き出し、頂点カラーにする
    if(mesh.has_textures()):
        pil_image = open3d_image_to_pil_image(mesh.textures[0])
        image = np.array(pil_image)
        #mesh = extract_vertex_color_from_uvmap(mesh, image) # not save vertexcolor in ply file(the function has not problem. just don't use it.)

    save_path = f"{output_dir}/{save_name}/{save_name}.ply"
    o3d.io.write_triangle_mesh(save_path, mesh, write_ascii=True)
    print("saved", save_path)

    # 保存したplyファイルを表示して色がついているか確認する
    if(args.debug):
        mesh = o3d.io.read_triangle_mesh(f"{output_dir}/{save_name}/{save_name}.ply")
        o3d.visualization.draw_geometries([mesh])
