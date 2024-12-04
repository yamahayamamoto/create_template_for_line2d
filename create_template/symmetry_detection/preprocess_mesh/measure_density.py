import open3d as o3d
import argparse
import numpy as np
import scipy
import scipy.spatial


"""
python measure_density.py
"""

def loadOBJ(fliePath):
    """objファイルを読み込み、open3dのメッシュデータに必要な情報を返す

    Args:
        fliePath (_type_): objファイルのパス

    Returns:
        頂点座標データ: vertices (numVertices x 3)
        テクスチャ座標データ:uv (numUVs x 3)
        法線データ: normals (numNormals x 3)
        面を構成する頂点のID: faceVertIDs (numFaces x 3 or 4)
        面頂点と対応したテクスチャ座標ID: uvIDs (numFaces x 3 or 4)
        面を構成する法線のID: normalIDs (numFaces x 3 or 4)
        頂点カラーデータ: veruv_mapColors (numVertices x 3)
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
    for line in open(fliePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = [float(n) for n in vals[1:4]]
            vertices.append(v)
            if len(vals) == 7:
                vc = [float(n) for n in vals[4:7]]
                veruv_mapColors.append(vc)
            numVertices += 1
        if vals[0] == "vt":
            vt = [float(n) for n in vals[1:3]]
            uv_maps.append(vt)
            numUVs += 1
        if vals[0] == "vn":
            vn = [float(n) for n in vals[1:4]]
            normals.append(vn)
            numNormals += 1
        if vals[0] == "f":
            fvID = []
            uvID = []
            nvID = []
            for f in vals[1:]:
                w = f.split("/")
                if numVertices > 0:
                    fvID.append(int(w[0])-1)
                if numUVs > 0:
                    uvID.append(int(w[1])-1)
                if numNormals > 0:
                    nvID.append(int(w[2])-1)
            if(len(fvID)>0):
                faceVertIDs.append(fvID)
            if(len(uvID)>0):
                uvIDs.append(uvID)
            if(len(nvID)>0):
                normalIDs.append(nvID)
            numFaces += 1
            
    if(len(uvIDs)!=0):
        uvs=[]
                
        for uv_id in uvIDs:
            uvs.append([uv_maps[uv_id[0]][0],1-uv_maps[uv_id[0]][1]])
            uvs.append([uv_maps[uv_id[1]][0],1-uv_maps[uv_id[1]][1]])
            uvs.append([uv_maps[uv_id[2]][0],1-uv_maps[uv_id[2]][1]])
    else:
        uvs = None
        
    if(len(normals)!=0):
        normals_=[]
        for normal_id in normalIDs:
            # print(normal_id)
            normals_.append(normals[normal_id[0]])
            normals_.append(normals[normal_id[1]])
            normals_.append(normals[normal_id[2]])
        normals = normals_
        
        
    print("numVertices: ", numVertices)
    print("numUVs: ", numUVs)
    print("numNormals: ", numNormals)
    print("numFaces: ", numFaces)
    
    return vertices, uvs, normals, faceVertIDs, normalIDs, veruv_mapColors

def load_open3d_mesh_data_from_obj(path_obj:str=None, path_uv_map:str=None) -> o3d.geometry.TriangleMesh:
    """objファイルを読み込んで、open3dのメッシュデータにする

    Args:
        path_obj (str, optional): objファイルのパス. Defaults to None.
        path_uv_map (str, optional): uvマップ. Defaults to None.

    Returns:
        o3d.geometry.TriangleMesh: メッシュ
    """

    vertices, uvs, normals, faceVertIDs, normalIDs, veruv_mapColors = loadOBJ(path_obj)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                   o3d.utility.Vector3iVector(faceVertIDs))
    
    if(path_uv_map is not None):
        mesh.textures              = [o3d.io.read_image(path_uv_map)]
        mesh.triangle_uvs          = o3d.utility.Vector2dVector(uvs)
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(mesh.triangles),)).astype(int))

    if(len(normals) == 0):
        mesh.compute_vertex_normals()
    else:
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_vertex_normals()
    return mesh


def edges_to_lineset(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    ls.paint_uniform_color(color)
    return ls

def check_properties(mesh, debug=True):
    """
    Open3Dを使用して三角形メッシュのさまざまな特性をテストし、結果を視覚化します。

    Parameters:
        mesh (open3d.geometry.TriangleMesh): テストするメッシュ。
        debug (bool, optional): Trueの場合、結果を視覚化します。デフォルトはTrueです。

    メッシュの特性:
    三角形メッシュには、Open3Dでテストできるいくつかの特性があります。重要な特性の1つは、マニホールド特性です。
    ここでは、三角形メッシュがエッジマニホールドであるかどうかをテストできます（is_edge_manifold）、
    また頂点マニホールドであるかどうかをテストできます（is_vertex_manifold）。三角形メッシュがエッジマニホールドであるとは、
    各エッジが1つまたは2つの三角形を囲んでいることを意味します。関数is_edge_manifoldには、境界エッジを許可するかどうかを定義する
    boolパラメータallow_boundary_edgesがあります。さらに、三角形メッシュが頂点マニホールドであるとは、頂点のスターがエッジマニホールドであり、
    エッジで接続されていることを意味します。つまり、2つ以上の面が頂点によってのみ接続され、エッジで接続されていないことです。

    もう1つの特性は、自己交差のテストです。関数is_self_intersectingは、メッシュ内の三角形が他のメッシュと交差している場合にTrueを返します。
    完全な水密メッシュは、エッジマニホールドであり、頂点マニホールドであり、自己交差していないメッシュと定義できます。関数is_watertightは、
    Open3Dでこのチェックを実装しています。

    また、三角形メッシュが向き付け可能かどうか、つまりすべての法線が外向きを向いているように三角形を向き付けることができるかどうかもテストできます。
    Open3Dの対応する関数はis_orientableです。

    以下のコードは、いくつかの三角形メッシュをこれらの特性に対してテストし、その結果を視覚化します。
    非マニホールドのエッジは赤で表示され、境界エッジは緑で表示され、非マニホールドの頂点は緑の点で表示され、自己交差している三角形はピンクで表示されます。

    参考:
    https://www.open3d.org/docs/latest/tutorial/Basic/mesh.html
    """
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(edges_to_lineset(mesh, edges, (1, 0, 1)))

    if(debug):
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)


def calculate_mesh_density_in_volume(mesh):

    # 頂点数の取得
    num_vertices = len(mesh.vertices)

    # 体積の計算
    volume = mesh.get_volume()

    # 密度の計算
    density = num_vertices / volume

    return density

def calculate_convexhullmesh_density_in_volume(mesh):
    """_summary_

    Args:
        mesh (_type_): _description_

        
    Returns:
        _type_: _description_
    # """
    # # 頂点数の取得
    # num_vertices = len(mesh.vertices)
    # # 体積の計算
    # convex = mesh.compute_convex_hull()[0]
    # o3d.visualization.draw_geometries([convex], mesh_show_back_face=True)
    # volume = convex.get_volume()
    # print(volume)

    # # 密度の計算
    # density = num_vertices / volume

    # o3d.visualization.draw_geometries([convex], mesh_show_back_face=True)
    # return density
    vertices = np.array(mesh.vertices)
    hull = scipy.spatial.ConvexHull(vertices)
    return vertices.shape[0] / hull.volume


def calculate_mesh_density_in_surface_area(mesh):

    # 頂点数の取得
    num_vertices = len(mesh.vertices)

    # 体積の計算
    surface_area = mesh.get_surface_area()

    # 密度の計算
    density = num_vertices / surface_area

    return density

def calculate_convexhullmesh_density_in_surface_area(mesh):
    """_summary_

    Args:
        mesh (_type_): _description_

    Returns:
        _type_: _description_
    https://gist.github.com/0xLeon/a3975fd9b011a9495470445b94670d28
    # """
    # 頂点数の取得
    num_vertices = len(mesh.vertices)
    # 体積の計算
    convex = mesh.compute_convex_hull()[0]
    o3d.visualization.draw_geometries([convex], mesh_show_back_face=True)
    surface_area = convex.get_surface_area()
    print(surface_area)

    # 密度の計算
    density = num_vertices / surface_area

    o3d.visualization.draw_geometries([convex], mesh_show_back_face=True)
    return density



if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Calculate mesh density")
    parser.add_argument("--mesh", help="Path of CAD model")
    parser.add_argument('--use_volume', action='store_true', help='Enable debug mode.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    args = parser.parse_args()

    # メッシュのファイルパス
    input_file_path = args.mesh



    # CADファイル読み込み
    if input_file_path.lower().endswith((".stl", ".STL", "ply")):
        mesh = o3d.io.read_triangle_mesh(input_file_path)
    elif input_file_path.lower().endswith(".obj"):
        mesh = load_open3d_mesh_data_from_obj(input_file_path)
    else:
        print("Unsupported file format. Please provide an STL or OBJ file.")


    if(args.debug):
        check_properties(mesh)

    if(args.use_volume):
        # メッシュの密度の計算
        if(mesh.is_watertight()):
            mesh_density = calculate_mesh_density_in_volume(mesh)
        else:
            mesh_density = calculate_convexhullmesh_density_in_volume(mesh)
        
    else:
        # メッシュの密度の計算(表面積で計算)
        mesh_density = calculate_mesh_density_in_surface_area(mesh)

    print("Mesh Density:", mesh_density)
