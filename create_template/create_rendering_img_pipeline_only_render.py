import argparse
import os
import subprocess
import shutil
from tqdm import tqdm
from pathlib import Path


#python create_rendering_img_pipeline.py "../../dataset/input" "../../dataset/output2"


def main():
    parser = argparse.ArgumentParser(description="Process mesh data and perform various tasks.")
    parser.add_argument("input_dir", type=str, help="Path to the mesh file (STL, OBJ, or folder)")
    parser.add_argument("output_dir", type=str, default="../../dataset/reduce_polygon", help="Path to the output mesh file")
    parser.add_argument("--use_grasp_pose", action='store_true', help="Whether to use grasp pose or not")
    args = parser.parse_args()

    if not args.input_dir:
        print("Error: Please provide the path to the mesh file or folder using the --mesh option.")
        return

    # 指定された拡張子のリスト
    valid_extensions = ['.STL', '.stl', '.obj', '.ply']

    # フォルダであるかを判定し、フォルダ内のファイルのリストを再帰的に取得する
    if os.path.isdir(args.input_dir):
        mesh_files = []
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                file_path = os.path.join(root, f)
                _, ext = os.path.splitext(file_path)
                ext_lower = ext.lower()
                if ext_lower in valid_extensions:
                    mesh_files.append(file_path)
    else:
        _, ext = os.path.splitext(args.input_dir)
        ext_lower = ext.lower()
        if ext_lower in valid_extensions:
            mesh_files = [args.input_dir]
        else:
            mesh_files = []

    num_files = len(mesh_files)  # ファイルの数を取得

    print(f"Number of files: {num_files}")

    for mesh_file in tqdm(mesh_files):
        # ファイルごとに処理を行う
        process_mesh(mesh_file=mesh_file, 
                     input_dir=args.input_dir, 
                     output_dir=args.output_dir, 
                     use_grasp_pose=args.use_grasp_pose)


def get_absolute_path(output_dir):
    # 現在の作業ディレクトリを取得
    current_dir = os.getcwd()

    # 相対パスを絶対パスに変換
    absolute_path = os.path.abspath(os.path.join(current_dir, output_dir))

    return absolute_path


def process_mesh(*, mesh_file, input_dir, output_dir, use_grasp_pose=False, is_scan=True):

    mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]

    aligned_mesh_dir = os.path.join(output_dir,"aligned_obj")
    reduction_mesh_dir = os.path.join(output_dir,"reduction_obj")
    mesh_ply_path = os.path.join(reduction_mesh_dir, f"{mesh_name}", f"{mesh_name}.ply")

    # フォルダが存在しない場合は作成
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    output_dir = get_absolute_path(output_dir)

    _, ext = os.path.splitext(mesh_file)
    if(ext == ".obj"):
        is_scan = True
        print("mesh is created by scan")
    else:
        is_scan = False
        print("mesh is created by 3d design tool")

    if(is_scan):
        points_ply_dir = os.path.join(output_dir, "observability")
        points_ply_path = os.path.join(points_ply_dir, f"{mesh_name}_observable.ply")
    else:
        points_ply_dir = os.path.join(output_dir, "points")
        points_ply_path = os.path.join(points_ply_dir, f"{mesh_name}_points.ply")


    result_dir                                 = os.path.join(output_dir, "viewpoint_restricted_result")

    graspdataset_path                          = os.path.join(output_dir, "grasp_database")

    viewpoints_dir                                = os.path.join(result_dir, "viewpoints")
    restricted_viewpoint_dir_by_rotation_symmetry = os.path.join(viewpoints_dir, "restricted_rotation_symmetry")
    restricted_viewpoint_dir_by_grasp_pose   = os.path.join(viewpoints_dir, "restricted_grasp_pose")
    raw_viewpoint_path                            = os.path.join(viewpoints_dir, "origin_viewpoints_324.npy")
    restricted_viewpoint_dir_by_perspected_area    = os.path.join(viewpoints_dir, "restricted_perspected_area")

    rendered_image_dir                         = os.path.join(output_dir, "rendered_image")


    path_camera_coordinates_dir = os.path.join(rendered_image_dir, "camera_coordinates")
    rendering_mesh_path = os.path.join(output_dir, "dataset", "mydata", "models", mesh_name, f"{mesh_name}.ply")



    # obj2ply.pyを実行
    # # STLもしくはobjファイルをplyデータに変換
    # obj2ply_command = ["python", "symmetry_detection/preprocess_mesh/obj2ply.py",
    #                    "--mesh", mesh_file, "--output_dir", aligned_mesh_dir]
    # print(obj2ply_command)
    # subprocess.run(obj2ply_command)

    # # polygon_reduction.pyを実行
    # # ポリゴンリダクション
    
    # polygon_reduction_command = ["python", "symmetry_detection/preprocess_mesh/polygon_reduction.py","--input_dir", aligned_mesh_dir, "--output_dir", reduction_mesh_dir]
    # print(polygon_reduction_command)
    # subprocess.run(polygon_reduction_command)
        
    #if(is_scan):
    #    polygon_reduction_command = ["python", "symmetry_detection/preprocess_mesh/polygon_reduction.py","--input_dir", aligned_mesh_dir, "--output_dir", reduction_mesh_dir]
    #    print(polygon_reduction_command)
    #    subprocess.run(polygon_reduction_command)
    #else:

     #   # コピー先のディレクトリが存在しない場合は作成する
    #    #os.makedirs(reduction_mesh_dir, exist_ok=True)
    #    os.makedirs(os.path.join(reduction_mesh_dir, f"{mesh_name}"), exist_ok=True)

    #    # mesh_ply_path の ply ファイルを rendering_mesh_path にコピーする
    #    mesh_ply_path = os.path.join(aligned_mesh_dir, f"{mesh_name}", f"{mesh_name}.ply")
    #    reduced_mesh_ply_path = os.path.join(reduction_mesh_dir, f"{mesh_name}", f"{mesh_name}.ply")
    #    shutil.copy(mesh_ply_path, reduced_mesh_ply_path)

    # if(is_scan):
    #     # observability_map.pyを実行
    #     # 球面上での視点から物体の観測性を評価し、自動分離（閾値はヒストグラムの最初の谷から判定する）
    #     observability_map_command = ["python", "symmetry_detection/preprocess_mesh/observability_map.py",
    #                                  "--ply", mesh_ply_path,
    #                                  "--output_dir", points_ply_dir]
    #     print(observability_map_command)
    #     subprocess.run(observability_map_command)
    # else:
    #     observability_map_command = ["python", "symmetry_detection/preprocess_mesh/create_pointcloud_from_mesh.py",
    #                                 "--ply", mesh_ply_path,
    #                                 "--output_dir", points_ply_dir]
    #     print(observability_map_command)
    #     subprocess.run(observability_map_command)




    # # preliminary_symmetry_detection.pyを実行
    # # 対称面の候補を計算する
    # preliminary_symmetry_detection_command = ["python", "symmetry_detection/preliminary_symmetry_detection.py",
    #                                         "--mesh", mesh_ply_path,
    #                                         "--points", points_ply_path,
    #                                         "--output_dir", result_dir]
    # print(preliminary_symmetry_detection_command)
    # subprocess.run(preliminary_symmetry_detection_command)

    # # rotational_symmetry_detection.pyを実行
    # # 回転対称軸とその回転角度を求める
    # rotational_symmetry_detection_command = ["python", "symmetry_detection/rotational_symmetry_detection.py",
    #                                         "--mesh", mesh_ply_path,
    #                                         "--points", points_ply_path,
    #                                         "--output_dir", result_dir,
    #                                         #"--debug",
    #                                         ]
    # subprocess.run(rotational_symmetry_detection_command)

    # if(not os.path.exists(raw_viewpoint_path)):
    #     # create_sphere_viewpoints.pyの実行
    #     create_sphere_viewpoints_command = ["python", "viewpoints_reduction/create_sphere_viewpoints.py",
    #                                         "--output_dir", viewpoints_dir,
    #                                         #    "--gif"
    #                                     ]
    #     subprocess.run(create_sphere_viewpoints_command)


    # # restrict_view_point_by_perspective_area.pyを実行
    # # 物体の形状のシャープさに応じて視点を制限
    # restrict_view_point_by_perspective_area_command = ["python", "viewpoints_reduction/restrict_viewpoints_by_perspective_area.py",
    #                                                    "--mesh", mesh_ply_path,
    #                                                    "--viewpoint", raw_viewpoint_path,
    #                                                    "--output_dir", restricted_viewpoint_dir_by_perspected_area
    #                                                 #    "--gif"
    #                                                    ]
    # subprocess.run(restrict_view_point_by_perspective_area_command)


    # if(use_grasp_pose):
    #     print("把持情報を用いて視点を制限")
    #     # 把持情報を用いて視点を制限
    #     restrict_view_points_by_grasp_pose_command   = ["python", "viewpoints_reduction/restrict_viewpoints_by_grasp_pose.py",
    #                                                     "--mesh", rendering_mesh_path,
    #                                                     "--grasp_dataset_dir",  graspdataset_path,
    #                                                     "--viewpoint", os.path.join(restricted_viewpoint_dir_by_perspected_area, f"{mesh_name}.npy"),
    #                                                     "--output_dir", restricted_viewpoint_dir_by_grasp_pose,]
    #     subprocess.run(restrict_view_points_by_grasp_pose_command)
    #     restricted_viewpoint_dir_by_grasp_pose_path = os.path.join(restricted_viewpoint_dir_by_grasp_pose, f"{mesh_name}.npy")
    #     print(restrict_view_points_by_grasp_pose_command)
    # else:
    #     print("把持情報を用いて視点を制限なし")
    restricted_viewpoint_dir_by_grasp_pose_path = raw_viewpoint_path #os.path.join(restricted_viewpoint_dir_by_perspected_area, f"{mesh_name}.npy")


    # # コピー先のディレクトリが存在しない場合は作成する
    # os.makedirs(os.path.dirname(rendering_mesh_path), exist_ok=True)

    # # mesh_ply_path の ply ファイルを rendering_mesh_path にコピーする
    # mesh_ply_path = os.path.join(aligned_mesh_dir, f"{mesh_name}", f"{mesh_name}.ply")
    # shutil.copy(mesh_ply_path, rendering_mesh_path)

    # # テンプレート画像を格納するフォルダがすでに存在するなら消す
    # save_dir = os.path.join(rendered_image_dir, f"train/{mesh_name}")
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    #     print(f"Removed existing directory: {save_dir}")
    # else:
    #     print(f"Directory does not exist: {save_dir}")
        
     
        
        
     
   
        
    # restrict_view_points_by_rotational_symmetry.pyを実行
    # 回転対称情報を用いて視点を制限
    restrict_view_points_by_rotational_symmetry_command = ["python", "viewpoints_reduction/restrict_viewpoints_by_rotational_symmetry.py",
                                                           "--mesh", mesh_ply_path,
                                                           "--viewpoint", restricted_viewpoint_dir_by_grasp_pose_path,
                                                           "--output_dir", restricted_viewpoint_dir_by_rotation_symmetry,
                                                           "--rot_sym_info_path", os.path.join(result_dir, "rotational_symmetry", f"{mesh_name}_rotational_symmetry.json")]
    subprocess.run(restrict_view_points_by_rotational_symmetry_command)
    print(restrict_view_points_by_rotational_symmetry_command)
    restricted_viewpoint_path_by_rotation_symmetry = os.path.join(restricted_viewpoint_dir_by_rotation_symmetry, f"{mesh_name}.npy")
        
     # パスの存在を確認します
    if os.path.exists(restricted_viewpoint_path_by_rotation_symmetry):
        print(f"The path '{restricted_viewpoint_dir_by_grasp_pose_path}' exists.")
          # パスが存在する場合に実行する処理をここに記述します
        rendering_viewpoint_path = restricted_viewpoint_path_by_rotation_symmetry
    else:
        print(f"The path '{restricted_viewpoint_dir_by_grasp_pose_path}' does not exist.")
          # パスが存在しない場合に実行する処理をここに記述します
        rendering_viewpoint_path = restricted_viewpoint_dir_by_grasp_pose_path
    
    # create_camera_coodinate_from_points.pyを実行
    # 視点からカメラの姿勢を作成する
    create_camera_coodinate_from_points_command = ["python", "rendering/create_camera_coodinate_from_points.py",
                                                   #"--viewpoints_file", restricted_viewpoint_path_by_rotation_symmetry,
                                                   "--viewpoints_file", rendering_viewpoint_path,
                                                   "--ply", rendering_mesh_path,
                                                   "--axial_angle_division_number", "1",
                                                   "--output_dir", path_camera_coordinates_dir,
                                                   #"--debug",
                                                   ]
    subprocess.run(create_camera_coodinate_from_points_command)


    # rendering_images_by_pyrender.pyを実行
    # カメラの姿勢からpyrenderでレンダリングする
    rendering_images_by_pyrender_command = ["python", "rendering/rendering_images_by_pyrender_material.py",
                                            "--path_camera_intrinsic_param", os.path.join(input_dir, "camera_param", "camera.json"),
                                            "--path_camera_coordinates", os.path.join(path_camera_coordinates_dir, f"{mesh_name}.npy"),
                                            "--mesh", rendering_mesh_path,
                                            "--output_dir", rendered_image_dir]
    subprocess.run(rendering_images_by_pyrender_command)


if __name__ == "__main__":
    main()
