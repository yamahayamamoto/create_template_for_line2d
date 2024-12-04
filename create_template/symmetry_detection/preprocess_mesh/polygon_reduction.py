import os
import argparse
import pymeshlab as ml
import numpy as np

"""
python polygon_reduction.py --input_dir './dataset/mydata/models_obj' --output_dir './dataset/mydata/models_obj_reduce' 
"""

def get_obj_file_paths(input_dir):
    obj_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith(".obj"):  # 小文字で終わるかどうかを確認
                obj_paths.append(os.path.join(root, file_name))
    return obj_paths

def simplify_mesh_by_ratio(input_path, output_path, reduction_ratio):
    print("meshlab load:")
    print(input_path)
    ms = ml.MeshSet()
    ms.load_new_mesh(input_path)

    # 現在の面数を取得
    original_face_count = ms.current_mesh().face_number()

    # 簡約率を計算
    target_face_count = int(original_face_count * reduction_ratio)

    if(target_face_count<10000):
        target_face_count=10000


    # 簡約化
    ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=target_face_count)

    # 結果の保存
    ms.save_current_mesh(output_path)
    print("save:::",output_path)

def simplify_mesh_by_number(input_path, output_path, target_face_count):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_path)

    # 簡約化
    ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=target_face_count)

    # 結果の保存
    ms.save_current_mesh(output_path)
    print("save:::",output_path)

def simplify_all_meshes(input_dir, output_dir, reduction_ratio):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 入力フォルダ内のobjファイルのパスを取得
    obj_paths = get_obj_file_paths(input_dir)
    print("obj_paths",obj_paths)

    # 各objファイルを簡約化して保存
    for obj_path in obj_paths:
        file_name = os.path.basename(obj_path)
        
          # 拡張子を変更
        if file_name.endswith('.obj'):
           file_name = file_name[:-4] + '.ply'
        
        # output_subfolder = os.path.join(output_dir, os.path.dirname(obj_path)[len(input_dir):])
        output_subfolder = os.path.join(output_dir, os.path.basename(os.path.dirname(obj_path)))
        output_path = os.path.join(output_subfolder, file_name)
        print(output_subfolder,output_path)
        # 出力サブフォルダが存在しない場合は作成
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        simplify_mesh_by_ratio(obj_path, output_path, reduction_ratio)
        # simplify_mesh_by_number(obj_path, output_path, target_face_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Simplification Script")
    parser.add_argument("--input_dir", required=True, help="Input folder containing obj files")
    parser.add_argument("--output_dir", required=True, help="Output folder for simplified obj files")
    parser.add_argument("--reduction_ratio", type=float, default=0.02, help="Reduction ratio for simplification")
    parser.add_argument("--reduction_number", type=float, default=2000, help="Reduction ratio for simplification")

    args = parser.parse_args()

    # すべてのメッシュを簡約化
    simplify_all_meshes(args.input_dir, args.output_dir, args.reduction_ratio)
