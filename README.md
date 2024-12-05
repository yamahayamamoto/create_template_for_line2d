
# Create Template

このリポジトリは、3D CADデータとカメラパラメータを使用してレンダリングされた画像を生成するためのパイプラインを提供します。

---

## **依存パッケージのインストール**

このリポジトリで必要なライブラリは、以下の手順でインストールできます。

1. リポジトリをクローンし、プロジェクトディレクトリに移動します：
   ```bash
   git clone https://github.com/yamahayamamoto/create_template_for_line2d.git
   cd create_template
   ```

2. 以下のコマンドを使用して必要なパッケージをインストールします：
   ```bash
   pip install -e .
   ```
---

## **データ構造**

入力データセットは、以下の構造に従ってください。

```
dataset/input
├── camera_param
│   └── camera.json        # Pyrenderで使用するカメラパラメータ
├── object                 # 3D CADデータ (.STLまたは .OBJ形式) ※(重要)ここに新しいデータを追加してください
│   ├── ex)95D95-06010
│   │   └── 95D95-06010.STL
│   ├── ex)ao
│   │   ├── ao.mtl
│   │   ├── ao.obj
│   │   └── ao.png
│   :
│   
└── viewpoints             
    └── origin_viewpoints.npy  # カメラの視点(球をsubdivisionしたメッシュの頂点)
```
### (補足)
- **`camera_param/camera.json`**: レンダリングに使用するカメラの設定が記載されています。
- **`object`**: 3D CADモデルを格納するディレクトリ。サブディレクトリごとに異なるオブジェクトを配置できます。
- **`viewpoints`**: 3D空間でのカメラの視点位置を記載しています。視点数を変更したい場合は以下の方法で視点数を変更したファイルが生成されます。
   ```bash
     python create_template/viewpoints_reduction/create_sphere_viewpoints.py　--viewpoint_num 364 --output_dir ../dataset/input/viewpoints --debug
   ```

---

## **使用方法**

3Dモデルを入力とし、対称性を判定してレンダリング画像を生成するには、以下の手順に従ってください。

1. プロジェクトディレクトリに移動します：
   ```bash
   cd create_template
   ```

2. 以下のコマンドでスクリプトを実行します：
   ```bash
   python create_rendering_img_pipeline.py "../dataset/input" "../dataset/output"
   ```

- 第一引数（`"../dataset/input"`）は、入力データが格納されているディレクトリを指定します。
- 第二引数（`"../dataset/output"`）は、レンダリングされた画像を保存する出力ディレクトリを指定します。



