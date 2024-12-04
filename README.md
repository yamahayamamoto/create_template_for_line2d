
# Create Template

このリポジトリは、3D CADデータとカメラパラメータを使用してレンダリングされた画像を生成するためのフレームワークを提供します。最小限の設定でレンダリングプロセスを簡素化することを目的としています。

---

## **データ構造**

入力データセットは、以下の構造に従ってください。

```
dataset
└── input
    ├── camera_param
    │   └── camera.json    # Pyrenderで使用するカメラパラメータ
    └── object             # 3D CADデータ (.STLまたは .OBJ形式)
        └── ex) ao
             └── ao.obj
```

- **`camera_param/camera.json`**: レンダリングに使用するカメラの設定が記載されています。
- **`object`**: 3D CADモデルを格納するディレクトリ。サブディレクトリごとに異なるオブジェクトを配置できます。

---

## **使用方法**

レンダリング画像を生成するには、以下の手順に従ってください。

1. プロジェクトディレクトリに移動します：
   ```bash
   cd create_template
   ```

2. 以下のコマンドでスクリプトを実行します：
   ```bash
   python create_rendering_img_pipeline.py "../../dataset/input" "../../dataset/output"
   ```

- 第一引数（`"../../dataset/input"`）は、入力データが格納されているディレクトリを指定します。
- 第二引数（`"../../dataset/output"`）は、レンダリングされた画像を保存する出力ディレクトリを指定します。

---

## **必要なライブラリ**

- Python 3.8 以上
- 以下のPythonライブラリが必要です：
  - `pyrender`
  - `numpy`
  - `open3d`
  - スクリプトで必要なその他の依存関係

---

