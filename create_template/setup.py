from setuptools import find_packages, setup

setup(
    name="create template for Line2D",  # プロジェクト名を指定してください
    version="0.1",
    author="Atsushi Yamamoto",  # 必要に応じて作者名を記載
    author_email="yamamotoats@yamaha-motor.co.jp",  # 必要に応じてメールアドレスを記載
    description="Line2D用のテンプレートを作成するためのソースコード",
    python_requires=">=3.8",
    install_requires=[
        "pyrender",
        "numpy",
        "open3d",
    ],
    packages=find_packages(),
)
