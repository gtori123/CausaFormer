from setuptools import setup, find_packages

setup(
    name="causaformer",  # パッケージ名
    version="0.1.0",  # バージョン
    author="gtori123",  # あなたの名前
    author_email="omission",  # あなたのメールアドレス
    description="A Transformer-based model for causal reasoning.",  # プロジェクト概要
    long_description=open("README.md", "r", encoding="utf-8").read(),  # 長い説明 (README.md を使用)
    long_description_content_type="text/markdown",  # README.md のフォーマット
    url="https://github.com/gtori123/CausaFormer",  # GitHub リポジトリの URL
    packages=find_packages(),  # `causaformer` 以下の Python パッケージを自動検出
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "scipy>=1.10.1"
    ],  # 必要な依存ライブラリ
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # パッケージのメタデータ
    python_requires=">=3.7",  # 必要なPythonのバージョン
)
