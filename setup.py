from setuptools import setup, find_packages

setup(
    name="embanker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "onnxruntime>=1.11.0",
        "transformers>=4.20.0",
        "sanic>=21.12.0",
    ],
)