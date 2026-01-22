"""
Setup script para instalação do pacote HeartRate-PPG.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="heartrate-ppg",
    version="1.0.0",
    author="Projeto Acadêmico",
    author_email="",
    description="Sistema de Reconhecimento de Batimentos Cardíacos via Fotopletismografia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "tensorflow>=2.8.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "jupyter>=1.0.0",
        ],
        "gui": [
            "PyQt5>=5.15.0",
        ],
        "pytorch": [
            "torch>=1.10.0",
            "torchvision>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "heartrate-monitor=app.main_app:main",
            "heartrate-train=src.train:main",
        ],
    },
)
