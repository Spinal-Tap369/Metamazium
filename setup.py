# setup.py

from setuptools import setup, find_packages
import glob

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metamazium",
    version="1.0.0",
    author="Samuel Verghese",
    author_email="samvd123@gmail.com",
    description="A package demonstrating SNAIL as a meta reinforcement learning tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Spinal-Tap369/Metamazium", 
    packages=find_packages(exclude=["tests", "scripts"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "packaging",
        "torch",
        "pygame",
        "gymnasium",
        "numba",
        "einops",
        "tqdm",
        "axial-positional-embedding",  
        "local-attention",            
        "flash-attn",               
        "jsonschema",               
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
        ],
    },
    entry_points={
        "console_scripts": [
            "generate-mazes=metamazium.scripts.generate_mazes:main",
            "train-snail=metamazium.train_model.train_snail_trpo_fo:main",
            "train-lstm=metamazium.train_model.train_lstm_trpo_fo:main"
        ],
    },
    include_package_data=True,
    package_data={
        "metamazium": [
            "env/img/*.png",          # maze render assets
            "mazes_data/*.json",      # training and testing data generated using scripts/generate_mazes.py
        ],
    },
    zip_safe=False,
)
