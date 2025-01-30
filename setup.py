# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metamazium",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive package for meta-learning mazes with SNAIL and Performer models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Spinal-Tap369/Metamazium", 
    packages=find_packages(exclude=["tests", "scripts"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "packaging",
        "torch",
        "pygame",
        "gymnasium",
        "numba",
        "einops",
        "tqdm",
        "AxialPositionalEmbedding",  
        "LocalAttention",            
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
            "train-snail=metamazium.scripts.train_snail_performer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "metamazium": ["mazes_data/*.json", "metamazium/env/img/*.png"], 
    },
    zip_safe=False,
)
