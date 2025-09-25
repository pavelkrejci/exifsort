from setuptools import setup, find_packages

setup(
    name="exifsort",
    version="1.0.0",
    description="A photo hierarchical sorting CLI tool based on EXIF data",
    author="Pavel Krejci",
    packages=find_packages(),
    install_requires=[
        "Pillow>=9.0.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "exifsort=exifsort.cli:main",
        ],
    },
    python_requires=">=3.7",
)