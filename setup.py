from setuptools import setup

setup(
    name="noisypatho",
    author="Kenji Harada",
    maintainer="Kenji Harada",
    description="Create Noisy Annotations for Pathological Semantic Segmentation Data",
    license="CC-BY-NC-SA 4.0",
    url="https://github.com/kenjhara/noisypatho",
    version="0.1",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",
        "scipy",
        "opencv"
    ]
)
