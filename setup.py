from setuptools import setup, find_packages  # type: ignore

setup(
    name="sabnetra_ai",
    version="2.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "sabnetra=sabnetra_ai.cli:main",
        ],
    },
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
        "scipy>=1.10.0",
        "faiss-cpu>=1.7.0",
        "insightface>=0.7.0",
        "onnxruntime>=1.15.0",
    ],
    extras_require={
        "reid": ["torchreid>=1.4.0"],
        "gait": ["opengaitext>=0.1.0"],
        "full": ["torchreid>=1.4.0", "opengaitext>=0.1.0"],
    },
    author="SabNetra AI",
    description="CCTV-first real-time surveillance intelligence system",
    python_requires=">=3.9",
)
