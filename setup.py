from setuptools import setup, find_packages

setup(
    name="daa",
    version="0.1.0",
    description="Domain-specific Adapter-based Adaptation for Event Detection (ACL 2021)",
    packages=find_packages(exclude=["tests*", "preprocessing*", "scripts*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.12.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
    ],
)
