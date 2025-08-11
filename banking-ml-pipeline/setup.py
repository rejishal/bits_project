# setup.py
"""Setup configuration for Banking ML Pipeline"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="banking-ml-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning pipeline for customer segmentation and loan eligibility prediction in banking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/banking-ml-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-banking-model=scripts.train_model:main",
            "evaluate-banking-model=scripts.evaluate_model:main",
            "banking-api=src.api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)