"""Package setup configuration for ChunIVision."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chunivision",
    version="1.0.0",
    author="Misaka 19465",
    description="Vision-based Chunithm controller using dual infrared cameras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/misaka19465/chunivision",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pyyaml>=5.4.0",
        "scipy>=1.7.0",
        "pyserial>=3.5",
        "pynput>=1.7.6",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "memory-profiler>=0.60.0",
        ],
        "linux": [
            "evdev>=1.4.0",
            "python-uinput>=0.11.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "chunivision=chunivision.main:main",
        ],
    },
    package_data={
        "chunivision": [
            "configs/*.yaml",
        ],
    },
    include_package_data=True,
)
