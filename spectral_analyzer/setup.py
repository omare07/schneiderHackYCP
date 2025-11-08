"""
Setup script for Spectral Analyzer.

Installation and packaging configuration for the AI-powered
spectral analysis desktop application.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "AI-powered spectral analysis desktop application for MRG Labs"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f.readlines()
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]
        # Filter out built-in modules
        requirements = [
            req for req in requirements
            if not any(builtin in req for builtin in [
                'sqlite3', 'configparser', 'asyncio', 'pathlib', 'hashlib',
                'json', 'uuid', 'platform', 'os', 'sys', 'time', 'datetime',
                'threading', 'multiprocessing', 'queue', 'contextlib',
                'functools', 'itertools', 'collections', 'dataclasses',
                'enum', 'abc', 'typing', 'warnings', 'traceback',
                'logging', 're', 'base64', 'tempfile', 'shutil',
                'glob', 'csv', 'io', 'gzip', 'zipfile'
            ])
        ]
else:
    requirements = [
        "PyQt6>=6.6.1",
        "pandas>=2.1.4",
        "numpy>=1.25.2",
        "matplotlib>=3.8.2",
        "httpx>=0.25.2",
        "cryptography>=41.0.8",
        "redis>=5.0.1",
        "pytest>=7.4.3"
    ]

setup(
    name="spectral-analyzer",
    version="1.0.0",
    author="MRG Labs Development Team",
    author_email="dev@mrglabs.com",
    description="AI-powered spectral analysis desktop application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrglabs/spectral-analyzer",
    project_urls={
        "Bug Tracker": "https://github.com/mrglabs/spectral-analyzer/issues",
        "Documentation": "https://docs.mrglabs.com/spectral-analyzer",
        "Source Code": "https://github.com/mrglabs/spectral-analyzer"
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: Qt",
        "Environment :: Win32 (MS Windows)",
        "Environment :: MacOS X"
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-qt>=4.2.0",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "isort>=5.12.0"
        ],
        "redis": [
            "redis>=5.0.1"
        ],
        "performance": [
            "numba>=0.58.1",
            "cython>=3.0.6"
        ]
    },
    entry_points={
        "console_scripts": [
            "spectral-analyzer=main:main",
            "spectral-analyzer-cli=cli.main:main"
        ],
        "gui_scripts": [
            "spectral-analyzer-gui=main:main"
        ]
    },
    package_data={
        "spectral_analyzer": [
            "resources/icons/*.png",
            "resources/icons/*.ico",
            "resources/styles/*.qss",
            "resources/templates/*.csv",
            "resources/templates/*.json"
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "spectroscopy", "ftir", "infrared", "csv", "analysis",
        "ai", "machine-learning", "normalization", "desktop",
        "pyqt", "matplotlib", "data-processing"
    ],
    platforms=["Windows", "macOS", "Linux"],
    license="Proprietary",
    maintainer="MRG Labs",
    maintainer_email="support@mrglabs.com"
)