#!/usr/bin/env python3
"""Setup script for Knowledge Distillation Framework."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_path, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
        return requirements

setup(
    name="knowledge-distillation",
    version="1.0.0",
    author="Knowledge Distillation Team",
    author_email="team@kd-framework.com",
    description="A production-ready knowledge distillation framework for neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/knowledge-distillation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "wandb": ["wandb>=0.12.0"],
        "timm": ["timm>=0.6.0"],
        "compression": ["torch-pruning>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "kd-train=main:main",
            "kd-eval=scripts.evaluate:main",
            "kd-export=scripts.export:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
