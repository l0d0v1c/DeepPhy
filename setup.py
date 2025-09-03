"""Setup configuration for DeepPhiELM."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.3.0",
    "scikit-optimize>=0.9.0",  # Optional for Bayesian optimization
]

setup(
    name="deepphielm",
    version="0.1.0",
    author="DeepPhiELM Team",
    author_email="contact@deepphielm.org",
    description="Physics-Informed Extreme Learning Machine for solving PDEs with numerical differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepphielm/deepphielm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
        "examples": [
            "jupyter",
            "seaborn",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/deepphielm/deepphielm/issues",
        "Source": "https://github.com/deepphielm/deepphielm",
        "Documentation": "https://deepphielm.readthedocs.io/",
    },
)