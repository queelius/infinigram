"""
Setup configuration for the infinigram package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="infinigram",
    version="0.1.0",
    author="Alex Towell",
    author_email="lex@metafunctor.com",
    description="Variable-length n-gram language models using suffix arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/queelius/infinigram",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "infinigram-demo=infinigram.demo:main",
        ],
    },
    include_package_data=True,
    keywords="language-model ngram suffix-array nlp text-prediction",
    project_urls={
        "Bug Reports": "https://github.com/queelius/infinigram/issues",
        "Source": "https://github.com/queelius/infinigram",
        "Documentation": "https://infinigram.readthedocs.io/",
    },
)
