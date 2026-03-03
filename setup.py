from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="planck-integrals",
    version="1.0.0",
    author="Ryan McClarren",
    description="Fast computation of incomplete Planck and Rosseland integrals using rational approximations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/planck-integrals",  # Update with your repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.16.0",
        "numba>=0.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    keywords="planck rosseland blackbody radiation thermodynamics integration",
)
