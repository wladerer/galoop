from setuptools import setup, find_packages

setup(
    name="galoop",
    version="0.1.0",
    description="Lean genetic algorithm for electrochemical surface adsorbate structure search",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/wladerer/galoop",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "ase>=3.22",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "click>=8.0",
        "dscribe>=1.2",
    ],
    extras_require={
        "mace": ["mace-torch>=0.3"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "galoop=galoop.cli:cli",
        ],
    },
)
