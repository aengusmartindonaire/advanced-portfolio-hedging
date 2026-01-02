from setuptools import setup, find_packages

setup(
    name="adv_hedging",
    version="0.1.0",
    description="Advanced Portfolio Hedging Project",
    author="Martin",
    package_dir={"": "src"},          # Tells setuptools that packages are under src
    packages=find_packages(where="src"), # Finds all packages inside src
    python_requires=">=3.10",
)