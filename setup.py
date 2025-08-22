import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List, Dict

# Constants
PROJECT_NAME = "enhanced_cs"
VERSION = "1.0.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your@email.com"
DESCRIPTION = "A Central Chilled Water Plant Model for Designing Learning-Based Controllers"
LICENSE = "MIT"
URL = "https://github.com/your-username/your-repo-name"

# Requirements
INSTALL_REQUIRES: List[str] = [
    "torch",
    "numpy",
    "pandas",
]

TEST_REQUIRES: List[str] = [
    "pytest",
    "pytest-cov",
]

EXTRA_REQUIRES: Dict[str, List[str]] = {
    "dev": [
        "flake8",
        "isort",
        "black",
    ],
}

# Setup configuration
class CustomInstallCommand(install):
    def run(self):
        install.run(self)

class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)

class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)

def read_file(filename: str) -> str:
    """Read the contents of a file."""
    with open(filename, "r") as file:
        return file.read()

def read_requirements(filename: str) -> List[str]:
    """Read the requirements from a file."""
    return read_file(filename).splitlines()

def main():
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=read_file("README.md"),
        long_description_content_type="text/markdown",
        license=LICENSE,
        url=URL,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        tests_require=TEST_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        keywords="central chilled water plant model learning-based controllers",
        project_urls={
            "Documentation": URL + "/blob/main/README.md",
            "Issue Tracker": URL + "/issues",
        },
    )

if __name__ == "__main__":
    main()