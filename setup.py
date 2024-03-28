import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="CARZero",
    py_modules=["CARZero"],
    version="0.1",
    description="",
    author="Haoran Lai",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    dependency_links=[
        'https://pypi.tuna.tsinghua.edu.cn/simple/',  # 清华源
    ],
    include_package_data=True,
    license="Apache License",
)
