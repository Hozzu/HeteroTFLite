from setuptools import Extension, find_packages, setup

# Parse requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


# Polyiou extension
polyiou_module = Extension(
    "polyiou._polyiou",
    sources=["./polyiou/polyiou_wrap.cxx", "./polyiou/polyiou.cpp"],
    include_dirs=["./polyiou"],
    language="c++",
)

setup(
    name="dotadevkit",
    author="Ashwin Nair",
    author_email="ash1995@gmail.com",
    description="""DOTA Devkit CLI""",
    version="1.3.0",
    url="https://github.com/ashnair1/dotadevkit",
    packages=find_packages(),
    package_dir={"dotadevkit": "dotadevkit"},
    python_requires=">=3.6",
    ext_modules=[polyiou_module],
    install_requires=requirements,
    include_package_data=True,
    entry_points="""
        [console_scripts]
        dotadevkit=dotadevkit.cli.cli:cli
    """,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
