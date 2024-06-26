from setuptools import setup

setup(
    name="cbm_pack",
    version="0.1.0",
    description="""
        A library of analysis scripts for CbmSim, a product
        of the Mauk lab at the University of Texas at Austin
    """,
    author="Sean Gallogly",
    author_email="sean.gallo@austin.texas.edu",
    url="https://github.com/gawdSim/cbm_pack",
    packages=["cbm_pack", "benchmarks", "tests"],
    install_requires=[
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
