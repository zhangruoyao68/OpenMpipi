from setuptools import setup, find_packages

setup(
    name="OpenMpipi",
    version="0.1.0",
    author="Kieran Russell",
    description="OpenMM implementation of the Mpipi recharged forcefield",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",      
        "scipy>=1.7",
        "mdtraj>=1.9",     
        "openmm>=7.7",     
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or whichever license you use
        "Operating System :: OS Independent",
    ],
)
