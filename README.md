# Scalable_ML for Astrophysics

This project template can be used to work on the programming exercises. Note that the following routines have been tested on a Linux machine and might not work on other OS.

## Create a virtual environment or a conda environment
It is common and recommended practice to use virtual environments for work in Python. I use a conda-forge Python distribution (similar to Anaconda that can also be used instead) in which I setup and activate a virtual environment using the following code that has to be executed before the pip install statement

conda update conda

conda create -n scalable_ml_env python=3.9

conda activate scalable_ml_env

## Install the project locally

pip install -e ".[test]"

If you don’t do this “editable installation” then your tests won’t run because the package will not be installed. An editable install means that changes in the code will be immediately reflected in the functionality of the package.
During the installation process, Python setuptools reads the file pyproject.toml and installs all packages that are listed there. If you require further packages, than you should add it to the pyproject.toml files. In the past, an alternative for the configuration of setuptools was done using a file called setup.cfg. It is only necessary to use on of these files, either pyproject.toml (newer) or setup.cfg (older).
