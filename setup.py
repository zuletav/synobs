"""
Setup file for package `myfuncs`.
"""
from setuptools import setup
import pathlib

PACKAGENAME = 'synobs'

# the directory where this setup.py resides

HERE = pathlib.Path(__file__).absolute().parent

# function to parse the version from

def read_version():
    with (HERE / PACKAGENAME / '__init__.py').open() as fid:
        for line in fid:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":

    setup(
        name=PACKAGENAME,
        description='Function to create synthetic observations from a model fits file.',
        version=read_version(),
        long_description=(HERE / "README.md").read_text(),
        long_description_content_type='text/markdown',
        url='https://github.com/zuletav/' + PACKAGENAME.lower(),
        author='Andres Zuleta',
        author_email='azuleta@usm.lmu.de',
        license='GPLv3',
        packages=[PACKAGENAME],
        install_requires=[
            'astropy',
            'numpy',
            'scipy',
            'gofish',]
    )