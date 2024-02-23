#!/usr/bin/python

from setuptools import setup, find_packages

setup(
    name   ='PathFillingPoints',
    version='0.0.1',
    author='Fernando Pujaico Rivera',
    author_email='fernando.pujaico.rivera@gmail.com',
    packages=[  'PathFillingPoints',
                'PathFillingPoints.Splines',
                'PathFillingPoints.Splines.Cubic3DSolverMethod1',
                'PathFillingPoints.Splines.Cubic3DSolverMethod2'],
    #scripts=['bin/script1','bin/script2'],
    url='https://github.com/trucomanx/PathFillingPoints',
    license='GPLv3',
    description='Path Filling Points',
    #long_description=open('README.txt').read(),
    install_requires=[
       "numpy"
    ],
)

#! python setup.py sdist bdist_wheel
# Upload to PyPi
# or 
#! pip3 install dist/PathFillingPoints-0.1.tar.gz 
