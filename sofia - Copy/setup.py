import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='smarter',
    version='0.1.29',
    packages=find_packages(),
    include_package_data=True,
    license='BSD License',  # example license
    description='A simple Tensorflow app to predict customer support calls outcome on telecom industry.',
    long_description=README,
    url='http://www.webdisplay.pt',
    author='Bruno Oliveira',
    author_email='bruno.oliveira@webdisplay.pt',
    classifiers=[
        'Environment :: Machine Learning',
        'Framework :: Tensorflow',
        'Framework :: Tensorflow :: 1.0.0rc2',  # replace "X.Y" as appropriate
        'Intended Audience :: Vodafone PT',
        'License :: OSI Approved :: BSD License',  # example license
        'Operating System :: Ubuntu',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
    ],
)
