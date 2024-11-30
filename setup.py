from setuptools import setup, find_packages

setup(
    name='semester_project',
    version='0.1',
    packages=find_packages(where='src'),
    author='Mastroddi Giacomo',
    package_dir={'': 'src'},
    author_email='giacomma@student.ethz.ch',
    description='Semester project on DIYA',
)