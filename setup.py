from setuptools import setup, find_packages

setup(
    name='value_capture',
    version='0.1',
    url='https://github.com/Gilles86/value_capture',
    author='Gilles de Hollander',
    author_email='gilles.de.hollander@gmail.com',
    description='Analysis code for the value capture fMRI experiment',
    packages=find_packages(),
    package_data={'value_capture': ['data/*.yml']},
    install_requires=[],
)
