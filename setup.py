from setuptools import setup

setup(
    name='PSFEM',
    version='1.0',
    packages=['PSFEM'],
    url='https://github.com/qTipTip/PSFEM',
    license='MIT',
    author='Ivar Stangeby',
    author_email='',
    description='A small library for the assembly of finite element solutions using S-splines',
    install_requires=[
        'SSplines'
    ]
)
