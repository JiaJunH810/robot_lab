from setuptools import find_packages, setup

setup(
    name='rsl_rl_amp',
    version='1.0.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        "torch",
        "numpy",
    ],
)
