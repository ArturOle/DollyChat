
from setuptools import setup, find_packages


with open("requirements/base.txt") as req_file:
    reqs = req_file.read().splitlines()

setup(
    name='DollyChat',
    version='0.0.1',
    description='''
        Cross-platform app for talking with Dolly LLM models.
    ''',
    author='Artur Oleksiński',
    url='https://github.com/ArturOle',
    install_requires=reqs,
    packages=find_packages()
)
