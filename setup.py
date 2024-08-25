import logging
import os
from setuptools import setup, find_namespace_packages

log_format = "%(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
log = logging.getLogger("Key Data Tools Install")

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="world_models",
    version="0.0.1",
    description="World Models",
    url="https://github.com/jeferal/world_models",
    author='Jesus Ferrandiz',
    author_email='jeferrandiz98@gmail.com',
    license="Copyright",
    packages=find_namespace_packages(include=["world_models.*"]),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
