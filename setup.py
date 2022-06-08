from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="meshflip",
    version="0.01",
    packages=["meshflip",],
    license="MIT",
    description="A toolbox for mesh reorientation.",
    url="https://github.com/nikwl/meshflip",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
)