import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name='graspDataToolkit',
    version='0.1',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    url='',
    license='',
    author='Martin Rudorfer',
    author_email='m.rudorfer@bham.ac.uk',
    description='robotic grasping toolkit',
    long_description=long_description,
    python_requires='>=3.6',
)
