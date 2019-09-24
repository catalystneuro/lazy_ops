from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lazy_ops",
    version="0.1.1",
    url="https://github.com/ben-dichter-consulting/lazy_ops",
    description="Lazy slicing and transpose operations for h5py",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Sotoude",
    author_email="dsot@protonmail.com",
    packages=find_packages(),
    install_requires=['numpy', 'h5py'],
)
