import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fourierimaging", 
    version="0.0.1",
    author="Samira Kabri and Tim Roith",
    author_email="tim.roith@fau.de",
    description="Python package for Forier Imaging Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samirak98/FourierImaging",
    packages=['fourierimaging'],
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"],
    install_requires=[  'numpy', 
                        'torch', 
                        'torchvision', 
                        'matplotlib',
                        'hydra-core',
                        'omegaconf',
                        'tqdm',
                        'pyyaml'],
    python_requires='>=3.6',
)
