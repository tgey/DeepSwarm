import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepswarm2",
    version="0.0.1",
    author="Thomas Gey",
    author_email="thomas.gey.dev@gmail.com",
    description="Neural Architecture Search Powered by Swarm Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tgey/DeepSwarm",
    packages=setuptools.find_packages(),
    package_data={'deepswarm': ['../settings/default.yaml']},
    install_requires=[
        'colorama==0.4.1',
        'pyyaml==5.1',
        'scikit-learn==0.20.3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
