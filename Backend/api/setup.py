import setuptools
setuptools.setup(
    name="validation",
    version="0.0.0",
    author="MABADATA",
    author_email="mabadatabgu@gmail.com",
    description="Handle files from local and form bucket",
    url="https://github.com/MABADATA/validation",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    dependency_links=[
        'https://pypi.python.org/simple'
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)