import setuptools

setuptools.setup(
    name="pymathtoolbox",
    version="1.0.dev0",
    author="Yuki Koyama",
    author_email="yuki@koyama.xyz",
    description="Python bindings of mathtoolbox: mathematical tools (interpolation, dimensionality reduction, optimization, etc.) written in C++11 with Eigen",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/yuki-koyama/mathtoolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
