# mathtoolbox

A library of mathematical tools (regression, interpolation, dimensionality reduction, clustering, etc.) written in C++11. Eigen <http://eigen.tuxfamily.org/> is used for the interface and internal vector/matrix representation.

![](header.png)

## Algorithms

### Scattered Data Interpolation and Function Approximation
- [`rbf-interpolation`: Radial basis function (RBF) network](./rbf-interpolation/)
- [`gaussian-process-regression`: Gaussian process regression (GPR)](./gaussian-process-regression/)

### Dimensionality Reduction and Low-Dimensional Embedding
- [`classical-mds`: Classical multi-dimensional scaling (MDS)](./classical-mds/)

## Dependencies

- Eigen <http://eigen.tuxfamily.org/>
- NLopt <https://nlopt.readthedocs.io/> (included as gitsubmodule)
- nlopt-util <https://github.com/yuki-koyama/nlopt-util> (included as gitsubmodule)

## Build and Installation

mathtoolbox uses CMake <https://cmake.org/> for building source codes. This library can be built, for example, by
```
git clone https://github.com/yuki-koyama/mathtoolbox.git --recursive
cd mathtoolbox
mkdir build
cd build
cmake ../
make
```
and optionally it can be installed to the system by
```
make install
```

When the CMake parameter `MATHTOOLBOX_BUILD_EXAMPLES` is set `ON`, the example applications are also built. (The default setting is `OFF`.) This is done by, for example,
```
cmake ../ -DMATHTOOLBOX_BUILD_EXAMPLES=ON
make
```

### Installing Eigen

If you are using macOS, Eigen can be easily installed by
```
brew install eigen
```

## Projects Using mathtoolbox

- SelPh <https://github.com/yuki-koyama/selph> (for `classical-mds`)

## Licensing

The MIT License.
