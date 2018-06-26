# mathtoolbox

A library of mathematical tools (regression, interpolation, dimensionality reduction, clustering, etc.) written in C++11. Eigen <http://eigen.tuxfamily.org/> is used for the interface and internal vector/matrix representation.

![](docs/header.png)

## Algorithms

### Scattered Data Interpolation and Function Approximation
- [`rbf-interpolation`: Radial basis function (RBF) network](https://yuki-koyama.github.io/mathtoolbox/docs/rbf-interpolation)
- [`gaussian-process-regression`: Gaussian process regression (GPR)](https://yuki-koyama.github.io/mathtoolbox/docs/gaussian-process-regression)

### Dimensionality Reduction and Low-Dimensional Embedding
- [`classical-mds`: Classical multi-dimensional scaling (MDS)](https://yuki-koyama.github.io/mathtoolbox/docs/classical-mds)

## Dependency

- Eigen <http://eigen.tuxfamily.org/>

## Build and Installation

mathtoolbox uses CMake <https://cmake.org/> for building source codes. This library can be built, for example, by
```
git clone https://github.com/yuki-koyama/mathtoolbox.git
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

## Projects Using mathtoolbox

- SelPh <https://github.com/yuki-koyama/selph> (for `classical-mds`)

## Licensing

The MIT License.
