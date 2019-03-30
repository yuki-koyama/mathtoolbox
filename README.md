# mathtoolbox

[![Build Status](https://travis-ci.com/yuki-koyama/mathtoolbox.svg?branch=master)](https://travis-ci.com/yuki-koyama/mathtoolbox)

A library of mathematical tools (regression, interpolation, dimensionality reduction, clustering, etc.) written in C++11. Eigen <http://eigen.tuxfamily.org/> is used for the interface and internal vector/matrix representation.

![](docs/header.png)

## Algorithms

### Scattered Data Interpolation and Function Approximation

- [`rbf-interpolation`: Radial basis function (RBF) network](https://yuki-koyama.github.io/mathtoolbox/rbf-interpolation/)
- [`gaussian-process-regression`: Gaussian process regression (GPR)](https://yuki-koyama.github.io/mathtoolbox/gaussian-process-regression/)

### Dimensionality Reduction and Low-Dimensional Embedding

- [`classical-mds`: Classical multi-dimensional scaling (MDS)](https://yuki-koyama.github.io/mathtoolbox/classical-mds/)

### Numerical Optimization

- [`backtracking-line-search`: Backtracking line search](https://yuki-koyama.github.io/mathtoolbox/backtracking-line-search/)
- [`bfgs`: BFGS method](https://yuki-koyama.github.io/mathtoolbox/bfgs/)
- [`l-bfgs`: Limited-memory BFGS method](https://yuki-koyama.github.io/mathtoolbox/l-bfgs/)
- [`strong-wolfe-conditions-line-search`: Strong Wolfe conditions line search](https://yuki-koyama.github.io/mathtoolbox/strong-wolfe-conditions-line-search/)

## Dependencies

- Eigen <http://eigen.tuxfamily.org/>
- NLopt <https://nlopt.readthedocs.io/> (included as gitsubmodule)
- nlopt-util <https://github.com/yuki-koyama/nlopt-util> (included as gitsubmodule)
- optimization-test-function <https://github.com/yuki-koyama/optimization-test-functions> (included as githsubmodule)

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

### Installing Prerequisites

macOS:
```
brew install eigen
```

Ubuntu:
```
sudo apt install libeigen3-dev
```

## Projects Using mathtoolbox

- SelPh <https://github.com/yuki-koyama/selph> (for `classical-mds`)

## Licensing

The MIT License.
