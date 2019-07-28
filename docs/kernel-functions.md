# kernel-functions

Kernel functions for various techniques.

## Header

```cpp
#include <mathtoolbox/kernel-functions.hpp>
```

## Overview

### Automatic Relevance Determination (ARD) Squared Exponential Kernel

This kernel is defined as

$$
k(\mathbf{x}_p, \mathbf{x}_q ; \boldsymbol{\theta}) = \sigma_f^{2} \exp \left( - \frac{1}{2} (\mathbf{x}_p - \mathbf{x}_q)^{T} \text{diag}(\boldsymbol{\ell})^{-2} (\mathbf{x}_p - \mathbf{x}_q) \right),
$$

where $ \sigma_f^{2} $ (the signal variance) and $ \boldsymbol{\ell} $ (the characteristic length-scales) are its hyperparameters. That is,

$$
\boldsymbol{\theta} = \begin{bmatrix} \sigma_{f}^{2} \\ \boldsymbol{\ell} \end{bmatrix} \in \mathbb{R}^{n + 1}_{> 0}.
$$

## Useful Resources

- Carl Edward Rasmussen and Christopher K. I. Williams. 2006. Gaussian Processes for Machine Learning. The MIT Press. Online version: <http://www.gaussianprocess.org/gpml/>
