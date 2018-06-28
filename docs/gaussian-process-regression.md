# gaussian-process-regression

Gaussian process regression (GPR) for scattered data interpolation and function approximation.

## Header

```
#include <mathtoolbox/gaussian-process-regression.hpp>
```

## Math

### Input

$$
\{ (\mathbf{x}_i, y_i) \}_{i = 1, \ldots, n},
$$

### Kernel

The automatic relevance determination (ARD) squared exponential kernel is used.

$$
k(\mathbf{x}_p, \mathbf{x}_q) = \sigma_f^{2} \exp \left( - \frac{1}{2} (\mathbf{x}_p - \mathbf{x}_q)^{T} \text{diag}(\boldsymbol{\ell})^{-2} (\mathbf{x}_p - \mathbf{x}_q) \right) + \sigma_s^{2} \delta_{pq},
$$

where $$ \sigma_f^{2} $$, $$ \sigma_n^{2} $$, and $$ \boldsymbol{\ell} $$ are hyperparameters.

### Hyperparameters

Options:
- Set manually
- Determined by the maximum likelihood method

## Useful Resources

- Mark Ebden. 2015. Gaussian Processes: A Quick Introduction. [arXiv:1505.02965](https://arxiv.org/abs/1505.02965).
- Carl Edward Rasmussen and Christopher K. I. Williams. 2006. Gaussian Processes for Machine Learning. The MIT Press. Online version: <http://www.gaussianprocess.org/gpml/>

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
