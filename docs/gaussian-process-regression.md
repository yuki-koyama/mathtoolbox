# gaussian-process-regression

Gaussian process regression (GPR) for scattered data interpolation and function approximation.

![](gaussian-process-regression/examples.png)

## Header

```cpp
#include <mathtoolbox/gaussian-process-regression.hpp>
```

## Overview

### Input

The input consists of a set of $ N $ scattered data points:

$$
\{ (\mathbf{x}_i, y_i) \}_{i = 1}^{N},
$$

where $ \mathbf{x}_i \in \mathbb{R}^D $ is the $ i $-th data point location in a $ D $-dimensional space and $ y_i \in \mathbb{R} $ is its associated value. This input data is also denoted as

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_{1} & \cdots & \mathbf{x}_{N} \end{bmatrix} \in \mathbb{R}^{D \times N}
$$

and

$$
\mathbf{y} = \begin{bmatrix} y_1 \\ \vdots \\ y_N \end{bmatrix} \in \mathbb{R}^{N}.
$$

Here, each observed values $ y_i $ is assumed to be a noisy version of the corresponding latent function value $ f_i = f(\mathbf{x}_i) $. More specifically,

$$
y_i = f_i + \delta,
$$

where

$$
\delta \sim \mathcal{N}(0, \sigma_n^{2}).
$$

$ \sigma_n^{2} $ (the noise variance) is considered as one of the hyperparamters of this model.

### Output

Given the data and the _Gaussian process_ assumption, GPR can calculate the most likely value $ f_{*} $ and its variance $ \text{Var}(f_{*}) $ for an arbitrary location $ \mathbf{x}_{*} $.

The variance roughly indicates how uncertain the estimation is. For example, when this value is large, the estimated value may not be very trustful (this often occurs in regions with less data points).

Note that, as the predicted value follows a Gaussian, its 95%-confidence interval can be obtained by $ [ f_{*} - 1.96 \sqrt{\text{Var}(f_{*})}, f_{*} + 1.96 \sqrt{\text{Var}(f_{*})} ] $.

## Math

### Covariance Function

The automatic relevance determination (ARD) Matern 5/2 kernel is the default choice:

$$
k(\mathbf{x}_p, \mathbf{x}_q ; \sigma_f^{2}, \boldsymbol{\ell}) = \sigma_f^{2} \left( 1 + \sqrt{5 r^{2}(\mathbf{x}_{p}, \mathbf{x}_{q}; \boldsymbol{\ell})} + \frac{5}{3} r^{2}(\mathbf{x}_{p}, \mathbf{x}_{q}; \boldsymbol{\ell}) \right) \exp \left\{ - \sqrt{5 r^{2}(\mathbf{x}_{p}, \mathbf{x}_{q}; \boldsymbol{\ell})} \right\},
$$

where

$$
r^{2}(\mathbf{x}_{p}, \mathbf{x}_{q}; \boldsymbol{\ell}) = (\mathbf{x}_p - \mathbf{x}_q)^{T} \text{diag}(\boldsymbol{\ell})^{-2} (\mathbf{x}_p - \mathbf{x}_q)
$$

and $ \sigma_f^{2} $ (the signal variance) and $ \boldsymbol{\ell} $ (the characteristic length-scales) are its hyperparameters.

### Mean Function

A constant-value function is used:

$$
m(\mathbf{x}) = 0.
$$

### Selecting Hyperparameters

There are two options for setting hyperparameters:

- Set manually
- Determined by the maximum likelihood estimation

#### Maximum Likelihood Estimation

Let $ \boldsymbol{\theta} $ be a concatenation of hyperparameters; that is,

$$
\boldsymbol{\theta} = \begin{bmatrix} \sigma_{f}^{2} \\ \sigma_{n}^{2} \\ \boldsymbol{\ell} \end{bmatrix} \in \mathbb{R}^{D + 2}.
$$

In this approach, these hyperparameters are determined by solving the following numerical optimization problem:

$$
\boldsymbol{\theta}^\text{ML} = \mathop{\rm arg~max}\limits_{\boldsymbol{\theta}} p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}).
$$

In this implementation, this maximization problem is solved by the L-BFGS method (a gradient-based local optimization algorithm) from the NLopt library <https://nlopt.readthedocs.io/>. Initial solutions for this maximization need to be specified.

## Usage

### Instantiation and Data Specification

A GPR object is instantiated with data specification in its constructor:
```cpp
GaussianProcessRegression(const Eigen::MatrixXd& X,
                          const Eigen::VectorXd& y,
                          const KernelType       kernel_type = KernelType::ArdMatern52);
```

### Hyperparameter Selection

Hyperparameters are set by either
```cpp
void SetHyperparameters(double sigma_squared_f,
                        double sigma_squared_n,
                        const Eigen::VectorXd& length_scales);
```
or
```cpp
void PerformMaximumLikelihood(double sigma_squared_f_initial,
                              double sigma_squared_n_initial,
                              const Eigen::VectorXd& length_scales_initial);
```

### Estimation

Once a GPR object is instantiated and its hyperparameters are set, it is ready for estimation. For an unknown location $ \mathbf{x} $, the GPR object estimates the most likely value $ f $ by the following method:
```cpp
double PredictMean(const Eigen::VectorXd& x) const;
```
It also estimates the variance $ \text{Var}(f) $ by the following method:
```cpp
double PredictVariance(const Eigen::VectorXd& x) const;
```

## Useful Resources

- Mark Ebden. 2015. Gaussian Processes: A Quick Introduction. [arXiv:1505.02965](https://arxiv.org/abs/1505.02965).
- Carl Edward Rasmussen and Christopher K. I. Williams. 2006. Gaussian Processes for Machine Learning. The MIT Press. Online version: <http://www.gaussianprocess.org/gpml/>
