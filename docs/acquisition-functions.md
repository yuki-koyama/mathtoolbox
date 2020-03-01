# acquisition-functions

Acquisition functions for Bayesian optimization

## Header

```cpp
#include <mathtoolbox/acquisition-functions.hpp>
```

## Overview

The following acquisition functions (and their derivatives with respect to the data point) are supported:

- Expected improvement (EI)
- Gaussian process upper confidence bound (GP-UCB)

## Useful Resources

- Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. 2012. Practical Bayesian optimization of machine learning algorithms. In Proc. NIPS '12, pp.2951--2959.
- Yuki Koyama, Issei Sato, Daisuke Sakamoto, and Takeo Igarashi. 2017. Sequential line search for efficient visual design optimization by crowds. ACM Trans. Graph. 36, 4, pp.48:1--48:11 (2017). DOI: <https://doi.org/10.1145/3072959.3073598>
- Niranjan Srinivas, Andreas Krause, Sham Kakade, and Matthias Seeger. 2010. Gaussian process optimization in the bandit setting: no regret and experimental design. In Proc. ICML '10, pp.1015--1022.
