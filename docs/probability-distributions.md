# probability-distributions

Probability distributions for statistical estimation.

## Header

```cpp
#include <mathtoolbox/probability-distributions.hpp>
```

## Overview

The following probability distributions and their first derivatives are supported:

- Standard normal distribution: $ \mathcal{N}(x \mid 0, 1) $
- Normal distribution: $ \mathcal{N}(x \mid \mu, \sigma^{2}) $
- Log-normal distribution: $ \mathcal{LN}(x \mid \mu, \sigma^{2}) $

In statistical estimation, taking logarithms of probabilities is often necessary. For this purpose, the following probability distributions and their derivatives are supported:

- Log of log-normal distribution: $ \log \{ \mathcal{LN}(x \mid \mu, \sigma^{2}) \} $

## Useful Resources

- Normal distribution - Wikipedia. <https://en.wikipedia.org/wiki/Normal_distribution>.
- Log-normal distribution - Wikipedia. <https://en.wikipedia.org/wiki/Log-normal_distribution>.
