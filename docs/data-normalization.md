# data-normalization

Data normalization for preprocessing.

## Header

```cpp
#include <mathtoolbox/data-normalization.hpp>
```

## Math

This implementation is based on simple standardization for each dimension $d$:

$$
X_{i}^{(d)} \leftarrow \frac{X_{i}^{(d)} - \text{E}[X^{(d)}]}{\sigma(X^{(d)})}.
$$
