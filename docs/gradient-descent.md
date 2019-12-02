# gradient-descent

Gradient descent is a gradient-based local optimization method. This is probably the simplest method in this category.

## Header

```cpp
#include <mathtoolbox/gradient-descent.hpp>
```

## Internal Dependencies

- [backtracking-line-search](../backtracking-line-search/)

## Math and Algorithm

### Line Search

This implementation uses [backtracking-line-search](../backtracking-line-search/) to find an appropriate step size. The initial step size needs to be specified as `default_alpha`.

### Stopping Criteria

(TODO)

### Bound Conditions

This implementation supports simple lower/upper bound conditions.

## Useful Resources

- Gradient descent - Wikipedia. <https://en.wikipedia.org/wiki/Gradient_descent>.
