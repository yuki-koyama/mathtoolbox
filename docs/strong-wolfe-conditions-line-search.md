# strong-wolfe-conditions-line-search

A line search method for finding a step size that satisfies the strong Wolfe conditions (i.e., the Armijo (i.e., sufficient decrease) condition and the curvature condition).

## Header

```cpp
#include <mathtoolbox/strong-wolfe-conditions-line-search.hpp>
```

## Math and Algorithm

We follow [Nocedal and Wright (2006)](https://doi.org/10.1007/978-0-387-40065-5) (Chapter 3, specifically Algorithm 3.5).

## Useful Resources

- Jorge Nocedal and Stephen J. Wright. 2006. Numerical optimization (2nd ed.). Springer. DOI: <https://doi.org/10.1007/978-0-387-40065-5>
