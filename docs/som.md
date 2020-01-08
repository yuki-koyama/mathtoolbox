# som

Self-organizing map (SOM) for dimensionality reduction and low-dimensional embedding. This is also useful for visualizing the similarities of individual items in a 2-dimensional scattered plot.

## Header

```cpp
#include <mathtoolbox/som.hpp>
```

## Math

(TODO)

## Example

The following is an example of applying the algorithm to the pixel RGB values of a target image and learning its 2D color manifold.
![](./som/som-image.jpg)

The above map was generated through 30 iterations. The learning process is visualized as below.
<video width="100%" src="./som-image.mp4" controls autoplay loop type="video/mp4">(Your browser doesn't support video playing.)</video>

## Useful Resources

- Chuong H. Nguyen, Tobias Ritschel, and Hans-Peter Seidel. 2015. Data-Driven Color Manifolds. ACM Trans. Graph. 34, 2, 20:1--20:9 (March 2015). DOI: <https://doi.org/10.1145/2699645>
