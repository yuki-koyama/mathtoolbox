# som

Self-organizing map (SOM) for dimensionality reduction and low-dimensional embedding. This technique is useful for understanding high-dimensional data via visualization.

## Header

```cpp
#include <mathtoolbox/som.hpp>
```

## Internal Dependencies

- [data-normalization](../data-normalization)

## Math

### Terms

In this library, the following terms are used:

- __Data space:__ The space that the data points exist. It is typically high-dimensional.
- __Latent space:__ The space that the map is constructed. It is typically (and in this library restricted to) one- or two-dimensional.
- __Node:__ The element of the map. It has fixed coordinates in the latent space and also has coordinates in the data space that will be learned by the SOM algorithm.

### Update

This library implements a batch-style SOM algorithm rather than online-style ones. That is, for each update step, all the data points are handled equally and contribute to the update of the node coordinates simultaneously.

Let $\mathbf{X} \in \mathbb{R}^{D \times N}$ be the data matrix (each column represents a data point) and $\mathbf{Y} \in \mathbb{R}^{D \times K}$ be the map matrix (each column represents coordinates of a node). The batch-style update is written as

$$
\mathbf{Y} = ( \mathbf{G}^{-1} \mathbf{H} \mathbf{B} \mathbf{X}^{T} )^{T}.
$$

See the source code for the definitions of $\mathbf{G}$, $\mathbf{H}$, and $\mathbf{B}$.

### Neighborhood Function

This library uses a Gaussian function with a decreasing variance:

$$
\sigma^{2}(t) = \max \left[ \sigma^{2}_{0} \exp \left(- \frac{t}{s} \right), \sigma^{2}_{\min} \right],
$$

where $t$ is the iteration count, $s$ is a user-specified parameter for controlling the speed of decrease, and $\sigma^{2}_{0}$ and $\sigma^{2}_{\min}$ are user-specified initial and minimum variances, respectively.

### Data Normalization

This library offers an option to perform data normalization, which is recommended to use in general.

## Examples

The following is an example of applying the algorithm to the pixel RGB values of a target image and learning its 2D color manifold.
![](./som/som-image.jpg)

The above map was generated through 30 iterations. The learning process is visualized as below.
<video width="100%" src="./som-image.mp4" controls autoplay loop type="video/mp4">(Your browser doesn't support video playing.)</video>

Self-organizing map is often assumed to be two-dimensional, but it is also possible to use other dimensionalities for the latent space. The below is an example of learning a one-dimensional map.

![](./som/som-image-1d.jpg)

## Useful Resources

- Chuong H. Nguyen, Tobias Ritschel, and Hans-Peter Seidel. 2015. Data-Driven Color Manifolds. ACM Trans. Graph. 34, 2, 20:1--20:9 (March 2015). DOI: <https://doi.org/10.1145/2699645>
