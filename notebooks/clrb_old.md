


### Image Model
We model the observed pixel values as:

$$
x_i = \mu_i(\theta) + \epsilon_i
$$

where:
* $x_i$ is the observed intensity at pixel $i$,
* $\mu_i(\theta)$ is the expected intensity at pixel $i$, given the position parameter $\theta = (\theta_x, \theta_y)$,
* $\epsilon_i$ is a zero-mean Gaussian noise term with variance $\sigma^2$, i.e.,

$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

The expected intensity is given by:

$$
\mu_i(\theta) = A \cdot h_i(\theta) + B
$$

where:
* $A$ is the signal amplitude,
* $h_i(\theta)$ is the PSF centered at $\theta$,
* $B$ is the constant background intensity.


### Likelihood Function
Given the Gaussian noise model, the joint probability of observing the entire image is:

$$
p(\mathbf{x} | \theta) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x_i - \mu_i(\theta))^2}{2 \sigma^2} \right)
$$

where $N$ is the total number of pixels.
The log-likelihood function is then:

$$
\log p(\mathbf{x} | \theta) = -\frac{N}{2} \log(2\pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^N (x_i - \mu_i(\theta))^2
$$


### Fisher Information Matrix (FIM)
The Fisher Information Matrix is given by [[1]](https://en.wikipedia.org/wiki/Fisher_information#:~:text=If%20log%E2%80%89f(x;%20θ)%20is%20twice%20differentiable%20with%20respect%20to%20θ%2C%20and%20under%20certain%20additional%20regularity%20conditions%2C%20then%20the%20Fisher%20information%20may%20also%20be%20written%20as[7]):

$$
\mathcal{I}(\theta) = -\mathbb{E} \left[ \frac{\partial^2 \log p(\mathbf{x} | \theta)}{\partial \theta \partial \theta^T} \right]  = \frac{1}{\sigma^2} \sum_{i=1}^N \left( \frac{\partial \mu_i(\theta)}{\partial \theta} \cdot \frac{\partial \mu_i(\theta)}{\partial \theta^T} \right)
$$

where the expectation $\mathbb{E}$ removes the noise-dependent term since the noise has zero mean.


### Practical Form for 2D Position
For a 2D position parameter $\theta = (x, y)$, this becomes a 2x2 matrix:

$$
\mathcal{I}(\theta) = \frac{A^2}{\sigma^2} \sum_{i=1}^N \begin{bmatrix}
\left(\frac{\partial h_i}{\partial x} \right)^2 & \frac{\partial h_i}{\partial x} \cdot \frac{\partial h_i}{\partial y} \\
\frac{\partial h_i}{\partial x} \cdot \frac{\partial h_i}{\partial y} & \left(\frac{\partial h_i}{\partial y} \right)^2
\end{bmatrix}
$$

The CRLB for the variance of the position estimate is given by the inverse of this matrix:

$$
\text{Var}(\hat{\theta}) \geq \mathcal{I}(\theta)^{-1}
$$
