# Parameter separation
For the project we ultimately aim at shape optimization and thus the parameter $\mu$ represents geometric properties of the unit cell.
As a simplification, one could first look at problems where the geometry is fixed, but the Young's Modulus is parameterized to mimick the increase in stiffness by putting more material.
Using the simplified model, one could first study how to deal with the training (oversampling) strategy and optimization process, without the complexity of the geometry transformation.

## ToDo

- [ ] add weak form for linear elasticity

## Linear elasticity
Assume $E$ or $\nu$ to be either components $\mu_i$ or some functions of the parameter value $\mu=(\mu_1, \ldots, \mu_p)$.

Lame constants:

$$
\begin{equation}
L(\mu)=L(E, \nu)=\frac{E\nu}{(1+\nu)(1-2\nu)}\,,\quad M(\mu)=M(E, \nu)=\frac{E}{2(1+\nu)}
\end{equation}
$$

Weak form:

$$
\begin{align}
\sigma_{ij} &= L(\mu) \varepsilon_{kk}\delta_{ij} + 2 M(\mu) \varepsilon_{ij}\\
a(w, v; \mu) &= \int_{\varOmega} \boldsymbol\varepsilon\boldsymbol\cdot\boldsymbol\sigma\,\mathrm{d}x\\
&= \int_{\varOmega} L(\mu)\varepsilon_{kk}^2\,\mathrm{d}x + 2 \int_{\varOmega} M(\mu)\varepsilon_{ij}\varepsilon_{ij}\,\mathrm{d}x
\end{align}
$$
### $E$ as piecewise constant function
Assume different Young's Modulus for each unit cell of the lattice structure, i.e. $\mu=(\mu_1,\ldots,\mu_N)=(E_1,\ldots,E_N)$ where $N$ is the number of cells in the lattice structure. Also, assume $\nu=\mathrm{const.}$.
In this case $L(\mu)$ and $M(\mu)$ can simply be put before the integral.

### $E$ as linear combination of coefficients and basis functions
Assume some $\mu=(\mu_1,\ldots,\mu_p)$ and $\nu=\mathrm{const.}$

$$
\begin{equation}
E(\mu, \boldsymbol{x}) = \sum_{i=1}^p \mu_i \boldsymbol{b}_i(\boldsymbol{x})\,.
\end{equation}
$$
The basis functions $\boldsymbol{b}_i$ may be chosen as a **polynomial (global) basis** or **pre-computed based on a global reduced model** of the structural problem.  This leads to $Q_a=2p$ terms in the parameter-separated form:

$$
\begin{equation}
a(w, v;\mu) = \frac{1}{(1+\nu)(1-2\nu)} \left( \sum_{i=1}^p \mu_i \int_{\varOmega} \boldsymbol{b}_i(x) \varepsilon_{kk}^2\,\mathrm{d}x\right)
+ \frac{1}{(1+\nu)} \left( \sum_{i=1}^p \mu_i \int_{\varOmega} \boldsymbol{b}_i(x) \varepsilon_{kl}\varepsilon_{kl}\,\mathrm{d}x \right)
\end{equation}
$$