# Getting started
In this example a heterogeneous beam structure is optimized.
The design variables are the Young's Moduli $E_q$, $q=1, \ldots, N$ for the $N$ subdomains.
Each subdomain corresponds to a unit cell.
The objective function is the compliance

$$
\begin{equation}
C(\boldsymbol{u}_{\mu}, \mu) = \boldsymbol{f}_{\mathrm{ext}}^T \boldsymbol{u}_{\mu}\,.
\end{equation}
$$

## Setup

- [ ] add image of the beam sketch?
- [ ] add image of mesh in paraview?

## Parametrization and weak form
Assume different Young's Modulus for each unit cell of the lattice structure, i.e. $\mu=(\mu_1,\ldots,\mu_N)=(E_1,\ldots,E_N)$ where $N$ is the number of cells in the lattice structure. Each unit cell is denoted by $\varOmega_i$, $i=1, \ldots, N$.
Assuming $\nu=\mathrm{const.}$, we can find a form such that $\theta_q(\mu)=E_q$ (leading to $Q_a=N$).

$$
\begin{align}
a(w, v;\mu) &= \sum_{q=1}^{N} \int_{\varOmega_q} L(\mu) \varepsilon_{kk}^2 + 2M(\mu) \varepsilon_{ij}\varepsilon_{ij}\,\mathrm{d}x\\
            &= \sum_{q=1}^{N} E_q \frac{1}{(1+\nu)} \int_{\varOmega_q} \frac{\nu}{(1-2\nu)} \varepsilon_{kk}^2 + \varepsilon_{ij}\varepsilon_{ij}\,\mathrm{d}x\,.
\end{align}
$$

$$
\begin{equation}
\theta_q(\mu) = E_q\,,\quad a_q(w, v)= \frac{1}{(1+\nu)} \int_{\varOmega_q} \frac{\nu}{(1-2\nu)} \varepsilon_{kk}^2 + \varepsilon_{ij}\varepsilon_{ij}\,\mathrm{d}x
\end{equation}
$$
