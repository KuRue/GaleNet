# Physics-based Metrics

This document defines the conserved quantities tracked during training and evaluation. The formulations follow standard meteorological treatments such as Holton & Hakim (2012).

## Total Mass

**Definition**

\[
M = \int_V \rho\,\mathrm{d}V
\]

**Computation**
1. Integrate the density field over the domain for both prediction ($M_p$) and truth ($M_t$).
2. Report the relative error $|M_p - M_t|/M_t$.

## Total Energy

**Definition**

\[
E = \int_V \left[\tfrac{1}{2}\rho\lVert\mathbf{u}\rVert^2 + c_p\rho T + \rho g z\right] \,\mathrm{d}V
\]

**Computation**
1. Compute the specific energy density $e = \tfrac{1}{2}\rho\lVert\mathbf{u}\rVert^2 + c_p\rho T + \rho g z$ at each grid cell.
2. Integrate $e$ over the domain for prediction ($E_p$) and truth ($E_t$).
3. Report the relative error $|E_p - E_t|/E_t$.

## Vertical Vorticity

**Definition**

\[
\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
\]

**Computation**
1. Use finite differences to compute spatial derivatives of the horizontal velocity components and obtain $\zeta$.
2. Integrate $\zeta$ over the domain for prediction ($\zeta_p$) and truth ($\zeta_t$).
3. Report the relative error $|\zeta_p - \zeta_t|/|\zeta_t|$.

