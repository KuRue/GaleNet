# Physics-based Metrics

This document defines physical consistency metrics that can be used as loss
functions during model training.

## Mass Conservation
Measures how well the predicted density field preserves total mass.

**Definition**

\[
L_\text{mass} = \left(\sum_i p_i - \sum_i t_i\right)^2
\]
where $p_i$ and $t_i$ denote predicted and target density values.

**Calculation**
1. Sum all elements of the predicted and target density fields.
2. Compute the squared difference of the two totals.

## Momentum Conservation
Ensures the integrated momentum of the velocity field matches the target.

**Definition**

\[
L_\text{mom} = \left(\sum_i \vec{p}_i - \sum_i \vec{t}_i\right)^2
\]
where $\vec{p}_i=(u_i, v_i)$ and $\vec{t}_i$ are predicted and target velocity
vectors.

**Calculation**
1. Sum the velocity vectors over all grid points for both fields.
2. Compute the squared Euclidean distance between the totals.

## Kinetic Energy
Penalises differences in kinetic energy between predicted and target
velocities.

**Definition**

\[
L_\text{ke} = \frac{1}{N} \sum_i \left( \tfrac{1}{2}\|\vec{p}_i\|^2 - \tfrac{1}{2}\|\vec{t}_i\|^2 \right)^2
\]
where $N$ is the number of grid points.

**Calculation**
1. Compute kinetic energy $\tfrac{1}{2}(u^2+v^2)$ for each velocity vector.
2. Subtract target from prediction and compute the mean squared difference.

## Total Energy
Extends kinetic energy with internal and gravitational potential energy.

**Definition**

\[
L_\text{te} = \left(\int_V e_p\,\mathrm{d}V - \int_V e_t\,\mathrm{d}V\right)^2,
\]
where $e = \tfrac{1}{2}\rho\lVert\mathbf{u}\rVert^2 + c_p\rho T + \rho g z$ is the specific energy density and subscripts $p$ and $t$ denote prediction and target.

**Calculation**
1. Compute $e$ at each grid cell using velocity $\mathbf{u}=(u,v)$, temperature $T$, and height $z$.
2. Integrate $e$ over the domain for prediction and target.
3. Take the squared difference of the two totals.

## Relative Vorticity
Measures rotation in the horizontal flow field.

**Definition**

\[
L_\text{vort} = \left(\sum_i \zeta_{p,i} - \sum_i \zeta_{t,i}\right)^2,
\]
where $\zeta = \partial v/\partial x - \partial u/\partial y$ is the vertical component of vorticity.

**Calculation**
1. Compute spatial derivatives of the velocity field to obtain $\zeta$.
2. Sum vorticity over the grid for prediction and target.
3. Compute the squared difference of the totals.
