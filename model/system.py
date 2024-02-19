import jax.numpy as jnp
from jax.nn import tanh, sigmoid, relu
from jax.experimental.ode import odeint
from jax import jit, pmap, jacfwd


# define system of equations
@jit
def system(x, t, params, inputs, s_cap, m_cap):
    # unpack species and mediators
    s, m = x

    # unpack params
    W1, b1 = params

    # growth rates
    dsdt = s * (W1 @ s + b1) * (1. - s / s_cap)

    # rate of change of mediators (empty for gLV)
    dmdt = jnp.zeros(len(m))

    return dsdt, dmdt


@jit
def J_system(x, t, params, inputs, s_cap, m_cap):
    # unpack augmented state
    s, m, J = x

    # time derivative of state
    dsdt, dmdt = system([s, m], t, params, inputs, s_cap, m_cap)

    # system Jacobian
    Js, Jm = jacfwd(system, 0)([s, m], t, params, inputs, s_cap, m_cap)
    Js = jnp.concatenate(Js, -1)
    Jm = jnp.concatenate(Jm, -1)
    Jx = jnp.concatenate((Js, Jm), 0)

    # time derivative of grad(state, initial condition)
    dJdt = jnp.einsum("ij,j...->i...", Jx, J)

    return dsdt, dmdt, dJdt


@jit
def aug_system(x, t, params, inputs, s_cap, m_cap):
    # unpack augmented state
    s, m, *Z = x

    # time derivative of state
    dsdt, dmdt = system([s, m], t, params, inputs, s_cap, m_cap)

    # system Jacobian
    Js, Jm = jacfwd(system, 0)([s, m], t, params, inputs, s_cap, m_cap)
    Js = jnp.concatenate(Js, -1)
    Jm = jnp.concatenate(Jm, -1)
    Jx = jnp.concatenate((Js, Jm), 0)

    # gradient of system w.r.t. parameters
    Gs, Gm = jacfwd(system, 2)([s, m], t, params, inputs, s_cap, m_cap)

    # time derivative of parameter sensitivity
    dZdt = [jnp.einsum("ij,j...->i...", Jx, Z_i) + jnp.concatenate((Gs_i, Gm_i), 0) for Z_i, Gs_i, Gm_i in
            zip(Z, Gs, Gm)]

    return dsdt, dmdt, *dZdt


# integrate system of equations
@jit
def runODE(t_eval, s, m, params, inputs, s_cap, m_cap):
    # get initial condition of species and mediators
    s_ic = s[0]
    m_ic = m[0]

    # integrate
    return odeint(system, [s_ic, m_ic], t_eval, params, inputs, s_cap, m_cap)


# integrate to compute Jacobian over time
@jit
def runODEJ(t_eval, s, m, J0, params, inputs, s_cap, m_cap):
    # get initial condition
    s_ic = s[0]
    # mediator transform preformed before function call in this case!
    m_ic = m[0]

    # integrate
    return odeint(J_system, [s_ic, m_ic, J0], t_eval, params, inputs, s_cap, m_cap)


# integrate forward sensitivity equations
@jit
def runODEZ(t_eval, s, m, Z0, params, inputs, s_cap, m_cap):
    # get initial condition
    s_ic = s[0]
    m_ic = m[0]

    # integrate
    return odeint(aug_system, [s_ic, m_ic, *Z0], t_eval, params, inputs, s_cap, m_cap)


### JIT compiled helper functions to integrate ODEs in parallel ###

# batch evaluation of system
batchODE = pmap(runODE, in_axes=(None, 0, 0, None, 0, None, None))

# batch evaluation of sensitivity equations
batchODEZ = pmap(runODEZ, in_axes=(None, 0, 0, None, None, 0, None, None))
