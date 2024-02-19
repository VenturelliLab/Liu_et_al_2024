from jax import jit, jacfwd, vmap
import jax.numpy as jnp
from jax.experimental.ode import odeint


### JIT compiled matrix operations ###

@jit
def GAinvG(G, Ainv):
    return jnp.einsum("tki,ij,tlj->tkl", G, Ainv, G)


@jit
def GAinvG_diag(G, Ainv):
    return jnp.einsum("tki,i,tli->tkl", G, Ainv, G)


@jit
def GGv(G, v):
    return jnp.einsum("tki,i->k", G ** 2, v)


@jit
def trace_GGM(G, M):
    # return Trace( g @ g.T @ M)
    # k indexes each output
    return jnp.einsum("tki,tkj,ij->k", G, G, M)

@jit
def TrBGVGT(lmbda, Beta, Gt):
    log_s = lmbda[(len(lmbda) // 2):]
    return jnp.trace(jnp.einsum('kl,li,ij,mj->km', Beta, Gt, jnp.diag(jnp.exp2(log_s)**2), Gt))

grad_TrBGVGT = jit(jacfwd(TrBGVGT))

@jit
def yCOV_next(Y_error, G, Ainv):
    # sum over time dimension
    return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.sum(GAinvG(G, Ainv), 0)


@jit
def yCOV_next_diag(Y_error, G, Ainv):
    # sum over time dimension
    return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.einsum("tki,i,tli->kl", G, Ainv, G)


@jit
def A_next(G, Beta):
    A_n = jnp.einsum('tki, kl, tlj->ij', G, Beta, G)
    A_n = (A_n + A_n.T) / 2.
    return A_n


@jit
def A_next_diag(y_precision, G):
    return jnp.einsum("k,tki->i", y_precision, G ** 2)


# jit compile inverse Hessian computation step
@jit
def Ainv_next(G, Ainv, BetaInv):
    # [G] = [n_out, n_params]
    GAinv = G @ Ainv  # [n_t, n_p]
    Ainv_step = GAinv.T @ jnp.linalg.inv(BetaInv + GAinv @ G.T) @ GAinv
    # Ainv_step = jnp.einsum("ti,tk,kj->ij", GAinv, jnp.linalg.inv(BetaInv + jnp.einsum("ti,ki->tk", GAinv, G)), GAinv)
    Ainv_step = (Ainv_step + Ainv_step.T) / 2.
    return Ainv_step


# jit compile inverse Hessian computation step
@jit
def Ainv_prev(G, Ainv, BetaInv):
    GAinv = G @ Ainv
    Ainv_step = GAinv.T @ jnp.linalg.inv(GAinv @ G.T - BetaInv) @ GAinv
    Ainv_step = (Ainv_step + Ainv_step.T) / 2.
    return Ainv_step


# jit compile function to compute log of determinant of a matrix
@jit
def log_det(A):
    # # using the SVD
    # u,s,v = jnp.linalg.svd(A)
    # return jnp.sum(jnp.log(s))

    # using a Cholesky decomposition
    L = jnp.linalg.cholesky(A)
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))

@jit
def Tyz(y, lmbda):
    # transform standard normal to posterior
    mu, log_s = lmbda[:(len(lmbda) // 2)], lmbda[(len(lmbda) // 2):]
    s = jnp.exp2(log_s)
    return mu + s*y
batch_T = jit(vmap(Tyz, (0, None)))

@jit
def grad_T(y, lmbda):
    # transform standard normal to posterior
    log_s = lmbda[(len(lmbda) // 2):]
    grad_s = jnp.log(2) * y * 2.**log_s
    return grad_s

@jit
def log_abs_det(lmbda):
    log_s = lmbda[(len(lmbda) // 2):]
    return jnp.sum(jnp.log(jnp.exp2(log_s)))


# gradient of entropy of approximating distribution w.r.t. lmbda
grad_log_abs_det = jit(jacfwd(log_abs_det))


# approximate inverse of A, where A = LL^T, Ainv = Linv^T Linv
@jit
def compute_Ainv(A):
    Linv = jnp.linalg.inv(jnp.linalg.cholesky(A))
    Ainv = Linv.T @ Linv
    return Ainv


@jit
def eval_grad_NLP(Y_error, Beta, G):
    return jnp.einsum('tk,kl,tli->i', Y_error, Beta, G)


# compute utility of each experimental condition
@jit
def utility(GBG, A_q):
    # G has dimensions [n_params]
    fim_diag = A_q + GBG
    return jnp.sum(jnp.log(fim_diag))


# determine what Alpha needs to be in order for A = H + diag(Alpha) to be positive definite
def make_pos_def(A, Alpha, beta=1e-8):
    # make sure matrix is not already NaN
    assert not jnp.isnan(A).any(), "Matrix contains NaN, cannot make positive definite."

    # make sure alpha is not zero
    Alpha = jnp.clip(Alpha, beta, jnp.inf)

    # initial amount to add to matrix
    tau = 0.

    # use cholesky decomposition to check positive-definiteness of A
    while jnp.isnan(jnp.linalg.cholesky(A + tau * jnp.diag(Alpha))).any():
        # increase prior precision
        # print("adding regularization")
        tau = max([2. * tau, beta])

    # tau*jnp.diag(Alpha) is the amount that needed to be added for A to be P.D.
    # make alpha = (1 + tau)*alpha so that alpha + H is P.D.
    if tau > 0:
        print("Added {:.3e} to matrix diagonal".format(tau))

    return A + tau * jnp.diag(Alpha), (1. + tau) * Alpha
