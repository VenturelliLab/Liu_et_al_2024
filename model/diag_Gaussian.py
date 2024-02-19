from .utilities import *

from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import random, jit, jacfwd, jacrev, grad, vmap
from jax.numpy.linalg import inv
from jax.nn import tanh, sigmoid, relu
from jax.experimental.ode import odeint

from scipy.optimize import minimize
from scipy.special import logsumexp

from tqdm import tqdm


# define model that takes as input the initial condition the latent variables
@partial(jit, static_argnums=(0,))
def model(system, Xi, z):
    # unpack data and integration time
    tf, x = Xi
    t_span = jnp.array([0., tf])

    # integrate ODE
    t_hat = odeint(system, jnp.array(x[0]), t_span, z)

    # t_hat is the model estimate of observed variable t
    return t_hat[1:]


# gradient of model w.r.t. latent variables z
@partial(jit, static_argnums=(0,))
def grad_model(system, Xi, z):
    return jacrev(model, 2)(system, Xi, z)


# invertible, differentiable function to map noise to model parameters
@partial(jit, static_argnums=(0,))
def T(transform, y, lmbda):
    # weights and biases of nn
    mu, log_s = lmbda[:len(lmbda) // 2], lmbda[len(lmbda) // 2:]

    # convert to z
    z = transform(mu + jnp.exp2(log_s) * y)

    return z


@partial(jit, static_argnums=(0,))
def batch_T(transform, y_batch, lmbda):
    return vmap(T, (None, 0, None))(transform, y_batch, lmbda)


# Jacobian of transform w.r.t. base distribution
@partial(jit, static_argnums=(0,))
def jac_T(transform, y, lmbda):
    return jacfwd(T, 1)(transform, y, lmbda)


# log of absolute value of determinant of Jacobian of T
@partial(jit, static_argnums=(0,))
def log_det_old(transform, y, lmbda):
    # return jnp.log(jnp.abs(jnp.linalg.det(jac_T(transform, y, lmbda))))
    # return jnp.sum(jnp.log(jnp.abs(jnp.linalg.eigvals(jac_T(transform, y, lmbda)))))
    return jnp.sum(jnp.log(jnp.abs(jnp.diag(jac_T(transform, y, lmbda)))))


@partial(jit, static_argnums=(0,))
def log_det(transform, y, lmbda):
    # unpack variational parameters
    mu, log_s = lmbda[:len(lmbda) // 2], lmbda[len(lmbda) // 2:]

    # convert to z
    z = mu + jnp.exp2(log_s) * y

    # compute log abs det
    eigs = jnp.diag(jacfwd(transform)(z)) * jnp.exp2(log_s)
    return jnp.sum(jnp.log(jnp.abs(eigs)))


# gradient of entropy of approximating distribution w.r.t. lmbda
@partial(jit, static_argnums=(0,))
def grad_log_det(transform, y, lmbda):
    return jacrev(log_det, 2)(transform, y, lmbda)


# evaluate log prior
@partial(jit, static_argnums=(0,))
def log_prior(transform, z_prior, y, lmbda, alpha):
    # map to sample from posterior
    z = T(transform, y, lmbda)

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of log prior
@partial(jit, static_argnums=(0,))
def grad_log_prior(transform, z_prior, y, lmbda, alpha):
    return jacrev(log_prior, 3)(transform, z_prior, y, lmbda, alpha)


# evaluate log likelihood
@partial(jit, static_argnums=(0, 1,))
def log_likelihood(system, transform, y, Xi, lmbda, beta):
    # map to sample from posterior
    z = T(transform, y, lmbda)

    # unpack condition
    tf, x = Xi

    # likelihood
    lp = jnp.nansum(beta * (x[1:] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0, 1,))
def grad_log_likelihood(system, transform, y, x, lmbda, beta):
    return jacrev(log_likelihood, 4)(system, transform, y, x, lmbda, beta)


# evaluate log posterior
@partial(jit, static_argnums=(0, 1))
def log_posterior(system, transform, y, Xi, z_prior, lmbda, alpha, beta, N):
    # map to sample from posterior
    z = T(transform, y, lmbda)

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2. / N

    # unpack condition
    tf, x = Xi

    # likelihood
    lp += jnp.nansum(beta * (x[1:] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0, 1,))
def grad_log_posterior(system, transform, y, Xi, z_prior, lmbda, alpha, beta, N):
    return jacrev(log_posterior, 5)(system, transform, y, Xi, z_prior, lmbda, alpha, beta, N)


# evaluate log posterior
@partial(jit, static_argnums=(0, 1))
def log_posterior_z(system, transform, z, Xi, z_prior, alpha, beta, N):
    # transform z
    z = transform(z)

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2. / N

    # unpack condition
    tf, x = Xi

    # likelihood
    lp += jnp.nansum(beta * (x[1:] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0, 1))
def grad_log_posterior_z(system, transform, z, Xi, z_prior, alpha, beta, N):
    return jacrev(log_posterior_z, 2)(system, transform, z, Xi, z_prior, alpha, beta, N)


# evaluate log prior
@partial(jit, static_argnums=(0,))
def log_prior_z(transform, z_prior, z, alpha):
    # transform z
    z = transform(z)

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of log prior
@partial(jit, static_argnums=(0,))
def grad_log_prior_z(transform, z_prior, z, alpha):
    return jacrev(log_prior_z, 2)(transform, z_prior, z, alpha)


# evaluate log likelihood
@partial(jit, static_argnums=(0, 1))
def log_likelihood_z(system, transform, z, Xi, beta):
    # unpack condition
    tf, x = Xi

    # transform z
    z = transform(z)

    # likelihood
    lp = jnp.nansum(beta * (x[1:] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0, 1))
def grad_log_likelihood_z(system, transform, z, x, beta):
    return jacrev(log_likelihood_z, 2)(system, transform, z, x, beta)


class ODE:
    def __init__(self,
                 system,
                 transform,
                 dataframe,
                 sys_vars,
                 prior_mean,
                 alpha=1., beta=100.):

        # system of differential equations
        self.system = system

        # function to reshape and transform parameters
        self.transform = transform

        # processed data
        self.sys_vars = sys_vars
        self.X = process_df(dataframe, sys_vars)

        # parameter prior
        self.prior_mean = prior_mean

        # prior and measurement precision
        self.alpha = alpha
        self.beta = beta

        # problem dimension
        self.d = len(self.prior_mean)

        # initial parameter guess
        self.z = np.random.randn(self.d) / 10.
        self.lmbda = jnp.append(self.z, jnp.log2(jnp.ones(self.d) / 100.))

    # negative log likelihood
    def nll(self):

        # prior
        self.NLL = np.nan_to_num(log_prior_z(self.transform, self.prior_mean, self.z, self.alpha))

        # likelihood
        for Xi in self.X:
            self.NLL += np.nan_to_num(log_likelihood_z(self.system, self.transform, self.z, Xi, self.beta))

        # return NLP
        return self.NLL

    # evidence lower bound
    def elbo(self, n_sample=21):

        # sample from posterior
        y = np.random.randn(n_sample, self.d)

        # entropy
        self.ELBO = 0.
        for yi in y:

            # entropy
            self.ELBO -= np.nan_to_num(log_det(self.transform, yi, self.lmbda)) / n_sample

            # prior
            self.ELBO += np.nan_to_num(log_prior(self.transform,
                                                 self.prior_mean,
                                                 yi,
                                                 self.lmbda,
                                                 self.alpha)) / n_sample

            # likelihood
            for Xi in self.X:
                self.ELBO += np.nan_to_num(log_likelihood(self.system,
                                                          self.transform,
                                                          yi,
                                                          Xi,
                                                          self.lmbda,
                                                          self.beta)) / n_sample

        # return NLP
        return self.ELBO

    # MLE update of measurement precision
    def update_precision(self):

        # init yCOV
        yCOV = 0.

        # loop over each sample in dataset
        N = np.zeros(len(self.sys_vars))
        for Xi in self.X:
            # predict condition
            tf, x = Xi

            # integrate ODE
            t_hat = model(self.system, Xi, self.z)

            # Determine error
            Y_error = np.nan_to_num(np.nan_to_num(t_hat) - x[-1])

            # sum of measurement error
            yCOV += Y_error ** 2

            # number of measurements
            N += np.array(x[-1] > 0, int)

        # update beta
        self.beta = N / (yCOV + 1e-4)

    def fit_MAP(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, max_epochs=1000, tol=1e-3, patience=3):
        """
        ADAM optimizer for minimizing a function.

        Parameters:
        - grad_f: Gradient function that returns the gradient of the objective function.
        - initial_params: Initial guess for the parameters.
        - learning_rate: Step size for the optimization (default: 0.001).
        - beta1: Exponential decay rate for the first moment estimate (default: 0.9).
        - beta2: Exponential decay rate for the second moment estimate (default: 0.999).
        - epsilon: Small constant to prevent division by zero (default: 1e-8).
        - max_iterations: Maximum number of iterations (default: 1000).
        - tol: Tolerance to stop optimization when the change in parameters is below this value (default: 1e-6).

        Returns:
        - Optimal parameters.
        """
        m = np.zeros_like(self.z)
        v = np.zeros_like(self.z)
        t = 0
        epoch = 0
        passes = 0

        # order of samples for sgd
        order = np.arange(len(self.X))
        f = []

        while epoch < max_epochs:
            epoch += 1
            f.append(self.nll())
            if len(f) > 10:

                # determine slope of elbo over time
                slope = check_convergence(f[-10:])

                # if performance is not improving, decrease lr
                if slope > 0:
                    lr *= .9
                    print("decrease lr to {:.3e}".format(lr))

                # try to accelerate
                if slope < -1e-2:
                    lr = np.min([1.1 * lr, .01])
                    print("set lr to {:.3e}".format(lr))

                # convergence criteria
                if lr < tol:
                    # increase pass count
                    passes += 1
                    # increase learning rate
                    lr *= 2.

                # return parameters after passing convergence criteria enough times
                if passes >= patience:
                    return

                print("Epoch {:.0f}, NLL: {:.3f}, Slope: {:.3f}".format(epoch, f[-1], slope))
            else:
                print("Epoch {:.0f}, NLL: {:.3f}".format(epoch, f[-1]))

            # sgd over each sample
            np.random.shuffle(order)
            for sample_index in order:
                # gradient of log posterior
                gradient = np.nan_to_num(grad_log_posterior_z(self.system,
                                                              self.transform,
                                                              self.z, self.X[sample_index],
                                                              self.prior_mean,
                                                              self.alpha, self.beta, N=len(self.X)))

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.z -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
                self.lmbda = self.lmbda.at[:self.d].set(self.z)

    def fit_MAP_EM(self, tol=.01):

        # init convergence
        self.nll()
        nll_prev = np.copy(self.NLL)
        convergence = 1.
        while convergence > tol:
            # optimize point estimate of parameters
            self.fit_MAP()

            # update measurement precision estimate
            self.update_precision()

            # update convergence
            convergence = abs(nll_prev - self.NLL) / self.NLL
            nll_prev = np.copy(self.NLL)
            print("NLL convergence: {:.3f}".format(convergence))

    def fit_posterior(self, n_sample=1, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, max_epochs=1000, tol=1e-3,
                      patience=3):
        """
        ADAM optimizer for minimizing a function.

        Parameters:
        - grad_f: Gradient function that returns the gradient of the objective function.
        - initial_params: Initial guess for the parameters.
        - learning_rate: Step size for the optimization (default: 0.001).
        - beta1: Exponential decay rate for the first moment estimate (default: 0.9).
        - beta2: Exponential decay rate for the second moment estimate (default: 0.999).
        - epsilon: Small constant to prevent division by zero (default: 1e-8).
        - max_iterations: Maximum number of iterations (default: 1000).
        - tol: Tolerance to stop optimization when the change in parameters is below this value (default: 1e-6).

        Returns:
        - Optimal parameters.
        """
        m = np.zeros_like(self.lmbda)
        v = np.zeros_like(self.lmbda)
        t = 0
        epoch = 0
        passes = 0

        # order of samples for sgd
        order = np.arange(len(self.X))

        # initialize function evaluations
        f = []

        while epoch < max_epochs:
            epoch += 1
            f.append(self.elbo(n_sample=n_sample))
            if len(f) > 10:

                # determine slope of elbo over time
                slope = check_convergence(f[-10:])

                # if performance is not improving, decrease lr
                if slope > 0:
                    lr *= .9
                    print("decrease lr to {:.3e}".format(lr))

                # try to accelerate
                if slope < -1e-2:
                    lr = np.min([1.1 * lr, .01])
                    print("set lr to {:.3e}".format(lr))

                # convergence criteria
                if lr < tol:
                    # increase pass count
                    passes += 1
                    # increase learning rate
                    lr *= 2.

                # return parameters after passing convergence criteria enough times
                if passes >= patience:
                    return

                print("Epoch {:.0f}, NEG ELBO: {:.3f}, Slope: {:.3f}".format(epoch, f[-1], slope))
            else:
                print("Epoch {:.0f}, NEG ELBO: {:.3f}".format(epoch, f[-1]))

            # sgd over each sample
            np.random.shuffle(order)
            for sample_index in order:

                # sample noise
                y = np.random.randn(n_sample, self.d)

                # stochastic evaluation of gradient
                for yi in y:

                    # gradient of entropy
                    gradient = -np.nan_to_num(grad_log_det(self.transform,
                                                           yi,
                                                           self.lmbda)) / len(self.X) / n_sample

                    # gradient of log posterior
                    grad_val = grad_log_posterior(self.system,
                                                  self.transform,
                                                  yi, self.X[sample_index],
                                                  self.prior_mean,
                                                  self.lmbda,
                                                  self.alpha, self.beta, N=len(self.X))

                    # ignore value for unstable parameter samples
                    # if not all(np.isnan(grad_val)):
                    #     if np.nanmax(np.abs(grad_val)) < 1000:
                    gradient += np.nan_to_num(grad_val) / n_sample

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    def fit_posterior_EM(self, n_sample_sgd=1, n_sample_hypers=32, n_sample_evidence=512, patience=3):

        # estimate model evidence
        print("Computing model evidence...")
        self.estimate_evidence(n_sample=n_sample_evidence)
        print("Log evidence: {:.3f}".format(self.log_evidence))

        # optimize evidence
        previdence = np.copy(self.log_evidence)
        fails = 0
        while fails < patience:

            # update prior and measurement precision estimate
            print("Updating hyperparameters...")
            self.update_hypers(n_sample=n_sample_hypers)

            # optimize parameter posterior
            print("Updating posterior...")
            self.fit_posterior(n_sample_sgd)

            # update evidence
            print("Computing model evidence...")
            self.estimate_evidence(n_sample=n_sample_evidence)
            print("Log evidence: {:.3f}".format(self.log_evidence))

            # check convergence
            if self.log_evidence > previdence:
                fails = 0
                previdence = np.copy(self.log_evidence)
            else:
                fails += 1

    # EM algorithm to update hyperparameters
    def update_hypers(self, n_sample=512):
        # init yCOV
        yCOV = 0.

        # current parameter guess
        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.transform, y, self.lmbda)

        # loop over each sample in dataset
        N = np.zeros(len(self.sys_vars))
        for zi in tqdm(z):
            for Xi in self.X:
                # predict condition
                tf, x = Xi

                # integrate ODE
                t_hat = model(self.system, Xi, zi)

                # Determine error
                Y_error = np.nan_to_num(np.nan_to_num(t_hat) - x[-1])

                # sum of measurement error
                yCOV += Y_error ** 2 / n_sample

                # number of measurements
                N += np.array(x[-1] != 0, int) / n_sample

        # update beta
        self.beta = N / (yCOV + 1e-4)
        # print("beta:", self.beta)

        # update alpha
        self.alpha = 1. / np.mean((z - self.prior_mean) ** 2, 0)
        # self.alpha = self.d * n_sample / np.sum((z - self.prior_mean) ** 2)
        # print("alpha:", self.alpha)

    def estimate_evidence(self, n_sample=512):

        # sample from prior
        y = np.random.randn(n_sample, self.d)
        z = self.prior_mean + np.sqrt(1. / self.alpha) * y

        # init log likelihoods
        log_likelihoods = []

        # for each parameter sample
        for zi in tqdm(z):

            # for each datapoint
            log_likelihood_val = 0.
            for Xi in self.X:
                # predict condition
                tf, x = Xi

                # integrate ODE
                t_hat = np.nan_to_num(model(self.system, Xi, zi))

                # Compute likelihood
                log_likelihood_val += .5 * np.sum(np.log(self.beta)) - .5 * len(self.sys_vars) * np.log(2 * np.pi)
                log_likelihood_val += -.5 * np.nansum(self.beta * (x[1:] - t_hat) ** 2)

            # append log_likelihood for parameter sample
            log_likelihoods.append(log_likelihood_val)

        # compute log evidence
        self.log_evidence = logsumexp(log_likelihoods) - np.log(n_sample)

    def predict_point(self, x0, t_eval):

        z = T(self.transform, np.zeros(self.d), self.lmbda)

        return odeint(self.system, x0, t_eval, z)

    def predict_sample(self, x0, t_eval, n_sample=21):

        # sample noise
        y = np.random.randn(n_sample, self.d)

        # posterior predictive
        predictions = []
        for yi in y:
            zi = T(self.transform, yi, self.lmbda)
            predictions.append(odeint(self.system, x0, t_eval, zi))

        return np.stack(predictions)

    # generate samples from posterior
    def sample_params(self, n_sample=100):

        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.transform, y, self.lmbda)

        return np.array(z, float)

    def param_df(self, n_sample=1000):
        # get mean of transformed parameter value
        mean = T(self.transform, np.zeros(self.d), self.lmbda)

        # standard deviation
        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.transform, y, self.lmbda)
        stdv = np.std(z, 0)

        # save parameter dataframe
        df_param = pd.DataFrame()
        df_param['mean'] = mean
        df_param['stdv'] = stdv

        return df_param
