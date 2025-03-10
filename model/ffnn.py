from .utilities import *

from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import random, jit, jacfwd, jacrev, grad, vmap
from jax.nn import tanh, sigmoid, relu
from jax.lax import dynamic_slice
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
    return jnp.nan_to_num(t_hat[-1])


# gradient of model w.r.t. latent variables z
@partial(jit, static_argnums=(0,))
def grad_model(system, Xi, z):
    return jacrev(model, 2)(system, Xi, z)


# outer product for approximating Hessian
@jit
def outer(beta, G):
    return jnp.einsum('k,ki,kj->ij', beta, G, G)


# reshape parameter vector into matrix/vectors
@partial(jit, static_argnums=(0,))
def reshape(shapes, params):
    return [jnp.reshape(dynamic_slice(params, (k1,), (k2 - k1,)), shape) for k1, k2, shape in
            zip(shapes[0], shapes[0][1:], shapes[1])]


# invertible, differentiable function to map noise to model parameters
@partial(jit, static_argnums=(0,))
def T(shapes, y, lmbda):
    # weights and biases of nn
    W1, b1, W4, b4 = reshape(shapes, lmbda)

    # hidden layer
    h = tanh(jnp.einsum('ij,...j->...i', W1, y) + b1)

    # hidden layer
    # h = tanh(jnp.einsum('ij,...j->...i', jnp.exp2(W2), h) + b2)

    # hidden layer
    # h = tanh(jnp.einsum('ij,...j->...i', jnp.exp2(W3), h) + b3)

    # output
    z = jnp.einsum('ij,...j->...i', W4, h) + b4

    return z


@partial(jit, static_argnums=(0,))
def batch_T(shapes, y_batch, lmbda):
    return vmap(T, (None, 0, None))(shapes, y_batch, lmbda)


# Jacobian of nn
@partial(jit, static_argnums=(0,))
def Jac_T(shapes, y, lmbda):
    return jacfwd(T, 1)(shapes, y, lmbda)


@partial(jit, static_argnums=(0,))
def log_det(shapes, y, lmbda):
    # return jnp.log(jnp.abs(jnp.linalg.det(Jac_T(shapes, y, lmbda))))
    return jnp.sum(jnp.log(jnp.abs(jnp.linalg.eigvals(Jac_T(shapes, y, lmbda)))))


# gradient of entropy of approximating distribution w.r.t. lmbda
@partial(jit, static_argnums=(0,))
def grad_log_det(shapes, y, lmbda):
    return jacrev(log_det, -1)(shapes, y, lmbda)


# log likelihood of standard Gaussian
@jit
def log_py(y):
    return -jnp.dot(y, y)/2. - len(y)/2 * jnp.log(2.*np.pi)


# evaluate log prior
@partial(jit, static_argnums=(0,))
def log_prior(shapes, z_prior, y, lmbda, alpha):
    # map to sample from posterior
    z = T(shapes, y, lmbda)

    # prior
    lp = -len(y)/2. * jnp.log(2.*np.pi) + jnp.sum(jnp.log(alpha)) / 2.
    lp -= jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of log prior
@partial(jit, static_argnums=(0,))
def grad_log_prior(shapes, z_prior, y, lmbda, alpha):
    return jacrev(log_prior, 3)(shapes, z_prior, y, lmbda, alpha)


# evaluate log likelihood
@partial(jit, static_argnums=(0, 1))
def log_likelihood(system, shapes, y, Xi, lmbda, beta):
    # map to sample from posterior
    z = T(shapes, y, lmbda)

    # unpack condition
    tf, x = Xi

    # likelihood
    lp = -len(beta)/2. * jnp.log(2.*jnp.pi) + jnp.sum(jnp.log(beta))/2.
    lp -= jnp.nansum(beta * (x[-1] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0, 1))
def grad_log_likelihood(system, shapes, y, x, lmbda, beta):
    return jacrev(log_likelihood, 4)(system, shapes, y, x, lmbda, beta)


# evaluate log posterior
@partial(jit, static_argnums=(0, 1))
def log_posterior(system, shapes, y, Xi, z_prior, lmbda, alpha, beta, N):
    # map to sample from posterior
    z = T(shapes, y, lmbda)

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2. / N

    # unpack condition
    tf, x = Xi

    # likelihood
    lp += jnp.nansum(beta * (x[-1] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0, 1))
def grad_log_posterior(system, shapes, y, Xi, z_prior, lmbda, alpha, beta, N):
    return jacrev(log_posterior, 5)(system, shapes, y, Xi, z_prior, lmbda, alpha, beta, N)


# evaluate log posterior
@partial(jit, static_argnums=(0,))
def log_posterior_z(system, z, Xi, z_prior, alpha, beta, N):
    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2. / N

    # unpack condition
    tf, x = Xi

    # likelihood
    lp += jnp.nansum(beta * (x[-1] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_log_posterior_z(system, z, Xi, z_prior, alpha, beta, N):
    return jacrev(log_posterior_z, 1)(system, z, Xi, z_prior, alpha, beta, N)


# evaluate log prior
@jit
def log_prior_z(z_prior, z, alpha):
    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of log prior
@jit
def grad_log_prior_z(z_prior, z, alpha):
    return jacrev(log_prior_z, 1)(z_prior, z, alpha)


# evaluate log likelihood
@partial(jit, static_argnums=(0,))
def log_likelihood_z(system, z, Xi, beta):
    # unpack condition
    tf, x = Xi

    # likelihood
    lp = jnp.nansum(beta * (x[-1] - model(system, Xi, z)) ** 2) / 2.

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_log_likelihood_z(system, z, x, beta):
    return jacrev(log_likelihood_z, 1)(system, z, x, beta)


class ODE:
    def __init__(self,
                 system,
                 transform,
                 dataframe,
                 sys_vars,
                 prior_mean,
                 n_h=8,
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

        # problem dimension
        self.d = len(self.prior_mean)

        # prior and measurement precision
        self.alpha = alpha * np.ones(self.d)
        self.beta = beta * np.ones(len(sys_vars))

        # dimension of neural network (at least as large as number of ODE parameters)
        self.h = np.max([n_h, self.d])

        # init params of neural network
        self.shapes = None
        self.lmbda = None
        self.init_params()

    # reinitialize parameters
    def init_params(self,):

        # determine shapes of weights/biases = [Wih, bih, Who, bho]
        # shapes = ((self.h, self.d), (self.h,), (self.d, self.h), (self.d,))
        shapes = ((self.h, self.d), (self.h,),
                  (self.d, self.h), (self.d,))
        k_params = []
        n_params = 0
        for shape in shapes:
            k_params.append(n_params)
            n_params += np.prod(shape)
        k_params.append(n_params)
        self.shapes = tuple((tuple(k_params), tuple(shapes)))

        # initialize parameters
        self.lmbda = np.zeros(n_params)
        for k1, k2, shape in zip(k_params, k_params[1:], shapes):
            if len(shape) > 1:
                stdv = 2. / np.sqrt(2 * shape[-1])
                # self.lmbda[k1:k2] = np.log2(np.random.uniform(0., stdv, k2 - k1, ))
                self.lmbda[k1:k2] = np.random.uniform(0., stdv, k2 - k1, )
            else:
                self.lmbda[k1:k2] = np.random.randn(k2 - k1,) / np.sqrt(2.)

    # evidence lower bound
    def warmup(self, n_sample=21):

        # sample from posterior
        y = np.random.randn(n_sample, self.d)

        # # prior
        # lp = -len(y)/2. * jnp.log(2.*np.pi) + jnp.sum(jnp.log(alpha)) / 2.
        # lp -= jnp.sum(alpha * (z - z_prior) ** 2) / 2.

        # log likelihood of prior
        WARMUP = -self.d/2. * jnp.log(2.*np.pi) + jnp.sum(jnp.log(self.alpha)) / 2.
        for yi in y:

            # prior
            WARMUP -= np.nan_to_num(log_prior(self.shapes,
                                              self.prior_mean,
                                              yi,
                                              self.lmbda,
                                              self.alpha)) / n_sample

        # return NLP
        return WARMUP

    # evidence lower bound
    def elbo(self, n_sample=21):

        # sample from posterior
        y = np.random.randn(n_sample, self.d)

        # entropy
        self.ELBO = 0.
        for yi in y:

            # entropy
            self.ELBO -= log_py(yi) / n_sample
            self.ELBO += np.nan_to_num(log_det(self.shapes, yi, self.lmbda)) / n_sample

            # prior
            self.ELBO += np.nan_to_num(log_prior(self.shapes,
                                                 self.prior_mean,
                                                 yi,
                                                 self.lmbda,
                                                 self.alpha)) / n_sample

            # likelihood
            for Xi in self.X:
                self.ELBO += np.nan_to_num(log_likelihood(self.system,
                                                          self.shapes,
                                                          yi,
                                                          Xi,
                                                          self.lmbda,
                                                          self.beta)) / n_sample

        # return NLP
        return self.ELBO


    def fit_prior(self, n_sample=1, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, max_epochs=1000, tol=1e-3,
                      patience=1):
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
            f.append(self.warmup(n_sample=n_sample))
            if len(f) > 10:

                # # determine slope of elbo over time
                # slope = check_convergence(f[-10:])
                #
                # # if performance is not improving, decrease lr
                # if slope > 0:
                #     lr *= .9
                #     print("decrease lr to {:.3e}".format(lr))
                #
                # # try to accelerate
                # if slope < -1e-2:
                #     lr = np.min([1.1 * lr, .01])
                #     print("set lr to {:.3e}".format(lr))
                #
                # # convergence criteria
                # if lr < tol:
                #     # increase pass count
                #     passes += 1
                #     # increase learning rate
                #     lr *= 2.
                #
                # # return parameters after passing convergence criteria enough times
                # if passes >= patience:
                #     return
                if f[-1] < .001:
                    return

                print("Epoch {:.0f}, Prior loss: {:.3f}".format(epoch, f[-1]))
            else:
                print("Epoch {:.0f}, Prior loss: {:.3f}".format(epoch, f[-1]))

            # sgd over each sample
            np.random.shuffle(order)
            for sample_index in order:

                # sample noise
                y = np.random.randn(n_sample, self.d)

                # stochastic evaluation of gradient
                gradient = 0.
                for yi in y:

                    # gradient of log prior (shapes, z_prior, y, lmbda, alpha)
                    grad_val = grad_log_prior(self.shapes,
                                              self.prior_mean,
                                              yi,
                                              self.lmbda,
                                              self.alpha)

                    # ignore value for unstable parameter samples
                    # if not all(np.isnan(grad_val)):
                    #     if np.nanmax(np.abs(grad_val)) < 1000:
                    # gradient += np.where(np.abs(grad_val) < 1e6, grad_val, 0.) / n_sample
                    gradient += np.nan_to_num(-grad_val) / n_sample

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)


    def fit_posterior(self, n_sample=1, lr=1e-3, beta1=0.5, beta2=0.5, epsilon=1e-8, max_epochs=1000, tol=1e-4,
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
                slope = -check_convergence(f[-10:])

                # if performance is not improving, decrease lr
                if slope > 0:
                    lr *= .95
                    print("decrease lr to {:.3e}".format(lr))

                # try to accelerate
                if slope < -1e-2:
                    lr = np.min([1.05 * lr, .001])
                    print("set lr to {:.3e}".format(lr))

                # convergence criteria
                if lr < tol:
                    # increase pass count
                    passes += 1
                    # increase learning rate
                    lr *= 2.

                # return parameters after passing convergence criteria enough times
                if passes >= patience:
                    return f

                print("Epoch {:.0f}, ELBO: {:.3f}, Slope: {:.3f}".format(epoch, f[-1], -slope))
            else:
                print("Epoch {:.0f}, ELBO: {:.3f}".format(epoch, f[-1]))

            # sgd over each sample
            np.random.shuffle(order)
            for sample_index in order:

                # sample noise
                y = np.random.randn(n_sample, self.d)

                # stochastic evaluation of gradient
                gradient = 0.
                for yi in y:
                    # gradient of entropy
                    gradient -= np.nan_to_num(grad_log_det(self.shapes, yi, self.lmbda)) / len(self.X) / n_sample

                    # gradient of log posterior
                    grad_val = grad_log_posterior(self.system,
                                                  self.shapes,
                                                  yi,
                                                  self.X[sample_index],
                                                  self.prior_mean,
                                                  self.lmbda,
                                                  self.alpha, self.beta, N=len(self.X))

                    # ignore value for unstable parameter samples
                    # if not all(np.isnan(grad_val)):
                    #     if np.nanmax(np.abs(grad_val)) < 1000:
                    gradient += np.where(np.abs(grad_val) < 1e6, grad_val, 0.) / n_sample
                    # gradient += np.nan_to_num(grad_val) / n_sample

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        return f

    def callback(self, x):
        print("Loss: {:.3f}".format(self.NLP))

    def fit_posterior_EM(self, n_sample_sgd=1, n_sample_hypers=1000, trials=3, patience=3, lr=1e-3, max_iterations=100):

        # initialize parameters of nn
        print("Warmup...")
        self.fit_prior(n_sample_sgd, lr=lr)

        # optimize parameter posterior
        print("Updating posterior...")
        f = self.fit_posterior(n_sample_sgd, lr=lr)

        # init evidence, fail count, iteration count
        self.approx_evidence()
        previdence = np.copy(self.log_evidence)
        passes = 0
        t = 0
        while passes < patience and t < max_iterations:

            # update iteration count
            t += 1

            # update prior and measurement precision estimate
            print("Updating hyperparameters...")
            self.update_hypers(n_sample=n_sample_hypers)

            # optimize parameter posterior
            print("Updating posterior...")
            f = self.fit_posterior(n_sample_sgd, lr=lr)

            # update evidence
            print("Computing model evidence...")
            # self.estimate_evidence(n_sample=n_sample_evidence)
            self.approx_evidence()

            # check convergence
            if np.mean(f)-np.std(f) < previdence < np.mean(f)+np.std(f):
                passes += 1
                previdence = np.copy(self.log_evidence)

    # EM algorithm to update hyperparameters
    def update_hypers(self, n_sample=1000):
        # init yCOV
        yCOV = 0.

        # current parameter guess
        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.shapes, y, self.lmbda)

        # loop over each sample in dataset
        N = np.zeros(len(self.sys_vars))
        for zi in tqdm(z):
            for Xi in self.X:
                # predict condition
                tf, x = Xi

                # integrate ODE
                t_hat = model(self.system, Xi, zi)

                # Determine error
                Y_error = np.nan_to_num(t_hat - x[-1])

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

    def estimate_evidence(self, n_sample=1000, n_trials=1):

        # compute evidence several times to reduce noise
        log_evidence_vals = []
        for trial in range(n_trials):

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
                    t_hat = model(self.system, Xi, zi)

                    # Compute likelihood
                    log_likelihood_val += .5 * np.sum(np.log(self.beta)) - .5 * len(self.sys_vars) * np.log(2 * np.pi)
                    log_likelihood_val -= .5 * np.sum(self.beta * (x[-1] - t_hat) ** 2)

                # append log_likelihood for parameter sample
                log_likelihoods.append(log_likelihood_val)

            # compute log evidence
            log_evidence_vals.append(logsumexp(log_likelihoods) - np.log(n_sample))

        # save average evidence
        self.log_evidence = np.mean(log_evidence_vals)
        print("Log evidence: {:.3f} +/- {:.3f}".format(self.log_evidence, np.std(log_evidence_vals)))

    def approx_evidence(self):

        # posterior entropy
        log_evidence = log_det(self.shapes, np.zeros(self.d), self.lmbda) - log_py(np.zeros(self.d))

        # deviation from prior
        # prior
        log_evidence += log_prior(self.shapes,
                                  self.prior_mean,
                                  np.zeros(self.d),
                                  self.lmbda,
                                  self.alpha)

        # likelihood
        for Xi in self.X:
            log_evidence += log_likelihood(self.system, self.shapes, np.zeros(self.d), Xi, self.lmbda, self.beta)

        # save average evidence
        self.log_evidence = log_evidence
        print("Log evidence: {:.3f}".format(log_evidence))

    def predict_point(self, x0, t_eval):

        z = T(self.shapes, np.zeros(self.d), self.lmbda)

        return odeint(self.system, x0, t_eval, z)

    def predict_sample(self, x0, t_eval, n_sample=21):

        # sample noise
        y = np.random.randn(n_sample, self.d)

        # posterior predictive
        predictions = []
        for yi in y:
            zi = T(self.shapes, yi, self.lmbda)
            predictions.append(odeint(self.system, x0, t_eval, zi))

        return np.stack(predictions)

    def predict_prior(self, x0, t_eval, n_sample=21):

        # sample from prior
        y = np.random.randn(n_sample, self.d)
        z = self.prior_mean + np.sqrt(1. / self.alpha) * y

        # posterior predictive
        predictions = []
        for zi in z:
            predictions.append(odeint(self.system, x0, t_eval, zi))

        return np.stack(predictions)

    # generate samples from posterior
    def sample_params(self, n_sample=100):

        y = np.random.randn(n_sample, self.d)
        z = vmap(self.transform)(batch_T(self.shapes, y, self.lmbda))

        return np.array(z, float)

    def param_df(self, n_sample=1000):
        # get mean of transformed parameter value
        mean = self.transform(T(self.shapes, np.zeros(self.d), self.lmbda))

        # standard deviation
        y = np.random.randn(n_sample, self.d)
        z = vmap(self.transform)(batch_T(self.shapes, y, self.lmbda))
        stdv = np.std(z, 0)

        # save parameter dataframe
        df_param = pd.DataFrame()
        df_param['mean'] = mean
        df_param['stdv'] = stdv

        return df_param
