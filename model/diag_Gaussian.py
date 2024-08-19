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
def model(system, tf, x, z):
    # unpack data and integration time
    t_span = jnp.array([0., tf])

    # integrate ODE
    t_hat = odeint(system, jnp.array(x[0]), t_span, z,
                   rtol=1.4e-8, atol=1.4e-8, mxstep=10000, hmax=jnp.inf)

    # t_hat is the model estimate of observed variable t
    return t_hat[-1]


# gradient of model w.r.t. latent variables z
@partial(jit, static_argnums=(0,))
def grad_model(system, tf, x, z):
    return jacrev(model, 3)(system, tf, x, z)


# outer product for approximating Hessian
@jit
def outer(beta, G):
    return jnp.einsum('k,ki,kj->ij', beta, G, G)


# invertible, differentiable function to map noise to model parameters
@jit
def T(y, lmbda):
    # weights and biases of nn
    mu, log_s = lmbda.at[:len(lmbda) // 2].get(), lmbda.at[len(lmbda) // 2:].get()

    # convert to z
    z = mu + jnp.exp(log_s) * y

    return z


@jit
def batch_T(y_batch, lmbda):
    return vmap(T, (0, None))(y_batch, lmbda)


@jit
def log_abs_det(lmbda):
    log_s = lmbda.at[(len(lmbda) // 2):].get()
    return jnp.sum(log_s)


# evaluate negative log prior
@jit
def neg_log_prior(z_prior, z, alpha):
    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of negative log prior
@jit
def grad_neg_log_prior(z_prior, z, alpha):
    return jacrev(neg_log_prior, 1)(z_prior, z, alpha)


# evaluate negative log likelihood
@partial(jit, static_argnums=(0,))
def neg_log_likelihood(system, z, tf, x, nu2, sigma2):

    # model prediction
    y = jnp.nan_to_num(model(system, tf, x, z))

    # residuals
    res = jnp.nan_to_num(x[-1] - y)

    # predicted variance
    var = nu2 + sigma2 * y ** 2

    # likelihood
    lp = jnp.sum(res ** 2 / var / 2. + jnp.log(var) / 2.)

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_neg_log_likelihood(system, z, tf, x, nu2, sigma2):
    return jacrev(neg_log_likelihood, 1)(system, z, tf, x, nu2, sigma2)


# evaluate negative log prior
@jit
def neg_log_prior_lmbda(z_prior, y, alpha, lmbda):
    # sample z
    z = T(y, lmbda)

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of negative log prior
@jit
def grad_neg_log_prior_lmbda(z_prior, y, alpha, lmbda):
    return jacrev(neg_log_prior_lmbda, -1)(z_prior, y, alpha, lmbda)


# evaluate negative log likelihood
@partial(jit, static_argnums=(0,))
def neg_log_likelihood_lmbda(system, y, tf, x, nu2, sigma2, lmbda):
    # sample z
    z = T(y, lmbda)

    # likelihood
    lp = neg_log_likelihood(system, z, tf, x, nu2, sigma2)

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_neg_log_likelihood_lmbda(system, y, tf, x, nu2, sigma2, lmbda):
    return jacrev(neg_log_likelihood_lmbda, -1)(system, y, tf, x, nu2, sigma2, lmbda)


class ODE:
    def __init__(self,
                 system,
                 transform,
                 dataframe,
                 sys_vars,
                 prior_mean,
                 alpha=1., nu2=.001, sigma2=.01):

        # system of differential equations
        self.system = system

        # function to reshape and transform parameters
        self.transform = transform

        # processed data
        self.sys_vars = sys_vars
        self.T, self.X, self.N = process_df(dataframe, sys_vars)

        # scale data based on max measured value
        self.X_scale = 1.  # np.max(self.X, 0)[-1]
        self.X /= self.X_scale

        # parameter prior
        self.prior_mean = prior_mean

        # problem dimension
        self.d = len(self.prior_mean)

        # prior and measurement precision
        self.alpha = alpha * np.ones(self.d)
        self.nu2 = nu2 * np.ones(len(sys_vars))
        self.sigma2 = sigma2 * np.ones(len(sys_vars))

        # initial parameter guess
        self.z = np.zeros(self.d)  # np.random.randn(self.d) / 10.
        self.lmbda = jnp.append(self.z, jnp.log(jnp.ones(self.d)))

    # reinitialize parameters
    def init_params(self, ):

        # initial parameter guess
        self.z = np.random.randn(self.d) / 10.
        self.lmbda = jnp.append(self.z, jnp.log(jnp.ones(self.d) / 10.))

    # negative log posterior
    def nlp(self, z):

        # prior
        self.NLP = neg_log_prior(self.prior_mean, z, self.alpha)

        # likelihood
        for tf, x in zip(self.T, self.X):
            self.NLP += neg_log_likelihood(self.system, z, tf, x, self.nu2, self.sigma2)

        # return NLP
        return self.NLP

    # gradient of negative log posterior
    def grad_nlp(self, z):

        # prior
        grad_NLP = grad_neg_log_prior(self.prior_mean, z, self.alpha)

        # likelihood
        for tf, x in zip(self.T, self.X):
            # gradient of negative log likelihood
            gradients = np.nan_to_num(grad_neg_log_likelihood(self.system, z, tf, x, self.nu2, self.sigma2))

            # ignore exploding gradients
            gradients = np.where(np.abs(gradients) < 1e6, gradients, 0.)

            # add to grad of NLP
            grad_NLP += gradients

        # return NLP
        return grad_NLP

    # gradient of negative log posterior
    def hess_nlp(self, z):

        # prior
        hess_NLP = np.diag(self.alpha)

        # likelihood
        for tf, x in zip(self.T, self.X):
            # Jacobian of model
            Gi = grad_model(self.system, tf, x, z)

            # outer product approximation of Hessian
            hess_NLP += outer(self.beta, Gi)

        # return NLP
        return hess_NLP

    def fit_posterior(self, n_sample=3, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, max_epochs=100000, tol=1e-3,
                      patience=5):
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

        """
        m = np.zeros_like(self.lmbda)
        v = np.zeros_like(self.lmbda)
        t = 0
        epoch = 0
        passes = 0
        fails = 0

        # order of samples
        N = len(self.X)
        order = np.arange(N)

        # save best parameters
        best_params = np.copy(self.lmbda)

        # initialize function evaluations
        f = []

        while epoch <= max_epochs and passes < patience:

            if epoch % 10 == 0:

                # check convergence
                f.append(self.approx_evidence())
                convergence = (f[-1] - np.mean(f[-10:])) / np.abs(np.mean(f[-10:]))

                # determine slope of elbo over time
                if len(f) > 2:
                    slope = check_convergence(f[-10:])
                else:
                    slope = 1.

                # check tolerance
                if abs(slope) < tol and epoch > 100:
                    passes += 1
                    print(f"pass {passes}")
                else:
                    passes = 0

                # save parameters if improved
                if f[-1] >= np.max(f):
                    best_params = np.copy(self.lmbda)

                # if slope is negative and not improving, add to fail count
                if slope < 0 and f[-1] < f[-2] and epoch > 100:
                    fails += 1
                    print(f"fail {fails}")
                else:
                    fails = 0

                # if fails exceeds patience, return best parameters
                if fails == patience:
                    self.lmbda = jnp.array(best_params)
                    self.z = self.lmbda.at[:self.d].get()
                    return f

                print("Epoch {:.0f}, ELBO: {:.3f}, Slope: {:.3f}, Convergence: {:.5f}".format(epoch, f[-1], slope,
                                                                                              convergence))
            epoch += 1

            # # gradient of entropy of approximate distribution w.r.t log_s
            # gradient = np.append(np.zeros(self.d), -np.ones(self.d))
            #
            # # sample parameters
            # y = np.random.randn(n_sample, self.d)
            #
            # # gradient of negative log posterior
            # for yi in y:
            #
            #     # prior
            #     gradient += grad_neg_log_prior_lmbda(self.prior_mean, yi, self.alpha, self.lmbda) / n_sample
            #
            #     # loop over each sample
            #     for tf, x in zip(self.T, self.X):
            #         # gradient of negative log likelihood
            #         grad_nll = np.nan_to_num(grad_neg_log_likelihood_lmbda(self.system,
            #                                                                yi,
            #                                                                tf,
            #                                                                x,
            #                                                                self.nu2,
            #                                                                self.sigma2,
            #                                                                self.lmbda))
            #
            #         # ignore exploding gradients
            #         gradient += np.where(np.abs(grad_nll) < 100, grad_nll, 0.) / n_sample
            #
            # # normalize gradient
            # gradient /= np.linalg.norm(gradient)
            #
            # # moment estimation
            # m = beta1 * m + (1 - beta1) * gradient
            # v = beta2 * v + (1 - beta2) * (gradient ** 2)
            #
            # # adjust moments based on number of iterations
            # t += 1
            # m_hat = m / (1 - beta1 ** t)
            # v_hat = v / (1 - beta2 ** t)
            #
            # # take step
            # self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)  # / np.sqrt(t)
            # self.z = self.lmbda.at[:self.d].get()

            # update at each sample
            np.random.shuffle(order)
            for sample_index in order:

                # gradient of entropy of approximate distribution w.r.t log_s
                gradient = np.append(np.zeros(self.d), -np.ones(self.d)) / N

                # sample parameters
                y = np.random.randn(n_sample, self.d)

                # gradient of negative log posterior
                for yi in y:

                    # prior
                    gradient += grad_neg_log_prior_lmbda(self.prior_mean,
                                                         yi,
                                                         self.alpha,
                                                         self.lmbda) / N / n_sample

                    # gradient of negative log likelihood
                    grad_nll = np.nan_to_num(grad_neg_log_likelihood_lmbda(self.system,
                                                                           yi,
                                                                           self.T[sample_index],
                                                                           self.X[sample_index],
                                                                           self.nu2, self.sigma2,
                                                                           self.lmbda)) / n_sample

                    # ignore exploding gradients
                    gradient += np.where(np.abs(grad_nll) < 1000, grad_nll, 0.)

                # normalize gradient: Does not work well in this case, magnitude of gradient per sample matters!
                # gradient /= np.linalg.norm(gradient)

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)  # / np.sqrt(t)
                self.z = self.lmbda.at[:self.d].get()

        return f

    def callback(self, x):
        print("Loss: {:.3f}".format(self.NLP))

    def fit_posterior_EM(self, n_sample_sgd=3, n_sample_hypers=500, patience=3, lr=1e-3, tol=1e-3, max_iterations=100):

        # optimize parameter posterior
        print("Updating posterior...")
        f = self.fit_posterior(n_sample_sgd, lr=lr, tol=tol)

        # init evidence, fail count, iteration count
        previdence = np.copy(f[-1])
        fails = 0
        t = 0
        while fails < patience and t < max_iterations:

            # update iteration count
            t += 1

            # update prior and measurement precision estimate
            print("Updating hyperparameters...")
            self.update_hypers(n_sample=n_sample_hypers)

            # optimize parameter posterior
            print("Updating posterior...")
            f = self.fit_posterior(n_sample_sgd, lr=lr, tol=tol)

            # check convergence
            if self.log_evidence <= previdence:
                fails += 1
            previdence = np.copy(self.log_evidence)

    # EM algorithm to update hyperparameters
    def update_hypers(self, n_sample=100):
        # create dictionaries of estimated/empirical moments for each output
        Z = {}
        Y2 = {}
        for j in range(len(self.N)):
            Z[j] = []
            Y2[j] = []

        # approximate expected error
        y = np.random.randn(n_sample, self.d)
        z = batch_T(y, self.lmbda)
        for zi in tqdm(z):

            # loop over each sample in dataset
            for tf, x in zip(self.T, self.X):
                # integrate ODE
                t_hat = model(self.system, tf, x, zi)

                # Determine error
                y_error = np.nan_to_num(x[-1] - np.nan_to_num(t_hat))

                # estimate noise
                for j, (y_j, f_j, e_j) in enumerate(zip(x[-1], t_hat, y_error)):
                    if y_j > 0:
                        Z[j].append((y_j - f_j) ** 2)
                        Y2[j].append(y_j ** 2)

        # solve for noise parameters
        for j in range(len(self.N)):
            y2 = np.array(Y2[j])
            z = np.array(Z[j])
            B = np.vstack((np.ones_like(y2), y2)).T
            a = (np.linalg.inv(B.T @ B) @ B.T) @ z
            self.nu2[j] = np.max([a[0], 1e-4])
            self.sigma2[j] = np.max([a[1], 1e-4])

        # update alpha
        var = jnp.exp(self.lmbda.at[self.d:].get()) ** 2
        self.alpha = 1. / ((self.z - self.prior_mean) ** 2 + var + 1e-4)

    def approx_evidence(self):

        # posterior entropy
        self.log_evidence = log_abs_det(self.lmbda) - self.nlp(self.z)

        return self.log_evidence

    def estimate_evidence(self, n_sample=1000, n_trials=3):

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
                for tf, x in zip(self.T, self.X):
                    # integrate ODE
                    t_hat = model(self.system, tf, x, zi)

                    # Compute likelihood
                    log_likelihood_val += .5 * np.sum(np.log(self.beta)) - .5 * len(self.beta) * np.log(2 * np.pi)
                    log_likelihood_val -= .5 * np.nansum(self.beta * (x[-1] - t_hat) ** 2)

                # append log_likelihood for parameter sample
                log_likelihoods.append(log_likelihood_val)

            # compute log evidence
            log_evidence_vals.append(logsumexp(log_likelihoods) - np.log(n_sample))

        # save average evidence
        self.log_evidence = np.mean(log_evidence_vals)
        print("Log evidence: {:.3f} +/- {:.3f}".format(self.log_evidence, np.std(log_evidence_vals)))

    def predict_point(self, x0, t_eval):

        z = T(np.zeros(self.d), self.lmbda)

        return self.X_scale * odeint(self.system, x0 / self.X_scale, t_eval, z)

    def predict_sample(self, x0, t_eval, n_sample=21):

        # sample noise
        y = np.random.randn(n_sample, self.d)

        # posterior predictive
        predictions = []
        for yi in y:
            zi = T(yi, self.lmbda)
            predictions.append(self.X_scale * odeint(self.system, x0 / self.X_scale, t_eval, zi))

        return np.stack(predictions)

    def predict_prior(self, x0, t_eval, n_sample=21):

        # sample from prior
        y = np.random.randn(n_sample, self.d)
        z = self.prior_mean + np.sqrt(1. / self.alpha) * y

        # posterior predictive
        predictions = []
        for zi in z:
            predictions.append(self.X_scale * odeint(self.system, x0 / self.X_scale, t_eval, zi))

        return np.stack(predictions)

    # generate samples from posterior
    def sample_params(self, n_sample=100):

        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.transform, y, self.lmbda)

        return np.array(z, float)

    def param_df(self, n_sample=1000):
        # get mean of transformed parameter value
        mean = self.transform(T(np.zeros(self.d), self.lmbda))

        # standard deviation
        y = np.random.randn(n_sample, self.d)
        z = vmap(self.transform)(batch_T(y, self.lmbda))
        stdv = np.std(z, 0)

        # save parameter dataframe
        df_param = pd.DataFrame()
        df_param['mean'] = mean
        df_param['stdv'] = stdv

        return df_param
