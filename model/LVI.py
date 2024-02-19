import os
from scipy.io import savemat, loadmat
import multiprocessing
from functools import partial

# for troubleshooting and animation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import libraries to compute gradients
import jax
from jax import vjp, jacfwd, vmap, pmap, random
from jax.experimental.ode import odeint
from jax.nn import tanh, sigmoid, softmax, relu
from scipy.optimize import minimize

# system of equations
from .glv_system import *

# matrix math
from .linalg import *
from .utilities import *

# set number of processors
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


class MODEL:

    # augmented system for forward sensitivity equations
    def __init__(self):
        self.species = None
        self.mediators = None
        self.a = 1e-4
        self.b = 1e-4
        self.n_devices = None
        self.n_params = None
        self.data_dict = None
        self.dataset = None
        self.params = None
        self.verbose = None
        self.m0 = None
        self.prior = None
        self.n_cr_params = None
        self.shapes = None
        self.k_params = None
        self.Z0 = None
        self.system = None
        self.n_m = None
        self.n_obs = None
        self.Y0 = None
        self.n_s = None

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        return [np.array(np.reshape(params[k1:k2], shape), dtype=np.float32) for k1, k2, shape in
                zip(self.k_params, self.k_params[1:], self.shapes)]

    # compute residual between parameter estimate and prior
    def param_res(self, params):
        # residuals
        res = params - self.prior
        return res

    def fit(self, nlp_tol=None, alpha=1e-3, trials=3, max_iterations=10):

        # initialize parameters over trials
        param_dict = {t: {} for t in range(trials)}
        for trial in range(trials):
            # initialize parameters
            self.init_params(trial)

            # initialize hyperparameters
            self.init_hypers(alpha)

            # scipy minimize works with a numpy vector of parameters
            params = np.concatenate([p.ravel() for p in self.params])

            # estimate parameters using gradient descent
            params = minimize(fun=self.objective,
                              jac=self.jacobian_fwd,
                              hess=self.hessian,
                              x0=params,
                              tol=nlp_tol,
                              method='Newton-CG',
                              callback=self.callback).x

            # save optimized parameter values and associated loss
            param_dict[trial]["NLP"] = self.NLP
            param_dict[trial]["params"] = params

        # pick the best parameter set
        NLPs = [param_dict[trial]["NLP"] for trial in range(trials)]
        best_trial = np.argmin(NLPs)
        print("\nLoading model with NLP: {:.3f}".format(NLPs[best_trial]))
        params = param_dict[best_trial]["params"]
        self.params = self.reshape(params)
        del param_dict

        # #return

        # optimize ELBO w.r.t. lambda (variational parameters) until convergence
        lmbda = np.concatenate((params, np.log2(np.abs(params) / 10.)))
        t = 0
        prelbo = np.inf
        self.posterior_stdv = None
        while t < max_iterations:

            # update hyperparameters
            self.update_hypers()

            # minimize negative elbo
            lmbda = minimize(fun=self.elbo,
                             jac=self.jacobian_elbo,
                             x0=lmbda,
                             tol=nlp_tol,
                             method='BFGS',
                             callback=self.callback_elbo).x
            # lmbda = adam_optimizer(self.elbo, self.jacobian_elbo, lmbda)

            # update params
            self.params = self.reshape(lmbda)
            self.posterior_stdv = np.exp2(lmbda[len(self.prior):])

            # check convergence
            convergence = np.abs(prelbo - self.ELBO) / np.max([1., np.abs(self.ELBO)])

            print("ELBO convergence {:.3f}".format(convergence))
            if np.linalg.norm(convergence) < 1e-2:
                break
            else:
                prelbo = np.copy(self.ELBO)

    def init_hypers(self, alpha):

        # count number of samples
        self.N = np.zeros(self.n_obs)
        yCOV = np.zeros(self.n_obs)

        for n_t, (t_eval, S_batch, M_batch, inputs) in self.data_dict.items():

            # combine species and mediators
            Y_batch = np.concatenate((S_batch, M_batch), -1)

            # loop over the batched outputs
            for Y_measured in Y_batch:

                # count number of observations
                for Y_t in Y_measured[1:, :self.n_obs]:
                    self.N += np.array(Y_t != 0, int)

                # rough initial guess of measurement covariance
                yCOV += np.nansum(Y_measured[1:, :self.n_obs] ** 2, 0)

        # throw warning if any system variables have no measurements
        if np.any(self.N == 0):
            sys_vars = np.array(self.species + self.mediators)
            print("Warning, no measurements for ", sys_vars[self.N == 0])
            yCOV[self.N == 0] = 1.
            self.N[self.N == 0] = 1

        # init output precision
        yCOV = (yCOV + self.b) / self.N
        self.Beta = jnp.diag(1. / yCOV)
        self.BetaInv = jnp.diag(yCOV)

        # initial guess of parameter precision
        self.alpha = alpha
        self.Alpha = alpha * np.ones(self.n_params)

        if self.verbose:
            print("\nTotal measurements: {:.0f}, Number of parameters: {:.0f}, Initial regularization: {:.2e}".format(
                sum(self.N),
                self.n_params,
                self.alpha))

    def update_hypers(self):

        # init yCOV
        yCOV = 0.

        # loop over each sample in dataset
        for n_t, (t_eval, S_batch, M_batch, inputs) in self.data_dict.items():
            # divide into batches
            n_samples = S_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):

                # integrate batch
                S_out, M_out = batchODE(t_eval,
                                        S_batch[batch_inds],
                                        M_batch[batch_inds],
                                        self.params,
                                        inputs[batch_inds],
                                        self.s_cap, self.m_cap)

                # concatenate measured and predicted values
                Y_meas = np.concatenate((S_batch[batch_inds], M_batch[batch_inds]), -1)
                Y_pred = np.concatenate((S_out, M_out), -1)

                # loop over the batched outputs
                for output, y_measured in zip(Y_pred, Y_meas):
                    # Determine error
                    Y_error = np.nan_to_num(np.nan_to_num(output[1:, :self.n_obs]) - y_measured[1:, :self.n_obs])

                    # sum of measurement covariance update
                    yCOV += np.sum(Y_error ** 2, 0)

        # divide by number of observations
        yCOV = (yCOV + self.b) / self.N

        # assume species have same measurement noise variance
        yCOV[:self.n_s] = np.mean(yCOV[:self.n_s])

        # update beta
        self.Beta = np.diag(1. / yCOV)
        self.BetaInv = np.diag(yCOV)

        # update alpha
        if self.posterior_stdv is not None:

            # vector of parameters
            params = np.concatenate([p.ravel() for p in self.params])

            # single alpha value (isotropic Gaussian prior)
            # self.alpha = len(params) / (np.sum(self.param_res(params) ** 2) + np.sum(self.posterior_stdv ** 2) + self.a)
            # self.Alpha = self.alpha * np.ones_like(params)

            # independent priors for each parameter
            self.Alpha = 1. / (self.param_res(params) ** 2 + self.posterior_stdv ** 2 + self.a)

    def objective(self, params):

        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha * self.param_res(params) ** 2) / 2.

        # compute residuals
        self.RES = 0.

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # loop over each sample in dataset
        # for treatment, t_eval, Y_measured, inputs in self.dataset:
        for n_t, (t_eval, S_batch, M_batch, inputs) in self.data_dict.items():
            # divide into batches
            n_samples = S_batch.shape[0]
            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):

                # integrate batch
                S_out, M_out = batchODE(t_eval,
                                        S_batch[batch_inds],
                                        M_batch[batch_inds],
                                        params,
                                        inputs[batch_inds],
                                        self.s_cap, self.m_cap)

                # concatenate measured and predicted values
                Y_meas = np.concatenate((S_batch[batch_inds], M_batch[batch_inds]), -1)
                Y_pred = np.concatenate((S_out, M_out), -1)

                # loop over the batched outputs
                for output, y_measured in zip(Y_pred, Y_meas):
                    # Determine error
                    Y_error = np.nan_to_num(np.nan_to_num(output[1:, :self.n_obs]) - y_measured[1:, :self.n_obs])

                    # Determine SSE and gradient of SSE
                    self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error) / 2.
                    self.RES += np.sum(Y_error) / sum(self.N)

        # return NLP
        return self.NLP

    def jacobian_fwd(self, params):

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha * self.param_res(params)

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # loop over each sample in dataset
        for n_t, (t_eval, S_batch, M_batch, inputs) in self.data_dict.items():
            # divide into batches
            n_samples = S_batch.shape[0]

            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):

                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_sb, out_mb, *Z_b = batchODEZ(t_eval,
                                                 S_batch[batch_inds],
                                                 M_batch[batch_inds],
                                                 self.Z0,
                                                 params,
                                                 inputs[batch_inds],
                                                 self.s_cap, self.m_cap)

                # concatenate S and M measurements
                Y_batch = np.concatenate((S_batch[batch_inds], M_batch[batch_inds]), -1)

                # concatenate species and mediator outputs
                out_b = np.concatenate((out_sb, out_mb), -1)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, Z, Y_measured, input in zip(np.array(out_b),
                                                        np.array(Z_b),
                                                        Y_batch,
                                                        inputs[batch_inds]):
                    # stack gradient matrices
                    G = np.nan_to_num(Z)

                    # ignore initial condition and NaNs
                    G = np.einsum("tk,tki->tki",
                                  np.array(~np.isnan(Y_measured[1:, :self.n_obs]), int),
                                  G[1:, :self.n_obs])

                    # determine error
                    Y_error = np.nan_to_num(np.nan_to_num(output[1:, :self.n_obs]) - Y_measured[1:, :self.n_obs])

                    # sum over time and outputs to get gradient w.r.t params
                    grad_NLP += eval_grad_NLP(Y_error, self.Beta, G)

        # return gradient of NLP
        return grad_NLP

    # evidence lower bound
    def elbo(self, lmbda):

        # unpack variational parameters
        params = lmbda[:(len(lmbda)//2)]
        stdv = jnp.exp2(lmbda[(len(lmbda)//2):])

        # entropy contribution to negative ELBO
        self.ELBO = -log_abs_det(lmbda)

        # expected log likelihood of prior
        self.ELBO += np.sum(self.Alpha * self.param_res(params) ** 2) / 2.
        self.ELBO += np.sum(self.Alpha * stdv**2)

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # loop over each sample in dataset
        for n_t, (t_eval, S_batch, M_batch, inputs) in self.data_dict.items():
            # divide into batches
            n_samples = S_batch.shape[0]

            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):

                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_sb, out_mb, *Z_b = batchODEZ(t_eval,
                                                 S_batch[batch_inds],
                                                 M_batch[batch_inds],
                                                 self.Z0,
                                                 params,
                                                 inputs[batch_inds],
                                                 self.s_cap, self.m_cap)

                # concatenate S and M measurements
                Y_batch = np.concatenate((S_batch[batch_inds], M_batch[batch_inds]), -1)

                # concatenate species and mediator outputs
                out_b = np.concatenate((out_sb, out_mb), -1)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, Z, Y_measured, input in zip(np.array(out_b),
                                                        np.array(Z_b),
                                                        Y_batch,
                                                        inputs[batch_inds]):
                    # stack gradient matrices
                    G = np.nan_to_num(Z)

                    # ignore initial condition and NaNs
                    G = np.einsum("tk,tki->tki",
                                  np.array(~np.isnan(Y_measured[1:, :self.n_obs]), int),
                                  G[1:, :self.n_obs])

                    # determine error
                    Y_error = np.nan_to_num(np.nan_to_num(output[1:, :self.n_obs]) - Y_measured[1:, :self.n_obs])

                    # Determine SSE and gradient of SSE
                    self.ELBO += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error) / 2.
                    for Gt in G:
                        self.ELBO += TrBGVGT(lmbda, self.Beta, Gt) / 2.

        # return NLP
        return self.ELBO

    def jacobian_elbo(self, lmbda):

        # unpack variational parameters
        params = lmbda[:(len(lmbda)//2)]
        stdv = jnp.exp2(lmbda[(len(lmbda)//2):])

        # gradient of entropy of posterior
        grad_ELBO = -grad_log_abs_det(lmbda)

        # gradient of expected log likelihood of prior
        grad_ELBO += np.concatenate((self.Alpha * self.param_res(params), np.zeros(len(lmbda) // 2)))
        grad_ELBO += np.concatenate((np.zeros(len(lmbda) // 2), self.Alpha * stdv))

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # loop over each sample in dataset
        for n_t, (t_eval, S_batch, M_batch, inputs) in self.data_dict.items():
            # divide into batches
            n_samples = S_batch.shape[0]

            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):

                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_sb, out_mb, *Z_b = batchODEZ(t_eval,
                                                 S_batch[batch_inds],
                                                 M_batch[batch_inds],
                                                 self.Z0,
                                                 params,
                                                 inputs[batch_inds],
                                                 self.s_cap, self.m_cap)

                # concatenate S and M measurements
                Y_batch = np.concatenate((S_batch[batch_inds], M_batch[batch_inds]), -1)

                # concatenate species and mediator outputs
                out_b = np.concatenate((out_sb, out_mb), -1)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, Z, Y_measured, input in zip(np.array(out_b),
                                                        np.array(Z_b),
                                                        Y_batch,
                                                        inputs[batch_inds]):
                    # stack gradient matrices
                    G = np.nan_to_num(Z)

                    # ignore initial condition and NaNs
                    G = np.einsum("tk,tki->tki",
                                  np.array(~np.isnan(Y_measured[1:, :self.n_obs]), int),
                                  G[1:, :self.n_obs])

                    # determine error
                    Y_error = np.nan_to_num(np.nan_to_num(output[1:, :self.n_obs]) - Y_measured[1:, :self.n_obs])

                    # sum over time and outputs to get gradient w.r.t params
                    grad_ELBO += np.concatenate((eval_grad_NLP(Y_error, self.Beta, G), np.zeros(len(lmbda)//2)))
                    for Gt in G:
                        grad_ELBO += grad_TrBGVGT(lmbda, self.Beta, Gt) / 2.

        # return gradient of negative elbo
        return grad_ELBO

    def hessian(self, params):

        # reshape params and convert to JAX tensors
        params = self.reshape(params)

        # compute Hessian of NLP
        A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for n_t, (t_eval, S_batch, M_batch, inputs) in self.data_dict.items():
            # divide into batches
            n_samples = S_batch.shape[0]

            for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / self.n_devices)):

                # batches of outputs, initial condition sensitivity, parameter sensitivity
                out_sb, out_mb, *Z_b = batchODEZ(t_eval,
                                                 S_batch[batch_inds],
                                                 M_batch[batch_inds],
                                                 self.Z0,
                                                 params,
                                                 inputs[batch_inds],
                                                 self.s_cap, self.m_cap)

                # concatenate S and M measurements
                Y_batch = np.concatenate((S_batch[batch_inds], M_batch[batch_inds]), -1)

                # concatenate species and mediator outputs
                out_b = np.concatenate((out_sb, out_mb), -1)

                # collect gradients and reshape
                Z_b = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], Z_i.shape[2], -1) for Z_i in Z_b], -1)

                # loop over the batched outputs
                for output, Z, Y_measured, input in zip(np.array(out_b),
                                                        np.array(Z_b),
                                                        Y_batch,
                                                        inputs[batch_inds]):
                    # stack gradient matrices
                    G = np.nan_to_num(Z)

                    # ignore initial condition and NaNs
                    G = np.einsum("tk,tki->tki",
                                  np.array(~np.isnan(Y_measured[1:, :self.n_obs]), int),
                                  G[1:, :self.n_obs])

                    # compute Hessian
                    A += A_next(G, self.Beta)

                    # make sure precision is symmetric
                    A = (A + A.T) / 2.

        # make sure precision is positive definite
        # A, _ = make_pos_def(A, np.ones_like(self.Alpha))

        # return Hessian
        return A

    def callback(self, xk, res=None):
        if self.verbose:
            print("Loss: {:.3f}, Residuals: {:.3f}".format(self.NLP, self.RES))
        return True

    def callback_elbo(self, xk, res=None):
        if self.verbose:
            print("NEG ELBO: {:.3f}".format(self.ELBO))
        return True

    def predict_point(self, x_test, t_eval, inputs=None):

        # set inputs to empty array if None
        if inputs is None:
            inputs = np.array([])

        # convert to arrays
        t_eval = np.array(t_eval, dtype=np.float32)

        # separate state
        s_test = np.atleast_2d(x_test)[:, :self.n_s]
        m_test = np.atleast_2d(x_test)[:, self.n_s:]

        # make predictions given initial conditions and evaluation times
        s_out, m_out = runODE(t_eval, s_test, m_test, self.params, inputs, self.s_cap, self.m_cap)

        return s_out, m_out

    def save(self, fname):
        # save model parameters needed to make predictions
        save_dict = {'m0': self.m0, 'BetaInv': self.BetaInv, 'Ainv': self.Ainv}

        # save list of params
        for i, p in enumerate(self.params):
            save_dict[f'param_{i}'] = p

        savemat(fname, save_dict)

    def load(self, fname):
        # load model parameters
        load_dict = loadmat(fname)

        # set params
        self.m0 = load_dict['m0'].ravel()
        self.BetaInv = load_dict['BetaInv']
        self.Ainv = load_dict['Ainv']

        # determine number of parameter matrices
        n_items = 1 + max(int(p.split('_')[-1]) for p in load_dict if 'param' in p)

        self.params = []
        for i in range(n_items):
            param = load_dict[f'param_{i}']
            if param.shape[0] > 1:
                self.params.append(param)
            else:
                self.params.append(param.ravel())

        # initial condition of grad system w.r.t. initial latent mediators
        self.Y0 = np.zeros([self.n_s + self.n_m, self.n_lm])
        if self.n_lm > 0:
            self.Y0[-self.n_lm:, :] = np.eye(self.n_lm)
        self.Z0 = [np.zeros([self.n_s + self.n_m] + list(param.shape)) for param in self.params]

        # initial condition of Jacobian
        self.J0 = np.eye(self.n_s + self.n_m)
