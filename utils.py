import jax.numpy as jnp
import numpy as np
import jax
import itertools as it
import matplotlib.pyplot as plt

# this slows down the computation but is necessary for numerical precision
jax.config.update("jax_enable_x64", True)

# for generating readable figures
plt.rcParams.update({"font.size": 15})

N = 10
e = jnp.zeros(2**N).at[-1].set(1)

M = it.product([0.0, 1.0], repeat=N)
M = jnp.array(list(M))

pre, post = jnp.where(jnp.eye(N) == 0)


@jax.jit
def get_nondale_net(x):
    return x.at[jnp.diag_indices(N)].set(0)


@jax.jit
def get_dale_net(x, signs):
    tmp = jnp.abs(x) * signs[:, None]
    return tmp.at[jnp.diag_indices(N)].set(0)


@jax.jit
def mat_to_params(W):
    return W[pre, post]


@jax.jit
def params_to_mat(ps):
    W = jnp.zeros((N, N))
    return W.at[pre, post].set(ps)


@jax.jit
def get_pi(W, s):
    P_unnorm = jnp.exp((M @ W @ M.T) + jnp.ones_like(M) @ jnp.diag(s) @ M.T)
    P = P_unnorm / P_unnorm.sum(1)[:, None]

    P2 = P - jnp.eye(2**N)
    P2 = P2.at[:, -1].set(1)
    return jnp.linalg.solve(P2.T, e)


@jax.jit
def dkl(p, q):
    return (p * jnp.log2(p / q)).sum()


@jax.jit
def dkl_from_ref(W_off_diag, s_new, P1):
    P_new = get_pi(params_to_mat(W_off_diag), s_new)
    return dkl(P1, P_new)


hess_w = jax.jit(jax.hessian(lambda w, s, p: dkl_from_ref(w, s, p)))
hess_s = jax.jit(jax.hessian(lambda s, w, p: dkl_from_ref(w, s, p)))


def create_stim(n_stim, s_scale=1, s_mean=0, seed=1, size=N):
    rng = np.random.RandomState(seed)
    ss = rng.randn(n_stim, size) * s_scale + s_mean
    return ss


def create_nondale_nets(n_nets, w_scale=1, seed=1, size=N):
    rng = np.random.RandomState(seed)
    nondale_Ws = []
    for _ in range(n_nets):
        W = rng.randn(size, size) * w_scale / np.sqrt(size)
        W[np.diag_indices(size)] = 0
        signs = np.ones(size * size)
        signs[: (size * size // 2)] *= -1
        signs = rng.permutation(signs).reshape(size, size)
        W = jnp.abs(W) * signs
        nondale_Ws.append(W)
    return jnp.array(nondale_Ws)


def create_dale_nets(n_nets, w_scale=1, seed=1, size=N):
    rng = np.random.RandomState(seed)
    dale_Ws = []
    for _ in range(n_nets):
        W = rng.randn(size, size) * w_scale / np.sqrt(size)
        W[np.diag_indices(size)] = 0
        signs = np.ones(size)
        signs[: (size // 2)] *= -1
        signs = rng.permutation(signs)
        W = jnp.abs(W) * signs[:, None]
        dale_Ws.append(W)
    return jnp.array(dale_Ws)


def create_permuted_dale_nets(dale_Ws, seed=1):
    rng = np.random.RandomState(seed)
    nondale_Ws = jnp.zeros_like(dale_Ws)
    for i in range(dale_Ws.shape[0]):
        d = np.array(dale_Ws[i])
        for n in range(N):
            d[:, n][d[:, n] != 0] = rng.permutation(d[:, n][d[:, n] != 0])
        nondale_Ws = nondale_Ws.at[i].set(d)
    return nondale_Ws


@jax.jit
def get_fr(p):
    return (M * p[:, None]).sum(0)


@jax.jit
def _get_cofiring(x):
    return jnp.outer(x, x)[jnp.triu_indices(N, 1)]


M_cofiring = jax.vmap(_get_cofiring)(M)


@jax.jit
def get_cofiring(p):
    return (M_cofiring * p[:, None]).sum(0)


@jax.jit
def get_pairwise_correlations(p):
    return get_cofiring(p) - _get_cofiring(get_fr(p))


@jax.jit
def ent(p):
    return -(p * jnp.log2(p)).sum()


@jax.jit
def calc_I(ps):
    ents = jax.vmap(ent)(ps)
    mean_p = jnp.nanmean(ps, axis=0)
    mean_p_ent = ent(mean_p)
    return mean_p_ent - jnp.nanmean(ents)


@jax.jit
def djs(p, q):
    m = (p + q) / 2
    return (dkl(p, m) + dkl(q, m)) / 2


@jax.jit
def create_dm(ps):
    return jax.vmap(jax.vmap(djs, in_axes=(None, 0)), in_axes=(0, None))(ps, ps)


@jax.jit
def create_dale_nets_log(n_nets, w_scale=1, seed=1):
    rng = np.random.RandomState(seed)
    dale_Ws = []
    for _ in range(n_nets):
        W = np.exp(rng.randn(N, N) * w_scale / np.sqrt(N))
        W[np.diag_indices(N)] = 0
        signs = np.ones(N)
        signs[: (N // 2)] *= -1
        signs = rng.permutation(signs)
        W = jnp.abs(W) * signs[:, None]
        dale_Ws.append(W)
    return jnp.array(dale_Ws)


@jax.jit
def get_pi_with_scales(W, stim, scales):
    W2 = W * scales[:, None]
    P_unnorm = jnp.exp((M @ W2 @ M.T) + jnp.ones_like(M) @ jnp.diag(stim) @ M.T)
    P = P_unnorm / P_unnorm.sum(1)[:, None]

    P2 = P - jnp.eye(2**N)
    P2 = P2.at[:, -1].set(1)
    return jnp.linalg.solve(P2.T, e)


@jax.jit
def dkl_from_ref_scales(W, s, P1, scales):
    P_new = get_pi_with_scales(W, s, scales)
    return dkl(P1, P_new)


hess_scales = jax.jit(
    jax.hessian(lambda scales, w, s, p: dkl_from_ref_scales(w, s, p, scales))
)
