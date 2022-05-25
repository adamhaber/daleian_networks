from utils import *
from jax.example_libraries.optimizers import adam

delta_in_init = np.ones(N)
delta_out_init = np.ones(N)
delta_th_init = np.zeros(N)


@jax.jit
def get_rescaled_w(w, delta_in, delta_out, delta_th):
    return jnp.diag(jnp.abs(delta_in)) @ w @ jnp.diag(jnp.abs(delta_out))


@jax.jit
def neuronal_loss_dale(params, w_init, stim, p_target, signs):
    delta_in, delta_out, delta_th = params
    w = get_rescaled_w(get_dale_net(w_init, signs), delta_in, delta_out, delta_th)
    p_new = get_pi(w, stim + delta_th)
    return djs(p_new, p_target)


v_and_g_neuronal_dale = jax.jit(jax.value_and_grad(neuronal_loss_dale))


def get_closest_dale(p_target, w_init, stim, signs, n_optim=2500, lr=1e-2):
    opt_init, opt_update, get_params = adam(lr)
    opt_state = opt_init((delta_in_init, delta_out_init, delta_th_init))

    @jax.jit
    def step(i, opt_state):
        value, grads = v_and_g_neuronal_dale(
            get_params(opt_state), w_init, stim, p_target, signs
        )
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state

    vals = []
    for i in range(n_optim):
        value, opt_state = step(i, opt_state)
        vals.append(value)
    return get_rescaled_w(get_dale_net(w_init, signs), *get_params(opt_state)), vals


@jax.jit
def neuronal_loss_nondale(params, w_init, stim, p_target):
    delta_in, delta_out, delta_th = params
    w = get_rescaled_w(get_nondale_net(w_init), delta_in, delta_out, delta_th)
    p_new = get_pi(w, stim + delta_th)
    return djs(p_new, p_target)


v_and_g_neuronal_nondale = jax.jit(jax.value_and_grad(neuronal_loss_nondale))


def get_closest_nondale(p_target, w_init, stim, n_optim=2500, lr=1e-2):
    opt_init, opt_update, get_params = adam(lr)
    opt_state = opt_init((delta_in_init, delta_out_init, delta_th_init))

    @jax.jit
    def step(i, opt_state):
        value, grads = v_and_g_neuronal_nondale(
            get_params(opt_state), w_init, stim, p_target
        )
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state

    vals = []
    for i in range(n_optim):
        value, opt_state = step(i, opt_state)
        vals.append(value)
    return get_rescaled_w(get_nondale_net(w_init), *get_params(opt_state)), vals
