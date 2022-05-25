from utils import *
from jax.example_libraries.optimizers import adam


## dale


@jax.jit
def synaptic_loss_dale_w_stim(params, orig_stim, p_target, signs):
    W, s = params
    p_new = get_pi(get_dale_net(W, signs), orig_stim + s)
    return djs(p_new, p_target)


v_and_g_synaptic_dale_w_stim = jax.jit(jax.value_and_grad(synaptic_loss_dale_w_stim))


@jax.jit
def synaptic_loss_dale_wo_stim(W, orig_stim, p_target, signs):
    p_new = get_pi(get_dale_net(W, signs), orig_stim)
    return djs(p_new, p_target)


v_and_g_synaptic_dale_wo_stim = jax.jit(jax.value_and_grad(synaptic_loss_dale_wo_stim))


def get_closest_dale(loss_fn, p_target, orig_stim, init, signs, n_optim=2500, lr=1e-2):
    opt_init, opt_update, get_params = adam(lr)
    opt_state = opt_init(init)

    @jax.jit
    def step(i, opt_state):
        value, grads = loss_fn(get_params(opt_state), orig_stim, p_target, signs)
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state

    vals = []
    for i in range(n_optim):
        value, opt_state = step(i, opt_state)
        vals.append(value)
    res = get_params(opt_state)
    if len(res) == 2:
        W_final, s_final = res
        return get_dale_net(W_final, signs), s_final, vals
    else:
        W_final = res
        return get_dale_net(W_final, signs), vals


### non dale


@jax.jit
def synaptic_loss_nondale_w_stim(params, orig_stim, p_target):
    W, s = params
    p_new = get_pi(get_nondale_net(W), orig_stim + s)
    return djs(p_new, p_target)


v_and_g_synaptic_nondale_w_stim = jax.jit(
    jax.value_and_grad(synaptic_loss_nondale_w_stim)
)


@jax.jit
def synaptic_loss_nondale_wo_stim(W, orig_stim, p_target):
    p_new = get_pi(get_nondale_net(W), orig_stim)
    return djs(p_new, p_target)


v_and_g_synaptic_nondale_wo_stim = jax.jit(
    jax.value_and_grad(synaptic_loss_nondale_wo_stim)
)


def get_closest_nondale(loss_fn, p_target, orig_stim, init, n_optim=2500, lr=1e-2):
    opt_init, opt_update, get_params = adam(lr)
    opt_state = opt_init(init)

    @jax.jit
    def step(i, opt_state):
        value, grads = loss_fn(get_params(opt_state), orig_stim, p_target)
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state

    vals = []
    for i in range(n_optim):
        value, opt_state = step(i, opt_state)
        vals.append(value)
    res = get_params(opt_state)
    if len(res) == 2:
        W_final, s_final = res
        return get_nondale_net(W_final), s_final, vals
    else:
        W_final = res
        return get_nondale_net(W_final), vals

