import jax
import jax.numpy as jnp
from functools import partial

# KAN model calculation:
# for each layer l
#   \psi(x_i) := scale_base * basis_fn(x_i) + scale_spline * spline(x_i)
#   x_{l+1,j} := \sum \psi(x_{l,i})


@partial(jax.jit, static_argnums=(4, 5, 6))
def psi(
    x: jnp.float32,
    t: jax.Array,
    params: jax.Array,
    params_idx: int,
    psi_param_length: int,
    k: int,
    basis_fn=jax.nn.silu,
) -> jnp.float32:
    # params_idx is the first index, (psi_param_length-2) is the length of the slice we want to take
    params_slice = jax.lax.dynamic_slice(params, (params_idx,), (psi_param_length - 2,))
    spline = bspline(x, t, params_slice, k)
    scale_base = params[params_idx + psi_param_length - 2]
    scale_spline = params[params_idx + psi_param_length - 1]
    return scale_base * basis_fn(x) + scale_spline * spline


def model(
    params: jax.Array, x: jax.Array, basis_fn, width_list: list, t: jax.Array, k: int
) -> jax.Array:
    psi_param_length = len(t) - k - 1 + 2
    current_idx = 0
    for l in range(len(width_list) - 1):
        P = jax.vmap(
            lambda i: jax.vmap(
                lambda j: psi(
                    x[i],
                    t,
                    params,
                    current_idx
                    + i * width_list[l + 1] * psi_param_length
                    + j * psi_param_length,
                    psi_param_length,
                    k,
                    basis_fn,
                )
            )(jnp.arange(width_list[l + 1]))
        )(jnp.arange(width_list[l]))
        assert P.shape == (width_list[l], width_list[l + 1])
        x = jnp.sum(P, axis=0)
        assert x.shape == (width_list[l + 1],)
        current_idx += width_list[l] * width_list[l + 1] * psi_param_length
        # assert post_activation.shape == (width_list[l + 1],)
    return x


# refer to https://github.com/scipy/scipy/blob/v1.13.0/scipy/interpolate/_bsplines.py#L150


@partial(jax.jit, static_argnums=(1,))
def B(x: jnp.float32, k: int, i: int, t: jax.Array) -> jnp.float32:
    # t corresponds to grid in the pykan implementation(maybe)
    def branch_true(x, t, i):
        return jnp.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)

    def branch_false(x, t, i):
        c1 = jnp.where(
            t[i + k] == t[i], 0.0, (x - t[i]) / (t[i + k] - t[i]) * B(x, k - 1, i, t)
        )
        c2 = jnp.where(
            t[i + k + 1] == t[i + 1],
            0.0,
            (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B(x, k - 1, i + 1, t),
        )
        return c1 + c2

    if k == 0:
        return branch_true(x, t, i)
    else:
        return branch_false(x, t, i)


@partial(jax.jit, static_argnums=(3,))
def bspline(x: jnp.float32, t: jax.Array, c: jax.Array, k: int) -> jnp.float32:
    n = len(t) - k - 1
    # assert (n >= k+1) and (len(c) >= n)
    B_list = [B(x, k, i, t) for i in range(n)]

    return jnp.sum(jnp.array(B_list) * jnp.array(c))


def test_bspline():
    k = 2
    t = jnp.array([0, 1, 2, 3, 4, 5, 6])
    c = jnp.array([-1, 2, 0, -1])
    spl = bspline(2.5, t, c, k)
    assert spl == 1.375
