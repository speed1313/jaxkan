import jax
import jax.numpy as jnp
from functools import partial

# KAN model calculation:
# for each layer l
#   \psi(x_i) := scale_base * basis_fn(x_i) + scale_spline * spline(x_i)
#   x_{l+1,j} := \sum \psi(x_{l,i})


@partial(jax.jit, static_argnums=(4, 5, 6))
def psi(x, t, coef, coef_idx, coef_length, k, basis_fn=jax.nn.silu):
    # coef_idx is the first index, (coef_length-2) is the length of the slice we want to take
    coef_slice = jax.lax.dynamic_slice(coef, (coef_idx,), (coef_length - 2,))
    spline = bspline(x, t, coef_slice, k)
    scale_base = coef[coef_idx + coef_length - 2]
    scale_spline = coef[coef_idx + coef_length - 1]
    return scale_base * basis_fn(x) + scale_spline * spline


def model(coef, x, basis_fn, width_list, t, k):
    coef_length = len(t) - k - 1 + 2
    post_activation = x
    current_idx = 0
    for l in range(len(width_list) - 1):
        P = jax.vmap(
            lambda i: jax.vmap(
                lambda j: psi(
                    post_activation[i],
                    t,
                    coef,
                    current_idx + i * width_list[l + 1] * coef_length + j * coef_length,
                    coef_length,
                    k,
                    basis_fn,
                )
            )(jnp.arange(width_list[l + 1]))
        )(jnp.arange(width_list[l]))
        assert P.shape == (width_list[l], width_list[l + 1])
        post_activation = jnp.sum(P, axis=0)
        assert post_activation.shape == (width_list[l + 1],)
        current_idx += width_list[l] * width_list[l + 1] * coef_length
        # assert post_activation.shape == (width_list[l + 1],)
    return post_activation


# refer to https://github.com/scipy/scipy/blob/v1.13.0/scipy/interpolate/_bsplines.py#L150


@partial(jax.jit, static_argnums=(1,))
def B(x, k, i, t):
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
def bspline(x, t, c, k):
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
