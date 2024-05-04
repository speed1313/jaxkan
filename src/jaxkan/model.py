import jax
import jax.numpy as jnp
from functools import partial

# l層目のi番目のneuronを(l,i)で示し, (l,i)-neuronのactivation valueをx_{l,i}とする
# lとl+1の間にはn_l * n_{l+1}個のactivation functionがある.
# (l,j) to (l+1,i) の activation functionを\phi_{l,i,j}とする
# \phi_{l,i,j}の入力はx{l,i}であるから, post-activation valueをx^~_{l,i,j}:=\phi_{l,i,j}(x_{l,i})とする
# x_{l+1, j} := \sum x^~_{l,i,j} = \sum \phi_{l,i,j}(x_{l,i}) とする


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def psi(x, t, coef, coef_idx, coef_length, k, basis_fn=jax.nn.silu):
    coef_slice = jax.lax.dynamic_slice(coef, (coef_idx,), (coef_length-1,))
    #assert coef_slice.shape == (coef_length-1,)
    spline = bspline(x, t, coef_slice, k)
    return coef[coef_idx + coef_length - 1] * (basis_fn(x) + spline)



def model(coef, x, basis_fn, width_list, t, k):
    coef_length = len(t) - k - 1 + 1
    post_activation = x
    current_idx = 0
    for l in range(len(width_list) - 1):
        def psi_matrix(i, j, l, current_idx):
            # j to i
            return psi(
                post_activation[j], t, coef, current_idx + (j*(l+1) + i) * coef_length, coef_length, k, basis_fn
            )

        def row(j, l, current_idx):
            return jax.vmap(partial(psi_matrix, j=j, l = l, current_idx = current_idx))(jnp.arange(width_list[l + 1]))

        def get_P(width_list, l, current_idx):
            return jax.vmap(partial(row, l=l, current_idx=current_idx))(jnp.arange(width_list[l]))
        P = get_P(width_list, l, current_idx)

        #assert P.shape == (width_list[l], width_list[l + 1])
        post_activation = jnp.sum(P, axis=0)
        current_idx += width_list[l] * width_list[l + 1] * coef_length
        #assert post_activation.shape == (width_list[l + 1],)
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
