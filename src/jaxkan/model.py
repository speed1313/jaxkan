import jax
import jax.numpy as jnp
from functools import partial

# l層目のi番目のneuronを(l,i)で示し, (l,i)-neuronのactivation valueをx_{l,i}とする
# lとl+1の間にはn_l * n_{l+1}個のactivation functionがある.
# (l,j) to (l+1,i) の activation functionを\phi_{l,i,j}とする
# \phi_{l,i,j}の入力はx{l,i}であるから, post-activation valueをx^~_{l,i,j}:=\phi_{l,i,j}(x_{l,i})とする
# x_{l+1, j} := \sum x^~_{l,i,j} = \sum \phi_{l,i,j}(x_{l,i}) とする


def test_train_parametrized_kan_layer():
    def f(x1, x2, x3, x4):
        return jnp.exp(jnp.sin(x1**2 + x2**2) + jnp.sin(x3**2 + x4**2))

    x = jnp.array([0.5, 0.1, 0.2, 0.3])
    y = f(*x)
    basis_fn = jax.nn.relu
    width_list = [4, 2, 1, 1]
    num_grid_interval = 10
    t = jnp.linspace(-1, 1, num_grid_interval + 1)
    k = 3
    coef_length = len(t) - k - 1
    param_size = sum(
        [
            width_list[l] * width_list[l + 1] * coef_length
            for l in range(len(width_list) - 1)
        ]
    )
    coef = jax.random.normal(
        jax.random.PRNGKey(0), shape=(param_size,), dtype=jnp.float32
    )
    print("coef:", coef)
    print("len(coef):", len(coef))

    def loss_fn(coef, x, y):
        prediction = model(coef, x, basis_fn, width_list, t, k)
        print(prediction)
        return jnp.mean((prediction - y) ** 2)

    for i in range(40):
        val, grad = jax.value_and_grad(loss_fn)(coef, x, y)
        print(val)
        print(grad)

        coef = coef - 0.1 * grad

    assert jnp.abs(model(coef, x, basis_fn, width_list, t, k) - y) < 1e-3


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def psi(x, t, coef, coef_idx, coef_length, k, basis_fn=jax.nn.silu):
    spline = bspline(x, t, coef[coef_idx : coef_idx + coef_length - 1], k)
    return coef[coef_idx + coef_length - 1] * (basis_fn(x) + spline)


def model(coef, x, basis_fn, width_list, t, k):
    coef_length = len(t) - k - 1 + 1
    coef_idx = 0
    post_activation = x
    for l in range(len(width_list) - 1):
        post_l_j = jnp.zeros(width_list[l + 1])
        for i in range(width_list[l]):
            for j in range(width_list[l + 1]):
                phi_l_i_j = psi(
                    post_activation[i], t, coef, coef_idx, coef_length, k, basis_fn
                )
                post_l_j = post_l_j.at[j].add(phi_l_i_j)
                coef_idx += coef_length
        post_activation = post_l_j
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


# from functools import partial
@partial(jax.jit, static_argnums=(3,))
def bspline(x, t, c, k):
    n = len(t) - k - 1
    # assert (n >= k+1) and (len(c) >= n)
    B_list = [B(x, k, i, t) for i in range(n)]

    return jnp.sum(jnp.array(B_list) * jnp.array(c))


def test_bspline():
    k = 2
    t = [0, 1, 2, 3, 4, 5, 6]
    c = [-1, 2, 0, -1]
    spl = bspline(2.5, t, c, k)
    assert spl == 1.375

    k = 3
    t = jnp.linspace(-1, 1, 11)
    c = jax.random.normal(jax.random.PRNGKey(0), shape=(len(t) - k - 1,))
    spl = bspline(0.4, t, c, k)
    assert spl == 0.0


def test_grad_bspline():
    k = 2
    t = [0, 1, 2, 3, 4, 5, 6]
    c = [-1, 2, 0, -1]
    grad_fn = jax.grad(lambda c: bspline(2.5, t, c, k))
    grad = grad_fn(jnp.array(c, dtype=jnp.float32))
    assert jnp.sum(grad) != 0.0
