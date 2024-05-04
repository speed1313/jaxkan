import jax
import jax.numpy as jnp
from jaxkan.model import model
import matplotlib.pyplot as plt

dataset_size = 100
input_dim = 2


@jax.jit
def f(x):
    # return x[0] * 2 + x[1] * 3
    return jnp.exp(jnp.sin(jnp.pi * x[0]) + x[1] ** 2)
    # return jnp.exp(jnp.sin(x[0]**2 + x[1]**2) + jnp.sin(x[2]**2 + x[3]**2))


X = jax.random.uniform(
    jax.random.PRNGKey(0),
    shape=(dataset_size, input_dim),
    dtype=jnp.float32,
    minval=-1.0,
    maxval=1.0,
)
# normalize
Y = jnp.array([f(x) for x in X])

basis_fn = jax.nn.silu
width_list = [2, 5, 1]
grid_size = 20
k = 3
grid_range = [-1, 1]
t = jnp.arange(grid_range[0], grid_range[1], 1/grid_size)

coef_length = len(t) - k - 1 + 1
param_size = sum(
    [
        width_list[l] * width_list[l + 1] * coef_length
        for l in range(len(width_list) - 1)
    ]
)
coef = (
    jax.random.normal(jax.random.PRNGKey(0), shape=(param_size,), dtype=jnp.float32)
    * 0.1
)


def loss_fn(coef, x, y):
    predict = model(coef, x, basis_fn, width_list, t, k)
    return (predict - y) ** 2


def batched_loss_fn(coef, X, Y):
    return jnp.mean(jax.vmap(lambda x, y: loss_fn(coef, x, y))(X, Y))



loss_history = []
for i in range(2000):
    val, grad = jax.value_and_grad(batched_loss_fn)(coef, X, Y)
    coef = coef - 0.1 * grad
    if i % 100 == 0:
        print(f"(step {i}) loss: {val}")
        loss_history.append(val)


plt.plot([i for i in range(0, 2000, 100)], loss_history)
# log scale
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss")
plt.xticks([i for i in range(0, 2000, 500)])
plt.savefig("loss.png")
plt.show()