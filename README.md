# jaxkan
JAX implementation of Kolmogorov Arnold Networks (KANs). This implementation is just for learning KANs.
The original implementation of KANs in PyTorch is [here](https://github.com/KindXiaoming/pykan). More efficient implementation of KANs in PyTorch is [here](https://github.com/Blealtan/efficient-kan)

# TODO
- [ ] Support for Adam optimizer
- [ ] Support for update grid

# How to use

```
$ rye run python3 src/jaxkan/model.py
(step 0) loss: 7.8534674644470215
(step 100) loss: 0.6264676451683044
(step 200) loss: 0.6532149314880371
...
x: [-0.58792615 -0.09436536] y: 0.38551077 predict: [0.56417537]
x: [ 0.2627325 -0.9413302] y: 5.0577445 predict: [5.027388]
x: [-0.25400543 -0.58075714] y: 0.68477184 predict: [0.9167945]
x: [0.42758942 0.80969167] y: 5.103045 predict: [5.1872888]
x: [-0.7591913  -0.03807139] y: 0.5041167 predict: [0.7283126]

```



# References
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)