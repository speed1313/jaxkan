# jaxkan
JAX implementation of Kolmogorov Arnold Networks (KANs). This implementation is just for learning KANs.
The original implementation of KANs in PyTorch is [here](https://github.com/KindXiaoming/pykan). More efficient implementation of KANs in PyTorch is [here](https://github.com/Blealtan/efficient-kan)

# TODO
- [ ] Support for Adam optimizer
- [ ] Support for update grid
- [ ] more efficient implementation

# How to use

```
$ rye run python3 src/jaxkan/model.py
(step 0) loss: 7.962317943572998
(step 100) loss: 0.562587559223175
(step 200) loss: 0.3345603048801422
...
x: [-0.25400543 -0.58075714] y: 0.68477184 predict: [0.6672076]
x: [0.42758942 0.80969167] y: 5.103045 predict: [5.210634]
x: [-0.7591913  -0.03807139] y: 0.5041167 predict: [0.6054907]
```



# References
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)