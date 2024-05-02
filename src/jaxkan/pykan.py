from kan import *
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[4,2,1,1], grid=5, k=3, seed=0)

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(x[:,[0]]**2 + x[:,[1]]**2) + torch.sin(x[:,[2]]**2 + x[:,[3]]**2))
dataset = create_dataset(f, n_var=4)
dataset['train_input'].shape, dataset['train_label'].shape

# plot KAN at initialization
model(dataset['train_input'])
model.plot(beta=100)

# train the model
model.train(dataset, opt="Adam", steps=200, lamb=0.01, lamb_entropy=10.)

model.plot()

