import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collapsed

np.random.seed(sum(map(ord, 'trick coin')))

n_states, n_symbols = 2, 2
pi = [1.0, 0.0]
A = [[0.95, 0.05], 
     [0.05, 0.95]]
B = [[0.5, 0.5], 
     [0.0, 1.0]]
latent = collapsed.MarkovChain(n_states, pi, A)
observed = collapsed.DirCat(n_states, n_symbols, B)
trick_coin = collapsed.HMM(latent, observed)
y_true, x = trick_coin.generate_sample(150)

latent = collapsed.MarkovChain(n_states)
observed = collapsed.DirCat(n_states, n_symbols)
model = collapsed.HMM(latent, observed)

score, y = model.propose(x)
model.add_data(y, x)
model.gibbs(20)
model.burning = False
model.gibbs(200)
print 'acceptance ratio:', model.acceptance_ratio()
frozen_params = model.expected_params()

x1 = np.random.randint(0, 2, 50)
_, x2 = trick_coin.generate_sample(50)
print model.likelihood(x1, frozen_params=frozen_params)
print model.likelihood(x2, frozen_params=frozen_params)

# y_pred, _ = model.data[0]
score, y_pred = model.map(x, frozen_params=frozen_params)
pal = sns.color_palette()
for t in range(len(y_true)):
    plt.bar(t, 1, width=1, linewidth=0, color=pal[y_true[t]])
    plt.bar(t, -1, width=1, linewidth=0, color=pal[y_pred[t] + 2])
plt.savefig('output.png')
