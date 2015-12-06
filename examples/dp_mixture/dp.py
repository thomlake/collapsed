import cPickle
import numpy as np
from prettytable import PrettyTable
import collapsed


def sample_dataset():
    print 'creating data'
    np.random.seed(sum(map(ord, 'synthetic')))
    
    # model1
    pi = [1.0, 0.0, 0.0, 0.0]
    A = [[0.9, 0.1, 0.0, 0.0],
         [0.0, 0.9, 0.1, 0.0],
         [0.0, 0.0, 0.9, 0.1],
         [0.1, 0.0, 0.0, 0.9]]
    mu = [[-10.0], [0.0], [10.0], [0.0]]
    Sigma = np.ones((4, 1, 1))
    markovchain = collapsed.MarkovChain(4, pi, A)
    emission = collapsed.emission.NIW(4, 1, 1.0, mu, 1.0, Sigma)
    hmm1 = collapsed.HMM(markovchain, emission)

    # model2
    pi = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    A = [[0.3, 0.3, 0.3, 0.1, 0.0, 0.0],
         [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
         [0.1, 0.1, 0.8, 0.0, 0.0, 0.0],
         [0.1, 0.0, 0.0, 0.3, 0.3, 0.3],
         [0.0, 0.0, 0.0, 0.3, 0.4, 0.3],
         [0.0, 0.0, 0.0, 0.3, 0.3, 0.4]]
    mu = [[-10.0], [-6.0], [0.0], [10.0], [6.0], [0.0]]
    Sigma = [[[1.0]], [[0.01]], [[0.01]], [[1.0]], [[0.01]], [[0.01]]]
    markovchain = collapsed.MarkovChain(6, pi, A)
    emission = collapsed.emission.NIW(6, 1, 1.0, mu, 1.0, Sigma)
    hmm2 = collapsed.HMM(markovchain, emission)

    # model3
    pi = [0.5, 0.5]
    A = [[0.98, 0.02],
         [0.02, 0.98]]
    mu = [[0.0], [0.0]]
    Sigma = [[[10.0]], [[0.01]]]
    markovchain = collapsed.MarkovChain(2, pi, A)
    emission = collapsed.emission.NIW(2, 1, 1.0, mu, 1.0, Sigma)
    hmm3 = collapsed.HMM(markovchain, emission)

    # sample data from the models
    mix = np.array([0.25, 0.35, 0.4])
    n = 100
    dataset = []
    components = [hmm1, hmm2, hmm3]
    for i in range(n):
        k = mix.cumsum().searchsorted(np.random.random())
        T = np.random.geometric(0.01) + 40
        _, x = components[k].generate_sample(T)
        dataset.append((k, x))

    with open('data.pkl', 'w') as f:
        cPickle.dump(dataset, f)
    return dataset

# load data
try:
    with open('data.pkl') as f:
        dataset = cPickle.load(f)
except IOError as err:
    dataset = sample_dataset()

np.random.seed(sum(map(ord, 'dphmm')))

# mixture component initializer
def new_component():
    n_states = 3
    markovchain = collapsed.MarkovChain(n_states)
    emission = collapsed.emission.NIX(n_states, 1, 1.0, np.zeros(1), 2.1, np.ones(1))
    return collapsed.HMM(markovchain, emission)

# create model and initialize with samples
model = collapsed.DP(new_component, alpha=5.0)
for y, x in dataset:
    _, k, s = model.propose(x)
    model.add_data(k, s, x)

for i in range(50):
    model.gibbs()
    if i % 5 == 0:
        print '[{0}] logP(Z, Y, X) = {1}'.format(i, model.marginal_likelihood(log=True, normalize=True))
        print 'Components:', np.round(model.component_counts()).astype(int)

print 'acceptance ratio:', model.acceptance_ratio()

# show clusters
K = len(model.components)
type_counts = np.zeros((K, 3))
for (y, _), (k, _, _) in zip(dataset, model.data):
    type_counts[k][y] += 1

table = PrettyTable([''] + ['y={0}'.format(i) for i in range(3)])
for i in range(3):
    table.align['y={0}'.format(i)] = 'r'
for k, counts in enumerate(type_counts):
    table.add_row(['k={0}'.format(k)] + counts.astype(int).tolist())
print table
