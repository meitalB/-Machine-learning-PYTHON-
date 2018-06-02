# Train a Linear Classifier

# initialize parameters randomly
import numpy as np
import matplotlib.pyplot as plt

def f(a,x):
    return (1/(np.sqrt(2*np.pi)))*np.e**(-((x-2*a)**2)/2)


N = 100  # number of points per class
# D = 2  # dimensionality
D = 1
K = 3  # number of classes
X = np.zeros((N*K, D))  # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8')  # class labels

mu, sigma = 2, 1  # mean and standard deviation
s1 = np.random.normal(mu, sigma, 100)
for i in range(0, 100):
    X[i] = s1[i]
    y[i] = 1

mu = 4
s2 = np.random.normal(mu, sigma, 100)
j = 0
for i in range(100, 200):
    X[i] = s2[j]
    y[i] = 2
    j += 1

mu = 6
s3 = np.random.normal(mu, sigma, 100)
k = 0
for i in range(200, 300):
    X[i] = s3[k]
    y[i] = 3
    k += 1

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.get_cmap('Spectral'))
# plt.show()

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
# b = np.zeros((1, K))
b = np.random.randn(1, K)

# some hyperparameters
step_size = 1e-0
reg = 1e-3  # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(200):

    # evaluate class scores, [N x K]
    scores = np.dot(X, W) + b

    # compute the class probabilities
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples), y-1])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    # if i % 10 == 0:
    #     print "iteration %d: loss %f" % (i, loss)

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y-1] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg * W  # regularization gradient

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

# the new set
X_new = np.zeros((100, D))
pointsNewSet = np.random.random(100) * 10
for i in range(0, 100):
    X_new[i]=pointsNewSet[i]

scores_new = np.dot(X_new, W) + b
exp_scores_new = np.exp(scores_new)
probs_new = exp_scores_new / np.sum(exp_scores_new, axis=1, keepdims=True)  # [N x K]

yResult = np.zeros(100)
for i in range(0, 100):
    yResult[i] = probs_new[i][0]

yTrue = []
for i in pointsNewSet:
    fun1 = f(1, i)
    fun2 = f(2, i)
    fun3 = f(3, i)
    yTrue.append(fun1/(fun1+fun2+fun3))

plt.xlim(0, 10)
plt.ylim(0, 1)
plt.plot(pointsNewSet, yResult, "*")
plt.plot(pointsNewSet, yTrue, "*")
plt.show()

