from copy import copy

import numpy as np



sigmoid = lambda x: 1 / (1 + np.exp(-x))
def relu(x): return np.argmax(0,x)
# def sigmoid(x):
#   return (np.exp(x-np.amax(x)))/(np.sum(np.exp(x-np.amax(x))))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def functionDerivative(v):
    return (1 - v) * v


def fprop(x, y, params):
  # Follows procedure given in notes
  x= x.reshape(784,1)

  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  h2 = softmax(z2)
  # loss = -np.log(h2[y])
  loss=0
  ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss,'h0':x}
  for key in params:
    ret[key] = params[key]
  return ret

# def bprop(fprop_cache):
#   # Follows procedure given in notes
#   x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
#   dz2 = (h2 - y)                                #  dL/dz2
#   dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
#   db2 = dz2                                     #  dL/dz2 * dz2/db2
#   dz1 = np.dot(fprop_cache['W2'].T,(h2 - y))
#   # * sigmoid(z1) * (1-sigmoid(z1))   #  dL/dz2 * dz2/dh1 * dh1/dz1
#   dW1 = np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
#   db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
#   return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}
def backpropWhile(X, y_hat, params):
    grads = {}
    model = fprop(X,y_hat,params)
    v_L = model['h2']
    errorL = v_L - y_hat
    g_L = errorL
    for l in range(L, 0, -1):
        deltaW = np.matmul(g_L, model['h' + str(l-1)].T)
        deltaB = g_L
        grads['W' + str(l)] = deltaW
        grads['b' + str(l)] =  deltaB
        errorL = np.matmul(W[l].T, g_L)
        if (l > 1):
            g_L = np.multiply(functionDerivative(model['h' + str(l-1)]), errorL)

    return  grads

def update(grads, params):
  # print(grads.keys())
  w1 =params['W1']
  w2= params['W2']
  b1=params['b1']
  b2 =params['b2']
  w1-=learningRate*grads['W1']
  w2 -= learningRate * grads['W2']
  b1 -= learningRate * grads['b1']
  b2 -= learningRate * grads['b2']
  params['W1']=w1
  params['W2']=w2
  params['b1']=b1
  params['b2']=b2
  return params
def getAccurcy(dev_x,dev_y,params):
  good =0.0
  bad =0.0
  for i in range(len(dev_x)):
    ans= fprop(dev_x[i],dev_y[i],params)
    prediction= np.argmax(ans['h2'])
    if(prediction ==int(dev_y[i])):
      good+=1
    else:
      bad+=1

  print('accurcy', good/(good+bad))




if __name__ == '__main__':
  INPUT = 28 * 28
  HIDDEN1 = 100
  OUTPUT = 10
  learningRate=0.01
  L=2
  W1 = np.random.uniform(-0.01,0.01,(HIDDEN1,INPUT))
  b1 = np.random.uniform(-0.01,0.01,(HIDDEN1, 1))
  W2 = np.random.uniform(-0.01, 0.01, (OUTPUT, HIDDEN1))
  b2 = np.random.uniform(-0.01,0.01,(OUTPUT, 1))
  params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
  train_x = np.loadtxt("train_x")
  train_y = np.loadtxt("train_y", dtype=int)
  test_x = np.loadtxt("test_x")

  np.random.seed(0)
  np.random.shuffle(train_x)
  np.random.seed(0)
  np.random.shuffle(train_y)


  dev_size = len(train_x) * 0.2
  dev_size_int = int(dev_size)
  dev_x, dev_y = train_x[-dev_size_int:, :], train_y[-dev_size_int:]

  W = [1, W1, W2]
  b = [1, b1, b2]


for epoch in range(20):
  getAccurcy(dev_x,dev_y,params)
  for i in range(len(train_x)):
    fprop_cache = fprop(train_x[i]/255.0, train_y[i], params)
    y_hat = np.zeros((10, 1))
    correct = int(train_y[i])
    y_hat[correct] = 1
    grads = backpropWhile(train_x[i]/255.0, y_hat, params)
    params=update(grads,params)

string_ans=""
for k in range(5000):
    string_ans+="\n"
    fprop_cache = fprop(test_x[k] / 255.0, test_x[k], params)
    prediction = np.argmax(fprop_cache['h2'])
    string_ans+= str(prediction)

file = open('test.pred', 'w')
file.write(string_ans)
file.close()
