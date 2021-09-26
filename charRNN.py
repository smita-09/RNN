import numpy as np

# Reading Data
data = open('input.txt', 'r').read()
chars = list(set(data))

# print(chars) # This is to get all the chars
data_size, vocab_size = len(data), len(chars) # This is to get the datasize and to get the len of list of chars
print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(chars) } # This is also to create a map from characteres to numbers
ix_to_char = { i:ch for i,ch in enumerate(chars) } # This is to create a map from numbers to characters
# print(ix_to_char)

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for (I think that means that we are gonna for and back prop in 25 steps)
# NO, this is the length of the ouput sequence mostly
learning_rate = 1e-1 # ususal stuff

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden and shape is 100*21
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden and the shape is 100*100
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output and the shape is 21*100
bh = np.zeros((hidden_size, 1)) # hidden bias and the shape is just 100*1
by = np.zeros((vocab_size, 1)) # output bias and the shape is just 21*1


def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)): # t is going over the length of the input
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1 # all clear of what is happening oever here
    # print("This is the shape!")
    # print(np.dot(Wxh, xs[t]).shape)
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars which is again a vector of same size as ys
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by) # for hidden bias and  output bias
  dhnext = np.zeros_like(hs[0]) # cause need to have that one I guess, still not sure if thats true what I am thinking
  for t in reversed(range(len(inputs))):  # in reverse order.   
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T) # will become 15*1 to 1*15
    dby += dy # adding the dy thing to bias of y as it was zero in the starting and remember  that is is starting from the reversed order
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    # I guess we need to remember the steps, cause it gets tedious to calculate it evrty freaking time
    dbh += dhraw # adding shit we calculated above
    dWxh += np.dot(dhraw, xs[t].T) # the same thing except that now raw also got a role to play
    dWhh += np.dot(dhraw, hs[t-1].T) # same shit as above here as well
    dhnext = np.dot(Whh.T, dhraw) # to get doone with this thing
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients, which is very nice thing, that is values larget than 5 will become 5 and values less than -5 will become -5
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]  # and this whole thing concludes one step of forw prop and backward prop.


def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):  # n has been passed as 200
    # print(h.shape, np.dot(Whh, h).shape, sep = " , ") # yes, it makes some sense
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh) # (100* 15) and (15*1) , (100*100 and 100*1) so in total it will just be 100*1
    y = np.dot(Why, h) + by # makes sense
    p = np.exp(y) / np.sum(np.exp(y)) # makes sense
    print("This is the fucking shape dammit!!!!!!!!!!!!", y.shape)
    ix = np.random.choice(range(vocab_size), p=p.ravel()) # just to select the next index based on the probabilty of the indices, nice use of random.choice
    x = np.zeros((vocab_size, 1))   
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]] # list of indices of the chars in input sequence
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]] # makes complete sense

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  """
  *******************************
  I think this also needs to be the part of adamoptimizer cause they do the same or mayb momemtum
  """

  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))# print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    # calculating the square of differentials
    # mem is I guess some intermediate thing and it is getting multiplied to dparam which we calculated using backprop
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 


