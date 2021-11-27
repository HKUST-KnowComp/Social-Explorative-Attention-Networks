from keras.backend import tensorflow_backend as K
import util

my_dot = lambda a, b: K.batch_dot(a, b, axes=1)
my_l2_norm = lambda a, b: K.sqrt(K.sum((a - b) ** 2, axis=1, keepdims=True))


def similarity(x):
    H = util.args.hidden_size
    choice = util.args.choice
    x0 = x[:, :H]
    x1 = x[:, H:]
    y = [x0, x1]
    if choice == "poly":
        return polynomial(y)
    elif choice == "sig":
        return sigmoid(y)
    elif choice == "rbf":
        return rbf(y)
    elif choice == "euc":
        return euclidean(y)
    elif choice == "exp":
        return exponential(y)
    elif choice == "man":
        return manhattan(y)
    elif choice == "gesd":
        return gesd(y)
    elif choice == "aesd":
        return aesd(y)
    elif choice == "cos":
        return cosine(y)
    else:
        print("wrong choice input!")


# 0
def cosine(x):
    a = K.l2_normalize(x[0], axis=-1)
    b = K.l2_normalize(x[1], axis=-1)
    return -K.mean(a * b, axis=-1, keepdims=True)


# 1
def polynomial(x):
    return (0.5 * my_dot(x[0], x[1]) + 1) ** 2


# 2
def sigmoid(x):
    return K.tanh(0.5 * my_dot(x[0], x[1]) + 1)


# 3
def rbf(x):
    return K.exp(-1 * 0.5 * my_l2_norm(x[0], x[1]) ** 2)


# 4
def euclidean(x):
    return 1 / (1 + my_l2_norm(x[0], x[1]))


# 5
def exponential(x):
    return K.exp(-1 * 0.5 * my_l2_norm(x[0], x[1]))


# 6
def manhattan(x):
    return my_l2_norm(x[0], x[1])


# 7
def gesd(x):
    euclidean_ = 1 / (1 + my_l2_norm(x[0], x[1]))
    sigmoid_ = 1 / (1 + K.exp(-1 * 0.5 * (my_dot(x[0], x[1]) + 1)))
    return euclidean_ * sigmoid_


# 8
def aesd(x):
    euclidean_ = 0.5 / (1 + my_l2_norm(x[0], x[1]))
    sigmoid_ = 0.5 / (1 + K.exp(-1 * 0.5 * (my_dot(x[0], x[1]) + 1)))
    return euclidean_ + sigmoid_
