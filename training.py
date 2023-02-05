from network import *

# On importe les données et on les mélange
data = pd.read_csv("./mnist_train.csv")
data = np.array(data)
m, n = data.shape   # m: nombre de lot / n: nb d'inputs
np.random.shuffle(data)

d = data[0:m].T
Y = d[0]    # valeur à prédire
X = d[1:n]  # inputs
X = X/255.0   # on normalise


w1, b1, w2, b2, w3, b3 = gradient_descent(X, Y, 0.01, 5000)
np.savez("./NN_weights_and_biases.npz", arr_0=w1, arr_1=b1, arr_2=w2, arr_3=b2, arr_4=w3, arr_5=b3)
