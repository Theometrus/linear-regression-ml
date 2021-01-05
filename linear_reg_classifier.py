import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def preprocess(data, desired_label):
    """ Formats the data correctly (and flattens it). """

    # Attempt to keep number of yes and no examples equal
    count_yes = 0
    count_no = 0
    X_original = data[b"data"]
    labels = data[b'labels']

    X = []
    Y = []

    for idx, label in enumerate(labels):
        if label == desired_label:
            Y.append(1)
            count_yes += 1
        elif not count_no > count_yes:
            Y.append(0)
            count_no += 1
        else:
            continue

        X.append(X_original[idx])

    ret_X = np.array(X).T
    ret_Y = np.array(Y).reshape(1, len(Y))

    return (ret_X, ret_Y)


def standardize(x):
    """ Standardize the dataset. In the case of images, dividing
    by 255 is often enough """
    return x / 255


def init_zeroes(num_rows):
    return np.zeros((num_rows, 1)), 0


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def propagate(X, Y, w, b):
    m = X.shape[1]  # Number of training examples
    A = sigmoid(np.dot(w.T, X) + b)

    # Necessary to avoid log of zero errors due to machine precision
    offset = 1e-10
    cost = (-1 / m) * (np.sum((Y * np.log(A)) +
                              ((1 - Y) * (np.log((1 + offset) - A)))))
    db = (1 / m) * (np.sum(A - Y))
    dw = (1 / m) * np.dot(X, (A - Y).T)

    deltas = {"db": db,
              "dw": dw}

    return deltas, cost


def learn(X, Y, w, b, num_iters, learning_rate):
    costs = []

    for i in range(num_iters):
        deltas, cost = propagate(X, Y, w, b)
        w -= learning_rate * deltas["dw"]
        b -= learning_rate * deltas["db"]

        if i % 100 == 0:
            costs.append(cost)
            print("Iteration {i} cost: {cost}".format(i=i, cost=cost))

    params = {"w": w,
              "b": b}

    return params, costs


def predict(w, b, X):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    Y_predict = np.zeros((1, m))
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5:
            Y_predict[0][i] = 0
        else:
            Y_predict[0][i] = 1

    return Y_predict


def model(X_train, Y_train, X_test, Y_test, num_iters, learning_rate):
    w, b = init_zeroes(X_train.shape[0])
    params, costs = learn(X_train, Y_train, w, b, num_iters, learning_rate)
    w = params["w"]
    b = params["b"]
    Y_predict_train = predict(w, b, X_train)
    Y_predict_test = predict(w, b, X_test)

    print("Training set accuracy: {} %".format(
        100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))

    print("Test set accuracy: {} %".format(
        100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))

    return {"costs": costs,
            "w": w,
            "b": b,
            "Y_predict_test": Y_predict_test,
            "Y_predict_train": Y_predict_train}


def main():
    label = 8  # Label is used to select the category of CIFAR-10 images being trained
    name = "ship"

    # Training data
    data = unpickle("./ml_data/data_batch_1")  # dataset courtesy of CIFAR-10
    processed_data = preprocess(data, label)
    X_train = standardize(processed_data[0])
    Y_train = processed_data[1]

    # Test data
    data = unpickle("./ml_data/test_batch")
    processed_data = preprocess(data, label)
    X_test = standardize(processed_data[0])
    Y_test = processed_data[1]

    ret = model(X_train, Y_train, X_test, Y_test, 5000, 0.01)

    print("-----------------------------------------------")

    for i in range(Y_test.shape[1]):
      # Courtesy of Stackoverflow
        img = np.transpose(
            np.reshape(processed_data[0].T[i], (3, 32, 32)), (1, 2, 0))

        if Y_test[0][i] == 1:
            print("This is a {}.".format(name))
        else:
            print("This is not a {}.".format(name))

        if ret["Y_predict_test"][0][i] == 1:
            print("The Neural Network classified this as a {}.".format(name))
        else:
            print("The Neural Network classified this as not a {}.".format(name))

        plt.imshow(img)
        plt.show()
        print("-----------------------------------------------")


if __name__ == "__main__":
    main()
