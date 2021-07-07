from data_ import get_fashion_mnist_labels
from plot_ import show_image


def predict_ch3(net, test_iter, n=6):
    """Predict labels define in Chapter 3."""
    for X, y in test_iter:
        break

    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    show_image(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
