import torch
from utils import Accumulator
from metrics import accuracy
from plot_ import Animator
from metrics import evaluate_accuracy


def train_epoch_ch3(net, train_iter, loss, updater):
    """The training loop in chapter 3."""
    # Set the model to training mode.
    if isinstance(net, torch.nn.Module):
        net.train()

    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)

    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch built-in optimizer & loss criterion
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model defined in Chapter 3"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 0.7 < train_acc <= 1, train_acc
    assert 0.7 < test_acc <= 1, test_acc
