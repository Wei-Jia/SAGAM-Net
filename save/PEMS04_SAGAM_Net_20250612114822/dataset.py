import numpy as np
import torch


def normalize(x):
    '''

    :param x:
    :return:
    '''
    max = np.max(x, axis=(0, 2))
    min = np.min(x, axis=(0, 2))
    x = (x - min.reshape(1, -1, 1)) / (max.reshape(1, -1, 1) - min.reshape(1, -1, 1))
    return x, max, min


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
                in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
            torch.from_numpy(np.array(target))


def Z_ScoreNormalization(x):
    '''

    :param x:
    :return:
    '''
    mean = np.mean(x, axis=(0, 2))
    std = np.std(x, axis=(0, 2))
    x = (x - mean.reshape(1, -1, 1)) / (std.reshape(1, -1, 1))
    return x, mean, std


def masked_mae_loss(y_pred, y_true, scaler, opt):
    mask = (y_true > opt.min).float()
    mask /= mask.mean()
    loss = torch.abs(scaler.inverse_transform(y_pred) - scaler.inverse_transform(y_true))
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mape_loss(y_pred, y_true, scaler, opt):
    mask = (y_true > opt.min).float()
    mask /= mask.mean()
    loss = torch.abs(torch.div(scaler.inverse_transform(y_true) - scaler.inverse_transform(y_pred), scaler.inverse_transform(y_true)))
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_rmse_loss(y_pred, y_true, scaler, opt):
    mask = (y_true > opt.min).float()
    mask /= mask.mean()
    loss = torch.pow(scaler.inverse_transform(y_true) - scaler.inverse_transform(y_pred), 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mse_loss(y_pred, y_true, scaler, opt):
    mask = (y_true > opt.min).float()
    mask /= mask.mean()
    loss = torch.pow(scaler.inverse_transform(y_true) - scaler.inverse_transform(y_pred), 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()