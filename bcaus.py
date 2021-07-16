import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class Propensity(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        """
        Define Pytorch model
        :param input_size: Number of covariates
        :param hidden_size: Number of hidden layer neurons
        :param dropout_prob: Dropout probability
        """
        super(Propensity, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in')
        self.do1 = nn.Dropout(p=dropout_prob)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in')
        self.do2 = nn.Dropout(p=dropout_prob)
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.do1(self.layer1(x)))
        x = F.relu(self.do2(self.layer2(x)))
        x = torch.sigmoid(self.layer3(x))
        return x


class BCaus(BaseEstimator, ClassifierMixin):

    def __init__(self, random_state=271, hidden_layer_size=None,
                 batch_size=None, shuffle=True,
                 learning_rate_init=0.001, nu=1, max_iter=100,
                 alpha=0.0, dropout=0, eps=1e-5, early_stopping=False,
                 n_iter_no_change=10, balance_threshold=0.1,
                 device='cpu', verbose=False, logger=None):
        """
        :param random_state: Random seed for torch and numpy
        :param hidden_layer_size: Number of neurons in hidden layer. If None set to 2x number of covariates
        :param batch_size: If None all samples are used. To achieve max balance in covariates it is recommended
                           that this be set to None unless sample size is very large e.g. > 1000
        :param shuffle: If True batches are shuffled. Ignored if batch_size is None
        :param learning_rate_init: Initial learning rate. Learning rate scheduler is not implemented in this version.
        :param nu: Determines relative balance between classification loss and balance loss. Backpropagated loss
                   is the sum of two terms: BCELoss() + nu * (bceloss/mseloss)*MSELoss()
        :param max_iter: Number of epochs
        :param alpha: L2 penalty term in loss function
        :param dropout: Dropout probability
        :param eps: Constant to stabilize denominator while computing IPTW weights
        :param early_stopping: If True training terminates when all covariates are balanced and remain so
                               for n_iter_no_change epochs
        :param n_iter_no_change:
        :param balance_threshold: Threshold for considering covariates as being balanced; default 0.1
        :param device: Device to train model on; default 'cpu'
        :param verbose: Log losses every 50 epochs if True
        :param logger: Python logger object. Should be provided if verbose = True
        """

        self.random_state = random_state
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate_init = learning_rate_init
        self.nu = nu
        self.max_iter = max_iter
        self.alpha = alpha
        self.dropout = dropout
        self.eps = eps
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.balance_threshold = balance_threshold
        self.device = device
        self.verbose = verbose
        self.logger = logger

    def _step(self, X, y, optimizer, criterion):
        """
        Method to perform one step of forward + back + update weights
        :param X:
        :param y:
        :param optimizer: Pytorch optimizer
        :param criterion: Tuple of Pytorch loss functions (BCELoss(), MSELoss())
        :return: bceloss, mseloss
        """

        zeros = (y == 0).nonzero().squeeze()
        ones = y.nonzero().squeeze()

        prop_criterion, cov_criterion = criterion

        optimizer.zero_grad()

        score = self.model(X).squeeze()

        # Propensity BCE loss
        loss_prop = prop_criterion(score, y)

        # Covariates balance loss
        weight = (y / (score + self.eps) + (1 - y) / (1 - score + self.eps)).unsqueeze(-1).repeat(1, X.shape[1])

        weight_zeros = torch.index_select(weight, 0, zeros)
        weight_ones = torch.index_select(weight, 0, ones)

        X_zeros = torch.index_select(X, 0, zeros)
        X_ones = torch.index_select(X, 0, ones)

        zeros_mean = torch.sum(weight_zeros * X_zeros, 0) / torch.sum(weight_zeros, 0)
        ones_mean = torch.sum(weight_ones * X_ones, 0) / torch.sum(weight_ones, 0)

        loss_cov = cov_criterion(zeros_mean, ones_mean)

        loss_ratio = (loss_prop / loss_cov).item()

        loss = loss_prop + self.nu * loss_ratio * loss_cov

        loss.backward()
        optimizer.step()

        return loss_prop.item(), loss_cov.item()

    def _balanced_cov(self, X, y):
        """
        Method to compute number of balanced covariates. This should operate on all samples not on batches.
        :param X:
        :param y:
        :return: Number of balanced covariates
        """

        zeros = (y == 0).nonzero().squeeze()
        ones = y.nonzero().squeeze()

        with torch.no_grad():
            score = self.model(X).squeeze()
            weight = (y / (score + self.eps) + (1 - y) / (1 - score + self.eps)).unsqueeze(-1).repeat(1, X.shape[1])

            weight_zeros = torch.index_select(weight, 0, zeros)
            weight_ones = torch.index_select(weight, 0, ones)

            X_zeros = torch.index_select(X, 0, zeros)
            X_ones = torch.index_select(X, 0, ones)

            zeros_mean = torch.sum(weight_zeros * X_zeros, 0) / torch.sum(weight_zeros, 0)
            ones_mean = torch.sum(weight_ones * X_ones, 0) / torch.sum(weight_ones, 0)

            # Unbiased weighted variance (reliability weights)
            zeros_var = (torch.sum(weight_zeros, 0) /
                         ((torch.sum(weight_zeros, 0)) ** 2 - torch.sum(weight_zeros ** 2, 0)) *
                         torch.sum(weight_zeros * (X_zeros - zeros_mean) ** 2, 0))
            ones_var = (torch.sum(weight_ones, 0) /
                        ((torch.sum(weight_ones, 0)) ** 2 - torch.sum(weight_ones ** 2, 0)) *
                        torch.sum(weight_ones * (X_ones - ones_mean) ** 2, 0))

            # Handle calculation of norm_diff gracefully
            numer = torch.abs(zeros_mean - ones_mean)
            denom = torch.sqrt((zeros_var + ones_var) / 2)

            # Compute normalized difference where denominator is non_zero
            norm_diff = (torch.masked_select(numer, denom.ne(0)) / torch.masked_select(denom, denom.ne(0)))
            num_balanced = torch.sum(torch.le(norm_diff, self.balance_threshold)).item()

            # When denominator is zero compute cases where numerator is also zero
            num_numer_zero = torch.sum((torch.masked_select(numer, denom.eq(0))).eq(0)).item()
            num_balanced += num_numer_zero

            # When demoninator is zero and numerator is nonzero raise warning
            num_numer_nonzero = torch.sum((torch.masked_select(numer, denom.eq(0))).ne(0)).item()
            if num_numer_nonzero > 0 and self.verbose:
                self.logger.warning('Perfect separation detected for some covariates...')

        return num_balanced

    def fit(self, X, y):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if self.verbose and (self.logger is None):
            raise ValueError('If verbose is set to True, logger should be specified')

        device = torch.device(self.device)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)

        num_features = X.shape[1]

        if self.hidden_layer_size is not None:
            self.model = Propensity(num_features, self.hidden_layer_size, self.dropout)
        else:
            self.model = Propensity(num_features, 2 * num_features, self.dropout)

        self.model.to(device)

        criterion = (nn.BCELoss(), nn.MSELoss())
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, betas=(0.5, 0.999),
                               weight_decay=self.alpha)

        if self.batch_size is not None:
            ds = TensorDataset(X, y)
            trainloader = DataLoader(ds, self.batch_size, shuffle=self.shuffle, drop_last=True)

        prop_loss = []
        cov_loss = []
        num_balanced = []

        for i in range(self.max_iter):

            if self.batch_size is not None:
                prop_epoch_loss = 0.0
                cov_epoch_loss = 0.0
                for X0, y0 in trainloader:
                    loss_prop, loss_cov = self._step(X0, y0, optimizer, criterion)
                    prop_epoch_loss += loss_prop
                    cov_epoch_loss += loss_cov

                prop_loss.append(prop_epoch_loss / len(trainloader))
                cov_loss.append(cov_epoch_loss / len(trainloader))
            else:
                loss_prop, loss_cov = self._step(X, y, optimizer, criterion)
                prop_loss.append(loss_prop)
                cov_loss.append(loss_cov)

            num_balanced.append(self._balanced_cov(X, y))

            if self.early_stopping and len(num_balanced) > self.n_iter_no_change + 1:
                if ((num_balanced[-self.n_iter_no_change:] == num_balanced[-self.n_iter_no_change - 1:-1])
                        and (num_balanced[-1] == num_features)):
                    if self.verbose:
                        self.logger.info('All covariates balanced at epoch {}'.format(i))
                    break

            if self.verbose:
                if i % 50 == 0:
                    self.logger.info('Epoch ={}: Propensity Loss ={}, Covariate Loss ={}, Balanced covs ={}'
                                     .format(i, prop_loss[-1], cov_loss[-1], num_balanced[-1]))

        self.model = self.model.eval()
        self.loss_stats_ = (prop_loss, cov_loss, num_balanced)

        if self.verbose:
            self.logger.info('Number of balanced covariates at end of training:{}'.format(num_balanced[-1]))

        return self

    def predict(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        check_is_fitted(self, attributes=['loss_stats_'])
        # Input validation
        X = check_array(X)
        X = torch.tensor(X, dtype=torch.float32, device=torch.device(self.device))
        with torch.no_grad():
            score = self.model(X).squeeze().cpu().numpy()

        labels = (score >= 0.5).astype('int')

        return labels

    def predict_proba(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        check_is_fitted(self, attributes=['loss_stats_'])
        # Input validation
        X = check_array(X)
        X = torch.tensor(X, dtype=torch.float32, device=torch.device(self.device))
        with torch.no_grad():
            score = self.model(X).squeeze().cpu().numpy()

        return np.stack([1 - score, score], axis=1)

    def score(self, X, y):
        """
        Returns number of balanced covariates instead of accuracy score since
        during cross-validation, this is the metric we want to optimize.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        X = torch.tensor(X, dtype=torch.float32, device=torch.device(self.device))
        y = torch.tensor(y, dtype=torch.float32, device=torch.device(self.device))
        num_balanced = self._balanced_cov(X, y)

        return num_balanced
