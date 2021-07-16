from abc import ABC
from typing import Optional

import numpy as np  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    check_array,
    check_is_fitted,
    check_X_y,
)
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from causal_rec.settings.model import BcausSettings


class BCausNet(nn.Module, ABC):
    def __init__(self, input_size, hidden_size, dropout_prob):
        """
        Define Pytorch model
        :param input_size: Number of covariates
        :param hidden_size: Number of hidden layer neurons
        :param dropout_prob: Dropout probability
        """
        super(BCausNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.layer1.weight, mode="fan_in")
        self.do1 = nn.Dropout(p=dropout_prob)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.layer2.weight, mode="fan_in")
        self.do2 = nn.Dropout(p=dropout_prob)
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.do1(self.layer1(x)))
        x = F.relu(self.do2(self.layer2(x)))
        x = torch.sigmoid(self.layer3(x))
        return x


class BCausModel(BaseEstimator, ClassifierMixin):
    def __init__(self, settings: Optional[BcausSettings]):
        """
        :param settings: optional BcausSettings settings
        BcausSettings is comprised of the following variables
        random_state: Random seed for torch and numpy
        hidden_layer_size: Number of neurons in hidden layer. If None set to 2x number of covariates
        batch_size: If None all samples are used. To achieve max balance in covariates it is recommended
                           that this be set to None unless sample size is very large e.g. > 1000
        shuffle: If True batches are shuffled. Ignored if batch_size is None
        learning_rate_init: Initial learning rate. Learning rate scheduler is not implemented in this version.
        nu: Determines relative balance between classification loss and balance loss. Backpropagated loss
                   is the sum of two terms: BCELoss() + nu * (bceloss/mseloss)*MSELoss()
        max_iter: Number of epochs
        alpha: L2 penalty term in loss function
        dropout: Dropout probability
        eps: Constant to stabilize denominator while computing IPTW weights
        early_stopping: If True training terminates when all covariates are balanced and remain so
                               for n_iter_no_change epochs
        n_iter_no_change:
        balance_threshold: Threshold for considering covariates as being balanced; default 0.1
        device: Device to train model on; default 'cpu'
        verbose: Log losses every 50 epochs if True
        logger: Python logger object. Should be provided if verbose = True
        """
        if settings is None:
            settings = BcausSettings()
        self.random_state = settings.random_state
        self.hidden_layer_size = settings.hidden_layer_size
        self.batch_size = settings.batch_size
        self.shuffle = settings.shuffle
        self.learning_rate_init = settings.learning_rate_init
        self.nu = settings.nu
        self.max_iter = settings.max_iter
        self.alpha = settings.alpha
        self.dropout = settings.dropout
        self.eps = settings.eps
        self.early_stopping = settings.early_stopping
        self.n_iter_no_change = settings.n_iter_no_change
        self.balance_threshold = settings.balance_threshold
        self.device = settings.device
        self.verbose = settings.verbose
        self.logger = settings.logger
        self.model = None
        self.loss_stats_ = (None, None, None)

    def _zeros_and_ones(self, x, y, score):
        """
        computes weight indices for treated and untreated x
        :param x: covariate
        :param y: treatment variable (binary)
        :param score: weights (e.g. IPTW)
        :return: untreated/treated x and corresponding weight indices
        """

        zeros = (y == 0).nonzero(as_tuple=False).squeeze()
        ones = y.nonzero(as_tuple=False).squeeze()

        weight = (
            (y / (score + self.eps) + (1 - y) / (1 - score + self.eps))
            .unsqueeze(-1)
            .repeat(1, x.shape[1])
        )

        weight_zeros = torch.index_select(weight, 0, zeros)
        weight_ones = torch.index_select(weight, 0, ones)

        X_zeros = torch.index_select(x, 0, zeros)
        X_ones = torch.index_select(x, 0, ones)

        return X_zeros, X_ones, weight_zeros, weight_ones

    def _step(self, x, y, optimizer, criterion):
        """
        Method to perform one step of forward + back + update weights
        :param x:
        :param y:
        :param optimizer: Pytorch optimizer
        :param criterion: Tuple of Pytorch loss functions (BCELoss(), MSELoss())
        :return: bceloss, mseloss
        """

        prop_criterion, cov_criterion = criterion

        # set_to_none=True optimization described in:
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        optimizer.zero_grad(set_to_none=True)
        score = self.model(x).squeeze()

        # Propensity BCE loss
        loss_prop = prop_criterion(score, y)

        # Covariates balance loss
        X_zeros, X_ones, weight_zeros, weight_ones = self._zeros_and_ones(
            x, y, score
        )
        zeros_mean = torch.sum(weight_zeros * X_zeros, 0) / torch.sum(
            weight_zeros, 0
        )
        ones_mean = torch.sum(weight_ones * X_ones, 0) / torch.sum(
            weight_ones, 0
        )
        loss_cov = cov_criterion(zeros_mean, ones_mean)
        loss_ratio = (loss_prop / loss_cov).item()
        loss = loss_prop + self.nu * loss_ratio * loss_cov
        loss.backward()
        optimizer.step()

        return loss_prop.item(), loss_cov.item()

    def _balanced_cov(self, x, y):
        """
        Method to compute number of balanced covariates. This should operate on
        all samples not on batches.
        :param x:
        :param y:
        :return: Number of balanced covariates
        """

        with torch.no_grad():
            score = self.model(x).squeeze()

            X_zeros, X_ones, weight_zeros, weight_ones = self._zeros_and_ones(
                x, y, score
            )
            weight_zeros_sum = torch.sum(weight_zeros, 0)
            weight_ones_sum = torch.sum(weight_ones, 0)

            # Weighted means
            zeros_mean = torch.sum(weight_zeros * X_zeros, 0) / weight_zeros_sum
            ones_mean = torch.sum(weight_ones * X_ones, 0) / weight_ones_sum

            # Unbiased weighted variance (reliability weights)
            zeros_var = (
                weight_zeros_sum
                / (weight_zeros_sum ** 2 - torch.sum(weight_zeros ** 2, 0))
                * torch.sum(weight_zeros * (X_zeros - zeros_mean) ** 2, 0)
            )
            ones_var = (
                weight_ones_sum
                / (weight_ones_sum ** 2 - torch.sum(weight_ones ** 2, 0))
                * torch.sum(weight_ones * (X_ones - ones_mean) ** 2, 0)
            )

            # Handle calculation of norm_diff gracefully
            numer = torch.abs(zeros_mean - ones_mean)
            denom = torch.sqrt((zeros_var + ones_var) / 2)

            numer_nonzero = torch.masked_select(numer, denom.ne(0))
            denom_nonzero = torch.masked_select(denom, denom.ne(0))
            numer_zero = torch.masked_select(numer, denom.eq(0))

            # Compute normalized difference where denominator is non_zero
            norm_diff = numer_nonzero / denom_nonzero
            num_balanced = torch.sum(
                torch.le(norm_diff, self.balance_threshold)
            ).item()

            # When denominator is zero compute cases where numerator is also zero
            num_numer_zero = torch.sum((numer_zero).eq(0)).item()
            num_balanced += num_numer_zero

            # When demoninator is zero and numerator is nonzero raise warning
            num_numer_nonzero = torch.sum((numer_zero).ne(0)).item()
            if num_numer_nonzero > 0 and self.verbose:
                self.logger.warning(
                    "Perfect separation detected for some covariates..."
                )

        return num_balanced

    def fit(self, x, y):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if self.verbose and (self.logger is None):
            raise ValueError(
                "If verbose is set to True, logger should be specified"
            )

        device = torch.device(self.device)

        # Check that X and y have correct shape
        X, y = check_X_y(x, y)

        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        self.eps = torch.tensor(self.eps, dtype=torch.float32, device=device)

        num_features = X.shape[1]

        if self.hidden_layer_size is not None:
            self.model = BCausNet(
                num_features, self.hidden_layer_size, self.dropout
            )
        else:
            self.model = BCausNet(num_features, 2 * num_features, self.dropout)

        self.model.to(device)

        criterion = (nn.BCELoss(), nn.MSELoss())
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate_init,
            betas=(0.5, 0.999),
            weight_decay=self.alpha,
        )

        if self.batch_size is not None:
            ds = TensorDataset(X, y)
            trainloader = DataLoader(
                ds,
                self.batch_size,
                shuffle=self.shuffle,
                drop_last=True,
                pin_to_memory=True,
            )

        prop_loss = []
        cov_loss = []
        balanced = []

        for i in range(self.max_iter):
            if self.batch_size is not None:
                prop_epoch_loss = 0.0
                cov_epoch_loss = 0.0
                for X0, y0 in trainloader:
                    loss_prop, loss_cov = self._step(
                        X0, y0, optimizer, criterion
                    )
                    prop_epoch_loss += loss_prop
                    cov_epoch_loss += loss_cov

                prop_loss.append(prop_epoch_loss / len(trainloader))
                cov_loss.append(cov_epoch_loss / len(trainloader))
            else:
                loss_prop, loss_cov = self._step(X, y, optimizer, criterion)
                prop_loss.append(loss_prop)
                cov_loss.append(loss_cov)

            balanced.append(self._balanced_cov(X, y))

            if (
                self.early_stopping
                and len(balanced) > self.n_iter_no_change + 1
            ):
                if (
                    balanced[-self.n_iter_no_change :]
                    == balanced[-self.n_iter_no_change - 1 : -1]
                ) and (balanced[-1] == num_features):
                    if self.verbose:
                        self.logger.info(
                            "All covariates balanced at epoch {}".format(i)
                        )
                    break

            if self.verbose:
                if i % 50 == 0:
                    self.logger.info(
                        "Epoch ={}: Propensity Loss ={}, Covariate Loss ={}, Balanced covs ={}".format(
                            i, prop_loss[-1], cov_loss[-1], balanced[-1]
                        )
                    )

        self.model = self.model.eval()
        self.loss_stats_ = (prop_loss, cov_loss, balanced)

        if self.verbose:
            self.logger.info(
                "Number of balanced covariates at end of training:{}".format(
                    balanced[-1]
                )
            )

        return self

    def predict(self, x):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        check_is_fitted(self, attributes=["loss_stats_"])
        # Input validation
        X = check_array(x)
        X = torch.tensor(
            X, dtype=torch.float32, device=torch.device(self.device)
        )
        with torch.no_grad():
            score = self.model(X).squeeze().cpu().numpy()

        labels = (score >= 0.5).astype("int")

        return labels

    def predict_proba(self, x):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        check_is_fitted(self, attributes=["loss_stats_"])
        # Input validation
        X = check_array(x)
        X = torch.tensor(
            X, dtype=torch.float32, device=torch.device(self.device)
        )
        with torch.no_grad():
            score = self.model(X).squeeze().cpu().numpy()

        return np.stack([1 - score, score], axis=1)

    def score(self, x, y, sample_weight=None):
        """
        Returns number of balanced covariates instead of accuracy score since
        during cross-validation, this is the metric we want to optimize.
        """
        # Check that x and y have correct shape
        X, y = check_X_y(x, y)

        X = torch.tensor(
            X, dtype=torch.float32, device=torch.device(self.device)
        )
        y = torch.tensor(
            y, dtype=torch.float32, device=torch.device(self.device)
        )
        num_balanced = self._balanced_cov(X, y)

        return num_balanced
