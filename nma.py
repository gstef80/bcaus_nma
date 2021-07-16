import numpy as np  # type: ignore
import pymc3 as pm  # type: ignore

INPUT_COLS = [
    "stratum_name",
    "stratum_value",
    "control",
    "treatment",
    "TE",
    "seTE",
    "balanced",
]


class MetaRank:
    """
    Class to perform Bayesian Hierarchical Network Meta Analysis (NMA).
    This implements an NMA with a hierachical i.e. random-effects model

    References:
    For general description of the method:

    1) "Evidence Synthesis for Decision Making 2: A Generalized Linear Modeling Framework for
          Pairwise and Network Meta-analysis of Randomized Controlled Trials" by Dias et al.

    2) "Mixed Treatment Comparison Meta-Analysis of Complex Interventions:
          Psychological Interventions in Coronary Heart Disease" by Welton et al.

    For details on implementation, specifically choosing uninformative priors:

    1) "Automating network meta‐analysis" by Valkenhoef et al.

    For definition of SUCRA score as implemented here:

    1) Graphical methods and numerical summaries for presenting results from
          multiple-treatment meta-analysis: an overview and tutorial by Salanti et al.

    2) Ranking treatments in frequentist network meta-analysis works without resampling methods
          by Rucker et al.

    Note: This implementation handles only continuous outcomes. For binary outcomes this will
    need to be modified.

    How to use:
    Input dataframe X should have the following columns: "treatment", "control", "TE", "seTE"
    where TE and seTE are the treatment effect and the standard error of the treatment effect
    respectively.

    meta = MetaRank(baseline_tx = "Name_of_baseline_treatment_as_string", **kwargs)
    meta.fit(X).predict() will return a list of drugs and their ATEs wrt baseline
    sorted in descending order of SUCRA scores.
    """

    def __init__(
        self, baseline_tx, trace_samples=100000, burn_ratio=0.5, num_chains=4
    ):
        """
        baseline_tx: Name of treatment to use as baseline
        trace_samples: Number of samples to draw for MCMC trace
        burn_ratio: Fraction of samples to discard at the beginning of MCMC run
        num_chains: Number of MCMC chain to run simultaneously
        Note: The default values chosen above are based on NICE DSU TECHNICAL SUPPORT DOCUMENT 2.
              (http://nicedsu.org.uk/wp-content/uploads/2017/05/TSD2-General-meta-analysis-corrected-2Sep2016v2.pdf)
              Do not change them except when testing.
        """
        self.baseline_tx = baseline_tx
        self.trace_samples = trace_samples
        self.burn_ratio = burn_ratio
        self.num_chains = num_chains
        self.model = None
        self.trace = None

    def _generate_dict(self, X):
        """
        Generate a dictionary of treatments mapped to integers
        """
        tx = set(list(X["treatment"]) + list(X["control"])) - {self.baseline_tx}
        self.tx_dict = {self.baseline_tx: 0}
        self.tx_dict.update({k: v for k, v in zip(tx, range(1, len(tx) + 1))})

        return

    def _process_data(self, X):
        """
        Preprocess data before fitting
        """
        X0 = X.copy()
        # Map treatments to integers using dictionary
        X0["treatment_idx"] = X0["treatment"].map(self.tx_dict).astype("int")
        X0["control_idx"] = X0["control"].map(self.tx_dict).astype("int")

        return X0

    def _build_model(self, X):
        """
        Build Pymc3 model
        """

        with pm.Model() as model:
            # Specify uninformative hyper-priors a la Valkenhoef et al.
            # The baseline treatment is set to zero
            mu0 = pm.Normal("mu0", mu=0.0, sd=0.01, shape=(1, 1))
            # All other treatments have  broad priors
            mur = pm.Normal(
                "mur",
                mu=0,
                sd=15 * np.max(np.abs(X.TE)),
                shape=(len(self.tx_dict) - 1, 1),
            )
            mu = pm.math.concatenate((mu0, mur), axis=0)
            tau = pm.HalfCauchy("tau", beta=5.0)

            # Specify priors (non-centered)
            ate_std = pm.Normal("ate_std", mu=0, sigma=1, shape=len(X))
            mean_ate = (
                mu[list(X["treatment_idx"]), 0] - mu[list(X["control_idx"]), 0]
            )
            ate = pm.Deterministic("ate", mean_ate + tau * ate_std)

            # Compute Likelihood
            lklhd = pm.Normal(  # noqa: F841
                "lklhd", mu=ate, sigma=X["seTE"].values, observed=X["TE"].values
            )

        return model

    def _train_model(self):
        """
        Run MCMC with No U-Turn Sampler (NUTS)
        """

        with self.model:
            trace = pm.sample(
                draws=self.trace_samples,
                tune=10000,
                step=pm.NUTS(target_accept=0.99),
                chains=self.num_chains,
                init="auto",
                discard_tuned_samples=True,
                return_inferencedata=False,
            )

        # Check convergence with Gelman-Rubin diagnostic R.
        r_hat = np.max(pm.stats.summary(trace)["r_hat"])
        if r_hat > 1.1:
            print("Chains not converged...")

        # Discard samples at the beginning of trace
        trace = trace[int(self.burn_ratio * self.trace_samples) :]

        return trace

    def fit(self, X):
        """
        Fits models to data X
        """

        Xcols = set(X.columns)
        required_cols = {"treatment", "control", "TE", "seTE"}

        if Xcols.intersection(required_cols) != required_cols:
            raise KeyError(
                "Columns 'treatment', 'control', 'TE', 'seTE' not found in X"
            )

        # Generate dictionary
        self._generate_dict(X)

        # Process data
        X = self._process_data(X)

        # Build model
        self.model = self._build_model(X)

        # Train
        self.trace = self._train_model()

        return self

    def predict(self, ascending=True):
        """
        Returns list of drugs and ATEs wrt baseline treatment sorted by SUCRA score.
        If lower values of ATE are better set ascending to True,
        If higher values are better set to False
        """

        if not self.model:
            raise NameError("Fitted model not found. Run fit first")

        # To compute SUCRA score sample from the posterior
        with self.model:
            ppc = pm.sample_posterior_predictive(
                self.trace, var_names=["mu0", "mur"], random_seed=314
            )

        # For all draws from the posterior, compute rank
        joined_samples = np.squeeze(
            np.concatenate([ppc["mu0"], ppc["mur"]], axis=1)
        )
        ranks = np.argsort(np.argsort(joined_samples, axis=1), axis=1)

        # Compute mean rank across all draws
        mean_ranks = np.mean(ranks, axis=0)

        # Compute SUCRA a la Rucker et al
        num_treatments = len(self.tx_dict)
        sucra = (num_treatments - 1 - mean_ranks) / (num_treatments - 1)

        if not ascending:
            sucra = 1 - sucra

        summary = pm.stats.summary(self.trace)
        summary = summary.rename(
            {"mean": "ate", "hdi_3%": "ate_lb", "hdi_97%": "ate_ub"}, axis=1
        )
        summary = summary[:num_treatments]

        tx_inv_dict = {v: k for k, v in self.tx_dict.items()}
        summary["treatment"] = list(range(num_treatments))
        summary["treatment"] = summary["treatment"].map(tx_inv_dict)

        summary = summary[["treatment", "ate", "ate_lb", "ate_ub"]]
        summary["sucra"] = sucra

        return summary.sort_values("sucra", ascending=False).reset_index(
            drop=True
        )
