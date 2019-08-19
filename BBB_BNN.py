import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BNN(nn.Module):
    def __init__(self, settings):
        super().__init__()

        # init
        self.nb_classes = settings["nb_classes"]
        self.input_size = settings["input_size"]
        self.hidden_size = settings["hidden_dim"]
        self.nb_layers = settings["nb_layers"]

        # we could use different priors for all layers
        # e.g. Fortunato uses a wider prior in the output layer
        # this could help having larger uncertainties in the class prob.
        self.prior = Prior(
            settings["pi"], settings["log_sigma1"], settings["log_sigma2"]
        )

        print(self.prior)

        #
        self.layers = nn.ModuleDict()
        for i in range(self.nb_layers):
            input_size = self.hidden_size
            output_size = self.hidden_size
            if i == 0:
                input_size = self.input_size
            elif i == self.nb_layers - 1:
                output_size = self.nb_classes   

            self.layers[f"{i}"] = BayesLinear(
                input_size,
                output_size,
                self.prior,
                mu_lower=-0.2,
                mu_upper=0.2,
                rho_lower=math.log(
                    math.exp(
                        self.prior.sigma_mix / settings["rho_scale_lower"]
                    )
                    - 1.0
                ),
                rho_upper=math.log(
                    math.exp(
                        self.prior.sigma_mix / settings["rho_scale_upper"]
                    )
                    - 1.0
                ),
            )

    def forward(self, x, force_nonbayesian=False):

        self.kl = 0
        for i in range(self.nb_layers):
            x = self.layers[f"{i}"](x, force_nonbayesian=force_nonbayesian)
            if i != self.nb_layers - 1:
                x = F.relu(x)
            # KL
            self.kl += self.layers[f"{i}"].kl

        return x


class BayesLinear(nn.Module):
    def __init__(
        self, in_features, out_features, prior, mu_lower, mu_upper, rho_lower, rho_upper
    ):
        # initializing the parent class
        # (nn.Module)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior

        # we could initialize mu,rho in any way
        # empirically is better to use the priors as reference
        mu, rho = get_bbb_variable(
            (out_features, in_features), mu_lower, mu_upper, rho_lower, rho_upper
        )

        bias = nn.Parameter(torch.Tensor(out_features))
        bias.data.fill_(0.0)

        self.mu = mu
        self.rho = rho
        self.bias = bias
        self.kl = None

    def forward(self, input, force_nonbayesian=False):

        # Sample weight
        mean = self.mu
        sigma = F.softplus(self.rho) + 1e-5

        if force_nonbayesian:
            # when sampling the weight we just use the mean
            # deterministic weight
            weights = mean
        else:
            # Obtain posterior sample of the weights (gaussian)
            # by sampling epsilon normal distribution
            # This way of creating the epsilon variable is faster than
            # from numpy or torch.randn or FloatTensor.normal_ when mean is on the GPU
            eps = mean.data.new(mean.size()).normal_(0.0, 1.0)
            weights = mean + eps * sigma

        logits = F.linear(input, weights, self.bias)

        # Compute KL divergence
        self.kl = compute_KL(weights, mean, sigma, self.prior)

        return logits

    def __repr__(self):
        # printing function
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class Prior(object):
    def __init__(self, pi=0.25, log_sigma1=-1.0, log_sigma2=-7.0):
        # the prior is initialized as mixture of Gaussians

        self.pi_mixture = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = math.exp(log_sigma1)
        self.sigma2 = math.exp(log_sigma2)

        self.sigma_mix = math.sqrt(
            pi * math.pow(self.sigma1, 2) + (1.0 - pi) * math.pow(self.sigma2, 2)
        )

    def __repr__(self):
        # printing function
        return (
            self.__class__.__name__
            + " ("
            + str(self.pi_mixture)
            + ", "
            + str(self.log_sigma1)
            + ", "
            + str(self.log_sigma2)
            + ")"
        )


def get_bbb_variable(shape, mu_lower, mu_upper, rho_lower, rho_upper):
    # Initialize weights using prior initialization

    mu = nn.Parameter(torch.Tensor(*shape))
    rho = nn.Parameter(torch.Tensor(*shape))

    mu.data.uniform_(mu_lower, mu_upper)
    rho.data.uniform_(rho_lower, rho_upper)

    return mu, rho


def compute_KL(x, mu, sigma, prior):

    posterior = torch.distributions.Normal(mu.view(-1), sigma.view(-1))
    log_posterior = posterior.log_prob(x.view(-1)).sum()

    if x.is_cuda:
        n1 = torch.distributions.Normal(
            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma1]).cuda()
        )
        n2 = torch.distributions.Normal(
            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma2]).cuda()
        )
    else:
        n1 = torch.distributions.Normal(0.0, prior.sigma1)
        n2 = torch.distributions.Normal(0.0, prior.sigma2)

    mix1 = torch.sum(n1.log_prob(x)) + math.log(prior.pi_mixture)
    mix2 = torch.sum(n2.log_prob(x)) + math.log(1.0 - prior.pi_mixture)
    prior_mix = torch.stack([mix1, mix2])
    log_prior = torch.logsumexp(prior_mix, 0)

    return log_posterior - log_prior
