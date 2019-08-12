# # Exploring BNNs predictions for moon dataset
# 1. fetch moons
# 2. write simple BNN
# 3. study predictions (classification probability + uncertainties)
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from simple_BNN import BNN
from sklearn import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec

#
# Useful functions
#


def plot_df_w_labels(df, key="label", outprefix="original"):
    plt.figure()
    sel = df[df[key] == 0]
    plt.scatter(sel["x"], sel["y"], label=f"{key} 0", color="blue")
    sel = df[df[key] == 1]
    plt.scatter(sel["x"], sel["y"], label=f"{key} 1", color="orange")
    plt.legend()
    plt.savefig(f"{path_figures}/{outprefix}_{key}.png")
    plt.clf()
    plt.close("all")


def reformat_predictions(df_preds, df_data):
    # aggregate predictions
    df_preds_agg = df_preds.groupby(by="id").median()
    df_preds_agg[["class0_std", "class1_std"]] = df_preds.groupby(by="id").std()
    df_preds[df_preds["id"] == 0]

    # get predicted target
    df_preds_agg["predicted_target"] = (
        df_preds_agg[[k for k in df_preds_agg.keys() if "std" not in k]]
        .idxmax(axis=1)
        .str.strip("class")
        .astype(int)
    )

    # merge with original data
    merged_preds = df_data.merge(df_preds_agg, on="id")

    return merged_preds


def visualize_weights_params(rnn, outsufix="init"):
    # visualize weights parameters (mu,rho)
    fig = plt.figure(figsize=(30, 20))

    gs = gridspec.GridSpec(3, rnn.nb_layers)

    for i in range(rnn.nb_layers):
        ax = plt.subplot(gs[0, i])
        ax.hist(
            rnn.layers[f"{i}"].mu.detach().view(-1).cpu().numpy(), bins=30, density=True
        )
        ax.tick_params(labelsize=16)
        ax.set_xlabel(f"layer {i} mu", fontsize=18)

        ax = plt.subplot(gs[1, i])
        ax.hist(
            F.softplus(rnn.layers[f"{i}"].rho.detach()).view(-1).cpu().numpy(),
            bins=30,
            density=True,
        )
        ax.tick_params(labelsize=16)
        ax.set_xlabel(f"layer {i} sigma", fontsize=18)

        # Blundel S/N (importance of weights)
        ax = plt.subplot(gs[2, i])
        ax.hist(
        rnn.layers[f"{i}"].mu.detach().view(-1).cpu().numpy()
        / F.softplus(rnn.layers[f"{i}"].rho.detach()).view(-1).cpu().numpy(),
        bins=30,
        density=True,
        )
        ax.tick_params(labelsize=16)
        ax.set_xlabel(f"layer {i} mu/sigma (S/N)", fontsize=18)

    plt.savefig(f"{path_figures}/weights_{outsufix}.png")


def visualize_bayesian_indicators(merged_preds, outprefix="all"):
    # visualize uncertainty on classification
    plt.figure()
    plt.scatter(
        merged_preds["x"],
        merged_preds["y"],
        c=merged_preds["class0_std"],
        cmap=CMAP,
        s=20,
    )
    cbar = plt.colorbar()
    plt.savefig(f"{path_figures}/{outprefix}_uncertainty_colormap.png")
    plt.clf()
    plt.close("all")

    # visualize classification probability for each predicted target
    # colorbar hack
    zs = np.concatenate([merged_preds["class0"], merged_preds["class1"]], axis=0)
    min_, max_ = zs.min(), zs.max()
    norm = plt.Normalize(min_, max_)
    plt.figure()
    sel = merged_preds[merged_preds["predicted_target"] == 0]
    plt.scatter(sel["x"], sel["y"], c=sel["class0"], cmap=CMAP, s=20, norm=norm)
    cbar = plt.colorbar()
    sel = merged_preds[merged_preds["predicted_target"] == 1]
    plt.scatter(sel["x"], sel["y"], c=sel["class1"], cmap=CMAP2, s=20, norm=norm)
    cbar = plt.colorbar()
    plt.savefig(f"{path_figures}/{outprefix}_probability_colormap.png")
    plt.clf()
    plt.close("all")


#
# Init
#
path_figures = "./figures/"
os.makedirs(path_figures, exist_ok=True)
CMAP = plt.cm.YlOrBr
CMAP2 = plt.cm.winter

# If we want a non Bayesian trained network
force_nonbayesian = False

#
# Datatset
#
# Get dataset, reformat to pandas and visualize
n_samples = 1000
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.2)
X, y = noisy_moons

df_data = pd.DataFrame(X, columns=["x", "y"])
df_data["label"] = y.reshape(-1)
df_data["id"] = df_data.index

plot_df_w_labels(df_data, outprefix="original")

#
# Simple BNN
#

# torch data formatting
X_tensor = torch.tensor(X)
# y array of shape (n_samples,1)
y = np.reshape(y, (len(y), 1))
inputs = torch.tensor(X, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.long)
labels = labels.reshape(-1)

# network settings
settings = {}
settings["nb_layers"] = 3
settings["learning_rate"] = 1e-3
settings["input_size"] = 2
settings["nb_classes"] = 2
settings["hidden_dim"] = 128
settings["pi"] = 0.75
settings["log_sigma1"] = -1.0
settings["log_sigma2"] = -7.0
settings["rho_scale_lower"] = 4.0
settings["rho_scale_upper"] = 3.0

rnn = BNN
criterion = nn.CrossEntropyLoss(reduction="sum")
rnn = rnn(settings)
optimizer = torch.optim.Adam(rnn.parameters(), lr=settings["learning_rate"])

visualize_weights_params(rnn, outsufix="init")

# train network
for epochs in range(1000):
    # Set NN to train mode (deals with dropout and batchnorm)
    rnn.train()

    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass
    output = rnn(inputs, force_nonbayesian=force_nonbayesian)
    clf_loss = criterion(output.squeeze(), labels)

    if force_nonbayesian:
        loss = (clf_loss) / inputs.shape[0]
    else:
        # Special case for BayesianRNN, need to use KL loss
        loss = (clf_loss + rnn.kl) / inputs.shape[0]

    # Backward pass
    loss.backward()
    optimizer.step()

    print(
        "e",
        epochs,
        "loss",
        loss.item(),
        "clf_loss",
        clf_loss.item(),
        "KL",
        rnn.kl.item(),
    )

# visualize trained weights
visualize_weights_params(rnn, outsufix="trained")

#
# Predictions
#

# sample
n_inference_samples = 20
dic_df_preds = {}
for inf in tqdm(range(n_inference_samples)):
    rnn.eval()
    output = rnn(inputs)
    pred_proba = nn.functional.softmax(output, dim=1)

    dic_df_preds[inf] = pd.DataFrame(
        pred_proba.data.cpu().numpy(), columns=["class0", "class1"]
    )
    dic_df_preds[inf]["id"] = dic_df_preds[inf].index
df_preds = pd.concat(dic_df_preds)

# reformat, aggregate and visualize
reformatted_preds = reformat_predictions(df_preds, df_data)

# visualize
plot_df_w_labels(reformatted_preds, key="predicted_target", outprefix="preds")
visualize_bayesian_indicators(reformatted_preds, outprefix="preds")
# visualize missclasified
missclassified = reformatted_preds[
    reformatted_preds["predicted_target"] != reformatted_preds["label"]
]
visualize_bayesian_indicators(missclassified, outprefix="preds_missclassified")
