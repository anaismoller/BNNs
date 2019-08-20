# # Exploring BNNs predictions for moon dataset
# 1. fetch moons
# 2. write simple BNN
# 3. study predictions (classification probability + uncertainties)
import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from BBB_BNN import BNN
from sklearn import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec


#
# Init
#
path_figures = "./figures/"
os.makedirs(path_figures, exist_ok=True)
CMAP = plt.cm.YlOrBr
CMAP2 = plt.cm.winter

path_models = "./models/"
os.makedirs(path_models, exist_ok=True)

#
# Useful functions
#
def get_dataset(n_samples, noise=0.2, plots=False):
    #
    # Datatset
    #
    # Get dataset, reformat to pandas and visualize
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise)
    X, y = noisy_moons
    df_data = pd.DataFrame(X, columns=["x", "y"])
    df_data["label"] = y.reshape(-1)
    df_data["id"] = df_data.index

    if plots:
        plot_df_w_labels(df_data, outprefix="original")

    # torch data formatting
    X_tensor = torch.tensor(X)
    # y array of shape (n_samples,1)
    y = np.reshape(y, (len(y), 1))
    inputs = torch.tensor(X, dtype=torch.float)
    labels = torch.tensor(y, dtype=torch.long)
    labels = labels.reshape(-1)

    return inputs, labels, df_data


def train_network(rnn, settings, force_nonbayesian=False, plots=False):

    outprefix = 'NN' if force_nonbayesian else 'BNN'

    criterion = nn.CrossEntropyLoss(reduction="sum")
    rnn = rnn(settings)
    optimizer = torch.optim.Adam(
        rnn.parameters(), lr=settings["learning_rate"])

    if plots:
        # visualize untrained weigths
        visualize_weights_params(rnn, outprefix=outprefix, outsufix="init")

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

    if plots:
        # visualize trained weigths
        visualize_weights_params(rnn, outprefix=outprefix, outsufix="trained")


    torch.save(
                rnn.state_dict(),
                f"{path_models}/{outprefix}.pt",
            )

    return rnn

def get_rnn_predictions(rnn, force_nonbayesian=False):
    if force_nonbayesian:
        n_inference_samples = 1
    else:
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
    return df_preds

def plot_df_w_labels(df, df_original = pd.DataFrame(),key="label", outprefix="original"):
    plt.figure(figsize=(7,5))
    plot_bkgd(df_original)
    sel = df[df[key] == 0]
    plt.scatter(sel["x"], sel["y"], label=f"{key} 0", color="orange")
    sel = df[df[key] == 1]
    plt.scatter(sel["x"], sel["y"], label=f"{key} 1", color="indigo")
    plt.legend()
    plt.savefig(f"{path_figures}/{outprefix}_{key}.png")
    plt.clf()
    plt.close("all")

# useful function
def plot_bkgd(df_original):
    if len(df_original) != 0:
        # plot background all predictions
        # select by labels different grey colors
        sel = df_original[df_original['label'] == 0]
        plt.scatter(sel["x"], sel["y"], label=f"label 0", color="grey",s=40)
        sel = df_original[df_original['label'] == 1]
        plt.scatter(sel["x"], sel["y"], label=f"label 1", color="black",s=40)

def reformat_predictions(df_preds, df_data):
    # aggregate predictions
    df_preds_agg = df_preds.groupby(by="id").median()
    df_preds_agg[["class0_std", "class1_std"]
                 ] = df_preds.groupby(by="id").std()
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


def visualize_weights_params(rnn, outprefix='BNN', outsufix="init"):
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

    plt.savefig(f"{path_figures}/{outprefix}/weights_{outsufix}.png")


def visualize_predictions(merged_preds, df_original=pd.DataFrame(), outprefix="all"):
    """ visualize predictions
    """

    # predicted targets
    plot_df_w_labels(merged_preds,df_original=df_original,
                     key="predicted_target", outprefix=f"{outprefix}_preds")

    # colormap uncertainty
    plt.figure(figsize=(7,5))
    plot_bkgd(df_original)
    plt.scatter(
        merged_preds["x"],
        merged_preds["y"],
        c=merged_preds["class0_std"],
        cmap=CMAP,
        s=20,
        vmin=0,
        vmax=0.4,
    )
    cbar = plt.colorbar()
    plt.title(f"#{outprefix}:{len(merged_preds)}")
    plt.savefig(f"{path_figures}/{outprefix}_uncertainty_colormap.png")
    plt.clf()
    plt.close("all")

    # visualize classification probability for each predicted target
    # colorbar hack
    zs = np.concatenate(
        [merged_preds["class0"], merged_preds["class1"]], axis=0)
    min_, max_ = zs.min(), zs.max()
    norm = plt.Normalize(min_, max_)
    plt.figure(figsize=(10, 5))
    plot_bkgd(df_original)
    sel = merged_preds[merged_preds["predicted_target"] == 0]
    plt.scatter(sel["x"], sel["y"], c=sel["class0"],
                cmap=CMAP, s=20, norm=norm)
    cbar = plt.colorbar()
    sel = merged_preds[merged_preds["predicted_target"] == 1]
    plt.scatter(sel["x"], sel["y"], c=sel["class1"],
                cmap=CMAP2, s=20, norm=norm)
    cbar = plt.colorbar()
    plt.title(f"#{outprefix}:{len(merged_preds)}")
    plt.savefig(f"{path_figures}/{outprefix}_probability_colormap.png")
    plt.clf()
    plt.close("all")


if __name__ == '__main__':

    '''Parse arguments
    '''
    parser = argparse.ArgumentParser(
        description='Selection function data vs simulations')
    parser.add_argument('--n_samples', type=int,
                        default=1000,
                        help="generated samples")
    parser.add_argument('--retrain',
                        action="store_true",
                        help="retrain the network")
    parser.add_argument('--plot', 
                        action="store_true",
                        help="do plots")
    parser.add_argument('--nonbayesian', 
                        action="store_true",
                        help="non Bayesian NN")

    args = parser.parse_args()

    n_samples = args.n_samples
    retrain = args.retrain
    plots = args.plot
    force_nonbayesian = args.nonbayesian

    outprefix = 'NN' if force_nonbayesian else 'BNN'
    os.makedirs(f"{path_figures}/{outprefix}/", exist_ok=True)

    # dataset
    inputs, labels, df_data = get_dataset(n_samples, plots=plots)

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

    # network BBB
    rnn = BNN
    if retrain:
        rnn = train_network(rnn, settings, force_nonbayesian=force_nonbayesian, plots=plots)
    else:
        rnn = rnn(settings)
        rnn_state = torch.load(f'{path_models}/{outprefix}.pt', map_location=lambda storage, loc: storage)
        rnn.load_state_dict(rnn_state)
    #
    # Predictions
    #
    df_preds = get_rnn_predictions(rnn,force_nonbayesian=force_nonbayesian)

    # reformat, aggregate and visualize
    reformatted_preds = reformat_predictions(df_preds, df_data)

    # visualize
    visualize_predictions(reformatted_preds, outprefix=f"{outprefix}/preds")

    # visualize missclassified
    missclassified = reformatted_preds[
        reformatted_preds["predicted_target"] != reformatted_preds["label"]
    ]
    visualize_predictions(
        missclassified, df_original=reformatted_preds, outprefix=f"{outprefix}/missclassified")

    # visualize missclassified . small uncertainties
    if not force_nonbayesian:
        # set threshold to 1 sigma
        uncertainty_threshold = reformatted_preds['class0_std'].std()
        missclassified_w_smallstd = reformatted_preds[
            (reformatted_preds["predicted_target"] != reformatted_preds["label"]) & ((reformatted_preds['class0_std']<uncertainty_threshold) | (reformatted_preds['class1_std']<uncertainty_threshold))
        ]
        visualize_predictions(
            missclassified_w_smallstd, df_original=reformatted_preds, outprefix=f"{outprefix}/missclassified_w_std_lt_{round(uncertainty_threshold,2)}")


