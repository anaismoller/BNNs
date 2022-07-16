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
from matplotlib.colorbar import Colorbar

plt.style.use(f"file://{os.getcwd()}/pltstyle_supernnova.mplstyle")

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


def get_dataset(typ, n_samples, noise=0.2, plots=False):
    #
    # Datatset
    #
    # Get dataset, reformat to pandas and visualize
    if typ == "moons":
        dataset = datasets.make_moons(n_samples=n_samples, noise=noise)
    if typ == "circles":
        dataset = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    if typ == "blobs":
        dataset = datasets.make_blobs(
            n_samples=n_samples, centers=[[-0.25, 0.5], [1.2, 0.25]], cluster_std=noise
        )
    if typ == "displaced_blobs":
        dataset = datasets.make_blobs(
            n_samples=n_samples, centers=[[0, 0.5], [1, 0.25]], cluster_std=noise
        )
    X, y = dataset
    # # fixing label for blobs
    # if typ == 'blobs':
    #     y=(~y.astype(bool)).astype(int)

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


def train_network(
    rnn,
    settings,
    inputs,
    labels,
    force_nonbayesian=False,
    plots=False,
    outmodel_name="./models/BNN/TRmoon_TEmoon.pt",
):

    criterion = nn.CrossEntropyLoss(reduction="sum")
    rnn = rnn(settings)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=settings["learning_rate"])

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

    torch.save(rnn.state_dict(), outmodel_name)

    return rnn


def train_network_SWA(
    rnn, settings, inputs, labels, outmodel_name="./models/BNN/TRmoon_TEmoon_SWA.pt",
):

    from torch.optim.swa_utils import AveragedModel, SWALR
    from torch.optim.lr_scheduler import CosineAnnealingLR

    criterion = nn.CrossEntropyLoss(reduction="sum")
    rnn = rnn(settings)
    swa_model = AveragedModel(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=settings["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    swa_start = 5
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    for epoch in range(1000):
        #   rnn.train()
        optimizer.zero_grad()
        # Forward pass
        output = rnn(inputs, force_nonbayesian=True)
        criterion(output.squeeze(), labels).backward()
        optimizer.step()
        if epoch > swa_start:
            swa_model.update_parameters(rnn)
            swa_scheduler.step()
        else:
            scheduler.step()

    # Update bn statistics for the swa_model at the end
    torch.optim.swa_utils.update_bn((input, labels), swa_model)

    torch.save(swa_model.state_dict(), outmodel_name)

    return swa_model


def get_rnn_predictions(rnn, inputs, force_nonbayesian=False):
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


#
# pLotting functions
#


def plot_bkgd(df_original):
    if len(df_original) != 0:
        # plot background all predictions
        # select by labels different grey colors
        sel = df_original[df_original["label"] == 0]
        plt.scatter(sel["x"], sel["y"], label=f"label 0", color="grey")
        sel = df_original[df_original["label"] == 1]
        plt.scatter(sel["x"], sel["y"], label=f"label 1", color="black")


def plot_df_w_labels(df, df_original=pd.DataFrame(), key="label", outprefix="original"):
    plt.figure(figsize=(7, 5))
    plot_bkgd(df_original)
    sel = df[df[key] == 0]
    plt.scatter(sel["x"], sel["y"], label=f"{key} 0", color="orange")
    sel = df[df[key] == 1]
    plt.scatter(sel["x"], sel["y"], label=f"{key} 1", color="indigo")
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


def visualize_weights_params(rnn, outprefix="BNN", outsufix="init"):
    # visualize weights parameters (mu,rho)
    fig = plt.figure(figsize=(30, 20))

    gs = gridspec.GridSpec(3, rnn.nb_layers)

    for i in range(rnn.nb_layers):
        ax = plt.subplot(gs[0, i])
        ax.hist(
            rnn.layers[f"{i}"].mu.detach().view(-1).cpu().numpy(), bins=30, density=True
        )
        ax.tick_params(labelsize=16)
        ax.set_xlabel(f"layer {i} mu")

        ax = plt.subplot(gs[1, i])
        ax.hist(
            F.softplus(rnn.layers[f"{i}"].rho.detach()).view(-1).cpu().numpy(),
            bins=30,
            density=True,
        )
        ax.tick_params(labelsize=16)
        ax.set_xlabel(f"layer {i} sigma")

        # Blundel S/N (importance of weights)
        ax = plt.subplot(gs[2, i])
        ax.hist(
            rnn.layers[f"{i}"].mu.detach().view(-1).cpu().numpy()
            / F.softplus(rnn.layers[f"{i}"].rho.detach()).view(-1).cpu().numpy(),
            bins=30,
            density=True,
        )
        ax.tick_params(labelsize=16)
        ax.set_xlabel(f"layer {i} mu/sigma (S/N)")

    plt.savefig(f"{path_figures}/{outprefix}/weights_{outsufix}.png")


def paper_visualize_predictions(
    train_dataset_tuple, dic_preds, outnamefix, nn_type="BNN"
):

    keys = ["class0", "class0_std"]
    if nn_type == "BNN":
        outnames = ["probability", "uncertainty"]
    else:
        outnames = ["probability"]
    for outname, key in zip(outnames, keys):
        cmap = plt.cm.RdYlBu if outname == "probability" else plt.cm.YlOrBr_r
        fig = plt.figure(figsize=(50, 15))
        gs = gridspec.GridSpec(
            3,
            4,
            width_ratios=[0.25, 0.25, 0.25, 0.25],
            height_ratios=[0.45, 0.45, 0.1],
            wspace=0.01,
            hspace=0.1,
        )
        #
        ax0 = plt.subplot(gs[:-1, 0])
        train_noise, train_inputs, train_labels = train_dataset_tuple
        list_colors_to_use = ["darkblue", "darkred"]
        list_colors = [list_colors_to_use[k] for k in train_labels.data.numpy()]
        plt.scatter(train_inputs[:, 0], train_inputs[:, 1], c=list_colors, s=350)
        ax0.set_xlim(-1.5, 2.5)
        ax0.set_ylim(-1, 1.5)
        ax0.tick_params(labelsize=16)
        ax0.set_title(f"training set (noise {train_noise})", fontsize=26)
        # preds
        for i in range(3):
            noise_value = [k for k in dic_preds.keys()][i]
            df = dic_preds[noise_value]
            ax = plt.subplot(gs[:-1, i + 1], sharey=ax0)
            plt1 = ax.scatter(
                df["x"],
                df["y"],
                c=df[key],
                cmap=cmap,
                s=350,
                vmin=0.0,
                vmax=df[key].max(),
            )
            # overimpose missclassified points
            sel = df[df["label"] != df["predicted_target"]]
            if len(sel) > 0:
                edgecolors = [
                    "darkblue" if sel["label"].iloc[i] == 0 else "darkred"
                    for i in range(len(sel))
                ]
                plt2 = ax.scatter(
                    sel["x"],
                    sel["y"],
                    c=sel[key],
                    cmap=cmap,
                    s=400,
                    vmin=0.0,
                    vmax=df[key].max(),
                    edgecolors=edgecolors,
                    linewidths=5,
                )
            ax.tick_params(labelsize=16)
            ax.set_xlim(-1.5, 2.5)
            ax.set_ylim(-1, 1.5)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_title(f"noise {round(noise_value,1)}", fontsize=26)

        plt.tight_layout()

        cb = Colorbar(
            ax=plt.subplot(gs[-1, 1:]),
            mappable=plt1,
            orientation="horizontal",
            ticklocation="bottom",
        )
        cb.ax.tick_params(labelsize=16)
        cb.set_label(f"classification {outname}", fontsize=26)
        plt.tight_layout()
        plt.savefig(f"{path_figures}/{outnamefix}_{outname}.png")
        plt.clf()
        plt.close("all")


if __name__ == "__main__":

    """Parse arguments
    """
    parser = argparse.ArgumentParser(
        description="Selection function data vs simulations"
    )
    parser.add_argument(
        "--test_samples", type=int, default=1000, help="generated samples"
    )
    parser.add_argument(
        "--train_samples", type=int, default=10000, help="generated samples"
    )
    parser.add_argument(
        "--train_type", type=str, default="moons", help="train dataset type"
    )
    parser.add_argument(
        "--test_type", type=str, default="moons", help="test dataset type"
    )
    parser.add_argument("--train_noise", type=float, default=0.2, help="train noise")
    parser.add_argument(
        "--no_train", action="store_true", help="do not train the network"
    )
    parser.add_argument("--plot", action="store_true", help="do plots")
    parser.add_argument("--nonbayesian", action="store_true", help="non Bayesian NN")
    parser.add_argument("--SWA", action="store_true", help="SWA")

    args = parser.parse_args()

    train_samples = args.train_samples
    test_samples = args.test_samples
    train = True if not args.no_train else False
    plots = args.plot
    force_nonbayesian = args.nonbayesian
    train_type = args.train_type
    test_type = args.test_type
    train_noise = args.train_noise

    nn_type = "NN" if force_nonbayesian else "BNN"
    extra_prefix = "_SWA" if args.SWA else ""
    trainprefix = (
        f"{nn_type}{extra_prefix}_TR{train_type}_N{str(train_noise).strip('0.')}"
    )
    outprefix = f"{trainprefix}{extra_prefix}_TE{test_type}"
    os.makedirs(f"{path_figures}/{nn_type}/", exist_ok=True)
    os.makedirs(f"{path_models}/{nn_type}/", exist_ok=True)

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

    # training dataset
    train_inputs, train_labels, _ = get_dataset(
        train_type, train_samples, noise=train_noise, plots=plots
    )
    train_dataset_tuple = (train_noise, train_inputs, train_labels)

    # network BBB
    rnn = BNN
    if train:
        if args.SWA:
            rnn = train_network_SWA(
                rnn,
                settings,
                train_inputs,
                train_labels,
                outmodel_name=f"{path_models}/{nn_type}/{trainprefix}.pt",
            )
        else:
            rnn = train_network(
                rnn,
                settings,
                train_inputs,
                train_labels,
                outmodel_name=f"{path_models}/{nn_type}/{trainprefix}.pt",
                force_nonbayesian=force_nonbayesian,
                plots=plots,
            )
    else:
        rnn = rnn(settings)
        rnn_state = torch.load(
            f"{path_models}/{nn_type}/{trainprefix}.pt",
            map_location=lambda storage, loc: storage,
        )
        rnn.load_state_dict(rnn_state)
    #
    # Predictions
    #
    dic_preds = {}
    list_noise_values = [0.1, 0.2, 0.3] if "blobs" not in test_type else [0.3, 0.4, 0.5]
    for noise_value in list_noise_values:
        # get preds
        test_inputs, test_labels, test_df_data = get_dataset(
            test_type, test_samples, noise=noise_value
        )
        df_preds = get_rnn_predictions(
            rnn, test_inputs, force_nonbayesian=force_nonbayesian
        )
        dic_preds[noise_value] = reformat_predictions(df_preds, test_df_data)

    # paper visualizations
    outname = f"{nn_type}/{outprefix}_baseline"
    print(outname)
    paper_visualize_predictions(
        train_dataset_tuple, dic_preds, outname, nn_type=nn_type
    )

