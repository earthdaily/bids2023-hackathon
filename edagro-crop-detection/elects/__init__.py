from torch.utils.data import DataLoader
from .earlyrnn import EarlyRNN
import torch
from tqdm import tqdm
from .loss import EarlyRewardLoss
import numpy as np
from .utils import VisdomLogger
import sklearn.metrics
import pandas as pd
import os


def train(
    train_ds,
    test_ds,
    n_classes,
    n_bands,
    batchsize=256,
    device="cpu",
    use_visdom=False,
    learning_rate=1e-3,
    epsilon=10,
    alpha=0.5,
    epochs=100,
    patience=30,
    weight_decay=0,
    snapshot="snapshots/model.pth",
    resume=False,
):
    sequencelength = train_ds[0][0].shape[0]
    traindataloader = DataLoader(train_ds, batch_size=batchsize)
    testdataloader = DataLoader(test_ds, batch_size=batchsize)

    model = EarlyRNN(nclasses=n_classes, input_dim=n_bands).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # exclude decision head linear bias from weight decay
    decay, no_decay = list(), list()
    for name, param in model.named_parameters():
        if name == "stopping_decision_head.projection.0.bias":
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": no_decay, "weight_decay": 0, "lr": learning_rate},
            {"params": decay},
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = EarlyRewardLoss(alpha=alpha, epsilon=epsilon)

    if resume and os.path.exists(snapshot):
        model.load_state_dict(torch.load(snapshot, map_location=device))
        optimizer_snapshot = os.path.join(
            os.path.dirname(snapshot),
            os.path.basename(snapshot).replace(".pth", "_optimizer.pth"),
        )
        optimizer.load_state_dict(
            torch.load(optimizer_snapshot, map_location=device)
        )
        df = pd.read_csv(snapshot + ".csv")
        train_stats = df.to_dict("records")
        start_epoch = train_stats[-1]["epoch"]
        print(f"resuming from {snapshot} epoch {start_epoch}")
    else:
        train_stats = []
        start_epoch = 1

    if use_visdom:
        visdom_logger = VisdomLogger()

    not_improved = 0
    with tqdm(range(start_epoch, epochs + 1)) as pbar:
        for epoch in pbar:
            trainloss = train_epoch(
                model,
                traindataloader,
                optimizer,
                criterion,
                device=device,
            )
            testloss, stats = test_epoch(
                model, testdataloader, criterion, device
            )

            # statistic logging and visualization...
            (
                precision,
                recall,
                fscore,
                support,
            ) = sklearn.metrics.precision_recall_fscore_support(
                y_pred=stats["predictions_at_t_stop"][:, 0],
                y_true=stats["targets"][:, 0],
                average="macro",
                zero_division=0,
            )
            accuracy = sklearn.metrics.accuracy_score(
                y_pred=stats["predictions_at_t_stop"][:, 0],
                y_true=stats["targets"][:, 0],
            )
            kappa = sklearn.metrics.cohen_kappa_score(
                stats["predictions_at_t_stop"][:, 0], stats["targets"][:, 0]
            )

            classification_loss = stats["classification_loss"].mean()
            earliness_reward = stats["earliness_reward"].mean()
            earliness = 1 - (stats["t_stop"].mean() / (sequencelength - 1))

            stats["confusion_matrix"] = sklearn.metrics.confusion_matrix(
                y_pred=stats["predictions_at_t_stop"][:, 0],
                y_true=stats["targets"][:, 0],
            )

            train_stats.append(
                dict(
                    epoch=epoch,
                    trainloss=trainloss,
                    testloss=testloss,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    kappa=kappa,
                    earliness=earliness,
                    classification_loss=classification_loss,
                    earliness_reward=earliness_reward,
                )
            )
            df = pd.DataFrame(train_stats).set_index("epoch")
            if use_visdom:
                visdom_logger(stats)
                visdom_logger.plot_boxplot(
                    stats["targets"][:, 0],
                    stats["t_stop"][:, 0],
                    tmin=0,
                    tmax=sequencelength,
                )
                visdom_logger.plot_epochs(
                    df[["precision", "recall", "fscore", "kappa"]],
                    name="accuracy metrics",
                )
                visdom_logger.plot_epochs(
                    df[["trainloss", "testloss"]], name="losses"
                )
                visdom_logger.plot_epochs(
                    df[["accuracy", "earliness"]], name="accuracy, earliness"
                )
                visdom_logger.plot_epochs(
                    df[["classification_loss", "earliness_reward"]],
                    name="loss components",
                )

            savemsg = ""
            if len(df) > 2:
                if testloss < df.testloss[:-1].values.min():
                    savemsg = f"saving model to {snapshot}"
                    os.makedirs(os.path.dirname(snapshot), exist_ok=True)
                    torch.save(model.state_dict(), snapshot)

                    optimizer_snapshot = os.path.join(
                        os.path.dirname(snapshot),
                        os.path.basename(snapshot).replace(
                            ".pth", "_optimizer.pth"
                        ),
                    )
                    torch.save(optimizer.state_dict(), optimizer_snapshot)

                    df.to_csv(snapshot + ".csv")
                    not_improved = 0  # reset early stopping counter
                else:
                    not_improved += 1  # increment early stopping counter
                    if patience is not None:
                        savemsg = f"early stopping in {patience - not_improved} epochs."
                    else:
                        savemsg = ""

            pbar.set_description(
                f"epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, "
                f"accuracy {accuracy:.2f}, earliness {earliness:.2f}. "
                f"classification loss {classification_loss:.2f}, earliness reward {earliness_reward:.2f}. {savemsg}"
            )

            if patience is not None:
                if not_improved > patience:
                    print(
                        f"stopping training. testloss {testloss:.2f} did not improve in {patience} epochs."
                    )
                    return model
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    losses = []
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)
        log_class_probabilities, probability_stopping = model(X)

        loss = criterion(
            log_class_probabilities, probability_stopping, y_true
        )

        # assert not loss.isnan().any()
        if not loss.isnan().any():
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())

    return np.stack(losses).mean()


def test_epoch(model, dataloader, criterion, device):
    model.eval()

    stats = []
    losses = []
    for batch in dataloader:
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)

        (
            log_class_probabilities,
            probability_stopping,
            predictions_at_t_stop,
            t_stop,
        ) = model.predict(X)
        loss, stat = criterion(
            log_class_probabilities,
            probability_stopping,
            y_true,
            return_stats=True,
        )

        stat["loss"] = loss.cpu().detach().numpy()
        stat["probability_stopping"] = (
            probability_stopping.cpu().detach().numpy()
        )
        stat["class_probabilities"] = (
            log_class_probabilities.exp().cpu().detach().numpy()
        )
        stat["predictions_at_t_stop"] = (
            predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
        )
        stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["targets"] = y_true.cpu().detach().numpy()

        stats.append(stat)

        losses.append(loss.cpu().detach().numpy())

    # list of dicts to dict of lists
    stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}

    return np.stack(losses).mean(), stats
