"""Train small PyTorch models and visualize what their activations learned.

Run from this repository's root:
    python AIDesign/AI/Pytorch/1th/activation_xor_regression.py

Dependencies: torch, numpy, matplotlib.
The output directory defaults to the script folder's outputs/ folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


torch.set_num_threads(1)
torch.manual_seed(7)
np.random.seed(7)
plt.rcParams.update({"figure.dpi": 145, "savefig.dpi": 170, "font.size": 10})


class OneHiddenLayer(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, activation: str):
        super().__init__()
        self.first = nn.Linear(n_input, n_hidden)
        self.activation = {"relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        self.last = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last(self.activation(self.first(x)))

    def parts(self, x: torch.Tensor):
        z = self.first(x)
        h = self.activation(z)
        contribution = h * self.last.weight[0]
        return z, h, contribution, self.last(h)


def fit(model, x, y, loss_fn, steps, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return np.asarray(losses)


def plot_activations(output):
    z = torch.linspace(-5, 5, 1001, requires_grad=True)
    activations = [
        ("ReLU", nn.ReLU(), "#2563eb"),
        ("Leaky ReLU (slope 0.1)", nn.LeakyReLU(0.1), "#8b5cf6"),
        ("Tanh", nn.Tanh(), "#059669"),
        ("Sigmoid", nn.Sigmoid(), "#dc2626"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), sharex=True)
    for (name, fn, color), ax in zip(activations, axes.flat):
        h = fn(z)
        derivative = torch.autograd.grad(h.sum(), z, retain_graph=True)[0]
        ax.plot(z.detach(), h.detach(), color=color, lw=2.5, label="output g(z)")
        ax.plot(z.detach(), derivative.detach(), color=color, ls="--", lw=1.7,
                label="gradient dg/dz")
        ax.axhline(0, color="black", lw=.7)
        ax.axvline(0, color="black", lw=.7)
        ax.set_title(name)
        ax.set_ylabel("Output / gradient")
        ax.grid(alpha=.22)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Pre-activation z = wx + b")
    fig.suptitle("Activation outputs (solid) and local gradients (dashed)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "01_activation_and_gradients.png", bbox_inches="tight")
    plt.close(fig)


def xor_data():
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return x, y


def train_xor():
    x, y = xor_data()
    models, histories = {}, {}
    linear = nn.Linear(2, 1)
    histories["linear"] = fit(linear, x, y, nn.BCEWithLogitsLoss(), 1200, .05)
    models["linear"] = linear
    for name, seed in [("relu", 11), ("tanh", 12)]:
        torch.manual_seed(seed)
        model = OneHiddenLayer(2, 8, name)
        histories[name] = fit(model, x, y, nn.BCEWithLogitsLoss(), 1800, .04)
        models[name] = model
    return x, y, models, histories


def plot_xor(output, x, y, models, histories):
    xx, yy = np.meshgrid(np.linspace(-.3, 1.3, 201), np.linspace(-.3, 1.3, 201))
    points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        with torch.no_grad():
            p = torch.sigmoid(model(points)).reshape(xx.shape).numpy()
            sample_p = torch.sigmoid(model(x)).squeeze().numpy()
        ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 21), cmap="RdBu_r", vmin=0, vmax=1)
        if p.min() < .5 < p.max():
            ax.contour(xx, yy, p, levels=[.5], colors="black", linewidths=1.7)
        ax.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap="RdBu_r", vmin=0, vmax=1,
                   edgecolors="black", s=155, zorder=4)
        for j, ((a, b), prob) in enumerate(zip(x.tolist(), sample_p)):
            ax.text(a + (.05 if a == 0 else -.23), b + (.08 if b == 0 else -.13),
                    f"p={prob:.2f}", fontsize=8, weight="bold")
        ax.set_title({"linear": "Affine + sigmoid", "relu": "ReLU hidden layer",
                      "tanh": "Tanh hidden layer"}[name])
        ax.set_xlabel("x1")
        ax.set_xlim(-.3, 1.3)
        ax.set_ylim(-.3, 1.3)
        ax.set_aspect("equal")
    axes[0].set_ylabel("x2")
    fig.suptitle("XOR: black line is the 0.5 decision boundary; red means p(y=1) is high")
    fig.tight_layout()
    fig.savefig(output / "02_xor_decision_boundaries.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, history in histories.items():
        ax.plot(history, label=name)
    ax.set(xlabel="Optimization step", ylabel="Binary cross-entropy",
           title="XOR training loss: a linear model cannot fit all four labels")
    ax.set_yscale("log")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "03_xor_training_loss.png", bbox_inches="tight")
    plt.close(fig)

    model = models["relu"]
    with torch.no_grad():
        z, h, c, logits = model.parts(x)
    # Select two neurons with the largest contribution range across the four examples.
    ranking = torch.argsort(c.max(dim=0).values - c.min(dim=0).values, descending=True)
    selected = ranking[:2].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, index in zip(axes, selected):
        ax.bar(range(4), h[:, index].numpy(), color="#2563eb", alpha=.78)
        ax.set_xticks(range(4), ["00", "01", "10", "11"])
        ax.set(xlabel="Input (x1,x2)", ylabel="Hidden output h",
               title=f"Neuron {index}: output weight={model.last.weight[0,index].item():+.2f}")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Two learned ReLU detectors (chosen by contribution range)")
    fig.tight_layout()
    fig.savefig(output / "04_xor_hidden_neurons.png", bbox_inches="tight")
    plt.close(fig)


def plot_xor_layer_comparison(output, x, y, models):
    """Compare learned hidden features with the final XOR probability map."""
    xx, yy = np.meshgrid(np.linspace(-.25, 1.25, 181),
                         np.linspace(-.25, 1.25, 181))
    points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row, name in enumerate(("relu", "tanh")):
        model = models[name]
        with torch.no_grad():
            _, hidden, _, logits = model.parts(points)
            _, _, sample_contributions, _ = model.parts(x)
        spread = sample_contributions.max(dim=0).values - sample_contributions.min(dim=0).values
        selected = torch.argsort(spread, descending=True)[:2].tolist()
        for col, index in enumerate(selected):
            ax = axes[row, col]
            values = hidden[:, index].reshape(xx.shape).numpy()
            im = ax.imshow(values, origin="lower", extent=(-.25, 1.25, -.25, 1.25),
                           cmap="viridis", aspect="equal")
            fig.colorbar(im, ax=ax, shrink=.75, label=f"h{index}")
            weight = model.last.weight[0, index].item()
            ax.set_title(f"{name.upper()} hidden h{index} (output weight {weight:+.2f})")
        ax = axes[row, 2]
        probability = torch.sigmoid(logits).reshape(xx.shape).numpy()
        im = ax.imshow(probability, origin="lower", extent=(-.25, 1.25, -.25, 1.25),
                       cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
        ax.contour(xx, yy, probability, levels=[.5], colors="black", linewidths=1.4)
        fig.colorbar(im, ax=ax, shrink=.75, label="p(y=1)")
        ax.set_title(f"{name.upper()} final output probability")
        for panel in axes[row]:
            panel.scatter(x[:, 0], x[:, 1], s=82, facecolors="white",
                          edgecolors="black", zorder=5)
            for (x1, x2), label in zip(x.tolist(), y[:, 0].tolist()):
                panel.text(x1, x2, str(int(label)), ha="center", va="center",
                           fontsize=8, weight="bold", zorder=6)
            panel.set(xlabel="x1", ylabel="x2", xlim=(-.25, 1.25), ylim=(-.25, 1.25))
    fig.suptitle("XOR: two hidden units versus the final output (which uses all 8 units)")
    fig.tight_layout()
    fig.savefig(output / "11_xor_hidden_to_output.png", bbox_inches="tight")
    plt.close(fig)


def regression_data():
    # A deterministic target isolates representation capacity from sampling noise.
    x = torch.linspace(-2, 2, 81).unsqueeze(1)
    y = x.square()
    return x, y


def train_regression():
    x, y = regression_data()
    models, histories = {}, {}
    linear = nn.Linear(1, 1)
    histories["linear"] = fit(linear, x, y, nn.MSELoss(), 1500, .04)
    models["linear"] = linear
    for name, seed in [("relu", 21), ("tanh", 22)]:
        torch.manual_seed(seed)
        model = OneHiddenLayer(1, 16, name)
        learning_rate = .025 if name == "relu" else .01
        histories[name] = fit(model, x, y, nn.MSELoss(), 2500, learning_rate)
        models[name] = model
    return x, y, models, histories


def plot_regression(output, x, y, models, histories):
    grid = torch.linspace(-2.5, 2.5, 501).unsqueeze(1)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].scatter(x, y, s=14, c="black", alpha=.55, label="Training targets: y=x^2")
    axes[0].plot(grid, grid.square(), c="black", lw=1.3, ls="--", label="True y=x^2")
    colors = {"linear": "#ea580c", "relu": "#2563eb", "tanh": "#059669"}
    for name, model in models.items():
        with torch.no_grad():
            pred = model(grid).numpy()
            residual = (model(x) - y).numpy()
        axes[0].plot(grid, pred, color=colors[name], lw=2, label=name)
        axes[1].plot(x, residual, color=colors[name], lw=1.5, label=name)
    axes[0].set(ylabel="Predicted y", title="Regression: learn the curved function y=x^2")
    axes[1].set(xlabel="Input x", ylabel="Prediction error", title="Residual on training points")
    axes[1].axhline(0, color="black", lw=.8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "05_regression_fit_and_residuals.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, history in histories.items():
        ax.plot(history, label=name)
    ax.set(xlabel="Optimization step", ylabel="Mean squared error",
           title="Regression training loss: x^2 cannot be represented by ax+b")
    ax.set_yscale("log")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "06_regression_training_loss.png", bbox_inches="tight")
    plt.close(fig)

    model = models["relu"]
    dense = torch.linspace(-2.1, 2.1, 841).unsqueeze(1)
    with torch.no_grad():
        z, h, contribution, prediction = model.parts(dense)
        weights = model.first.weight[:, 0].numpy()
        biases = model.first.bias.numpy()
        output_weights = model.last.weight[0].numpy()
        bias_out = model.last.bias.item()
        variation = np.ptp(contribution.numpy(), axis=0)
        top = np.argsort(variation)[-4:][::-1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for index in top:
        axes[0].plot(dense, contribution[:, index], lw=1.8,
                     label=f"neuron {index}: {output_weights[index]:+.2f} ReLU({weights[index]:+.2f}x{biases[index]:+.2f})")
        if abs(weights[index]) > 1e-7:
            threshold = -biases[index] / weights[index]
            if -2.1 <= threshold <= 2.1:
                axes[0].axvline(threshold, color="gray", ls=":", lw=.7)
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set(ylabel="Contribution v_j h_j(x)",
                title="Four largest learned ReLU contributions; dotted lines are activation thresholds")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=.2)
    axes[1].plot(dense, prediction, color="#2563eb", lw=2.4, label="Sum of all 16 contributions + bias")
    axes[1].plot(dense, dense.square(), color="black", ls="--", label="Target x^2")
    axes[1].set(xlabel="Input x", ylabel="Output", title=f"Full network: output bias={bias_out:+.2f}")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "07_regression_neuron_contributions.png", bbox_inches="tight")
    plt.close(fig)

    # Local derivative is piecewise constant for a one-hidden-layer ReLU model.
    x_grad = torch.linspace(-2, 2, 801).unsqueeze(1).requires_grad_(True)
    for name in ("relu", "tanh"):
        derivative = torch.autograd.grad(models[name](x_grad).sum(), x_grad,
                                         retain_graph=True)[0]
        if name == "relu": relu_grad = derivative.detach().numpy()
        else: tanh_grad = derivative.detach().numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_grad.detach(), 2*x_grad.detach(), "k--", label="True derivative 2x")
    ax.plot(x_grad.detach(), relu_grad, color="#2563eb", label="ReLU model")
    ax.plot(x_grad.detach(), tanh_grad, color="#059669", label="Tanh model")
    ax.set(xlabel="Input x", ylabel="d(prediction)/dx",
           title="Local sensitivity: ReLU steps versus smooth Tanh")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "08_regression_input_gradients.png", bbox_inches="tight")
    plt.close(fig)

    tanh_model = models["tanh"]
    extended = torch.linspace(-5, 5, 1001).unsqueeze(1)
    with torch.no_grad():
        z_tanh, h_tanh, c_tanh, _ = tanh_model.parts(extended)
        _, _, c_train, _ = tanh_model.parts(x)
        tanh_top = torch.argsort(c_train.max(dim=0).values - c_train.min(dim=0).values,
                                  descending=True)[:3].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
    for ax, index in zip(axes, tanh_top):
        w = tanh_model.first.weight[index, 0].item()
        b = tanh_model.first.bias[index].item()
        ax.plot(extended, h_tanh[:, index], color="#059669", lw=2)
        ax.axhline(1, color="gray", ls=":", lw=.8)
        ax.axhline(-1, color="gray", ls=":", lw=.8)
        ax.axvspan(-2, 2, color="#e5e7eb", alpha=.6)
        ax.set(xlabel="Input x", title=f"Neuron {index}: tanh({w:+.2f}x{b:+.2f})")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Hidden output h_j(x)")
    fig.suptitle("Learned Tanh neurons: shaded region is the training interval")
    fig.tight_layout()
    fig.savefig(output / "09_regression_tanh_hidden_neurons.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(extended, extended.square(), "k--", lw=1.7, label="True x^2")
    with torch.no_grad():
        for name, color in [("relu", "#2563eb"), ("tanh", "#059669")]:
            ax.plot(extended, models[name](extended), color=color, lw=2, label=name)
    ax.axvspan(-2, 2, color="#e5e7eb", alpha=.65, label="Training interval [-2, 2]")
    ax.set(xlabel="Input x", ylabel="Prediction", ylim=(-1, 26),
           title="Extrapolation: fitting the training interval does not determine behavior outside it")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "10_regression_extrapolation.png", bbox_inches="tight")
    plt.close(fig)


def plot_regression_layer_comparison(output, models):
    """Show h_j, v_j h_j, and the complete prediction on the same x-axis."""
    grid = torch.linspace(-2.2, 2.2, 601).unsqueeze(1)
    x_values = grid[:, 0].numpy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, name in enumerate(("relu", "tanh")):
        model = models[name]
        with torch.no_grad():
            _, hidden, contributions, prediction = model.parts(grid)
        spread = np.ptp(contributions.numpy(), axis=0)
        selected = np.argsort(spread)[-3:][::-1]
        for index in selected:
            axes[row, 0].plot(x_values, hidden[:, index], lw=1.8, label=f"h{index}")
            weight = model.last.weight[0, index].item()
            axes[row, 1].plot(x_values, contributions[:, index], lw=1.8,
                              label=f"v{index} h{index} (v={weight:+.2f})")
        axes[row, 2].plot(x_values, prediction[:, 0], color="#2563eb", lw=2.4,
                          label="output: bias + all 16 contributions")
        axes[row, 2].plot(x_values, x_values**2, "k--", lw=1.5, label="target x^2")
        axes[row, 0].set_title(f"{name.upper()}: hidden outputs h_j(x)")
        axes[row, 1].set_title(f"{name.upper()}: weighted v_j h_j(x)")
        axes[row, 2].set_title(f"{name.upper()}: final output")
        for col in range(3):
            ax = axes[row, col]
            ax.axhline(0, color="black", lw=.65)
            ax.set_xlabel("input x")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Regression: selected hidden units, their contributions, and the full prediction")
    fig.tight_layout()
    fig.savefig(output / "12_regression_hidden_to_output.png", bbox_inches="tight")
    plt.close(fig)


def metrics(xor_x, xor_y, xor_models, reg_x, reg_y, reg_models):
    result = {"torch_version": torch.__version__, "xor": {}, "regression": {}}
    with torch.no_grad():
        for name, model in xor_models.items():
            probabilities = torch.sigmoid(model(xor_x)).squeeze().numpy()
            accuracy = np.mean((probabilities >= .5) == xor_y.squeeze().numpy())
            result["xor"][name] = {
                "probabilities_00_01_10_11": probabilities.round(6).tolist(),
                "accuracy": float(accuracy),
                "bce": float(nn.BCEWithLogitsLoss()(model(xor_x), xor_y)),
            }
        xor_relu = xor_models["relu"]
        z_xor, h_xor, c_xor, _ = xor_relu.parts(xor_x)
        result["xor"]["relu_hidden"] = {
            "input_order": ["00", "01", "10", "11"],
            "first_weight": xor_relu.first.weight.numpy().round(6).tolist(),
            "first_bias": xor_relu.first.bias.numpy().round(6).tolist(),
            "output_weight": xor_relu.last.weight[0].numpy().round(6).tolist(),
            "output_bias": float(xor_relu.last.bias),
            "hidden_outputs_by_input": h_xor.numpy().round(6).tolist(),
        }
        for name, model in reg_models.items():
            pred = model(reg_x)
            result["regression"][name] = {
                "rmse_on_training_grid": float(torch.sqrt(((pred-reg_y)**2).mean())),
                "predictions_at_minus2_0_2": model(torch.tensor([[-2.], [0.], [2.]])).squeeze().numpy().round(6).tolist(),
            }
        relu = reg_models["relu"]
        result["regression"]["relu_parameters"] = {
            "first_weight": relu.first.weight[:, 0].numpy().round(6).tolist(),
            "first_bias": relu.first.bias.numpy().round(6).tolist(),
            "output_weight": relu.last.weight[0].numpy().round(6).tolist(),
            "output_bias": float(relu.last.bias),
        }
        example_x = torch.tensor([[1.]])
        z_example, h_example, c_example, pred_example = relu.parts(example_x)
        result["regression"]["relu_at_x_1"] = {
            "hidden_outputs": h_example.squeeze().numpy().round(6).tolist(),
            "contributions": c_example.squeeze().numpy().round(6).tolist(),
            "prediction": float(pred_example.squeeze()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    default_output = Path(__file__).resolve().parent / "outputs"
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_activations(args.output_dir)
    xor_x, xor_y, xor_models, xor_histories = train_xor()
    plot_xor(args.output_dir, xor_x, xor_y, xor_models, xor_histories)
    plot_xor_layer_comparison(args.output_dir, xor_x, xor_y, xor_models)
    reg_x, reg_y, reg_models, reg_histories = train_regression()
    plot_regression(args.output_dir, reg_x, reg_y, reg_models, reg_histories)
    plot_regression_layer_comparison(args.output_dir, reg_models)
    summary = metrics(xor_x, xor_y, xor_models, reg_x, reg_y, reg_models)
    (args.output_dir / "experiment_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
