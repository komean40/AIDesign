"""Iris: explain ReLU kinks using finite input changes.

Run: python iris_relu_input_changes.py
Optional: --show (open figures), --output-dir PATH
Dependencies: torch, numpy, matplotlib (no scikit-learn required).
First run downloads the public UCI Iris CSV and caches it in the output folder.
Architecture: all four input features -> 8 ReLU units -> 3 class logits.
"""
import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn

NAMES = ['setosa', 'versicolor', 'virginica']
COLORS = ['#2563eb', '#059669', '#dc2626']
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'iris_relu')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('TkAgg' if args.show else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 160})
    torch.set_num_threads(1)
    torch.manual_seed(21)
    data_path = out / 'iris.data'
    if not data_path.exists():
        with urlopen(URL, timeout=30) as response:
            data_path.write_bytes(response.read())
    rows = [r for r in csv.reader(io.StringIO(data_path.read_text())) if r]
    raw = np.array([[float(v) for v in r[:4]] for r in rows], dtype=np.float32)
    labels = np.array([NAMES.index(r[4].replace('Iris-', '')) for r in rows])
    if raw.shape != (150, 4):
        raise ValueError(f'Unexpected dataset shape: {raw.shape}')
    rng = np.random.default_rng(21)
    train_ids, test_ids = [], []
    for c in range(3):
        ids = rng.permutation(np.flatnonzero(labels == c))
        train_ids.extend(ids[:35])
        test_ids.extend(ids[35:])
    train_ids, test_ids = np.array(train_ids), np.array(test_ids)
    mean = raw[train_ids].mean(0)
    scale = raw[train_ids].std(0)
    X = torch.tensor((raw-mean)/scale)
    Y = torch.tensor(labels, dtype=torch.long)
    model = IrisModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses = []
    for _ in range(1800):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(X[train_ids]), Y[train_ids])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        predicted = model(X).argmax(1).numpy()
    train_accuracy = float(np.mean(predicted[train_ids] == labels[train_ids]))
    test_accuracy = float(np.mean(predicted[test_ids] == labels[test_ids]))

    # Convert standardized hidden parameters into original centimeter coordinates.
    # W((x-mean)/scale)+b = W_raw x + b_raw.
    W = model.hidden.weight.detach().numpy()
    b = model.hidden.bias.detach().numpy()
    V = model.output.weight.detach().numpy()
    bout = model.output.bias.detach().numpy()
    Wraw = W / scale
    braw = b - Wraw @ mean

    # One-dimensional slice: vary only petal length t, fix all other features.
    fixed = mean.copy()
    fixed[3] = np.median(raw[train_ids, 3])
    lo, hi = .7, 7.4
    t = np.linspace(lo, hi, 1401, dtype=np.float32)
    delta = .10  # centimeters
    a = Wraw[:, 2]
    c = Wraw @ fixed + braw - a*fixed[2]
    thresholds = np.divide(-c, a, out=np.full_like(a, np.nan), where=np.abs(a)>1e-9)
    def along(values):
        points = np.tile(fixed, (len(values), 1))
        points[:, 2] = values
        with torch.no_grad():
            z = model.hidden(torch.tensor((points-mean)/scale))
            h = torch.relu(z)
            scores = model.output(h)
            probs = torch.softmax(scores, dim=1)
        return z.numpy(), h.numpy(), scores.numpy(), probs.numpy()
    z, h, scores, probs = along(t)
    z_next, h_next, scores_next, _ = along(t+delta)
    h_change = h_next-h
    score_change = scores_next-scores
    slopes_h = (z > 0)*a[None, :]
    slopes_scores = slopes_h @ V.T
    # Exact finite change: works even when the interval crosses a ReLU threshold.
    exact_h_change = np.maximum(0, z+delta*a)-np.maximum(0, z)
    np.testing.assert_allclose(h_change, exact_h_change, atol=1e-5, rtol=2e-4)
    np.testing.assert_allclose(score_change, h_change @ V.T, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(z, t[:, None]*a+c, atol=1e-5, rtol=2e-4)
    # Confirm the analytic local logit slopes against PyTorch autograd.
    midpoint_points = [lo+(hi-lo)*q for q in [.13, .37, .61, .87]]
    for value in midpoint_points:
        point = fixed.copy()
        point[2] = value
        xt = torch.tensor(point, requires_grad=True)
        zt = model.hidden((xt-torch.tensor(mean))/torch.tensor(scale))
        yt = model.output(torch.relu(zt))
        analytic = V @ ((zt.detach().numpy()>0)*a)
        for cls in range(3):
            derivative = torch.autograd.grad(yt[cls], xt, retain_graph=True)[0][2].item()
            np.testing.assert_allclose(derivative, analytic[cls], atol=1e-5, rtol=1e-5)

    def mark(ax, j=None):
        selected = range(8) if j is None else [j]
        for unit in selected:
            threshold = float(thresholds[unit])
            if lo < threshold < hi:
                ax.axvline(threshold, color='#f59e0b', ls=':', alpha=.6, lw=1)
        ax.axhline(0, color='black', lw=.6)
        ax.set_xlim(lo, hi)
        ax.grid(alpha=.2)
        ax.set_xlabel('Petal length t (cm)')

    figures = []
    def save(fig, name):
        fig.savefig(out / name, bbox_inches='tight')
        figures.append(fig)
        if not args.show:
            plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, z[:, j], '--', color='#64748b', label='Before ReLU z_j')
        ax.plot(t, h[:, j], color='#2563eb', lw=2, label='After ReLU h_j')
        mark(ax, j)
        ax.set_title(f'Neuron {j}: z={a[j]:+.3f}t{c[j]:+.3f}\nThreshold={thresholds[j]:.3f} cm')
        if j % 4 == 0:
            ax.set_ylabel('Neuron value')
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle('Iris: all 8 hidden units along one input path\nGray dashed: affine line | Blue: ReLU | Other 3 features held fixed', fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91))
    save(fig, '01_hidden_relu_kinks.png')

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, h_change[:,j], color='#dc2626', lw=2, label='Exact h(t+0.1)-h(t)')
        ax.plot(t, delta*slopes_h[:,j], '--', color='#2563eb', label="Local slope x 0.1")
        mark(ax, j)
        ax.set_title(f'Neuron {j}: same input change +0.1 cm')
        if j % 4 == 0:
            ax.set_ylabel('Hidden output change')
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle('Same input change, different output changes\nSolid red is exact; dashed blue agrees if the step does not cross a kink', fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91))
    save(fig, '02_same_input_change.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for cls, name in enumerate(NAMES):
        axes[0,0].plot(t, scores[:,cls], color=COLORS[cls], lw=2, label=name)
        axes[0,1].plot(t, probs[:,cls], color=COLORS[cls], lw=2, label=name)
        axes[1,0].plot(t, slopes_scores[:,cls], color=COLORS[cls], lw=2, label=name)
        axes[1,1].plot(t, score_change[:,cls], color=COLORS[cls], lw=2, label=name)
    for ax, title, ylabel in zip(axes.flat,
            ['Class scores: sums of ReLU units', 'Softmax probabilities', 'Local score slope changes at kinks', 'Score change for the same +0.1 cm step'],
            ['Logit', 'Probability', 'd(logit)/dt', 'Logit(t+0.1)-logit(t)']):
        mark(ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    axes[0,1].set_ylim(-.03,1.03)
    fig.suptitle('Hidden kinks -> piecewise affine class scores -> probabilities', fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig, '03_scores_probabilities_changes.png')

    # 2-D slice: hold sepal measurements fixed, vary petal length and width.
    gx = np.linspace(.7,7.4,280)
    gy = np.linspace(.05,2.7,240)
    xx, yy = np.meshgrid(gx,gy)
    points = np.tile(fixed,(xx.size,1))
    points[:,2], points[:,3] = xx.ravel(), yy.ravel()
    with torch.no_grad():
        grid_scores = model(torch.tensor((points-mean)/scale, dtype=torch.float32)).numpy().reshape(*xx.shape,3)
    region = grid_scores.argmax(2)
    constant = Wraw[:,0]*fixed[0]+Wraw[:,1]*fixed[1]+braw
    fig, axes = plt.subplots(1,2,figsize=(13,5.4),sharex=True,sharey=True)
    for ax in axes:
        ax.contourf(xx,yy,region, levels=[-.5,.5,1.5,2.5], cmap=ListedColormap(COLORS), alpha=.16)
        ax.axhline(fixed[3],color='black',ls='--',lw=1,label='1-D path used above')
        ax.set_xlabel('Petal length (cm)')
        ax.set_xlim(gx.min(),gx.max())
        ax.set_ylim(gy.min(),gy.max())
    axes[0].set_ylabel('Petal width (cm)')
    for j in range(8):
        plane = Wraw[j,2]*xx+Wraw[j,3]*yy+constant[j]
        if plane.min()<0<plane.max():
            contour = axes[0].contour(xx,yy,plane,levels=[0],linewidths=1.2)
            axes[0].clabel(contour,fmt={0:f'unit {j}'},fontsize=8)
    axes[0].set_title('Straight activation thresholds: z_j=0')
    # Contour only the actual predicted labels, avoiding nonwinning score ties.
    axes[1].contour(xx,yy,region,levels=[.5,1.5],colors='black',linewidths=1.5)
    for cls, name in enumerate(NAMES):
        ids = labels==cls
        axes[1].scatter(raw[ids,2],raw[ids,3],s=20,color=COLORS[cls],edgecolor='white',lw=.3,label=name)
    axes[1].legend(fontsize=8)
    axes[1].set_title('Predicted classes on a 2-D slice')
    fig.suptitle('All 4 features used for training; sepal features fixed for this slice\nDots: projection of real samples, whose sepal measurements differ from this slice',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.90))
    save(fig,'04_activation_regions_and_classification.png')

    example_rows = []
    for val in [1.,3.,5.,6.5]:
        zz,hh,ss,_ = along(np.array([val,val+delta],dtype=np.float32))
        example_rows.append(dict(start_cm=val, delta_cm=delta, active_units=np.flatnonzero(zz[0]>0).tolist(),
                                 hidden_changes=(hh[1]-hh[0]).tolist(), score_changes=(ss[1]-ss[0]).tolist()))
    report = dict(data_source=URL, model='4 -> 8 ReLU -> 3 logits', seed=21,
                  train_size=105,test_size=45,train_accuracy=train_accuracy,test_accuracy=test_accuracy,
                  fixed_features_cm=fixed.tolist(),mean=mean.tolist(),scale=scale.tolist(),
                  output_weights=V.tolist(),output_bias=bout.tolist(),
                  neurons=[dict(index=j,path_slope=float(a[j]),path_intercept=float(c[j]),
                                threshold_cm=float(thresholds[j]),raw_input_weights=Wraw[j].tolist(),
                                raw_bias=float(braw[j])) for j in range(8)], examples=example_rows)
    (out/'experiment_parameters.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'path_changes.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['petal_length_cm']+[f'h{j}' for j in range(8)]+[f'delta_h{j}' for j in range(8)]+[f'logit_{n}' for n in NAMES]+[f'delta_logit_{n}' for n in NAMES])
        for i in range(len(t)):
            writer.writerow([float(t[i])]+h[i].tolist()+h_change[i].tolist()+scores[i].tolist()+score_change[i].tolist())
    torch.save(dict(model=model.state_dict(),mean=mean,scale=scale),out/'model_weights.pt')
    print(json.dumps(report,indent=2))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
