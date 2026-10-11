"""Circle classification: compare ReLU/Tanh and 4/8 hidden neurons.

Run: python circle_activation_comparison.py
Optional: --show, --steps 3500, --output-dir PATH
Requires: torch, numpy, matplotlib.
The same deterministic 400-point dataset and 280/120 split are used for all models.
PNG figures, CSV data, parameters and checkpoints are saved in outputs/circle_comparison.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class CircleModel(nn.Module):
    def __init__(self, width, activation):
        super().__init__()
        self.hidden = nn.Linear(2, width)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.output = nn.Linear(width, 1)

    def parts(self, x):
        z = self.hidden(x)
        h = self.activation(z)
        contribution = h * self.output.weight[0]
        score = self.output(h)
        return z, h, contribution, score

    def forward(self, x):
        return self.parts(x)[-1]


def make_data():
    rng = np.random.default_rng(42)
    pool = rng.uniform(-1.8, 1.8, (10000, 2))
    r2 = np.sum(pool**2, axis=1)
    X = np.vstack([pool[r2 <= .85**2][:200], pool[r2 >= 1.15**2][:200]])
    y = np.repeat([0, 1], 200)
    order = rng.permutation(400)
    return X[order].astype(np.float32), y[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--steps', type=int, default=3500)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'outputs'/'circle_comparison')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('TkAgg' if args.show else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size': 9, 'savefig.dpi': 145})
    torch.set_num_threads(1)
    X, y = make_data()
    np.savetxt(out/'circle_data.csv',np.column_stack([X,y]),delimiter=',',
               header='x1,x2,label',comments='',fmt=['%.8f','%.8f','%d'])
    rng = np.random.default_rng(17)
    train, test = [], []
    for cls in [0, 1]:
        ids = rng.permutation(np.flatnonzero(y==cls))
        train.extend(ids[:140]); test.extend(ids[140:])
    train, test = np.array(train), np.array(test)
    np.savetxt(out/'split_indices.csv',np.column_stack([np.arange(400),np.isin(np.arange(400),train).astype(int)]),
               delimiter=',',header='index,is_train',comments='',fmt='%d')
    tx, ty = torch.tensor(X), torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    line = np.linspace(-1.8,1.8,1001,dtype=np.float32)
    path = torch.tensor(np.column_stack([line,np.zeros_like(line)]))
    axis = np.linspace(-1.8,1.8,241,dtype=np.float32)
    xx, yy = np.meshgrid(axis,axis)
    grid = torch.tensor(np.column_stack([xx.ravel(),yy.ravel()]))
    truth = (xx**2+yy**2>1).astype(int)
    angle = np.linspace(0,2*np.pi,500)
    colors = ['#2563eb','#dc2626']
    results, overview = {}, []

    def save(fig,name):
        fig.savefig(out/name,bbox_inches='tight')
        if not args.show:
            plt.close(fig)

    def decorate(ax):
        ax.plot(np.cos(angle),np.sin(angle),'k--',lw=1,alpha=.7)
        ax.set_aspect('equal')
        ax.set_xlim(-1.8,1.8); ax.set_ylim(-1.8,1.8)
        ax.set_xlabel('Input x1'); ax.set_ylabel('Input x2')

    def boundary(ax,score):
        if score.min()<0<score.max():
            ax.contour(xx,yy,score,levels=[0],colors='black',linewidths=1.7)

    for activation,width in [('relu',4),('relu',8),('tanh',4),('tanh',8)]:
        key=f'{activation}_{width}'
        torch.manual_seed(21)
        model=CircleModel(width,activation)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        losses=[]
        for _ in range(args.steps):
            optimizer.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(model(tx[train]),ty[train])
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            _,hg,cg,sg=model.parts(grid)
            zg,hl,cl,sl=model.parts(path)
            _,hn,_,sn=model.parts(path+torch.tensor([.1,0.]))
            predictions=(model(tx).squeeze(1)>0).numpy().astype(int)
            torch.testing.assert_close(cg.sum(1,keepdim=True)+model.output.bias,sg)
            torch.testing.assert_close(cl.sum(1,keepdim=True)+model.output.bias,sl)
        W=model.hidden.weight.detach().numpy()
        b=model.hidden.bias.detach().numpy()
        v=model.output.weight.detach().numpy()[0]
        bias=float(model.output.bias.detach()[0])
        H=hg.numpy().reshape(*xx.shape,width)
        C=cg.numpy().reshape(*xx.shape,width)
        S=sg.numpy().reshape(xx.shape)
        P=torch.sigmoid(sg).numpy().reshape(xx.shape)
        trainacc=float(np.mean(predictions[train]==y[train]))
        testacc=float(np.mean(predictions[test]==y[test]))
        denseacc=float(np.mean((S>0)==truth))
        threshold=np.divide(-b,W[:,0],out=np.full_like(b,np.nan),where=np.abs(W[:,0])>1e-9)
        summary=dict(activation=activation,width=width,seed=21,steps=args.steps,learning_rate=.01,
                     train_accuracy=trainacc,test_accuracy=testacc,uniform_grid_agreement=denseacc,
                     output_bias=bias,neurons=[dict(index=j,input_weights=W[j].tolist(),hidden_bias=float(b[j]),
                                                   output_weight=float(v[j]),x2_zero_crossing=float(threshold[j])) for j in range(width)])
        results[key]=summary
        torch.save(model.state_dict(),out/f'{key}_weights.pt')

        # Each heatmap is a scalar function over the full two-dimensional input plane.
        # Equal color scales across neurons make their output magnitudes comparable.
        fig,axes=plt.subplots((width+4)//4,4,figsize=(15,3.5*((width+4)//4)))
        axes=axes.ravel()
        hmin,hmax=(0,float(H.max())) if activation=='relu' else (-1,1)
        for j in range(width):
            ax=axes[j]
            im=ax.pcolormesh(xx,yy,H[:,:,j],cmap='viridis' if activation=='relu' else 'coolwarm',
                             vmin=hmin,vmax=hmax,shading='auto')
            plane=W[j,0]*xx+W[j,1]*yy+b[j]
            if plane.min()<0<plane.max():
                ax.contour(xx,yy,plane,levels=[0],colors='#f59e0b',linewidths=1)
            decorate(ax)
            ax.set_title(f'Unit {j}: h=g({W[j,0]:+.2f}x1{W[j,1]:+.2f}x2{b[j]:+.2f})\nOutput weight v={v[j]:+.3f}',fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.7)
        ax=axes[width]
        limit=float(np.abs(S).max())
        im=ax.pcolormesh(xx,yy,S,cmap='coolwarm',vmin=-limit,vmax=limit,shading='auto')
        boundary(ax,S);decorate(ax);fig.colorbar(im,ax=ax,shrink=.7)
        ax.set_title(f'Final score: sum(v_j h_j)+bias\nbias={bias:+.3f}')
        ax=axes[width+1]
        im=ax.pcolormesh(xx,yy,P,cmap='coolwarm',vmin=0,vmax=1,shading='auto')
        boundary(ax,S);decorate(ax);fig.colorbar(im,ax=ax,shrink=.7)
        ax.set_title('Final probability: sigmoid(score)\nBlack: p=0.5')
        ax=axes[width+2]
        ax.contourf(xx,yy,(S>0).astype(int),levels=[-.5,.5,1.5],cmap=ListedColormap(colors),alpha=.15)
        for cls in [0,1]:
            ids=test[y[test]==cls]
            ax.scatter(X[ids,0],X[ids,1],color=colors[cls],s=12,label=f'Test class {cls}')
        boundary(ax,S);decorate(ax);ax.legend(fontsize=7)
        ax.set_title(f'Final classes | Test accuracy {testacc:.1%}\nDashed circle: true boundary')
        ax=axes[width+3]
        ax.semilogy(np.arange(1,args.steps+1),losses,color='#2563eb')
        ax.set_xlabel('Optimization step');ax.set_ylabel('BCE loss');ax.grid(alpha=.2)
        ax.set_title(f'Training loss | Train accuracy {trainacc:.1%}')
        fig.suptitle(f'{activation.upper()} / {width} units: every hidden output and final prediction\nOrange: z_j=0 (activation threshold / Tanh zero crossing); dashed circle: reference only',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.93))
        fig.subplots_adjust(hspace=.58, wspace=.5)
        save(fig,f'{key}_01_neurons_and_output.png')

        # Weighted contributions must be summed at the SAME input coordinate.
        fig,axes=plt.subplots((width+4)//4,4,figsize=(15,3.4*((width+4)//4)))
        axes=axes.ravel();lim=max(float(np.abs(C).max()),.01)
        for j in range(width):
            im=axes[j].pcolormesh(xx,yy,C[:,:,j],cmap='coolwarm',vmin=-lim,vmax=lim,shading='auto')
            decorate(axes[j]);fig.colorbar(im,ax=axes[j],shrink=.7)
            axes[j].set_title(f'Unit {j}: contribution v_j h_j\nv={v[j]:+.3f}')
        for offset,values,title in [(0,C.sum(2),'Sum of contributions'),(1,S,'Sum + output bias = final score')]:
            ax=axes[width+offset]
            im=ax.pcolormesh(xx,yy,values,cmap='coolwarm',shading='auto')
            decorate(ax);fig.colorbar(im,ax=ax,shrink=.7);ax.set_title(title)
        boundary(axes[width+1],S)
        axes[width+2].axis('off');axes[width+3].axis('off')
        axes[width+2].text(.02,.8,'At each SAME input (x1,x2):\n\n1. Read all hidden outputs h_j\n2. Multiply by output weights v_j\n3. Sum the contributions\n4. Add output bias\n5. Predict outside if score > 0',va='top',fontsize=11)
        fig.suptitle(f'{activation.upper()} / {width} units: how hidden outputs form the final score',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.94))
        fig.subplots_adjust(hspace=.5, wspace=.5)
        save(fig,f'{key}_02_contributions.png')

        # A 1-D slice gives familiar line plots; x2 is exactly zero throughout.
        fig,axes=plt.subplots((width+4)//4,4,figsize=(15,3.2*((width+4)//4)))
        axes=axes.ravel()
        for j in range(width):
            ax=axes[j]
            ax.plot(line,zg[:,j],color='#64748b',ls='--',label='z before activation')
            ax.plot(line,hl[:,j],color='#2563eb',lw=2,label='h after activation')
            ax.plot(line,cl[:,j],color='#9333ea',lw=1.8,label='weighted v*h')
            ax.axhline(0,color='black',lw=.6)
            if -1.8<threshold[j]<1.8:
                ax.axvline(threshold[j],color='#f59e0b',ls=':')
            ax.set_title(f'Unit {j} along x2=0');ax.set_xlabel('Input x1');ax.grid(alpha=.2)
            if j==0: ax.legend(fontsize=7)
        ax=axes[width]
        for j in range(width):ax.plot(line,cl[:,j],alpha=.65,lw=1,label=f'unit {j}')
        ax.plot(line,sl[:,0],color='black',lw=2.5,label='Final sum + bias')
        ax.axhline(0,color='black',ls=':',lw=.8);ax.legend(fontsize=6,ncol=2)
        ax.set_title('All contributions and final score')
        ax=axes[width+1]
        ax.plot(line,torch.sigmoid(sl)[:,0],color='#dc2626',lw=2)
        ax.axhline(.5,color='black',ls='--');ax.set_ylim(-.03,1.03)
        ax.set_title('Outside probability along x2=0')
        for value in [-1,1]:ax.axvline(value,color='#059669',ls=':')
        ax=axes[width+2]
        for j in range(width):ax.plot(line,(hn-hl)[:,j],label=f'unit {j}',lw=1.2)
        ax.set_title('Hidden changes for the same +0.1 x1');ax.legend(fontsize=6,ncol=2)
        ax=axes[width+3]
        ax.plot(line,(sn-sl)[:,0],color='black',lw=2)
        ax.set_title('Final score change for +0.1 x1')
        for ax in axes[width:]:ax.set_xlabel('Input x1');ax.grid(alpha=.2)
        fig.suptitle(f'{activation.upper()} / {width} units: affine -> activation -> weighted sum\nA slice of the 2-D model, with x2 fixed at zero',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.93))
        fig.subplots_adjust(hspace=.55)
        save(fig,f'{key}_03_input_path.png')
        overview.append((key,S,P,testacc,denseacc))
        print(f'{key}: train={trainacc:.4f}, test={testacc:.4f}, grid agreement={denseacc:.4f}')

    fig,axes=plt.subplots(2,2,figsize=(11,12),sharex=True,sharey=True)
    for ax,(key,S,P,testacc,denseacc) in zip(axes.ravel(),overview):
        im=ax.pcolormesh(xx,yy,P,cmap='coolwarm',vmin=0,vmax=1,shading='auto')
        boundary(ax,S);decorate(ax)
        for cls in [0,1]:
            ids=test[y[test]==cls]
            ax.scatter(X[ids,0],X[ids,1],s=14,color=colors[cls],edgecolor='white',lw=.3)
        fig.colorbar(im,ax=ax,shrink=.75,label='P(outside)')
        ax.set_title(f'{key.upper()} | Test {testacc:.1%}\nAgreement with circle on uniform grid {denseacc:.1%}')
    fig.suptitle('Same data and split: ReLU/Tanh with 4/8 hidden units\nBlack: learned boundary | Dashed: true circle | Dots: test samples',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.subplots_adjust(hspace=.38,wspace=.32)
    save(fig,'00_final_comparison.png')
    report=dict(data_seed=42,split_seed=17,train_size=280,test_size=120,
                inside_radius_max=.85,outside_radius_min=1.15,
                grid_note='Uniform-grid agreement measures boundary shape, not held-out sample accuracy.',models=results)
    (out/'experiment_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if args.show:plt.show()


if __name__=='__main__':
    main()
