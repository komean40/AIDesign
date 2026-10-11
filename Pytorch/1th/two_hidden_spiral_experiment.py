"""Two intertwined spirals: visualize every neuron in two hidden layers.

Run: python two_hidden_spiral_experiment.py
Options: --show, --steps 6000, --output-dir PATH
Dependencies: torch, numpy, matplotlib.
Both models: 2 -> 8 -> 8 -> 1, activation ReLU or Tanh in BOTH hidden layers.
Each color map is a neuron's output as a function of the ORIGINAL input (x1,x2).
Orange contours indicate z=0, not final class boundaries.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class SpiralModel(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.hidden1=nn.Linear(2,8)
        self.hidden2=nn.Linear(8,8)
        self.output=nn.Linear(8,1)
        self.activation=nn.ReLU() if activation=='relu' else nn.Tanh()

    def parts(self,x):
        z1=self.hidden1(x);h1=self.activation(z1)
        z2=self.hidden2(h1);h2=self.activation(z2)
        contribution=h2*self.output.weight[0]
        return z1,h1,z2,h2,contribution,self.output(h2)

    def forward(self,x):return self.parts(x)[-1]


def make_data():
    rng=np.random.default_rng(42)
    t=np.linspace(.45,3.5*np.pi,300)
    r=np.linspace(.15,1.55,300)
    points=np.column_stack([r*np.cos(t),r*np.sin(t)])
    X=np.vstack([points,-points])+rng.normal(0,.035,(600,2))
    y=np.repeat([0,1],300)
    order=rng.permutation(600)
    return X[order].astype(np.float32),y[order]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs'/'two_hidden_spiral')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('TkAgg' if args.show else 'Agg')
    import matplotlib.pyplot as plt
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    X,y=make_data()
    rng=np.random.default_rng(17);train=[];test=[]
    for cls in [0,1]:
        ids=rng.permutation(np.flatnonzero(y==cls))
        train.extend(ids[:210]);test.extend(ids[210:])
    train,test=np.array(train),np.array(test)
    tx=torch.tensor(X);ty=torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    np.savetxt(out/'spiral_data.csv',np.column_stack([X,y,np.isin(np.arange(600),train).astype(int)]),
               delimiter=',',header='x1,x2,label,is_train',comments='',fmt=['%.8f','%.8f','%d','%d'])
    axis=np.linspace(-1.8,1.8,241,dtype=np.float32)
    xx,yy=np.meshgrid(axis,axis)
    grid=torch.tensor(np.column_stack([xx.ravel(),yy.ravel()]))
    t=torch.linspace(-1.8,1.8,1001)
    path=torch.stack([t,torch.zeros_like(t)],dim=1)
    results={};overview=[]
    colors=['#2563eb','#dc2626']
    def save(fig,name):
        fig.savefig(out/name,dpi=145,bbox_inches='tight')
        if not args.show:plt.close(fig)
    def map_axes(ax):
        ax.set_aspect('equal');ax.set_xlabel('Original input x1');ax.set_ylabel('Original input x2')
        ax.set_xlim(-1.8,1.8);ax.set_ylim(-1.8,1.8)
    def sample_points(ax,ids,small=False):
        for cls in [0,1]:
            chosen=ids[y[ids]==cls]
            ax.scatter(X[chosen,0],X[chosen,1],s=6 if small else 15,color=colors[cls],alpha=.4 if small else .8,
                       edgecolor='none' if small else 'white',lw=.25)
    def contour_zero(ax,values,color='black'):
        if values.min()<0<values.max():ax.contour(xx,yy,values,levels=[0],colors=color,linewidths=.9)
    for activation in ['relu','tanh']:
        torch.manual_seed(21)
        model=SpiralModel(activation)
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        history=[]
        for _ in range(args.steps):
            opt.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(model(tx[train]),ty[train])
            loss.backward();opt.step();history.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            z1,h1,z2,h2,c,s=model.parts(grid)
            path_parts=model.parts(path)
            shifted_parts=model.parts(path+torch.tensor([.05,0.]))
            pred=(model(tx).squeeze(1)>0).numpy().astype(int)
            # Float32 reductions can differ slightly when large terms cancel.
            torch.testing.assert_close(c.sum(1,keepdim=True)+model.output.bias,s,atol=1e-4,rtol=1e-5)
            h64=h2.double();v64=model.output.weight.double();b64=model.output.bias.double()
            torch.testing.assert_close((h64*v64[0]).sum(1,keepdim=True)+b64,
                                       nn.functional.linear(h64,v64,b64),atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(model.hidden2(h1),z2)
        trainacc=float(np.mean(pred[train]==y[train]));testacc=float(np.mean(pred[test]==y[test]))
        Z1=z1.numpy().reshape(*xx.shape,8);H1=h1.numpy().reshape(*xx.shape,8)
        Z2=z2.numpy().reshape(*xx.shape,8);H2=h2.numpy().reshape(*xx.shape,8)
        S=s[:,0].numpy().reshape(xx.shape);P=torch.sigmoid(s).numpy().reshape(xx.shape)
        v=model.output.weight.detach().numpy()[0];bias=float(model.output.bias.detach()[0])
        metrics=dict(train_accuracy=trainacc,test_accuracy=testacc,train_size=420,test_size=180,
                     steps=args.steps,seed=21,learning_rate=.01,final_loss=history[-1],
                     layer1_weights=model.hidden1.weight.detach().tolist(),layer1_bias=model.hidden1.bias.detach().tolist(),
                     layer2_weights=model.hidden2.weight.detach().tolist(),layer2_bias=model.hidden2.bias.detach().tolist(),
                     output_weights=v.tolist(),output_bias=bias)
        results[activation]=metrics;overview.append((activation,S,P,history,testacc))
        torch.save(model.state_dict(),out/f'{activation}_weights.pt')
        for layer,Z,H in [(1,Z1,H1),(2,Z2,H2)]:
            fig,axes=plt.subplots(3,4,figsize=(15,11))
            axes=axes.ravel();lo,hi=(0,float(H.max())) if activation=='relu' else (-1,1)
            for j in range(8):
                im=axes[j].pcolormesh(xx,yy,H[:,:,j],cmap='viridis' if activation=='relu' else 'coolwarm',
                                     vmin=lo,vmax=hi,shading='auto')
                contour_zero(axes[j],Z[:,:,j],'#f59e0b');map_axes(axes[j])
                sample_points(axes[j],train,small=True)
                axes[j].set_title(f'Hidden layer {layer} / unit {j}',fontsize=10)
                fig.colorbar(im,ax=axes[j],shrink=.65)
            ax=axes[8];limit=float(np.abs(S).max())
            im=ax.pcolormesh(xx,yy,S,cmap='coolwarm',vmin=-limit,vmax=limit,shading='auto')
            contour_zero(ax,S);map_axes(ax);fig.colorbar(im,ax=ax,shrink=.65);ax.set_title('Final score (all layer-2 units)')
            ax=axes[9];im=ax.pcolormesh(xx,yy,P,cmap='coolwarm',vmin=0,vmax=1,shading='auto')
            contour_zero(ax,S);map_axes(ax);fig.colorbar(im,ax=ax,shrink=.65);ax.set_title('Final probability and boundary')
            ax=axes[10];ax.contourf(xx,yy,(S>0).astype(int),levels=[-.5,.5,1.5],colors=colors,alpha=.16)
            contour_zero(ax,S);sample_points(ax,test);map_axes(ax);ax.set_title(f'Test samples | Accuracy {testacc:.1%}')
            ax=axes[11];ax.semilogy(np.arange(1,args.steps+1),history);ax.set_title('Training loss')
            ax.set_xlabel('Step');ax.set_ylabel('BCE loss');ax.grid(alpha=.2)
            fig.suptitle(f'{activation.upper()} | 2 -> 8 -> 8 -> 1 | Hidden layer {layer} and final output\n'
                         'Colors show neuron output; orange: pre-activation z=0; black: final score=0\n'
                         'Every panel uses original input coordinates, including layer 2',fontsize=13)
            fig.tight_layout(rect=(0,0,1,.90));fig.subplots_adjust(hspace=.55,wspace=.42)
            save(fig,f'{activation}_0{layer}_neurons_and_final.png')

        # Explicitly show contributions from every second-layer neuron.
        C=c.numpy().reshape(*xx.shape,8)
        fig,axes=plt.subplots(3,4,figsize=(15,11));axes=axes.ravel();lim=float(np.abs(C).max())
        for j in range(8):
            im=axes[j].pcolormesh(xx,yy,C[:,:,j],cmap='coolwarm',vmin=-lim,vmax=lim,shading='auto')
            map_axes(axes[j]);fig.colorbar(im,ax=axes[j],shrink=.65)
            axes[j].set_title(f'Layer-2 unit {j}: v*h2\nv={v[j]:+.3f}')
        for k,values,title in [(8,C.sum(2),'Sum of contributions'),(9,S,f'Sum + bias ({bias:+.3f})')]:
            im=axes[k].pcolormesh(xx,yy,values,cmap='coolwarm',shading='auto')
            map_axes(axes[k]);fig.colorbar(im,ax=axes[k],shrink=.65);axes[k].set_title(title)
        contour_zero(axes[9],S)
        axes[10].axis('off');axes[11].axis('off')
        axes[10].text(0,.85,'At the SAME input (x1,x2):\n\nRead all 8 layer-2 outputs\nMultiply each by its output weight\nSum all contributions\nAdd output bias\nClass 1 if score > 0',va='top',fontsize=11)
        fig.suptitle(f'{activation.upper()}: layer-2 weighted contributions form the final score',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.94));fig.subplots_adjust(hspace=.5,wspace=.42)
        save(fig,f'{activation}_03_contributions.png')

        # Follow the same x2=0 path through both layers, then the output.
        fig,axes=plt.subplots(5,4,figsize=(15,16));axes=axes.ravel()
        for layer,zi,hi in [(1,0,1),(2,2,3)]:
            for j in range(8):
                ax=axes[(layer-1)*8+j]
                ax.plot(t,path_parts[zi][:,j],color='#64748b',ls='--',lw=1,label='z before activation')
                ax.plot(t,path_parts[hi][:,j],color='#2563eb',lw=2,label='h after activation')
                ax.axhline(0,color='black',lw=.6);ax.grid(alpha=.2)
                ax.set_title(f'Layer {layer} / unit {j}');ax.set_xlabel('x1 (x2=0)')
                if j==0:ax.legend(fontsize=7)
        for j in range(8):axes[16].plot(t,path_parts[4][:,j],alpha=.6,lw=.8)
        axes[16].plot(t,path_parts[5][:,0],color='black',lw=2,label='Final score');axes[16].legend(fontsize=8)
        axes[16].axhline(0,color='black',ls=':');axes[16].set_title('Contributions and final score')
        axes[17].plot(t,torch.sigmoid(path_parts[5])[:,0],color='#dc2626');axes[17].axhline(.5,color='black',ls='--')
        axes[17].set_title('Final probability');axes[17].set_ylim(-.03,1.03)
        for j in range(8):axes[18].plot(t,(shifted_parts[3]-path_parts[3])[:,j],lw=.8)
        axes[18].set_title('Layer-2 output changes for +0.05 x1')
        axes[19].plot(t,(shifted_parts[5]-path_parts[5])[:,0],color='black');axes[19].set_title('Final score change for +0.05 x1')
        for ax in axes[16:]:ax.set_xlabel('x1 (x2=0)');ax.grid(alpha=.2)
        fig.suptitle(f'{activation.upper()}: same input path through layer 1, layer 2, and output\n'
                     'Layer-1 pre-activation is a line; layer-2 pre-activation already combines nonlinear features',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.95));fig.subplots_adjust(hspace=.55)
        save(fig,f'{activation}_04_input_path.png')
        print(f'{activation}: train={trainacc:.4f}, test={testacc:.4f}, final loss={history[-1]:.6f}',flush=True)

    fig,axes=plt.subplots(1,3,figsize=(15,5))
    sample_points(axes[0],np.arange(600));map_axes(axes[0]);axes[0].set_title('Two intertwined spirals: 600 points')
    for ax,(activation,S,P,history,accuracy) in zip(axes[1:],overview):
        im=ax.pcolormesh(xx,yy,P,cmap='coolwarm',vmin=0,vmax=1,shading='auto')
        contour_zero(ax,S);sample_points(ax,test);map_axes(ax);fig.colorbar(im,ax=ax,shrink=.7)
        ax.set_title(f'{activation.upper()} | Test {accuracy:.1%}')
    fig.suptitle('Two hidden layers (8 + 8 units): learned spiral classification\nBlack: model boundary | Dots in prediction panels: held-out test data',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.89));save(fig,'00_overview.png')
    (out/'metrics_and_parameters.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if args.show:plt.show()


if __name__=='__main__':main()
