"""Retrain the circle model with g(z)=z+sin(3z).
Requires torch, numpy, matplotlib and existing circle_comparison outputs.
Run: python circle_custom_retraining.py
Uses identical data/split/Adam settings to the previous circle experiment.
Compares fine-tuning replaced ReLU weights and training from fresh initialization.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(2,8)
        self.output=nn.Linear(8,1)

    def parts(self,x):
        z=self.hidden(x)
        h=z+torch.sin(3*z)
        contributions=h*self.output.weight[0]
        return z,h,contributions,self.output(h)

    def forward(self,x):
        return self.parts(x)[-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,default=Path(__file__).resolve().parent/'outputs'/'circle_comparison')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs'/'circle_custom_retraining')
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('TkAgg' if args.show else 'Agg')
    import matplotlib.pyplot as plt
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    ref=args.reference_dir
    torch.set_num_threads(1)
    data=np.loadtxt(ref/'circle_data.csv',delimiter=',',skiprows=1)
    split=np.loadtxt(ref/'split_indices.csv',delimiter=',',skiprows=1)
    X=torch.tensor(data[:,:2],dtype=torch.float32)
    y=torch.tensor(data[:,2],dtype=torch.float32).unsqueeze(1)
    train=np.flatnonzero(split[:,1]==1);test=np.flatnonzero(split[:,1]==0)
    state=torch.load(ref/'relu_8_weights.pt',weights_only=True)
    axis=np.linspace(-1.8,1.8,241,dtype=np.float32)
    xx,yy=np.meshgrid(axis,axis)
    grid=torch.tensor(np.column_stack([xx.ravel(),yy.ravel()]))
    angle=np.linspace(0,2*np.pi,501)
    line=torch.linspace(-1.8,1.8,1001)
    path=torch.stack([line,torch.zeros_like(line)],dim=1)
    with torch.no_grad():
        z=grid@state['hidden.weight'].T+state['hidden.bias']
        relu_score=(torch.relu(z)@state['output.weight'].T+state['output.bias'])[:,0]
        replaced_score=((z+torch.sin(3*z))@state['output.weight'].T+state['output.bias'])[:,0]
    records=[];metrics={};trained={}
    def record(name,score):
        values=score.detach().numpy().reshape(xx.shape)
        records.append((name,values))
    record('Original trained ReLU',relu_score)
    record('Custom activation, before retraining',replaced_score)
    for mode in ['warm_start','fresh_start']:
        torch.manual_seed(21)
        model=CustomModel()
        if mode=='warm_start':model.load_state_dict(state)
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        history=[]
        for _ in range(3500):
            opt.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(model(X[train]),y[train])
            loss.backward();opt.step()
            history.append(float(loss.detach()))
        with torch.no_grad():
            _,h,c,s=model.parts(grid)
            torch.testing.assert_close(c.sum(1,keepdim=True)+model.output.bias,s)
            prediction=(model(X)>0).float()
            accuracy=float((prediction[test]==y[test]).float().mean())
            trainacc=float((prediction[train]==y[train]).float().mean())
            agreement=float(np.mean((s[:,0].numpy().reshape(xx.shape)>0)==(xx**2+yy**2>1)))
        metrics[mode]=dict(test_accuracy=accuracy,train_accuracy=trainacc,grid_agreement=agreement,
                           seed=21,steps=3500,learning_rate=.01,final_training_loss=history[-1])
        record('Custom: retrained from ReLU weights' if mode=='warm_start' else 'Custom: trained from fresh weights',s[:,0])
        trained[mode]=(model,history)
        torch.save(model.state_dict(),out/f'{mode}_weights.pt')

    def decorate(ax):
        ax.plot(np.cos(angle),np.sin(angle),'k--',lw=1)
        ax.set_aspect('equal');ax.set_xlabel('x1');ax.set_ylabel('x2')
    def save(fig,name):
        fig.savefig(out/name,dpi=155,bbox_inches='tight')
        if not args.show:plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,11))
    for ax,(name,S) in zip(axes.ravel(),records):
        P=1/(1+np.exp(-np.clip(S,-80,80)))
        im=ax.pcolormesh(xx,yy,P,cmap='coolwarm',vmin=0,vmax=1,shading='auto')
        if S.min()<0<S.max():ax.contour(xx,yy,S,levels=[0],colors='black',linewidths=1.5)
        for cls,col in [(0,'#2563eb'),(1,'#dc2626')]:
            ids=test[data[test,2]==cls]
            ax.scatter(data[ids,0],data[ids,1],s=13,color=col,edgecolor='white',lw=.3)
        decorate(ax);ax.set_title(name,fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.75,label='P(outside)')
    fig.suptitle('Custom nonlinear activation: g(z) = z + sin(3z)\nSame dataset and split; 8 hidden units | Black solid: model boundary; dashed: true circle',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));fig.subplots_adjust(hspace=.25)
    save(fig,'01_retraining_comparison.png')

    for mode,(model,history) in trained.items():
        with torch.no_grad():
            zg,hg,cg,sg=model.parts(grid)
            zp,hp,cp,sp=model.parts(path)
        W=model.hidden.weight.detach().numpy();b=model.hidden.bias.detach().numpy();v=model.output.weight.detach().numpy()[0]
        fig,axes=plt.subplots(3,4,figsize=(15,11))
        axes=axes.ravel();H=hg.numpy().reshape(*xx.shape,8);limit=float(np.abs(H).max())
        for j in range(8):
            im=axes[j].pcolormesh(xx,yy,H[:,:,j],cmap='coolwarm',vmin=-limit,vmax=limit,shading='auto')
            decorate(axes[j]);fig.colorbar(im,ax=axes[j],shrink=.6)
            axes[j].set_title(f'Unit {j}: g({W[j,0]:+.2f}x1{W[j,1]:+.2f}x2{b[j]:+.2f})\nv={v[j]:+.3f}',fontsize=9)
        S=sg[:,0].numpy().reshape(xx.shape)
        im=axes[8].pcolormesh(xx,yy,S,cmap='coolwarm',shading='auto');fig.colorbar(im,ax=axes[8],shrink=.6)
        axes[8].set_title('Final score: sum(v*h)+bias');decorate(axes[8])
        im=axes[9].pcolormesh(xx,yy,torch.sigmoid(sg).numpy().reshape(xx.shape),cmap='coolwarm',vmin=0,vmax=1,shading='auto')
        if S.min()<0<S.max():axes[9].contour(xx,yy,S,levels=[0],colors='black',linewidths=1.5)
        fig.colorbar(im,ax=axes[9],shrink=.6);decorate(axes[9]);axes[9].set_title('Final probability and boundary')
        for j in range(8):axes[10].plot(line,cp[:,j],lw=.8,alpha=.6)
        axes[10].plot(line,sp[:,0],color='black',lw=2,label='Final score')
        axes[10].axhline(0,color='black',ls=':',lw=.7)
        axes[10].set_title('1-D slice: x2=0');axes[10].set_xlabel('x1');axes[10].legend(fontsize=8)
        axes[11].semilogy(np.arange(1,3501),history);axes[11].set_title('Training loss');axes[11].set_xlabel('Step')
        fig.suptitle(f'g(z)=z+sin(3z) / 8 units / {mode}\nTest accuracy={metrics[mode]["test_accuracy"]:.1%}',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.94));fig.subplots_adjust(hspace=.55,wspace=.45)
        save(fig,f'02_{mode}_neurons_and_output.png')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    if args.show:plt.show()


if __name__=='__main__':main()
