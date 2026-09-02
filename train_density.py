"""Train density-feature CNNs on the frozen previous split; compare actual recall."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import CLASSES,LegacyUnpickler,representations,seed_all
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'results/polar_comparison'
OUT=ROOT/'results/density_comparison'

class DensityCNN(nn.Module):
    def __init__(self,pooling):
        super().__init__()
        pool=nn.AvgPool2d if pooling=='avg' else nn.MaxPool2d
        self.net=nn.Sequential(nn.Conv2d(3,32,3),nn.ReLU(),pool(2),nn.Conv2d(32,64,3),nn.ReLU(),pool(2),nn.Flatten(),nn.Linear(64*14*14,64),nn.ReLU(),nn.Linear(64,9))
    def forward(self,x):return self.net(x)

def features(x):
    valid=(x>0).float(); fail=(x==2).float()
    channels=[x]
    for k in (5,11):
        filt=torch.ones(1,1,k,k)
        count=F.conv2d(fail,filt,padding=k//2)
        n=F.conv2d(valid,filt,padding=k//2)
        channels.append(count/n.clamp_min(1)*valid)
    return torch.cat(channels,dim=1)

@torch.no_grad()
def predict(model,loader,device):
    model.eval();ys=[];ps=[]
    for x,y in loader:
        ys.extend(y.tolist());ps.extend(model(x.to(device)).argmax(1).cpu().tolist())
    return np.array(ys),np.array(ps)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    seed_all(42);torch.set_num_threads(min(8,torch.get_num_threads()))
    manifest=pd.read_csv(BASE/'split_manifest.csv',dtype={'source_row':str})
    print('Loading the exact previous split and data...',flush=True)
    with (ROOT/'LSWMD.pkl').open('rb') as f:df=LegacyUnpickler(f,encoding='latin1').load()
    maps=np.stack([representations(df.loc[int(i),'waferMap'])[0] for i in manifest.source_row])
    del df
    x=torch.from_numpy(maps[:,None].astype(np.float32))
    print('Computing masked 5x5 and 11x11 density channels...',flush=True)
    x=torch.cat([features(batch) for batch in x.split(256)])
    y=torch.tensor([CLASSES.index(c) for c in manifest.label])
    datasets={name:TensorDataset(x[np.where(manifest['split']==name)[0]],y[np.where(manifest['split']==name)[0]]) for name in ['train','valid','test']}
    del x,maps
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    val=DataLoader(datasets['valid'],batch_size=64);test=DataLoader(datasets['test'],batch_size=64)
    base_summary=pd.read_csv(BASE/'summary.csv');summaries=[base_summary[base_summary.model=='cartesian'].iloc[0].to_dict()]
    base_report=json.loads((BASE/'cartesian_report.json').read_text())
    reports={'cartesian':base_report}
    config={'seed':42,'epochs':20,'patience':5,'batch_size':64,'optimizer':'Adam','learning_rate':.001,'device':device,'split_manifest':str(BASE/'split_manifest.csv'),'channels':['original 0/1/2 map','masked 5x5 failure density','masked 11x11 failure density'],'notes':'Densities use resized 64x64 grid, not original physical die counts. No weights or augmentation. Two new variants differ only in pooling. Test reused from earlier exploration; this is not a new independent final test.'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    for pooling in ['max','avg']:
        name='density_'+pooling
        seed_all(42);model=DensityCNN(pooling).to(device)
        loader=DataLoader(datasets['train'],batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(42))
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loss_fn=nn.CrossEntropyLoss();best=-1;stale=0;history=[];start=time.time()
        for epoch in range(1,21):
            model.train();loss_sum=0
            for a,b in loader:
                a,b=a.to(device),b.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(a),b);loss.backward();optimizer.step()
                loss_sum+=loss.item()*len(b)
            truth,pred=predict(model,val,device)
            score=f1_score(truth,pred,labels=range(9),average='macro',zero_division=0)
            history.append({'epoch':epoch,'train_loss':loss_sum/len(datasets['train']),'valid_macro_f1':score,'valid_accuracy':accuracy_score(truth,pred)})
            pd.DataFrame(history).to_csv(OUT/f'{name}_history.csv',index=False)
            print(f'{name} epoch {epoch}: loss={history[-1]["train_loss"]:.4f}, validation Macro-F1={score:.4f}',flush=True)
            if score>best:
                best=score;best_epoch=epoch;stale=0
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(state,OUT/f'{name}_best.pt')
            else:stale+=1
            if stale>=5:break
        model.load_state_dict(state)
        truth,pred=predict(model,test,device)
        report=classification_report(truth,pred,labels=list(range(9)),target_names=CLASSES,output_dict=True,zero_division=0)
        reports[name]=report
        (OUT/f'{name}_report.json').write_text(json.dumps(report,indent=2))
        pd.DataFrame({'source_row':manifest.loc[manifest['split']=='test','source_row'].to_numpy(),'true':truth,'pred':pred}).to_csv(OUT/f'{name}_predictions.csv',index=False)
        pd.DataFrame(confusion_matrix(truth,pred,labels=range(9)),index=CLASSES,columns=CLASSES).to_csv(OUT/f'{name}_confusion.csv')
        summaries.append({'model':name,'test_accuracy':accuracy_score(truth,pred),'test_macro_f1':report['macro avg']['f1-score'],'test_macro_recall':report['macro avg']['recall'],'best_valid_macro_f1':best,'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.time()-start})
        pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
        del model,optimizer
    summary=pd.DataFrame(summaries)
    recall=pd.DataFrame({n:[r[c]['recall'] for c in CLASSES] for n,r in reports.items()},index=CLASSES)
    recall['test_support']=[base_report[c]['support'] for c in CLASSES]
    recall.to_csv(OUT/'recall_by_class.csv')
    names=['Original CNN (max pool)','Density CNN (max pool)','Density CNN (avg pool)']
    colors=['#627d98','#ef9850','#1b9e77']
    fig,axes=plt.subplots(2,1,figsize=(14,11),gridspec_kw={'height_ratios':[1,1.5]},layout='constrained')
    xs=np.arange(3);width=.24
    for j,(_,row) in enumerate(summary.iterrows()):
        vals=[row.test_accuracy*100,row.test_macro_f1*100,row.test_macro_recall*100]
        bars=axes[0].bar(xs+(j-1)*width,vals,width,label=names[j],color=colors[j])
        axes[0].bar_label(bars,fmt='%.2f',fontsize=10,padding=3)
    axes[0].set_xticks(xs,['Accuracy','Macro-F1 (x 100)','Macro Recall']);axes[0].set_ylim(0,105)
    axes[0].set_title('Overall test performance');axes[0].legend(loc='upper center',ncol=3)
    xs=np.arange(9)
    for j,n in enumerate(reports):
        bars=axes[1].bar(xs+(j-1)*width,recall[n]*100,width,label=names[j],color=colors[j])
        axes[1].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    axes[1].set_xticks(xs,[f'{c}\n(n={int(recall.loc[c,"test_support"])})' for c in CLASSES]);axes[1].set_ylim(0,110)
    axes[1].set_title('Recall by actual class');axes[1].set_ylabel('Recall (%)')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Does explicit local density help CNN classification?\nSame frozen split, seed, optimizer; checkpoints selected by validation Macro-F1',fontsize=15)
    fig.savefig(OUT/'performance_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(summary.to_string(index=False),flush=True)
    print('Saved graph:',OUT/'performance_comparison.png',flush=True)
if __name__=='__main__':main()
