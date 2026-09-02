"""Compare Cartesian, polar and fused CNNs on labeled WM-811K data."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']

class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('pandas.indexes'):
            module = module.replace('pandas.indexes', 'pandas.core.indexes', 1)
        return super().find_class(module, name)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def unwrap(value):
    a = np.asarray(value).reshape(-1)
    if not len(a):
        return None
    v = a[0]
    if isinstance(v, bytes):
        v = v.decode('utf-8')
    return str(v)

def representations(w, size=64):
    """Keep categorical values; polar r=0..R vertically and theta horizontally.

    Use original coordinates before either resize; estimate center and radius
    from the valid-die bounding box. Polar corners outside the wafer remain 0.
    """
    w = np.asarray(w)
    if w.ndim != 2 or not np.isin(w, [0, 1, 2]).all() or not np.any(w > 0):
        raise ValueError('Expected a nonempty 2D map with values 0, 1, 2')
    h, width = w.shape
    yi = np.minimum(((np.arange(size)+.5)*h/size).astype(int), h-1)
    xi = np.minimum(((np.arange(size)+.5)*width/size).astype(int), width-1)
    cart = w[yi[:, None], xi[None, :]]
    yy, xx = np.nonzero(w > 0)
    cy, cx = (yy.min()+yy.max())/2, (xx.min()+xx.max())/2
    radius = max(yy.max()-yy.min()+1, xx.max()-xx.min()+1)/2
    r = (np.arange(size)+.5)/size*radius
    theta = np.arange(size)/size*2*np.pi
    py = np.rint(cy-r[:, None]*np.sin(theta)).astype(int)
    px = np.rint(cx+r[:, None]*np.cos(theta)).astype(int)
    inside = (py >= 0) & (py < h) & (px >= 0) & (px < width)
    polar = np.zeros((size, size), dtype=np.uint8)
    polar[inside] = w[py[inside], px[inside]]
    return cart.astype(np.uint8), polar

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*14*14,64), nn.ReLU())
    def forward(self,x):
        return self.layers(x)

class Classifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.encoder = Encoder()
        if mode == 'fusion':
            self.polar_encoder = Encoder()
        self.head = nn.Linear(128 if mode == 'fusion' else 64,9)
    def forward(self, cart, polar):
        if self.mode == 'fusion':
            f = torch.cat([self.encoder(cart),self.polar_encoder(polar)],dim=1)
        else:
            f = self.encoder(polar if self.mode == 'polar' else cart)
        return self.head(f)

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    target, prediction = [], []
    for cart, polar, y in loader:
        logits = model(cart.to(device), polar.to(device))
        target.extend(y.tolist())
        prediction.extend(logits.argmax(1).cpu().tolist())
    return np.asarray(target), np.asarray(prediction)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/LSWMD.pkl'))
    ap.add_argument('--output', type=Path, default=Path('results/polar_comparison'))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--none-cap', type=int, default=5000)
    ap.add_argument('--split', choices=['lot','stratified'], default='lot')
    ap.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    args = ap.parse_args()
    if not args.data.is_file():
        raise SystemExit(f'Dataset missing: {args.data.resolve()}. Supply the real LSWMD.pkl with --data; no results have been generated.')
    if args.epochs < 1 or args.patience < 1:
        raise SystemExit('epochs and patience must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    print('Loading dataset...',flush=True)
    with args.data.open('rb') as f:
        df = LegacyUnpickler(f, encoding='latin1').load()
    original_count = len(df)
    labels = df.failureType.map(unwrap)
    df = df.loc[labels.isin(CLASSES)].copy()
    df['label'] = labels.loc[df.index]
    df['source_row'] = df.index.astype(str)
    excluded = original_count-len(df)
    # Never turn missing labels into the labeled class 'none'.
    normals = df[df.label == 'none']
    if args.none_cap > 0 and len(normals) > args.none_cap:
        normals = normals.sample(args.none_cap,random_state=args.seed)
    df = pd.concat([df[df.label != 'none'],normals]).reset_index(drop=True)
    # Remove identical maps before splitting. Exclude conflicting annotations.
    df['map_hash'] = [hashlib.sha256(str(np.asarray(w).shape).encode()+np.asarray(w,dtype=np.uint8).tobytes()).hexdigest() for w in df.waferMap]
    conflicts = df.groupby('map_hash').label.nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    before_dedup = len(df)
    df = df[~df.map_hash.isin(conflicting)].drop_duplicates('map_hash').reset_index(drop=True)
    y = np.array([CLASSES.index(v) for v in df.label])
    idx = np.arange(len(y))
    if args.split == 'lot':
        if 'lotName' not in df or df.lotName.isna().any():
            raise ValueError('lotName is required for lot split; use --split stratified explicitly if unavailable.')
        groups = df.lotName.astype(str).to_numpy()
        outer = StratifiedGroupKFold(5,shuffle=True,random_state=args.seed)
        remaining,test = next(outer.split(idx,y,groups))
        inner = StratifiedGroupKFold(4,shuffle=True,random_state=args.seed+1)
        tr,va = next(inner.split(remaining,y[remaining],groups[remaining]))
        train,valid = remaining[tr],remaining[va]
        assert not (set(groups[train]) & set(groups[valid]) or set(groups[train]) & set(groups[test]) or set(groups[valid]) & set(groups[test]))
    else:
        remaining,test = train_test_split(idx,test_size=.2,stratify=y,random_state=args.seed)
        train,valid = train_test_split(remaining,test_size=.25,stratify=y[remaining],random_state=args.seed)
    splits = {'train':train,'valid':valid,'test':test}
    for name,indices in splits.items():
        counts = np.bincount(y[indices],minlength=9)
        if np.any(counts == 0):
            raise ValueError(f'{name} lacks a class: {counts}. Review the split before training.')
    manifest = df[['source_row','label','map_hash']].copy()
    if 'lotName' in df:
        manifest['lotName'] = df.lotName.astype(str)
    for name,indices in splits.items():
        manifest.loc[indices,'split'] = name
    manifest.to_csv(out/'split_manifest.csv',index=False)
    counts = pd.DataFrame({k:np.bincount(y[v],minlength=9) for k,v in splits.items()},index=CLASSES)
    counts.to_csv(out/'class_counts.csv')
    print(counts,flush=True)
    print('Preparing Cartesian and polar maps...',flush=True)
    pairs = [representations(w) for w in df.waferMap]
    cart = np.stack([p[0] for p in pairs])
    polar = np.stack([p[1] for p in pairs])
    fig,axes = plt.subplots(9,2,figsize=(7,24))
    for c,label in enumerate(CLASSES):
        i = train[np.flatnonzero(y[train] == c)[0]]
        for j,m in enumerate([cart[i],polar[i]]):
            axes[c,j].imshow(m,cmap='inferno',vmin=0,vmax=2,aspect='auto')
            axes[c,j].set_title(f'{label}: '+('Cartesian' if j==0 else 'Polar (angle / radius)'))
            axes[c,j].axis('off')
    fig.tight_layout(); fig.savefig(out/'representations.png',dpi=140); plt.close(fig)
    datasets = {k:TensorDataset(torch.from_numpy(cart[v,None].astype(np.float32)),torch.from_numpy(polar[v,None].astype(np.float32)),torch.from_numpy(y[v])) for k,v in splits.items()}
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(min(8,torch.get_num_threads()))
    valid_loader = DataLoader(datasets['valid'],batch_size=args.batch_size)
    test_loader = DataLoader(datasets['test'],batch_size=args.batch_size)
    metadata = vars(args).copy()
    metadata.update(device_used=device, unlabeled_or_unknown_excluded=excluded, deduplicated_or_conflicting_removed=before_dedup-len(df), classes=CLASSES, torch_version=torch.__version__, split_note='Approximately 60/20/20. Lot grouping by default. Selected 5000 labeled none maps; prevalence is not production prevalence.', comparison_note='Fusion has two encoders and more parameters; its gain is not proof that coordinates alone caused the gain.')
    (out/'config.json').write_text(json.dumps(metadata,default=str,indent=2))
    summaries, recalls = [],[]
    for mode in ['cartesian','polar','fusion']:
        seed_all(args.seed)
        model = Classifier(mode).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        best,best_epoch,stale = -1,0,0
        history=[]
        start=time.time()
        for epoch in range(1,args.epochs+1):
            model.train(); total_loss=0
            for a,b,t in loader:
                a,b,t=a.to(device),b.to(device),t.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(a,b),t)
                loss.backward(); optimizer.step()
                total_loss+=loss.item()*len(t)
            truth,pred=predict(model,valid_loader,device)
            score=f1_score(truth,pred,labels=range(9),average='macro',zero_division=0)
            history.append({'epoch':epoch,'train_loss':total_loss/len(datasets['train']),'valid_macro_f1':score,'valid_accuracy':accuracy_score(truth,pred)})
            print(f'{mode} epoch {epoch}: loss={history[-1]["train_loss"]:.4f} valid Macro-F1={score:.4f}',flush=True)
            pd.DataFrame(history).to_csv(out/f'{mode}_history.csv',index=False)
            if score > best:
                best,best_epoch,stale=score,epoch,0
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(state,out/f'{mode}_best.pt')
            else:
                stale+=1
            if stale>=args.patience: break
        model.load_state_dict(state)
        truth,pred=predict(model,test_loader,device)
        report=classification_report(truth,pred,labels=list(range(9)),target_names=CLASSES,output_dict=True,zero_division=0)
        (out/f'{mode}_report.json').write_text(json.dumps(report,indent=2))
        pd.DataFrame({'source_row':df.iloc[test].source_row.to_numpy(),'true':truth,'pred':pred}).to_csv(out/f'{mode}_predictions.csv',index=False)
        pd.DataFrame(confusion_matrix(truth,pred,labels=range(9)),index=CLASSES,columns=CLASSES).to_csv(out/f'{mode}_confusion.csv')
        recalls.append([report[c]['recall'] for c in CLASSES])
        summaries.append({'model':mode,'test_accuracy':accuracy_score(truth,pred),'test_macro_f1':report['macro avg']['f1-score'],'test_macro_recall':report['macro avg']['recall'],'best_valid_macro_f1':best,'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.time()-start})
        pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
        del model,optimizer
    recall_df=pd.DataFrame(np.array(recalls).T,index=CLASSES,columns=['cartesian','polar','fusion'])
    recall_df['test_support']=counts['test']
    recall_df.to_csv(out/'recall_by_class.csv')
    fig,ax=plt.subplots(figsize=(14,6))
    x=np.arange(9); width=.25
    for j,mode in enumerate(['cartesian','polar','fusion']):
        bars=ax.bar(x+(j-1)*width,np.array(recalls[j])*100,width,label=mode)
        ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
    ax.set_xticks(x,[f'{c}\n(n={counts.loc[c,"test"]})' for c in CLASSES])
    ax.set_ylim(0,110);ax.set_ylabel('Test recall (%)');ax.set_title('Per-class recall | Same held-out test set | Best validation Macro-F1 checkpoint')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(out/'recall_comparison.png',dpi=180);plt.close(fig)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(f'Graph: {(out/"recall_comparison.png").resolve()}',flush=True)

if __name__ == '__main__':
    main()
