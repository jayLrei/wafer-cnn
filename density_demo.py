"""Demonstrate defect density with fixed convolutions and mask-aware pooling.
No classifier is trained; all output values are exact descriptive features.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from run_experiment import LegacyUnpickler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'density_demo'

def masks(wafer):
    w = torch.as_tensor(np.asarray(wafer).copy())
    return (w == 2).float()[None,None], (w > 0).float()[None,None]

def local_density(wafer, kernel_size):
    """All-ones filter counts failures and valid dies independently."""
    failed, valid = masks(wafer)
    kernel = torch.ones(1,1,kernel_size,kernel_size)
    numerator = F.conv2d(failed,kernel,padding=kernel_size//2)
    denominator = F.conv2d(valid,kernel,padding=kernel_size//2)
    density = numerator / denominator.clamp_min(1)
    # Render only centers located on the wafer; off-wafer is not zero density.
    return density[0,0].numpy(), (valid[0,0]>0).numpy()

def pooled_density(wafer, size=8):
    failed, valid = masks(wafer)
    numerator = F.adaptive_avg_pool2d(failed,(size,size))
    denominator = F.adaptive_avg_pool2d(valid,(size,size))
    density = numerator/denominator.clamp_min(1e-12)
    return density[0,0].numpy(), (denominator[0,0]>0).numpy()

def verify():
    # A single failure among nine valid dies must give 1/9.
    w=np.ones((3,3),dtype=np.uint8);w[1,1]=2
    d,m=local_density(w,3)
    assert abs(d[1,1]-1/9)<1e-6
    p,_=pooled_density(w,1)
    assert abs(p[0,0]-1/9)<1e-6
    # Mask-aware edge density: all actual dies failed, padding is not normal.
    w=np.zeros((5,5),dtype=np.uint8);w[1:4,1:4]=2
    d,m=local_density(w,3); assert np.allclose(d[m],1)
    p,m=pooled_density(w,3); assert np.allclose(p[m],1)
    z=np.ones((7,7),dtype=np.uint8)
    d,m=local_density(z,5); assert np.all(d[m]==0)
    print('PASS: exact density, background exclusion, pooling ratios, zero failures',flush=True)

def main():
    verify()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(ROOT/'results/polar_comparison/split_manifest.csv',dtype={'source_row':str})
    print('Loading real training samples...',flush=True)
    with (ROOT/'LSWMD.pkl').open('rb') as f:
        df=LegacyUnpickler(f,encoding='latin1').load()
    classes=['none','Scratch','Loc','Random']
    candidates={}
    # Select same native dimensions for a fair spatial-window comparison.
    for label in classes:
        indices=manifest.loc[(manifest['split']=='train') & (manifest.label==label),'source_row'].astype(int)
        candidates[label]=[(i,np.asarray(df.loc[i,'waferMap'])) for i in indices]
    common=set(w.shape for _,w in candidates[classes[0]])
    for label in classes[1:]: common &= set(w.shape for _,w in candidates[label])
    if not common: raise ValueError('No common native size among selected classes')
    shape=max(sorted(common),key=lambda shape:min(sum(w.shape==shape for _,w in candidates[c]) for c in classes))
    print('Common native dimensions:',shape,flush=True)
    selected=[]
    for label in classes:
        choices=[(i,w) for i,w in candidates[label] if w.shape==shape]
        # Median overall failure rate within this class/size, not a handpicked success.
        choices.sort(key=lambda pair:(np.mean(pair[1][pair[1]>0]==2),pair[0]))
        i,w=choices[len(choices)//2];selected.append((label,i,w))
    del df,candidates
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d9dde2')
    fig,axes=plt.subplots(4,4,figsize=(13,12),layout='constrained')
    names=['Original map','5 x 5 convolution density','11 x 11 convolution density','8 x 8 pooled density']
    summaries=[]
    saved={}
    for row,(label,i,w) in enumerate(selected):
        overall=float((w==2).sum()/(w>0).sum())
        axes[row,0].imshow(w,cmap=ListedColormap(['#090909','#b73759','#ffffa2']),vmin=0,vmax=2,interpolation='nearest')
        axes[row,0].set_ylabel(f'{label}\nOverall: {overall:.1%}',fontsize=12)
        info={'label':label,'source_row':int(i),'split':'train','shape':str(shape),'failed_dies':int((w==2).sum()),'valid_dies':int((w>0).sum()),'overall_density':overall}
        saved[label+'_original']=w
        for col,(d,m) in enumerate([local_density(w,5),local_density(w,11),pooled_density(w)],1):
            im=axes[row,col].imshow(np.ma.array(d,mask=~m),cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            info[f'{col}_max_density']=float(d[m].max())
            saved[label+'_'+str(col)+'_density']=d
            saved[label+'_'+str(col)+'_valid']=m
        summaries.append(info)
        for col in range(4):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(names[col],fontsize=11)
    fig.colorbar(im,ax=axes[:,1:],shrink=.7,label='Local failed / valid dies (0 = 0%, 1 = 100%)')
    fig.suptitle(f'Real wafer density | Same native size {shape[0]} x {shape[1]} | Fixed filters, no training\nGray = no valid die / background; heatmap colors are NOT original wafer colors',fontsize=13)
    fig.savefig(OUT/'real_wafer_density.png',dpi=150);plt.close(fig)
    pd.DataFrame(summaries).to_csv(OUT/'density_values.csv',index=False)
    np.savez_compressed(OUT/'density_maps.npz',**saved)
    # Controlled demonstration: same global density, different local arrangement.
    clustered=np.ones((8,8),dtype=np.uint8)
    for n in range(13): clustered[n//4,n%4]=2
    clustered[1,5]=clustered[5,1]=clustered[5,5]=2
    scattered=np.ones((8,8),dtype=np.uint8);scattered[::2,::2]=2
    assert (clustered==2).sum()==(scattered==2).sum()==16
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for row,(name,w) in enumerate([('Clustered',clustered),('Scattered',scattered)]):
        f,v=masks(w)
        avg=F.avg_pool2d(f,4)[0,0].numpy()
        maximum=F.max_pool2d(f,4)[0,0].numpy()
        d,m=local_density(w,5)
        arrays=[w==2,d,avg,maximum]
        for col,a in enumerate(arrays):
            im=axes[row,col].imshow(a,cmap='magma',vmin=0,vmax=1,interpolation='nearest')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if col==0: axes[row,col].set_ylabel(name+'\n16/64 failed = 25%')
            if row==0: axes[row,col].set_title(['Defect mask','5 x 5 convolution density','4 x 4 average pooling','4 x 4 max pooling'][col],fontsize=10)
            if col>=2:
                for y in range(a.shape[0]):
                    for x in range(a.shape[1]):axes[row,col].text(x,y,f'{a[y,x]:.2f}',ha='center',va='center',color='black' if a[y,x]>.6 else 'white',fontsize=9)
    fig.suptitle('Controlled example: equal total defects, different local density\nAverage pooling retains density; max pooling only detects presence within each block')
    fig.colorbar(im,ax=axes,shrink=.7)
    fig.savefig(OUT/'pooling_comparison.png',dpi=160);plt.close(fig)
    (OUT/'README.md').write_text('''# 밀도 비교 결과\n\n이 결과는 학습된 CNN의 판단이나 정확도가 아니라, 고정된 필터와 풀링으로 계산한 특징입니다.\n\n`real_wafer_density.png`: 기존 train 분할에서 none/Scratch/Loc/Random의 공통 원본 크기를 선택하고, 각 유형·크기 안에서 전체 불량률이 중앙 순위인 샘플을 선택했습니다. 라벨별 대표성이나 구분 가능성을 보장하지 않습니다.\n\n5×5 및 11×11 all-ones 합성곱을 불량 마스크와 유효 칩 마스크에 각각 적용한 뒤 나눕니다. 검은 배경과 padding은 분모에 포함하지 않습니다. 풀링도 두 마스크를 각각 평균 풀링하고 나눕니다. 8×8 adaptive pooling 구역은 입력 크기가 나누어떨어지지 않으면 일부 겹칠 수 있습니다.\n\n`pooling_comparison.png`는 원리를 분리해 보여주는 인공 예시입니다. 두 입력 모두 16/64칸이 불량입니다. 고정 4×4 블록에서 Max Pooling은 두 예시의 출력이 같지만 Average Pooling은 다릅니다. 일반적인 학습 CNN의 Max Pooling도 앞선 학습 필터가 추출한 정보를 전달할 수 있으므로 Max Pooling CNN 전체가 밀도를 구분 못한다는 뜻은 아닙니다.\n\n고정 필터로 밀도를 명시적으로 계산할 수 있다는 것과, 자유롭게 학습한 CNN 필터가 실제로 밀도를 사용한다는 것은 다릅니다. 밀도만으로 9개 유형의 분류 성능이 좋아지는지도 별도 학습·검증이 필요합니다.\n\n재현: 프로젝트 .venv Python으로 `density_demo.py` 실행. 숫자는 density_values.csv와 density_maps.npz에 저장했습니다.\n''')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print('Saved:',OUT,flush=True)

if __name__=='__main__': main()
