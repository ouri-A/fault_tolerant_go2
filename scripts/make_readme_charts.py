import json, sys, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({'font.size':10,'axes.edgecolor':'#c3c2b7','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e',
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e1e0d9','grid.linewidth':0.8,'axes.axisbelow':True,
 'legend.frameon':False,'axes.titlesize':11,'axes.titleweight':'semibold','savefig.dpi':160,'figure.facecolor':'white'})
C={'stock':'#1baf7a','policy2':'#eb6834','policy3':'#2a78d6'}
NAME={'stock':'Stock (trained healthy)','policy2':'Policy 2 (failures, no memory)','policy3':'Policy 3 (failures, 0.6 s memory)'}
J=[f'{l}_{p}' for l in ['FL','FR','RL','RR'] for p in ['hip','thigh','calf']]
def d(label): return json.load(open(os.path.join(R,label+'.json')))['summary']['mean_distance_m']
def lab(p,s,c,pre=''): return f'{pre}{p}_{c}' if s==42 else f'{pre}{p}_s{s}_{c}'
SEEDS={'stock':[42],'policy2':[42,43,44],'policy3':[42,43,44]}
P=['stock','policy2','policy3']
straight={p:np.array([[d(lab(p,s,c)) for s in SEEDS[p]] for c in J]) for p in P}
diag={p:np.array([d(f'diag_{p}_{c}') for c in J]) for p in P}
for p in P: print('straight',p,straight[p].mean().round(2),'per-seed',straight[p].mean(0).round(2),'| diag',diag[p].mean().round(2))

# chart 1: per joint, straight
fig,ax=plt.subplots(figsize=(11,4.2)); x=np.arange(len(J)); w=0.8/3
for i,p in enumerate(P):
    xi=x-0.4+w/2+i*w; v=straight[p]
    ax.bar(xi,v.mean(1),width=w*0.88,color=C[p],label=NAME[p]+('' if p=='stock' else ', mean of 3 seeds'))
    if v.shape[1]>1:
        for k in range(v.shape[1]): ax.scatter(xi,v[:,k],s=9,color='#0b0b0b',zorder=3,linewidths=0)
ax.axhline(12,color='#898781',ls='--',lw=1); ax.text(-0.45,12.15,'perfect: 12 m',ha='left',va='bottom',fontsize=8.5,color='#52514e')
ax.axhline(0,color='#c3c2b7',lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([j.replace('_',' ') for j in J],fontsize=9)
ax.set_ylabel('distance walked in 20 s (m)'); ax.set_ylim(-3,13.5)
ax.set_title('Distance walked with one joint locked, 0.6 m/s straight command (dots = individual seeds)',loc='left')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),ncol=3,fontsize=9)
fig.savefig(os.path.join(OUT,'per_joint_straight.png'),bbox_inches='tight'); plt.close(fig)

# chart 2: summary straight vs diagonal
fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,(title,data,perfect) in zip(axs,[('Straight, 0.6 m/s (3 seeds)',straight,12.0),('Diagonal, vx 0.5 + vy 0.3 m/s (seed 42)',diag,11.66)]):
    for i,p in enumerate(P):
        v=data[p]; m=v.mean()
        ax.bar(i,m,width=0.62,color=C[p])
        top=m
        if v.ndim==2 and v.shape[1]>1:
            sm=v.mean(0); ax.scatter([i]*len(sm),sm,s=14,color='#0b0b0b',zorder=3,linewidths=0); top=max(m,sm.max())
        ax.text(i,0.5,f'{m:.2f} m',ha='center',va='bottom',fontsize=10,color='white',fontweight='semibold')
    ax.axhline(perfect,color='#898781',ls='--',lw=1); ax.text(2.4,perfect+0.15,f'perfect: {perfect:g} m',ha='right',va='bottom',fontsize=8.5,color='#52514e')
    ax.set_xticks(range(3)); ax.set_xticklabels(['Stock','Policy 2\nno memory','Policy 3\n0.6 s memory'])
    ax.set_title(title,loc='left',fontsize=10)
axs[0].set_ylabel('mean distance over 12 locked joints (m)'); axs[0].set_ylim(0,13.5)
fig.suptitle('Mean distance along the command, averaged over all 12 locked joints (dots = seed means)',x=0.01,ha='left',fontsize=11,fontweight='semibold')
fig.tight_layout()
fig.savefig(os.path.join(OUT,'summary_straight_diagonal.png'),bbox_inches='tight'); plt.close(fig)
