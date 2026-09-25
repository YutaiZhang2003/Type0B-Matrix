"""Regenerate the figures from shipped numerical evidence; no fitted scale."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight'})
out=ROOT/'notes/figures';out.mkdir(exist_ok=True)
scan=json.loads((ROOT/'notes/evidence/sphere_amplitude_scan.json').read_text())
rows=scan['rows'];t=np.array([r['t'] for r in rows])
fig,axes=plt.subplots(1,2,figsize=(9.1,3.35),layout='constrained')
curve=np.linspace(min(t),max(t),400)
axes[0].plot(curve,curve**4*(1-2*curve)/27,color='#323d4a',label='Matrix prediction')
for key,label,color,marker in [('mixed','A → ATT','#136c9e','o'),('four_r','A → AAA','#bd591f','x')]:
    axes[0].plot(t,[r[key][1] for r in rows],marker=marker,linestyle='none',color=color,label=label,ms=5)
    axes[1].semilogy(t,[r[key+'_relative_difference'] for r in rows],marker=marker,color=color,label=label)
axes[0].set(xlabel=r'$t$, with $\omega=it$',ylabel=r'$\mathrm{Im}(\mu_F^2\mathcal{M})$',title='Type 0B sphere amplitudes')
axes[1].set(xlabel='t',ylabel='Relative difference from matrix prediction',title='Observed comparison error')
for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
fig.savefig(out/'type0b_sphere_amplitudes.pdf');fig.savefig(out/'type0b_sphere_amplitudes.png',dpi=180);plt.close(fig)

cross=json.loads((ROOT/'validation/cross_channel.json').read_text())
raw=cross['raw'];decode=lambda x:np.asarray(x)[...,0]+1j*np.asarray(x)[...,1]
fig,ax=plt.subplots(figsize=(7.7,3.05),layout='constrained')
x=np.arange(len(raw['points']))
for family,label,color in [('four_r','Four R','#bd591f'),('mixed','Mixed NS/R','#136c9e')]:
    s=decode(raw[family]['s'] if family=='four_r' else raw[family]['s']['one']['sum'])
    for channel,marker in [('t','o'),('u','s')]:
        v=decode(raw[family][channel] if family=='four_r' else raw[family][channel]['one']['sum'])
        err=abs(s-v)/np.maximum(np.maximum(abs(s),abs(v)),1e-300)
        ax.semilogy(x, np.maximum(err,1e-16),marker=marker,linestyle='-' if channel=='t' else '--',
                    color=color,label=label+', s vs '+channel)
ax.set_xticks(x,[f'{a:g}{b:+g}i' for a,b in raw['points']])
ax.set(xlabel='z in the base chart',ylabel='Relative channel discrepancy',
       title='Fresh assembly of the order-5 production banks at ω = i/4')
ax.grid(alpha=.2);ax.legend(frameon=False,ncol=2,fontsize=9)
fig.savefig(out/'sphere_crossing.pdf');fig.savefig(out/'sphere_crossing.png',dpi=180);plt.close(fig)
print('Wrote four figure files.')
