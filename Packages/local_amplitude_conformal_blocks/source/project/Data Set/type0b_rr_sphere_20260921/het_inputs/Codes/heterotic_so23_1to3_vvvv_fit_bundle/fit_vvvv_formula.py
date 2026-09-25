#!/usr/bin/env python3
"""Fit the numerical heterotic VVVV data to simple channel functions."""
import json, math
from pathlib import Path
import numpy as np

CODE_DIR=Path(__file__).resolve().parent
WORKSPACE_DIR=CODE_DIR.parents[1]
DATA_DIR=WORKSPACE_DIR/'Data Set'/CODE_DIR.name
DATA=DATA_DIR/'heterotic_vvvv_fit_data.json'
OUT=DATA_DIR/'heterotic_vvvv_fit_results.json'
data=json.loads(DATA.read_text())
xs=[]; ys=[]; vals=[]; preds=[]; point_ids=[]; channels=[]
for ip,row in enumerate(data['rows']):
    E=[complex(*z) for z in row['energies']]; w1,w2,w3,w0=E
    pair=[w1+w2,w1+w3,w2+w3]
    common=-math.pi*w0*w1*w2*w3
    for ic,(x,zv,zp) in enumerate(zip(pair,row['value'],row['prediction'])):
        v=complex(*zv); p=complex(*zp)
        xs.append(x); ys.append(common/v); vals.append(v); preds.append(p); point_ids.append(ip); channels.append(ic)
xs=np.asarray(xs); ys=np.asarray(ys); vals=np.asarray(vals); preds=np.asarray(preds); point_ids=np.asarray(point_ids); channels=np.asarray(channels)

# Complex least squares Y=alpha+beta*x.
M=np.column_stack([np.ones_like(xs),xs])
coef=np.linalg.lstsq(M,ys,rcond=None)[0]
alpha,beta=coef
fit_y=M@coef
# Convert back to amplitudes.
commons=[]
for ip,row in enumerate(data['rows']):
    E=[complex(*z) for z in row['energies']]
    commons.extend([-math.pi*np.prod(E)]*3)
commons=np.asarray(commons)
fit_amp=commons/fit_y
locked_y=1+1j*xs
locked_amp=commons/locked_y

# Hold out the last four entire kinematic points.
train=point_ids<6; test=~train
coef_train=np.linalg.lstsq(M[train],ys[train],rcond=None)[0]
cv_amp=commons/(M@coef_train)

def metrics(pred,mask=None):
    if mask is None: mask=np.ones(len(vals),dtype=bool)
    rel=np.abs((vals[mask]-pred[mask])/pred[mask])
    return dict(count=int(mask.sum()),max_relative=float(rel.max()),rms_relative=float(np.sqrt(np.mean(rel**2))),median_relative=float(np.median(rel)))

# Competitors.
# 1: naive continuation of the exact resonance pair products.
res_poly=[]
# 2: universal total-energy factor inferred from only the equal slice.
univ=[]
for row in data['rows']:
    E=[complex(*z) for z in row['energies']]; w1,w2,w3,w0=E
    res_poly.extend([math.pi*w1*w2,math.pi*w1*w3,math.pi*w2*w3])
    F=-w0*w0/(3+2j*w0)
    univ.extend((math.pi*F*np.array([w1*w2,w1*w3,w2*w3])).tolist())
res_poly=np.asarray(res_poly); univ=np.asarray(univ)

# Quadratic inverse model to test whether an extra term is numerically needed.
M2=np.column_stack([np.ones_like(xs),xs,xs**2])
coef2=np.linalg.lstsq(M2,ys,rcond=None)[0]
amp2=commons/(M2@coef2)

result={
 'model_definition': 'Y_ij=(-pi*omega0*omega1*omega2*omega3)/A_ij = alpha + beta*(omega_i+omega_j)',
 'unrestricted_complex_fit': {
   'alpha':[float(alpha.real),float(alpha.imag)],
   'beta':[float(beta.real),float(beta.imag)],
   'distance_alpha_from_1':float(abs(alpha-1)),
   'distance_beta_from_i':float(abs(beta-1j)),
   'metrics':metrics(fit_amp),
 },
 'locked_candidate': {
   'alpha':[1.0,0.0], 'beta':[0.0,1.0], 'metrics':metrics(locked_amp),
   'channel_metrics':{str(c):metrics(locked_amp,channels==c) for c in range(3)},
 },
 'cross_validation': {
   'training_points':data['rows'][0]['label']+' through '+data['rows'][5]['label'],
   'held_out_points':[r['label'] for r in data['rows'][6:]],
   'fit_alpha':[float(coef_train[0].real),float(coef_train[0].imag)],
   'fit_beta':[float(coef_train[1].real),float(coef_train[1].imag)],
   'training_metrics':metrics(cv_amp,train),
   'test_metrics':metrics(cv_amp,test),
 },
 'quadratic_inverse_check': {
   'alpha':[float(coef2[0].real),float(coef2[0].imag)],
   'beta':[float(coef2[1].real),float(coef2[1].imag)],
   'gamma':[float(coef2[2].real),float(coef2[2].imag)],
   'gamma_magnitude':float(abs(coef2[2])),
   'metrics':metrics(amp2),
 },
 'competitors': {
   'resonance_pair_polynomial_used_off_resonance':metrics(res_poly),
   'universal_total_energy_pair_factor':metrics(univ),
 },
 'pointwise_locked_relative_errors':[float(x) for x in np.abs((vals-locked_amp)/locked_amp)],
}
OUT.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
