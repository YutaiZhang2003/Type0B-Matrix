#!/usr/bin/env python3
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parents[1]
DATA_DIR = WORKSPACE_DIR / "Data Set" / CODE_DIR.name
import json, math
rows=[
(0.50,0.019271352900073854,0.009736686472813774),
(0.52,0.021129783891282346,0.012481859856881033),
(0.54,0.023062400410216802,0.015618180222318924),
(0.56,0.025068232876020426,0.019182841281596367),
(0.58,0.0271466290789497,0.023216155989555388),
(0.60,0.029297398070543806,0.02776184983185072),
(0.62,0.03152098446189293,0.032867382588879585),
(0.64,0.03381867899544899,0.03858430093688982),
(0.66,0.03619287237582014,0.04496862425401826),
(0.68,0.03864736067536122,0.05208126591930697),
(0.70,0.04118771221059207,0.05998849219297747),
]
out=[]
for a,re,im in rows:
 w=1/3+1j*a; A=re+1j*im; p=-math.pi*w**4/(27*(1+2j*w/3))
 out.append({'a':a,'omega0':[w.real,w.imag],'value':[re,im],'prediction':[p.real,p.imag], 'relative_error':abs((A-p)/p)})
json.dump({'settings':{'q_order':5,'p_quadrature':'segmented 124-node','p_max':4.0,'epsilon0':0.08,'epsilon1':0.06,'theta_orders':[24,24,60],'radial_order':24,'disk_total_order':14,'lens_radial_order':20,'lens_angular_order':56,'lens_power':3.0},'rows':out},open(DATA_DIR/'heterotic_equal_scan_segmented_complete.json','w'),indent=2)
print('wrote',len(out))
