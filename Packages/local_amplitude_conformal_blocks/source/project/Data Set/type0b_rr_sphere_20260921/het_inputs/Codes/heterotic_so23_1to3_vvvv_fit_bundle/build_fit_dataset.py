#!/usr/bin/env python3
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parents[1]
DATA_DIR = WORKSPACE_DIR / "Data Set" / CODE_DIR.name
import json, math
rows = [
  dict(label='slice_a035_r020_030_050',
       energies=[[0.06666666666666667,0.07],[0.1,0.105],[0.16666666666666666,0.175],[0.3333333333333333,0.35]],
       value=[[0.006079452028137991,-0.0006210047039395399],[0.006376228001826262,-0.0013071197518240026],[0.006478740726840955,-0.0017038242330484894]],
       q_order=5,p_quadrature='48-point global'),
  dict(label='slice_a035_r015_035_050',
       energies=[[0.05,0.0525],[0.11666666666666665,0.12249999999999998],[0.16666666666666666,0.175],[0.3333333333333333,0.35]],
       value=[[0.005319520263027238,-0.0005433791011476572],[0.005522939882692639,-0.0009816692567985125],[0.0057003989939039415,-0.0016751832595788218]],
       q_order=5,p_quadrature='48-point global'),
  dict(label='slice_a050_r020_030_050',
       energies=[[0.06666666666666667,0.1],[0.1,0.15],[0.16666666666666666,0.25],[0.3333333333333333,0.5]],
       value=[[0.013459569809106803,0.008644530470896368],[0.016063240248039862,0.007659325437564923],[0.017442058504593478,0.006792389802169005]],
       q_order=5,p_quadrature='48-point global'),
  dict(label='slice_a060_r020_030_050_high_precision',
       energies=[[0.06666666666666667,0.12],[0.1,0.18],[0.16666666666666666,0.3],[0.3333333333333333,0.6]],
       value=[[0.0185342280375305,0.022395352214606003],[0.024917065521813647,0.022330691164935267],[0.028724507872596273,0.02135752585343376]],
       q_order=6,p_quadrature='segmented 124-node'),
  dict(label='different_total_re025_im045',
       energies=[[0.055,0.099],[0.0775,0.1395],[0.1175,0.2115],[0.25,0.45]],
       value=[[0.005389181845507992,0.00739340512485278],[0.006430606383571165,0.00759226098957317],[0.007109220309228661,0.007639206867417688]],
       q_order=5,p_quadrature='64-point global'),
  dict(label='different_total_re045_im042',
       energies=[[0.0765,0.0714],[0.1305,0.1218],[0.243,0.2268],[0.45,0.42]],
       value=[[0.013338393556420096,-0.005467643534258029],[0.013150593742803227,-0.008338385024390348],[0.01264730308886192,-0.009785128731609026]],
       q_order=5,p_quadrature='72-point global'),
  dict(label='extreme_partition_a050',
       energies=[[0.03333333333333333,0.05],[0.08333333333333333,0.125],[0.21666666666666667,0.325],[0.3333333333333333,0.5]],
       value=[[0.006364830969632196,0.004829550351241247],[0.00907354462692073,0.003933678897097398],[0.01018147961834989,0.0030408923937593735]],
       q_order=5,p_quadrature='64-point global'),
  dict(label='noncollinear_1',
       energies=[[0.05,0.09],[0.11,0.16],[0.16,0.23],[0.32,0.48]],
       value=[[0.010724891792969043,0.007757760558665159],[0.012256379469326319,0.007294793223906769],[0.013896054215337739,0.0062005943831106155]],
       q_order=5,p_quadrature='segmented 124-node'),
  dict(label='noncollinear_2',
       energies=[[0.08,0.07],[0.13,0.18],[0.19,0.2],[0.4,0.45]],
       value=[[0.01574106482496678,-0.0015042332379075342],[0.015577316156398669,-0.002778674800366244],[0.016869810637271346,-0.0051950019123943566]],
       q_order=5,p_quadrature='segmented 124-node'),
  dict(label='euclidean_high_precision',
       energies=[[0.0,0.1],[0.0,0.12],[0.0,0.14],[0.0,0.36]],
       value=[[-0.0024359432197652146,6.780914407470344e-19],[-0.002500046990680185,-1.258638123544058e-18],[-0.0025676158364445227,-1.2351765098267175e-18]],
       q_order=6,p_quadrature='segmented 124-node'),
]
for row in rows:
    E=[complex(*z) for z in row['energies']]
    w1,w2,w3,w0=E
    common=-math.pi*w0*w1*w2*w3
    pred=[common/(1+1j*(w1+w2)),common/(1+1j*(w1+w3)),common/(1+1j*(w2+w3))]
    val=[complex(*z) for z in row['value']]
    row['prediction']=[[z.real,z.imag] for z in pred]
    row['relative_error']=[abs((v-p)/p) for v,p in zip(val,pred)]
json.dump({'tensor_basis':['delta03 delta12','delta02 delta13','delta01 delta23'],
           'common_settings':{'p_max':4.0,'epsilon0':0.08,'epsilon1':0.06,'theta_orders':[24,24,60],
                              'radial_order':24,'lens_radial_order':20,'lens_angular_order':64,
                              'regularization':'analytic small-z OPE disk plus direct crossed-channel lens'},
           'rows':rows},open(DATA_DIR/'heterotic_vvvv_fit_data.json','w'),indent=2)
print('wrote',len(rows),'kinematic points and',3*len(rows),'complex coefficients')
