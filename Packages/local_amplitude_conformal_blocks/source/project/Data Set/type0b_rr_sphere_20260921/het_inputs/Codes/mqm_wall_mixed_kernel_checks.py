#!/usr/bin/env python3
"""Independent closed kernels and precision checks for the SSVV audit."""
import json
from pathlib import Path
import mpmath as mp
import sympy as s
import mqm_wall_mixed_quartic_checks as direct


def jk(p,q):
    z=p-q
    ratio=z/mp.sinh(mp.pi*z/2) if z else 2/mp.pi
    common=-mp.pi*(p+q)*ratio/(4*mp.sinh(mp.pi*(p+q)/2))
    return common*mp.sinh(mp.pi*p),common*mp.sinh(mp.pi*q)


def thermal(q):return mp.pi*q*mp.coth(mp.pi*q)


def kernel_j(q):
    return q/2+q/(1+1j*q)+(1+q*q)/(8j)*(mp.polygamma(1,(1+1j*q)/2)-mp.polygamma(1,(1-1j*q)/2))


def kernel_v(q):return ((1+q*q)*thermal(q)-1)/3-1j*q


def integral_kernel(q,which,cutoff):
    def numerator(p):
        j,k=jk(p,q)
        return p*j*k/(1+p*p) if which=='J' else k*k
    pole=numerator(q)
    def smooth(p):
        if p==q:return -mp.diff(numerator,q)/(2*q)
        return (numerator(p)-pole)/(q*q-p*p)
    breaks=[0,q/2,q,q+1,q+8,cutoff]
    tail=pole/(2*q)*mp.log((cutoff+q)/(cutoff-q))
    value=2/mp.pi*(mp.quad(smooth,breaks)+tail)-1j*pole/q
    return value


def delta(q):
    return mp.mpf(2)/3-(q*q+4)*thermal(q)/3-1/(1+q*q)-q*mp.re(kernel_j(q))


def closed_ssvv(w1,w2,w3):
    w0=w1+w2+w3;q=w2+w3
    target=1+2j*w0-w0*w1/(1+1j*q)
    return target+delta(q)


def main():
    mp.mp.dps=50
    # Verify independently derived off-shell sources against the Fourier
    # generator, including several non-on-shell internal spatial momenta.
    A=s.Rational(17,10);B=s.Rational(2,5);q=mp.mpf('1.3')
    v,_=direct.make_vertex(['S','S','S'],[A,B],[-A,B,A-B])
    for p in map(mp.mpf,['.2','.8','1.3','1.9','3.1']):
        j,k=jk(p,q)
        expected=-mp.mpf(str(A*B))*((1+mp.mpf(str(A*B)))*j+q*p*k)
        assert abs(v(p)-expected)<mp.mpf('1e-43')
    out={'off_shell_vertex_checks':'pass','kernels':[], 'precision_checks':[], 'mixed_samples':[]}
    for q in map(mp.mpf,['.2','.7','1.2','1.3','2.1']):
        row={'q':str(q)}
        for which,closed in [('J',kernel_j(q)),('V',kernel_v(q))]:
            measured=integral_kernel(q,which,mp.mpf(65))
            error=abs(measured-closed)
            assert error<mp.mpf('1e-38'),(q,which,error)
            row[which]={'integral':mp.nstr(measured,40),'closed_form':mp.nstr(closed,40),'error':mp.nstr(error,5)}
        row['SSVV_difference']=mp.nstr(delta(q),40)
        out['kernels'].append(row)
    for digits,cutoff in [(40,45),(65,80)]:
        mp.mp.dps=digits;q=mp.mpf('1.3')
        measured=integral_kernel(q,'J',mp.mpf(cutoff))
        error=abs(measured-kernel_j(q))
        assert error<mp.power(10,-digits+10)
        out['precision_checks'].append({'digits':digits,'cutoff':cutoff,'J':mp.nstr(measured,digits-7),'error':mp.nstr(error,5)})
    for raw in [('.4','.6','.7'),('.3','.4','.8'),('.8','.3','1.0')]:
        w1,w2,w3=map(mp.mpf,raw)
        w0=w1+w2+w3;q=w2+w3
        value=closed_ssvv(w1,w2,w3);target=1+2j*w0-w0*w1/(1+1j*q)
        out['mixed_samples'].append({'outgoing':raw,'computed':mp.nstr(value,40),'target':mp.nstr(target,40),'difference':mp.nstr(value-target,40)})
    # The difference is already nonzero at zero pair energy, and has
    # nonconstant q² dependence; it is not one finite constant.
    for q in [mp.mpf('.0001'),mp.mpf('.00005')]:
        expansion=-mp.mpf(5)/3+q*q*(mp.mpf(7)*mp.zeta(3)/4-mp.mpf(5)/6-4*mp.pi**2/9)
        assert abs(delta(q)-expansion)<10*q**4
    # Continue the full retarded expression, rather than continuing a
    # complex-conjugation-based real-part operation away from the real axis.
    # The trigamma formula cancels 1/eps singular terms. Guard digits are
    # essential: working at 65 digits with eps=1e-18 is insufficient to
    # resolve its O(eps²) remainder through mpmath's reflection formulas.
    # Laurent expansion gives J(i+eps)=eps/2-i*zeta(3)*eps²/2+O(eps³).
    with mp.workdps(110):
        for eps in [mp.mpf('1e-14'),mp.mpf('1e-18')]:
            q=1j+eps
            continued_delta=mp.mpf(2)/3-(q*q+4)*thermal(q)/3-q*kernel_j(q)-1/(1+1j*q)-1j*q
            local_J=eps/2-1j*mp.zeta(3)*eps**2/2
            assert abs(kernel_j(q)-local_J)<10*eps**3
            assert abs(continued_delta-mp.mpf(4)/3)<10*eps
    out['continued_q_i_checks']={'J_i':'0','Delta_i':'4/3','status':'pass'}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_mixed_kernel_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
