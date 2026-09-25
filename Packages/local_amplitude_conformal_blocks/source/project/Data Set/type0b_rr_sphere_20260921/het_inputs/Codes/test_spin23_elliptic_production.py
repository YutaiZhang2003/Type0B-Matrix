"""Production nome conversion, descendant components and OPE consistency."""

from unittest.mock import patch

import mpmath as mp
import numpy as np
import pytest

import spin23_singlet_amplitudes as s
import ns_elliptic_conversion as e
from check_spin23_cft_repairs import legacy_value


ENERGIES = (.11+.15j,.17+.16j,.23+.18j,.51+.49j)


@pytest.fixture(scope="module")
def blocks():
    weights = tuple(s.fast.h_of_p(w) for w in ENERGIES)
    table = s._recursive_coefficient_table(weights,np.array([.73]),7,tuple(s.WORD_PATTERNS))
    return {name:s._generic_block(h_internal=s.fast.h_of_p(.73),external_weights=weights,words=words,
                even_coefficients=table.coefficients[name][0][0],odd_coefficients=table.coefficients[name][1][0],q_order=7)
            for name,words in s.WORD_PATTERNS.items()}


def test_lambda_and_inverse_are_independent_algebraic_inverses():
    z,u,theta = e.lambda_series(12)
    q,v,theta_z = e.inverse_lambda_series(12)
    np.testing.assert_allclose(z[:5],[0,16,-128,704,-3072],rtol=0,atol=0)
    np.testing.assert_allclose(q[:5],[0,1/16,1/32,21/1024,31/2048],rtol=0,atol=0)
    identity=np.zeros(13,dtype=complex); identity[1]=1
    # z(q(z)) is well conditioned (the inverse coefficients contain 16^-n).
    np.testing.assert_allclose(e.compose(z,q,13),identity,rtol=0,atol=2e-14)
    np.testing.assert_allclose(e.compose(theta,q,13),theta_z,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(e.multiply([1],e.power([1],.7,5),5),[1,0,0,0,0])


@pytest.mark.parametrize("name",tuple(s.WORD_PATTERNS))
@pytest.mark.parametrize("beta",(0,1))
def test_forward_and_reverse_conversion_recovers_coefficients_without_splicing(blocks,name,beta):
    block=blocks[name]; elliptic=block.elliptic; order=7; length=order+1
    q,v,theta=e.inverse_lambda_series(order)
    one_minus=np.zeros(length,dtype=complex); one_minus[:2]=[1,-1]
    normal=e.multiply(e.power(v,elliptic.h-elliptic.kappa+beta/2,length),
                      e.power(one_minus,elliptic.kappa-elliptic.weights[1]-elliptic.weights[2],length),length)
    normal=e.multiply(normal,e.power(theta,elliptic.theta_exponent,length),length)/4**beta
    recovered=e.multiply(normal,e.compose(elliptic.coefficients[beta],q,length),length)
    expected=(block.even_z,block.odd_z)[beta]
    np.testing.assert_allclose(recovered,expected,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("name",tuple(s.WORD_PATTERNS))
@pytest.mark.parametrize("beta",(0,1))
def test_plane_values_match_independent_historical_conversion(blocks,name,beta):
    z=np.array([.25+.1j,.49+.84j,-.8+.1j,.53+.1j])
    block=blocks[name]
    for order in (2,5,7):
        np.testing.assert_allclose(block.elliptic.value(z,beta,order),legacy_value(block,beta,z,order),
                                   rtol=3e-11,atol=2e-12)


@pytest.mark.parametrize("name",tuple(s.WORD_PATTERNS))
@pytest.mark.parametrize("beta",(0,1))
def test_local_patch_expands_the_truncated_nome_not_the_original_z_polynomial(blocks,name,beta):
    block=blocks[name]
    z=np.array([.16+.07j,.29+.06j])
    power,coeff=block.elliptic.local_data(beta,2,32)
    local=np.exp(power*np.log(z))*np.polynomial.polynomial.polyval(z,coeff)
    np.testing.assert_allclose(local,block.elliptic.value(z,beta,2),rtol=2e-13,atol=2e-13)
    assert len(coeff)==33
    assert np.max(abs(coeff[3:]))>1e-9  # resummation has a genuine higher-z tail
    assert abs(local[1]-block.direct_value(beta,z,2)[1])>1e-7


def test_nome_and_plane_derivatives_at_both_cut_lips(blocks):
    z=np.array([.25+.1j,-.7+.02j,-.7-.02j,.49+.84j])
    q,t,dt=e.nome_geometry(z)
    qc,tc,_=e.nome_geometry(z.conj())
    np.testing.assert_allclose(qc,q.conj(),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(tc,t.conj(),rtol=1e-14,atol=1e-14)
    for zz,qq in zip(z,q):
        with mp.workdps(60):
            expected=complex(mp.exp(-mp.pi*mp.ellipk(1-complex(zz))/mp.ellipk(complex(zz))))
        np.testing.assert_allclose(qq,expected,rtol=3e-14,atol=1e-16)
    delta=2e-6
    for block in blocks.values():
        for beta in (0,1):
            fd=(block.elliptic.value(z+delta,beta)-block.elliptic.value(z-delta,beta))/(2*delta)
            np.testing.assert_allclose(block.elliptic.value(z,beta,derivative=True),fd,rtol=3e-8,atol=3e-8)


def test_exact_real_cut_limits_need_no_artificial_regulator(blocks):
    z=np.array([complex(-.7,0.),complex(-.7,-0.),complex(1.7,0.),complex(1.7,-0.)])
    direction=np.array([1.,-1.,1.,-1.])
    nearby=z+1j*direction*1e-12
    np.testing.assert_allclose(e.nome_geometry(z)[0],e.nome_geometry(nearby)[0],rtol=2e-11,atol=1e-14)
    for beta in (0,1):
        np.testing.assert_allclose(blocks['M'].elliptic.value(z,beta),blocks['M'].elliptic.value(nearby,beta),rtol=2e-10,atol=2e-12)


def test_vector_adjacent_component_uses_ward_coefficients_before_nome_truncation(blocks):
    # Independently generated generic L coefficients versus the vector Ward route.
    weights=tuple(s.fast.h_of_p(w) for w in ENERGIES)
    hp=s.fast.h_of_p(.73)
    primary=s.ref.ResummedNSBlock(s.fast.FastNSBlockComputer(*weights[::-1],False,False,max_level2=15),hp,7)
    star=s.ref.ResummedNSBlock(s.fast.FastNSBlockComputer(*weights[::-1],True,True,max_level2=15),hp,7)
    adjacent=primary.adjacent_component(star)
    z=np.array([.23+.13j,.45+.6j])
    for beta in (0,1):
        np.testing.assert_allclose(adjacent.input_coefficients[beta],(blocks['L'].even_z,blocks['L'].odd_z)[beta],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(adjacent.value(z,beta),blocks['L'].elliptic.value(z,beta),rtol=2e-11,atol=2e-11)


def test_production_default_and_cache_identity_do_not_mix_representations():
    caches={}; diagnostic={}
    kwargs=dict(q_order=3,p_nodes=3,p_max=3,elliptic_cache=caches,diagnostics=diagnostic)
    sewing=s.fast.build_s_channel_data(ENERGIES,series_parameter="sewing",**kwargs)
    nome=s.fast.build_s_channel_data(ENERGIES,series_parameter="elliptic_nome",**kwargs)
    assert len(caches)==2*len(sewing)
    assert all(a.primary is not b.primary for a,b in zip(sewing,nome))
    assert diagnostic['series_parameter']=='elliptic_nome'
    assert diagnostic['numerical_algorithm']==e.ALGORITHM_VERSION
    with patch.object(s,"_gram_solvers",side_effect=AssertionError("production must not use Gram")):
        result=s.evaluate_singlet_amplitudes(ENERGIES,q_order=2,lower_q_order=1,p_nodes=3,
                  theta_orders=(4,4,8),radial_order=6,disk_total_order=8,crossed_disk_total_order=8)
    assert result.series_parameter=='elliptic_nome'
    assert result.block_backend=='c_recursion'
    assert result.momentum_quadrature['scheme']=='threshold_weighted'
    assert np.isfinite(result.values.ssvv_raw)


def test_vector_scan_options_and_new_profiles_are_explicit():
    from spin23_vvvv_scan_evaluator import _numerical_options
    from run_spin23_ssvv_equal_blind import PROFILES as equal
    from run_spin23_ssvv_real_blind import PROFILES as real
    assert _numerical_options({}, {})['series_parameter']=='elliptic_nome'
    for profiles in (equal,real):
        assert profiles['threshold_elliptic_q7']['series_parameter']=='elliptic_nome'
        assert profiles['threshold_sewing_q7']['series_parameter']=='sewing'
    with pytest.raises(ValueError,match="series_parameter"):
        _numerical_options({'series_parameter':'pillow'}, {})


@pytest.mark.parametrize('crossed',(False,True))
@pytest.mark.parametrize('process',('ssvv','ssss','v_to_vss'))
def test_analytic_ope_patches_integrate_the_same_nome_block(crossed,process):
    energies=(.05+.01j,.06+.01j,.07+.01j,.18+.03j)
    atlas=s._build_atlas(energies,q_order=2,p_nodes=3,p_max=3,p_cut=.03,gram_condition_limit=1e13)
    channel=atlas.original_t if crossed else atlas.original_s
    # Large internal P makes the radial integral absolutely convergent.
    data=s._ChannelData(energies=channel.energies,kernels=(channel.kernels[-1],))
    radius=.24
    coord,weights=(s.ref._lens_grid(radius,48,72,3.) if crossed else s.fast.disk_grid(radius,48,72,3.))
    numerical=s._grid_integral(data,coord,weights,original_z=1-coord if crossed else coord,
                q=None,theta3=None,order=2,process=process,series_parameter='elliptic_nome')
    analytic=(s._crossed_disk_integral if crossed else s._disk_integral)(data,radius,block_order=2,total_order=26,
                process=process,series_parameter='elliptic_nome')
    np.testing.assert_allclose(analytic,numerical,rtol=3e-11,atol=1e-100)
