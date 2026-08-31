"""Parity-resolved all-NS sphere elliptic h-recursion, human-note convention."""

from .recursion import (RecursionTable, compute_h_recursion, load_table,
                        total_degree_indices, KacPoleError, NORMALIZATION)
from .geometry import (AlignedNomes, elliptic_nome, cross_ratio_from_nome,
                       coordinates_from_segment_nomes, invert_aligned_coordinates)
from .block import (SphereBlockEvaluation, effective_plumbing_parameters,
                    ns_pillow_product, reconstruct_from_real_moduli,
                    reconstruct_from_segment_nomes)
from .numbers import central_charge_from_b

__version__ = "0.1.0"
__all__ = ["RecursionTable","compute_h_recursion","load_table","total_degree_indices",
           "KacPoleError","NORMALIZATION","AlignedNomes","elliptic_nome",
           "cross_ratio_from_nome","coordinates_from_segment_nomes",
           "invert_aligned_coordinates","SphereBlockEvaluation",
           "effective_plumbing_parameters","ns_pillow_product",
           "reconstruct_from_real_moduli","reconstruct_from_segment_nomes",
           "central_charge_from_b"]
