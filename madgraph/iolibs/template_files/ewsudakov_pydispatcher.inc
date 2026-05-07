"""
EW Sudakov Python Dispatcher

This module provides the interface between Python reweighting code and the
Fortran EW Sudakov routines. PDG mappings are filled in at generation time.
"""
import importlib
import os
import sys
import numpy as np

# =============================================================================
# Path discovery from __file__ (no hardcoded absolute paths)
# =============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RW_ME_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # bin/internal -> bin -> rw_me
_SUBPROCESSES_DIR = os.path.join(_RW_ME_DIR, 'SubProcesses')

if _SUBPROCESSES_DIR not in sys.path:
    sys.path.insert(0, _SUBPROCESSES_DIR)

# =============================================================================
# Generated at export time (filled in by MadGraph)
# =============================================================================

pdir_list = [%(pdir_list)s]

pdg2ewsud_dict = {%(pdg2sud)s}

original_pdg_list_dict = {%(pdgsorted)s}


# =============================================================================
# Public API
# =============================================================================

def ewsudakov(pdgs, p, g):
    """
    Compute scalar EW Sudakov correction.

    Args:
        pdgs: ((initial), (sorted_final)) tuple
        p: list of 4-momenta [[E,px,py,pz], ...]
        g: strong coupling constant

    Returns:
        numpy array [born, sud0, sud1, sud2, sud3, sud4]
    """
    if pdgs not in pdg2ewsud_dict:
        raise KeyError(f"No EW Sudakov module for PDGs: {pdgs}. "
                      f"Available: {list(pdg2ewsud_dict.keys())}")

    p_transp = np.array([[pp[i] for pp in p] for i in range(4)], order='F')
    res = np.array([0., 0., 0., 0., 0., 0.], order='F')

    pdg2ewsud_dict[pdgs].ewsudakov_py(p_transp, g, res)
    return res


# Maximum dimensions for density matrix arrays
MAX_DIM = 36   # For ttWW: 2*2*3*3 = 36
MAX_RES = 10   # Maximum number of resonances


def density_sudakov(pdgs, p, g, res_indices, res_dims):
    """
    Compute density matrix EW Sudakov for spin-correlated reweighting.

    Args:
        pdgs: ((initial), (sorted_final)) tuple
        p: list of 4-momenta [[E,px,py,pz], ...]
        g: strong coupling constant
        res_indices: list of resonance leg indices (1-based)
        res_dims: list of spin dimensions (2 for fermion, 3 for vector)

    Returns:
        dict with:
          - density_born: Born density matrix B[h,h'] (upper triangle)
          - density_delta: central Sudakov corrections (s_to_rij=True)
          - density_delta_var: variation Sudakov corrections (s_to_rij=False)
          - born_diag: diagonal Born elements (IDEN-normalized)
          - results: scalar results array
          - total_dim, nres: dimensions
    """
    if pdgs not in pdg2ewsud_dict:
        raise KeyError(f"No EW Sudakov module for PDGs: {pdgs}. "
                      f"Available: {list(pdg2ewsud_dict.keys())}")

    p_transp = np.array([[pp[i] for pp in p] for i in range(4)], order='F')

    nres = len(res_indices)
    res_idx_arr = np.zeros(MAX_RES, dtype=np.int32, order='F')
    res_dim_arr = np.zeros(MAX_RES, dtype=np.int32, order='F')
    for i, (idx, dim) in enumerate(zip(res_indices, res_dims)):
        res_idx_arr[i] = idx
        res_dim_arr[i] = dim

    density_born, density_delta, density_delta_var, born_diag, results = (
        pdg2ewsud_dict[pdgs].density_sudakov_py(
            p_transp, g, nres, res_idx_arr, res_dim_arr
        )
    )

    total_dim = 1
    for dim in res_dims:
        total_dim *= dim

    actual_born_size = total_dim * (total_dim + 1) // 2

    return {
        'density_born': density_born[:actual_born_size],
        'density_delta': density_delta[:total_dim],
        'density_delta_var': density_delta_var[:total_dim],
        'born_diag': born_diag[:total_dim],
        'results': results,
        'total_dim': total_dim,
        'nres': nres
    }
