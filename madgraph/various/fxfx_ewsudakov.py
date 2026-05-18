################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""
FxFx clustering and EW Sudakov reweighting for LHE events.

This module provides:
- FxFx event clustering (n+1 body -> n body Born-like events)
- EW Sudakov reweighting with scalar and density-matrix paths
- Helper functions for charge conservation and flavor identification

The main class FxFxEWSudakovMixin is designed to be mixed into ReweightInterface.
"""
from __future__ import division

import copy
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from typing import List, Optional, Tuple

import madgraph.various.lhe_parser as lhe_parser
import numpy as np

# Convenience alias for FourMomentum
FourMomentum = lhe_parser.FourMomentum

# =============================================================================
# Module-level debug flag and function
# =============================================================================
DEBUG = False # Set True to enable verbose debug output

# Event counter set by reweight_interface.py during event loop (1-based)
CURRENT_EVENT_ID = 0


def _dbg(*args, **kwargs):
    """Debug output controlled by DEBUG flag."""
    if DEBUG:
        print("[DEBUG]", *args, flush=True, **kwargs)


# =============================================================================
# Physical constants (pole masses in GeV)
# These are defaults, overridden from param_card at runtime by
# _init_pole_masses_from_banner() once the banner is available.
# =============================================================================
MW_POLE = 80.385  # W boson pole mass (default; overridden from param_card)
MZ_POLE = 91.1876  # Z boson pole mass (default; overridden from param_card)
MT_POLE = 173.3  # Top quark pole mass (default; overridden from param_card)
MH_POLE = 125.0  # Higgs boson pole mass (default; overridden from param_card)
_POLE_MASSES_INITIALIZED = False  # Set True after reading from param_card

# =============================================================================
# MadSpin Forced Clustering Data Structures
# =============================================================================


@dataclass
class ForcedClusterStep:
    """
    A single forced clustering step from MadSpin decay reversal.

    When MadSpin decays a resonance (e.g., W+ -> l+ nu), we need to reverse
    this by clustering the decay products back into the mother resonance.

    Attributes:
        children_lhe_idx: List of 1-based LHE indices of particles being clustered
        mother_lhe_idx: 1-based LHE index of the mother resonance
        mother_pdg: PDG code of the mother resonance
        scale: Clustering scale (invariant mass of children, in GeV)
        depth: Distance to leaves (lower depth = inner decay, clustered first)
    """

    children_lhe_idx: List[int]
    mother_lhe_idx: int
    mother_pdg: int
    scale: float
    depth: int


# =============================================================================
# Pre-compiled regex helpers for FxFx clustering parsing
# =============================================================================

FXFX_CLUSTER_STEP_RE = re.compile(r'step_(\d+)="([^"]+)"')
FXFX_CLUSTER_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# =============================================================================
# Flavor families for charge conservation fixes
# =============================================================================

FXFX_FLAVOR_FAMILIES = {
    "quark": set(range(1, 7)),
    "lepton": {11, 13, 15},
    "neutrino": {12, 14, 16},
}


# =============================================================================
# Module-level helper functions
# =============================================================================


def _choose_keep(i_pos, j_pos):
    """Choose which position to keep when clustering two particles.

    Prioritizes initial-state particles (positions 0, 1) over final-state.
    For same-category positions, keeps the one with lower index.
    """
    if i_pos <= 1 and j_pos > 1:
        return i_pos
    if j_pos <= 1 and i_pos > 1:
        return j_pos
    return min(i_pos, j_pos)


def _flavor_family(pdg):
    """Return a flavor family identifier for PDGs we can safely relabel."""
    try:
        pid = abs(int(pdg))
    except Exception:
        return None
    for family, pdgs in FXFX_FLAVOR_FAMILIES.items():
        if pid in pdgs:
            return family
    return None


def _get_charge_from_pdg(model, pdg):
    """Return electric charge for a PDG, handling antiparticles when needed."""
    if model is None:
        return 0.0
    try:
        part_dict = model.get("particle_dict") or {}
    except Exception:
        return 0.0
    abs_pdg = abs(int(pdg))
    part = part_dict.get(abs_pdg)
    if part is None:
        return 0.0
    try:
        charge = part.get("charge")
    except Exception:
        charge = 0.0
    if charge is None:
        charge = 0.0
    try:
        self_antipart = part.get("self_antipart")
    except Exception:
        self_antipart = False
    if pdg < 0 and not self_antipart:
        return -charge
    return charge


def _event_charges_after_merge(model, buff_event, keep_pos, drop_pos, mother_pdg):
    """
    Compute (Q_initial, Q_final) after clustering the legs at keep_pos and drop_pos
    into a single mother with PDG = mother_pdg (simulation only, no mutation).
    """
    q_init = 0.0
    q_final = 0.0
    for idx, p in enumerate(buff_event):
        if idx == drop_pos:
            continue

        status = int(getattr(p, "status", 0))
        pdg = int(mother_pdg) if idx == keep_pos else int(getattr(p, "pid"))
        q = _get_charge_from_pdg(model, pdg)

        if status == -1:
            q_init += q
        else:
            q_final += q

    return q_init, q_final


def _get_mother_lhe_idx(mother):
    """
    Get 1-based LHE index from a mother pointer.

    Handles both:
    - Particle object (with event_id attribute) -> returns event_id + 1
    - Raw number (float from LHE parsing) -> returns int(number)
    - None/0 -> returns 0
    """
    if mother is None:
        return 0
    if hasattr(mother, "event_id"):
        return mother.event_id + 1  # Convert 0-based to 1-based LHE
    elif isinstance(mother, (int, float)):
        return int(mother)
    return 0


def _fix_mother_sign_global(model, buff_event, pos_i, pos_j, mother_pdg, tol=1e-6):
    """
    Fix mother PDG sign/flavor using global charge conservation for this clustering step.

    - Compute (Q_init, Q_final) after merging pos_i and pos_j with mother_pdg.
    - If charges mismatch, try with -mother_pdg.
    - If neither works and mother is a quark/lepton, try other flavors in the family.
    - This handles the case where LHE coinflip picked a different flavor than FxFx clustering.
    """
    if mother_pdg is None:
        return mother_pdg, False

    keep_pos = _choose_keep(pos_i, pos_j)
    drop_pos = pos_j if keep_pos == pos_i else pos_i

    q_init_after, q_final_after = _event_charges_after_merge(
        model, buff_event, keep_pos, drop_pos, mother_pdg
    )

    if abs(q_init_after - q_final_after) < tol:
        return mother_pdg, False

    flipped = -int(mother_pdg)
    q_init_flip, q_final_flip = _event_charges_after_merge(
        model, buff_event, keep_pos, drop_pos, flipped
    )

    if abs(q_init_flip - q_final_flip) < tol:
        return flipped, True

    # Try other flavors in the same family (handles LHE coinflip vs FxFx mismatch)
    original_family = _flavor_family(mother_pdg)
    if original_family:
        family_pdgs = list(FXFX_FLAVOR_FAMILIES.get(original_family, set()))
        # Try both particles and antiparticles
        candidates = []
        for pdg in family_pdgs:
            candidates.extend([pdg, -pdg])

        for cand_pdg in candidates:
            if cand_pdg == mother_pdg or cand_pdg == flipped:
                continue  # Already tried these
            q_init_cand, q_final_cand = _event_charges_after_merge(
                model, buff_event, keep_pos, drop_pos, cand_pdg
            )
            if abs(q_init_cand - q_final_cand) < tol:
                _dbg(
                    f"FxFx clustering: fixed mother flavor {mother_pdg} -> {cand_pdg} for positions {pos_i+1},{pos_j+1}"
                )
                return cand_pdg, True

    msg = (
        "Global charge conservation FAILED for clustering of positions "
        f"{pos_i+1} and {pos_j+1} with mother PDG {mother_pdg}. "
        f"With mother={mother_pdg}: Q_init={q_init_after:.6f}, Q_final={q_final_after:.6f}; "
        f"with mother={flipped}: Q_init={q_init_flip:.6f}, Q_final={q_final_flip:.6f}"
    )
    raise RuntimeError(msg)


def suggest_mothers_from_interactions(model, perts, child_i, child_j, emitter_state):
    """Suggest possible mother legs from UFO 3-point interactions."""
    if model is None:
        return []

    try:
        part_dict = model.get("particle_dict") or {}
    except Exception:
        part_dict = {}

    if not part_dict:
        return []

    def _to_incoming_pdg(pdg, is_final):
        part = part_dict.get(pdg)
        if part is None:
            return None
        if is_final:
            return pdg
        return part.get_anti_pdg_code()

    pi_int = _to_incoming_pdg(child_i["pdg"], child_i["state"])
    pj_int = _to_incoming_pdg(child_j["pdg"], child_j["state"])
    if pi_int is None or pj_int is None:
        return []

    # Look up particles, falling back to abs(pdg) for models that don't have
    # explicit antiparticle entries in particle_dict
    part_i = part_dict.get(pi_int)
    if part_i is None and pi_int is not None:
        part_i = part_dict.get(abs(pi_int))
    part_j = part_dict.get(pj_int)
    if part_j is None and pj_int is not None:
        part_j = part_dict.get(abs(pj_int))
    if part_i is None or part_j is None:
        return []

    try:
        interactions = model.get("interactions") or []
    except Exception:
        interactions = []

    candidates = []
    for inter in interactions:
        try:
            participants = inter.get("particles") or []
        except Exception:
            continue
        if len(participants) != 3:
            continue

        if perts:
            try:
                orders = inter.get("orders") or {}
            except Exception:
                orders = {}
            if not any(orders.get(p, 0) > 0 for p in perts):
                continue

        tmp = list(participants)
        try:
            tmp.remove(part_i)
        except ValueError:
            continue
        try:
            tmp.remove(part_j)
        except ValueError:
            continue
        if len(tmp) != 1:
            continue

        remaining = tmp[0]
        if emitter_state:
            mother_pdg = remaining.get_anti_pdg_code()
        else:
            mother_pdg = remaining.get_pdg_code()

        candidates.append({"mother_pdg": mother_pdg})

    return candidates


# =============================================================================
# FxFx Clustering Kinematics (Catani-Seymour reshuffling)
# =============================================================================


# Debug flag for verbose clustering output
_cluster_dbg = _dbg  # Alias for clustering debug output


def _m2(p):
    """Compute invariant mass squared."""
    return p.E**2 - p.px**2 - p.py**2 - p.pz**2


def _get_mother_mass(mother_pid):
    """Get physical mass for a mother particle based on PDG code."""
    abs_pid = abs(mother_pid)
    if abs_pid == 6:  # top quark
        return MT_POLE
    elif abs_pid == 23:  # Z boson
        return MZ_POLE
    elif abs_pid == 24:  # W boson
        return MW_POLE
    elif abs_pid == 25:  # Higgs boson
        return MH_POLE
    else:  # photon, gluon, light quarks, etc.
        return 0.0


def _boost_along_direction(p, beta, direction):
    """
    Apply a Lorentz boost with velocity beta along a unit direction vector.

    This is the boost B from FKS Section 5.2 (Eq. 5.25-5.27).

    Convention:
        E' = γ(E - β·p·n̂)
        p'_∥ = γ(p·n̂ - β·E)

        This boosts TO a frame moving at velocity +β in direction n̂.

        For a particle with momentum p⃗ = |p|·n̂ and β = |p|/E:
        - TO rest frame: use +β  → E' = M (mass)
        - FROM rest frame: use -β → E' = γM, p'_∥ = γβM

    Args:
        p: FourMomentum to boost
        beta: Boost velocity (dimensionless, |beta| < 1)
               Positive β: boost TO frame moving in direction n̂
               Negative β: boost FROM frame moving in direction n̂
        direction: (nx, ny, nz) unit vector along boost direction

    Returns:
        Boosted FourMomentum
    """
    if abs(beta) < 1e-12:
        return FourMomentum(p)

    nx, ny, nz = direction
    gamma = 1.0 / math.sqrt(1.0 - beta**2)

    # Component of p parallel to boost direction
    p_dot_n = p.px * nx + p.py * ny + p.pz * nz

    # Boosted energy and parallel momentum
    E_new = gamma * (p.E - beta * p_dot_n)
    p_parallel_new = gamma * (p_dot_n - beta * p.E)

    # Perpendicular components unchanged
    p_perp_x = p.px - p_dot_n * nx
    p_perp_y = p.py - p_dot_n * ny
    p_perp_z = p.pz - p_dot_n * nz

    return FourMomentum(
        [
            E_new,
            p_parallel_new * nx + p_perp_x,
            p_parallel_new * ny + p_perp_y,
            p_parallel_new * nz + p_perp_z,
        ]
    )


def _transform_decay_products_to_onshell(
    decay_momenta, original_res_p4, onshell_res_p4,
):
    """
    Transform decay products from original (off-shell) resonance kinematics
    to on-shell resonance kinematics.

    Algorithm (scaling):
    1. Boost decay products to original resonance rest frame
    2. Scale momenta to match on-shell mass
    3. Boost from on-shell rest frame to lab frame

    Args:
        decay_momenta: List of FourMomentum for decay products
        original_res_p4: Original resonance 4-momentum (off-shell, from original event)
        onshell_res_p4: On-shell resonance 4-momentum (from FKS-clustered event)

    Returns:
        List of transformed FourMomentum for decay products (same order as input)
    """
    if not decay_momenta:
        return []

    n_particles = len(decay_momenta)

    # Compute original resonance properties
    orig_m2 = _m2(original_res_p4)
    orig_m = math.sqrt(max(0, orig_m2))
    orig_p = math.sqrt(original_res_p4.px**2 + original_res_p4.py**2 + original_res_p4.pz**2)

    # Compute on-shell resonance properties
    onshell_m2 = _m2(onshell_res_p4)
    onshell_m = math.sqrt(max(0, onshell_m2))
    onshell_p = math.sqrt(onshell_res_p4.px**2 + onshell_res_p4.py**2 + onshell_res_p4.pz**2)

    if DEBUG:
        _cluster_dbg("=" * 60)
        _cluster_dbg("  [C MATRIX ON-SHELL TRANSFORMATION]")
        _cluster_dbg("=" * 60)
        _cluster_dbg(f"  Original: p4=({original_res_p4.E:.4f}, {original_res_p4.px:.4f}, {original_res_p4.py:.4f}, {original_res_p4.pz:.4f}), m={orig_m:.4f}")
        _cluster_dbg(f"  On-shell: p4=({onshell_res_p4.E:.4f}, {onshell_res_p4.px:.4f}, {onshell_res_p4.py:.4f}, {onshell_res_p4.pz:.4f}), m={onshell_m:.4f}")
        _cluster_dbg(f"  Mass shift: {onshell_m - orig_m:+.4f} GeV (pole - invariant)")

    # Edge case: original resonance at rest or very low momentum
    if orig_p < 1e-10 and orig_m < 1e-10:
        _cluster_dbg("  [decay_transform] WARNING: degenerate original resonance, returning unchanged")
        return [FourMomentum(p) for p in decay_momenta]

    # Boost parameters for original resonance (boost TO rest frame)
    # β = p/E in direction of p
    if orig_p > 1e-10:
        orig_dir = (
            original_res_p4.px / orig_p,
            original_res_p4.py / orig_p,
            original_res_p4.pz / orig_p,
        )
        orig_beta = orig_p / original_res_p4.E  # boost to rest frame
    else:
        orig_dir = (0, 0, 1)
        orig_beta = 0

    # Boost decay products to original rest frame
    # The boost formula E' = γ(E - β·p·n̂) boosts TO a frame moving at +β in direction n̂
    # To go TO the rest frame of a particle with momentum p⃗, use β = |p|/E (positive)
    rest_momenta = []
    for p in decay_momenta:
        # Boost with +beta to go TO rest frame (frame moving at +β relative to lab)
        p_rest = _boost_along_direction(p, orig_beta, orig_dir)
        rest_momenta.append(p_rest)

    _cluster_dbg(f"  [decay_transform] Boosted {len(rest_momenta)} particles to rest frame")

    # In rest frame, get directions and masses of decay products
    rest_directions = []
    rest_masses = []
    for i, p_rest in enumerate(rest_momenta):
        p_mag = math.sqrt(p_rest.px**2 + p_rest.py**2 + p_rest.pz**2)
        m2 = _m2(p_rest)
        m = math.sqrt(max(0, m2))
        rest_masses.append(m)
        if p_mag > 1e-10:
            direction = (p_rest.px / p_mag, p_rest.py / p_mag, p_rest.pz / p_mag)
        else:
            direction = (0, 0, 1)
        rest_directions.append(direction)
        if DEBUG:
            _cluster_dbg(f"  [decay_transform] decay[{i}]: E_rest={p_rest.E:.4f}, |p|={p_mag:.4f}, m={m:.4f}")

    # Construct new momenta in on-shell rest frame
    # For 2-body decay: E1 + E2 = M, |p1| = |p2|
    # Using same directions but scaled for on-shell mass

    if len(decay_momenta) == 2:
        # Two-body decay: special handling for back-to-back
        m1, m2 = rest_masses[0], rest_masses[1]
        # Solve for momentum magnitude p in M rest frame:
        # E1 = sqrt(p^2 + m1^2), E2 = sqrt(p^2 + m2^2)
        # E1 + E2 = M_onshell
        # This is a standard kinematic formula
        if onshell_m > m1 + m2:
            # λ(M², m1², m2²) = (M² - (m1+m2)²)(M² - (m1-m2)²)
            lambda_val = (onshell_m2 - (m1 + m2) ** 2) * (onshell_m2 - (m1 - m2) ** 2)
            p_new = math.sqrt(max(0, lambda_val)) / (2 * onshell_m)
        else:
            # Kinematically impossible: on-shell mass < sum of decay product masses.
            # Return original momenta to preserve momentum conservation.
            _cluster_dbg(
                f"  [decay_transform] WARNING: M_onshell={onshell_m:.4f} < m1+m2={m1+m2:.4f}, "
                "returning original momenta"
            )
            return [FourMomentum(p) for p in decay_momenta]

        E1_new = math.sqrt(p_new**2 + m1**2)
        E2_new = math.sqrt(p_new**2 + m2**2)

        new_rest_momenta = [
            FourMomentum(
                [
                    E1_new,
                    p_new * rest_directions[0][0],
                    p_new * rest_directions[0][1],
                    p_new * rest_directions[0][2],
                ]
            ),
            FourMomentum(
                [
                    E2_new,
                    -p_new * rest_directions[0][0],
                    -p_new * rest_directions[0][1],
                    -p_new * rest_directions[0][2],
                ]
            ),
        ]
        if DEBUG:
            orig_p_mag = math.sqrt(rest_momenta[0].px**2 + rest_momenta[0].py**2 + rest_momenta[0].pz**2)
            _cluster_dbg(f"  2-body rescaling: |p| {orig_p_mag:.4f} -> {p_new:.4f}, E1+E2={E1_new + E2_new:.4f} (M={onshell_m:.4f})")
    else:
        # Multi-body decay: scale momenta uniformly
        # This is an approximation that preserves angles but not exact phase space
        sum_E_rest = sum(p.E for p in rest_momenta)
        scale = onshell_m / sum_E_rest if sum_E_rest > 0 else 1.0
        new_rest_momenta = []
        for p_rest in rest_momenta:
            p_new = FourMomentum(
                [p_rest.E * scale, p_rest.px * scale, p_rest.py * scale, p_rest.pz * scale]
            )
            new_rest_momenta.append(p_new)
        _cluster_dbg(f"  [decay_transform] multi-body: scale={scale:.6f}")

    # Boost from on-shell rest frame to lab frame
    if onshell_p > 1e-10:
        onshell_dir = (
            onshell_res_p4.px / onshell_p,
            onshell_res_p4.py / onshell_p,
            onshell_res_p4.pz / onshell_p,
        )
        onshell_beta = onshell_p / onshell_res_p4.E  # boost FROM rest frame
    else:
        onshell_dir = (0, 0, 1)
        onshell_beta = 0

    lab_momenta = []
    for p_rest in new_rest_momenta:
        # Boost with -beta to go FROM rest frame to lab (inverse of TO-rest-frame boost)
        # This gives the particle momentum in the +n̂ direction (matching the resonance)
        p_lab = _boost_along_direction(p_rest, -onshell_beta, onshell_dir)
        lab_momenta.append(p_lab)

    # Debug: verify momentum conservation
    if DEBUG:
        sum_p = FourMomentum()
        for p in lab_momenta:
            sum_p += p
        delta_E = abs(onshell_res_p4.E - sum_p.E)
        delta_px = abs(onshell_res_p4.px - sum_p.px)
        delta_py = abs(onshell_res_p4.py - sum_p.py)
        delta_pz = abs(onshell_res_p4.pz - sum_p.pz)
        _cluster_dbg(f"  Boost to lab: β={onshell_beta:.6f}")
        for i, p in enumerate(lab_momenta):
            m = math.sqrt(max(0, _m2(p)))
            _cluster_dbg(f"    [{i}] p4=({p.E:.4f}, {p.px:.4f}, {p.py:.4f}, {p.pz:.4f}), m={m:.4f}")
        _cluster_dbg(
            f"  Conservation: dE={delta_E:.2e}, dpx={delta_px:.2e}, dpy={delta_py:.2e}, dpz={delta_pz:.2e}"
        )
        _cluster_dbg("=" * 60)

    return lab_momenta


def _fks_isr_mapping(fks_i, fks_j, moth, orig_momenta):
    """
    FKS initial-state singularity mapping (Section 5.1 of FKS paper).

    For ISR clustering, we remove the radiation from the initial state and
    boost all final-state particles to the new CM frame with reduced √ŝ.

    The mapping:
        1. Subtract radiation momentum: q̄ = q - k_{n+1}
        2. Compute new CM energy: √ŝ' = √(q̄²)
        3. Boost FS particles to new CM frame (q̄ rest frame)
        4. Set beams along z-axis with E = √ŝ'/2
        5. Boost back to lab frame

    This effectively changes the parton momentum fractions x± while keeping
    the final-state kinematics consistent.

    Args:
        fks_i: Index of initial-state particle (0 or 1)
        fks_j: Index of radiation particle to remove
        moth: Mother info dict (beam parton after clustering)
        orig_momenta: List of FourMomentum for all particles [beam0, beam1, FS...]

    Returns dict with:
        'beams': [beam0, beam1] new beam momenta
        'final_state': dict {idx: FourMomentum} for all FS particles (except removed)
        'sqrt_s_new': new CM energy
        'method': description of method used
    """
    p_initial = orig_momenta[0] + orig_momenta[1]
    p_radiation = orig_momenta[fks_j]

    # New initial state: q̄ = q - k_{n+1} (subtract radiation)
    p_clustered = FourMomentum(
        [
            p_initial.E - p_radiation.E,
            p_initial.px - p_radiation.px,
            p_initial.py - p_radiation.py,
            p_initial.pz - p_radiation.pz,
        ]
    )

    s_hat = _m2(p_clustered)
    sqrt_s = math.sqrt(max(0, s_hat))

    if DEBUG:
        _cluster_dbg(f"  ISR FKS: √ŝ'={sqrt_s:.4f}, E_rad={p_radiation.E:.4f}, pT_rad={math.sqrt(p_radiation.px**2+p_radiation.py**2):.4f}")

    # GUARD: Nearly light-like p_clustered has no rest frame — boost is numerically unstable.
    # Threshold 0.1 GeV: below this, γ > E/0.1 makes the CM rescaling unreliable.
    # Caller (_cluster_fxfx_event) catches this and sets stop_clustering=True.
    if sqrt_s < 0.1:
        raise ValueError(f"ISR mapping: light-like p_clustered (√s={sqrt_s:.4f} GeV), cannot boost")

    result = {
        "final_state": {},
        "sqrt_s_new": sqrt_s,
        "method": "fks_isr_cm_boost",
    }

    # ISR MAPPING: Boost all particles to new CM frame (p_clustered rest frame)

    # Check if p_clustered already has zero transverse momentum
    p_clustered_pt = math.sqrt(p_clustered.px**2 + p_clustered.py**2)
    needs_pt_boost = p_clustered_pt > 1e-10

    # NOTE: The pT boost is deliberately NOT inverted at the end. The output
    # frame is defined by the clustered initial state (p_clustered along z),
    # which is the correct convention for a Born-level LHE event (beams along z).
    # The downstream Sudakov computation boosts to CM, which is frame-invariant.
    if needs_pt_boost:
        # Step 1: pT boost to remove transverse momentum of p_clustered
        after_pt = [p.pt_boost(pboost=p_clustered) for p in orig_momenta]
        p_clustered_after_pt = p_clustered.pt_boost(pboost=p_clustered)
    else:
        # Already along z - no pT boost needed
        after_pt = list(orig_momenta)  # Copy to avoid modifying original
        p_clustered_after_pt = FourMomentum(p_clustered)

    # Step 2: z-boost to CM frame
    cm_momenta = [p.zboost(pboost=p_clustered_after_pt) for p in after_pt]

    # Step 3: Set beams along z-axis with correct energy
    E_beam_cm = sqrt_s / 2.0
    if cm_momenta[0].pz >= 0:
        beam0_new = FourMomentum([E_beam_cm, 0, 0, E_beam_cm])
        beam1_new = FourMomentum([E_beam_cm, 0, 0, -E_beam_cm])
    else:
        beam0_new = FourMomentum([E_beam_cm, 0, 0, -E_beam_cm])
        beam1_new = FourMomentum([E_beam_cm, 0, 0, E_beam_cm])

    # Step 4: Boost FS particles back to lab frame
    # The lab frame here is defined by p_clustered (the new IS system)
    for ip, p_cm in enumerate(cm_momenta):
        if ip <= 1:  # Skip beams (handled separately)
            continue
        if ip == fks_j:  # Skip removed radiation
            continue
        p_lab = p_cm.zboost_inv(pboost=p_clustered_after_pt)
        result["final_state"][ip] = p_lab

    # Beams stay in CM frame (massless along z)
    # But we need to boost them back to lab too for consistency
    result["beams"] = [
        beam0_new.zboost_inv(pboost=p_clustered_after_pt),
        beam1_new.zboost_inv(pboost=p_clustered_after_pt),
    ]

    # Debug: verify momentum conservation
    if DEBUG:
        p_fs_total = FourMomentum()
        for idx, p in result["final_state"].items():
            p_fs_total += p
        p_is_new = result["beams"][0] + result["beams"][1]
        _cluster_dbg(
            f"    ISR conservation: dE={abs(p_is_new.E - p_fs_total.E):.2e}, "
            f"dpx={abs(p_is_new.px - p_fs_total.px):.2e}, dpy={abs(p_is_new.py - p_fs_total.py):.2e}, "
            f"dpz={abs(p_is_new.pz - p_fs_total.pz):.2e}"
        )

    return result


def _fks_fsr_mapping(fks_i, fks_j, moth, orig_momenta):
    """
    Modified FKS final-state singularity mapping for MASSIVE resonances.

    Based on Section 5.2 of FKS paper, but modified to put the mother
    on-shell with mass M instead of making it lightlike.

    Original FKS condition: (q - B k_rec)² = 0        (lightlike mother)
    Modified condition:     (q - B k_rec)² = M²       (massive mother)

    The boost parameter β is found by solving a quadratic equation.
    Define:
        a = q · k_rec                                  (4-vector dot product)
        b = E_q |k̄_rec| - (q̄ · n̂) E_rec             (mixed term)
        C = (q² + m²_rec - M²) / 2                    (RHS of boost equation)

    Then: β = (-ab ± C√(a² - b² + C²)) / (b² + C²)

    Args:
        fks_i: Index of "emitter" particle (will become mother)
        fks_j: Index of "radiation" particle (will be removed)
        moth: Mother info dict with moth[0]['id'] = PDG code
        orig_momenta: List of FourMomentum for all particles [beam0, beam1, FS...]

    Returns dict with:
        'mother': FourMomentum of clustered mother (on mass shell M)
        'beams': [beam0, beam1] (unchanged for FSR)
        'spectators': dict {idx: FourMomentum} for boosted spectators
        'method': description of method used
    """
    mother_pid = abs(moth[0]["id"])
    mother_mass = _get_mother_mass(mother_pid)
    M2 = mother_mass**2

    # q = total initial state momentum
    q = orig_momenta[0] + orig_momenta[1]
    q2 = _m2(q)

    # Identify spectators (all FS particles except fks_i, fks_j)
    spectator_indices = [
        idx for idx in range(len(orig_momenta)) if idx >= 2 and idx != fks_i and idx != fks_j
    ]

    result = {
        "beams": [FourMomentum(orig_momenta[0]), FourMomentum(orig_momenta[1])],
        "spectators": {},
    }

    # Compute invariant mass of the clustering pair (before any reshuffling)
    mother_naive = orig_momenta[fks_i] + orig_momenta[fks_j]
    invariant_m2 = _m2(mother_naive)
    invariant_m = math.sqrt(max(0, invariant_m2))

    if DEBUG:
        _cluster_dbg(f"  FKS FSR: pid={mother_pid}, M={mother_mass:.4f}, m_inv={invariant_m:.4f}, spectators={spectator_indices}")

    if len(spectator_indices) == 0:
        # NO SPECTATORS: k_rec = 0, mother = q (total initial state)
        # To put mother on pole mass M, we scale beam energies.
        # For massless beams along z: q² = 4 E1 E2
        # Scale factor: s = M / sqrt(q²) = M / m_inv
        # New beams: E1' = E1 * s, E2' = E2 * s
        # This preserves rapidity (E1/E2 ratio) and puts mother on-shell.

        actual_m = math.sqrt(max(0, q2))

        # NOTE: Massless mothers (γ, g) in no-spectator case should NOT occur.
        # Clustering is stopped BEFORE creating 2→1 diagrams with massless mothers.
        # If we reach here with a massless mother, something is wrong upstream.
        if mother_mass < 1e-6:
            raise ValueError(
                f"FSR no-spectators: massless mother (M={mother_mass:.2e}) should have been "
                "prevented by clustering stop logic. This is a bug."
            )

        if actual_m > 1e-10:
            scale = mother_mass / actual_m
        else:
            scale = 1.0

        # Scale beams
        beam0_new = FourMomentum(
            [
                orig_momenta[0].E * scale,
                orig_momenta[0].px * scale,
                orig_momenta[0].py * scale,
                orig_momenta[0].pz * scale,
            ]
        )
        beam1_new = FourMomentum(
            [
                orig_momenta[1].E * scale,
                orig_momenta[1].px * scale,
                orig_momenta[1].py * scale,
                orig_momenta[1].pz * scale,
            ]
        )
        result["beams"] = [beam0_new, beam1_new]

        # Mother = scaled q (now at pole mass)
        q_new = beam0_new + beam1_new
        result["mother"] = q_new
        result["method"] = "fks_fsr_no_spectators_beam_scaled"

        if DEBUG:
            mother_m_check = math.sqrt(max(0, _m2(q_new)))
            _cluster_dbg(f"  No-spectator: scale={scale:.6f}, m_check={mother_m_check:.4f}, Δ={abs(mother_m_check - mother_mass):.2e}")

        return result

    # WITH SPECTATORS: Apply modified FKS mapping for massive mother
    # k_rec = sum of spectator momenta
    k_rec = FourMomentum()
    for idx in spectator_indices:
        k_rec += orig_momenta[idx]

    k_rec_E = k_rec.E
    k_rec_p = math.sqrt(k_rec.px**2 + k_rec.py**2 + k_rec.pz**2)
    k_rec_m2 = _m2(k_rec)  # Invariant mass squared of recoil system

    _cluster_dbg(f"  k_rec: E={k_rec_E:.4f}, |p|={k_rec_p:.4f}, m²={k_rec_m2:.4f}")

    if k_rec_p < 1e-10:
        # Spectators have zero total 3-momentum: no boost needed
        # Mother = q - k_rec (momentum conservation)
        result["mother"] = FourMomentum(
            [
                q.E - k_rec.E,
                q.px - k_rec.px,
                q.py - k_rec.py,
                q.pz - k_rec.pz,
            ]
        )
        for idx in spectator_indices:
            result["spectators"][idx] = FourMomentum(orig_momenta[idx])
        result["method"] = "fks_fsr_zero_krec_momentum"
        _cluster_dbg("  Zero k_rec momentum: no boost needed")
        return result

    # NOTE: We do NOT reject here even if naive_m² < M². With negative β,
    # we can ADD momentum from spectators to the mother and INCREASE its mass.
    # The quadratic solver will determine if a physical solution (|β| < 1) exists.

    # Boost direction: along k_rec spatial momentum
    n_rec = (k_rec.px / k_rec_p, k_rec.py / k_rec_p, k_rec.pz / k_rec_p)

    # Compute β from the massive mother condition: (q - B k_rec)² = M²
    #
    # For the general case, we solve this directly using the constraint that
    # mother² = M². The boosted spectator k̄_rec = B(β) k_rec satisfies:
    #   mother = q - k̄_rec
    #   mother² = q² - 2 q·k̄_rec + k̄_rec² = M²
    #
    # For massless spectator (k_rec² = 0), the boosted spectator is also massless.
    # The constraint becomes: q·k̄_rec = (q² - M²) / 2
    #
    # For a boost along direction n̂ with parameter β, a massless particle
    # with 4-momentum k = (E, E n̂) transforms to:
    #   k̄ = (γE(1-β), γE(1-β) n̂)  if k moves along +n̂ (k·n > 0)
    #   k̄ = (γE(1+β), γE(1+β) n̂)  if k moves along -n̂ (k·n < 0)
    #
    # In our case, n̂ = k_rec direction, so k_rec·n̂ = |k_rec| > 0.
    # Thus k̄_rec_E = γ k_E (1 - β) where γ = 1/√(1-β²).
    #
    # The constraint q·k̄_rec = (q² - M²)/2 can be solved for β.

    # Check if spectator is massless (simplifies the calculation)
    if abs(k_rec_m2) < 1e-6:  # Massless spectator
        # For massless spectator, use the simplified formula
        # k̄_rec_E = γ k_E (1 - β) for particle moving along n̂
        #
        # q·k̄_rec = q_E k̄_E - q⃗·k̄⃗
        # For k̄ moving along n̂: k̄⃗ = k̄_E n̂
        # q·k̄_rec = q_E k̄_E - k̄_E (q⃗·n̂) = k̄_E (q_E - q⃗·n̂)
        #
        # Setting this equal to (q² - M²)/2:
        # k̄_E (q_E - q⃗·n̂) = (q² - M²)/2
        # k̄_E = (q² - M²) / (2(q_E - q⃗·n̂))

        q_dot_n = q.px * n_rec[0] + q.py * n_rec[1] + q.pz * n_rec[2]
        denom_qn = q.E - q_dot_n

        if abs(denom_qn) < 1e-10:
            _cluster_dbg("  WARNING: Degenerate q_E - q·n̂ = 0")
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_degenerate"
            return result

        k_bar_E_target = (q2 - M2) / (2.0 * denom_qn)
        _cluster_dbg(f"  Massless spectator: k̄_E target = {k_bar_E_target:.4f}")

        if k_bar_E_target < 0:
            _cluster_dbg("  WARNING: k̄_E < 0 not physical")
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_negative_energy"
            return result

        # Now solve for β from: k̄_E = γ k_E (1 - β)
        # Let r = k̄_E / k_E:
        # r = (1 - β) / √(1 - β²) = √((1-β)/(1+β))
        # r² = (1-β)/(1+β)
        # β = (1 - r²) / (1 + r²)

        r = k_bar_E_target / k_rec_E
        _cluster_dbg(f"  r = k̄_E/k_E = {r:.6f}")

        if r <= 0:
            _cluster_dbg("  WARNING: r ≤ 0, no physical solution")
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_negative_r"
            return result

        r2 = r * r
        beta = (1.0 - r2) / (1.0 + r2)
        _cluster_dbg(f"  β = (1-r²)/(1+r²) = {beta:.6f}")

    else:  # Massive spectator - use general quadratic approach
        # For massive spectator, the algebra is more complex.
        # We use: q·k̄_rec = (q² + k_rec² - M²) / 2
        #
        # After boost B(β) along n̂, the spectator becomes:
        #   k̄_E = γ(E - β p_n)
        #   k̄_n = γ(p_n - β E)
        # where p_n = k⃗·n̂ and k̄⃗ = k̄_n n̂ + k⃗_perp
        #
        # This leads to a quadratic in β. We'll solve numerically if needed.

        _cluster_dbg(f"  Massive spectator (m²={k_rec_m2:.4f}), using general approach")

        # q·k̄ = q_E k̄_E - q⃗·k̄⃗
        #      = q_E γ(E - β p_n) - q_n γ(p_n - β E) - q⃗_perp·k⃗_perp
        # where q_n = q⃗·n̂, p_n = k⃗·n̂

        q_dot_n = q.px * n_rec[0] + q.py * n_rec[1] + q.pz * n_rec[2]
        p_n = k_rec.px * n_rec[0] + k_rec.py * n_rec[1] + k_rec.pz * n_rec[2]

        # k_perp components
        k_perp_x = k_rec.px - p_n * n_rec[0]
        k_perp_y = k_rec.py - p_n * n_rec[1]
        k_perp_z = k_rec.pz - p_n * n_rec[2]
        q_dot_k_perp = q.px * k_perp_x + q.py * k_perp_y + q.pz * k_perp_z

        target = (q2 + k_rec_m2 - M2) / 2.0

        # q·k̄ = γ[q_E(E - β p_n) - q_n(p_n - β E)] - q·k_perp
        #      = γ[q_E E - β q_E p_n - q_n p_n + β q_n E] - q·k_perp
        #      = γ[(q_E E - q_n p_n) + β(q_n E - q_E p_n)] - q·k_perp
        # Let A = q_E E - q_n p_n, B = q_n E - q_E p_n
        # Then: γ(A + β B) = target + q·k_perp

        A = q.E * k_rec_E - q_dot_n * p_n
        B = q_dot_n * k_rec_E - q.E * p_n
        rhs = target + q_dot_k_perp

        _cluster_dbg(f"  A={A:.4f}, B={B:.4f}, rhs={rhs:.4f}")

        # Check for numerical instability: B ≈ 0 means spectators nearly collinear with beam
        # In this limit, the quadratic becomes degenerate. Use simplified formula.
        if abs(B) < 1e-8 * max(abs(A), abs(rhs), 1.0):
            _cluster_dbg(
                "  WARNING: B ≈ 0 (spectators collinear with beam), using simplified formula"
            )
            # When B → 0: γ A = rhs, so γ = rhs/A
            # γ² = rhs²/A² = 1/(1-β²)
            # 1 - β² = A²/rhs²
            # β² = 1 - A²/rhs²
            if abs(rhs) < 1e-10:
                _cluster_dbg("  WARNING: rhs ≈ 0, degenerate case")
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_degenerate_collinear"
                return result
            gamma_sq = (rhs / A) ** 2 if abs(A) > 1e-10 else float("inf")
            if gamma_sq < 1.0:
                _cluster_dbg("  WARNING: γ² < 1, unphysical")
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_collinear_unphysical"
                return result
            beta_sq = 1.0 - 1.0 / gamma_sq
            beta = math.sqrt(beta_sq) if beta_sq > 0 else 0.0
            # Sign of β: need γ(A + β B) = rhs, with B ≈ 0 this is γ A ≈ rhs
            # If A and rhs have same sign, β can be either; choose small |β|
            _cluster_dbg(f"  Collinear limit: β = {beta:.6f}")
        else:
            # γ(A + β B) = rhs
            # (A + β B)² = rhs² (1 - β²)
            # A² + 2AB β + B² β² = rhs² - rhs² β²
            # (B² + rhs²) β² + 2AB β + (A² - rhs²) = 0

            coef_a = B * B + rhs * rhs
            coef_b = 2 * A * B
            coef_c = A * A - rhs * rhs

            disc = coef_b * coef_b - 4 * coef_a * coef_c
            if disc < 0:
                _cluster_dbg("  WARNING: Negative discriminant")
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_no_solution"
                return result

            sqrt_disc = math.sqrt(disc)
            beta1 = (-coef_b + sqrt_disc) / (2 * coef_a)
            beta2 = (-coef_b - sqrt_disc) / (2 * coef_a)
            _cluster_dbg(f"  Two solutions: β₁={beta1:.6f}, β₂={beta2:.6f}")

            # Choose valid solution with |β| < 1
            valid_betas = [b for b in [beta1, beta2] if abs(b) < 1.0]
            if not valid_betas:
                _cluster_dbg("  WARNING: No valid β with |β| < 1")
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_superluminal"
                return result

            beta = min(valid_betas, key=abs)
            _cluster_dbg(f"  Chosen β = {beta:.6f}")

    # Verify |β| < 1
    if abs(beta) >= 1.0:
        _cluster_dbg(f"  WARNING: |β| = {abs(beta):.6f} >= 1 (superluminal)")
        result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
        for idx in spectator_indices:
            result["spectators"][idx] = FourMomentum(orig_momenta[idx])
        result["method"] = "fks_fsr_superluminal"
        return result

    if DEBUG:
        _cluster_dbg(f"  [FKS FSR] β={beta:.6f}, n̂=({n_rec[0]:.4f}, {n_rec[1]:.4f}, {n_rec[2]:.4f})")

    # Apply boost B to each spectator: k̄ᵢ = B kᵢ
    k_rec_boosted = _boost_along_direction(k_rec, beta, n_rec)

    for idx in spectator_indices:
        p_orig = orig_momenta[idx]
        k_bar_i = _boost_along_direction(p_orig, beta, n_rec)
        result["spectators"][idx] = k_bar_i

    # Mother momentum by conservation: k̄_mother = q - B k_rec
    mother = FourMomentum(
        [
            q.E - k_rec_boosted.E,
            q.px - k_rec_boosted.px,
            q.py - k_rec_boosted.py,
            q.pz - k_rec_boosted.pz,
        ]
    )
    result["mother"] = mother
    result["method"] = "fks_fsr_massive_onshell"

    if DEBUG:
        mother_m2 = _m2(mother)
        mother_m = math.sqrt(max(0, mother_m2))
        _cluster_dbg(f"  Mother: m={mother_m:.4f} (target={mother_mass:.4f}), Δm²={mother_m2 - M2:.2e}")
        _cluster_dbg(f"  k_rec boost: E {k_rec.E:.4f} -> {k_rec_boosted.E:.4f}")
        p_final = FourMomentum(mother)
        for idx in spectator_indices:
            p_final += result["spectators"][idx]
        _cluster_dbg(
            f"  Conservation: dE={abs(q.E - p_final.E):.2e}, dpx={abs(q.px - p_final.px):.2e}, "
            f"dpy={abs(q.py - p_final.py):.2e}, dpz={abs(q.pz - p_final.pz):.2e}"
        )

    return result


def fxfx_merge_particles_kinematics(event, i, j, moth):
    """
    FxFx kinematic reshuffling for clustering particles i and j.

    Implements FKS mapping (Frixione-Kunszt-Signer) for both ISR and FSR.

    For ISR: Section 5.1 of FKS paper
        - Remove radiation, adjust initial state x_±
        - Boost FS to match new CM frame

    For FSR: Section 5.2 of FKS paper
        - Boost spectators with β from Eq. 5.26
        - Mother = q - B·k_rec (Eq. 5.27)

    Args:
        event: lhe_parser.Event to modify in-place
        i, j: Particle indices to merge
        moth: Mother info list with moth[0] = {'number': ..., 'id': PDG}
    """
    # Determine which particle becomes the mother
    if i == moth[0].get("number") - 1:
        fks_i = i
        fks_j = j
    elif j == moth[0].get("number") - 1:
        fks_i = j
        fks_j = i
    else:
        fks_i = min(i, j)
        fks_j = max(i, j)

    to_remove = fks_j

    # Save original momenta
    orig_momenta = [FourMomentum(p) for p in event]

    cluster_type = "ISR" if fks_i <= 1 else "FSR"
    _cluster_dbg(
        f"FxFx merge: {cluster_type} clustering [{fks_i}]+[{fks_j}] -> mother (pid={moth[0]['id']})"
    )

    if fks_i <= 1:
        # ISR: FKS Section 5.1 mapping
        isr_result = _fks_isr_mapping(fks_i, fks_j, moth, orig_momenta)

        # Apply results to event
        event[0].set_momentum(isr_result["beams"][0])
        event[1].set_momentum(isr_result["beams"][1])

        for idx, p in isr_result["final_state"].items():
            event[idx].set_momentum(p)

        event[fks_i].pid = moth[0]["id"]

    else:
        # FSR: FKS Section 5.2 mapping
        fks_result = _fks_fsr_mapping(fks_i, fks_j, moth, orig_momenta)

        # Apply results to event
        event[0].set_momentum(fks_result["beams"][0])
        event[1].set_momentum(fks_result["beams"][1])
        event[fks_i].set_momentum(fks_result["mother"])
        event[fks_i].pid = moth[0]["id"]

        # Set mother mass (pole mass from PDG lookup)
        event[fks_i].mass = _get_mother_mass(abs(moth[0]["id"]))

        # Apply spectator momenta
        for idx, p in fks_result["spectators"].items():
            event[idx].set_momentum(p)

    # Remove radiation particle
    event.pop(to_remove)

    # Debug: before/after comparison and momentum conservation verification
    if DEBUG:
        _cluster_dbg("-" * 70)
        _cluster_dbg("RESHUFFLING SUMMARY")
        _cluster_dbg("-" * 70)
        _cluster_dbg("BEFORE reshuffling (original momenta):")
        p_in_sum_orig = FourMomentum()
        p_out_sum_orig = FourMomentum()
        for idx, p in enumerate(orig_momenta):
            m = math.sqrt(max(0, _m2(p)))
            is_in = idx <= 1
            _cluster_dbg(
                f"  [{idx}] {'IS' if is_in else 'FS'}: E={p.E:10.4f}, px={p.px:10.4f}, py={p.py:10.4f}, pz={p.pz:10.4f}, m={m:8.4f}"
            )
            if is_in:
                p_in_sum_orig += p
            else:
                p_out_sum_orig += p
        _cluster_dbg(
            f"  SUM IS: E={p_in_sum_orig.E:10.4f}, px={p_in_sum_orig.px:10.4f}, py={p_in_sum_orig.py:10.4f}, pz={p_in_sum_orig.pz:10.4f}"
        )
        _cluster_dbg(
            f"  SUM FS: E={p_out_sum_orig.E:10.4f}, px={p_out_sum_orig.px:10.4f}, py={p_out_sum_orig.py:10.4f}, pz={p_out_sum_orig.pz:10.4f}"
        )
        _cluster_dbg("AFTER reshuffling (new momenta):")
        p_in_sum_new = FourMomentum()
        p_out_sum_new = FourMomentum()
        for idx, p in enumerate(event):
            m = math.sqrt(max(0, _m2(p)))
            is_in = idx <= 1
            pdg = getattr(p, "pid", "?")
            _cluster_dbg(
                f"  [{idx}] {'IS' if is_in else 'FS'} pdg={pdg:4}: E={p.E:10.4f}, px={p.px:10.4f}, py={p.py:10.4f}, pz={p.pz:10.4f}, m={m:8.4f}"
            )
            if is_in:
                p_in_sum_new += p
            else:
                p_out_sum_new += p
        _cluster_dbg(
            f"  SUM IS: E={p_in_sum_new.E:10.4f}, px={p_in_sum_new.px:10.4f}, py={p_in_sum_new.py:10.4f}, pz={p_in_sum_new.pz:10.4f}"
        )
        _cluster_dbg(
            f"  SUM FS: E={p_out_sum_new.E:10.4f}, px={p_out_sum_new.px:10.4f}, py={p_out_sum_new.py:10.4f}, pz={p_out_sum_new.pz:10.4f}"
        )
        delta_E = abs(p_in_sum_new.E - p_out_sum_new.E)
        delta_px = abs(p_in_sum_new.px - p_out_sum_new.px)
        delta_py = abs(p_in_sum_new.py - p_out_sum_new.py)
        delta_pz = abs(p_in_sum_new.pz - p_out_sum_new.pz)
        _cluster_dbg(
            f"  CONSERVATION: |ΔE|={delta_E:.2e}, |Δpx|={delta_px:.2e}, |Δpy|={delta_py:.2e}, |Δpz|={delta_pz:.2e}"
        )
        if cluster_type == "FSR":
            mother_m = math.sqrt(max(0, _m2(event[fks_i])))
            target_m = _get_mother_mass(abs(moth[0]["id"]))
            _cluster_dbg(
                f"  MOTHER MASS: computed={mother_m:.4f}, target={target_m:.4f}, Δ={abs(mother_m - target_m):.2e} GeV"
            )
        _cluster_dbg("=" * 70)


# =============================================================================
# Density Matrix Constants and Configuration
# =============================================================================

# Spin dimensions for resonances (use abs(pdg) for lookup)
# Fermions (spin-1/2): 2 states
# Vector bosons (spin-1): 3 states
SPIN_DIMS = {
    6: 2,  # t / t̄
    23: 3,  # Z
    24: 3,  # W+ / W-
}

# PDG codes that are treated as resonances for density matrix formalism
# Include both particle and antiparticle codes for membership testing
RESONANCE_PDGS = frozenset([6, -6, 23, 24, -24])


def _total_dimension(dims: List[int]) -> int:
    """Compute total Hilbert space dimension as product of individual dims."""
    return reduce(mul, dims, 1)


def _fortran_to_python_hel_permutation(res_dims):
    """Build permutation to reorder Fortran delta indices to Python B indices.

    The Fortran density_matrix_wrapper enumerates helicity combinations using
    little-endian mixed-radix (first resonance varies fastest) with ascending
    helicity values (0->-1, 1->+1 for fermions; 0->-1, 1->0, 2->+1 for vectors).

    The Python _build_allowed_helicities uses itertools.product (big-endian,
    first resonance varies slowest) with descending values ([+1,-1] for fermions;
    [+1,0,-1] for vectors).

    Returns perm such that delta_python[i] = delta_fortran[perm[i]],
    i.e. perm[i] is the Fortran index corresponding to Python index i.
    """
    total_dim = 1
    for d in res_dims:
        total_dim *= d

    perm = [0] * total_dim
    for ih_fortran in range(total_dim):
        # Decode Fortran index: little-endian, ascending values
        k = ih_fortran
        hels = []
        for d in res_dims:
            idx = k % d
            k //= d
            if d == 2:
                hels.append(2 * idx - 1)  # 0->-1, 1->+1
            elif d == 3:
                hels.append(idx - 1)  # 0->-1, 1->0, 2->+1
            else:
                hels.append(idx - (d - 1) // 2)

        # Encode as Python index: big-endian, descending values
        ih_python = 0
        for n, d in enumerate(res_dims):
            h = hels[n]
            if d == 2:
                local_idx = 0 if h == +1 else 1
            elif d == 3:
                local_idx = {+1: 0, 0: 1, -1: 2}[h]
            else:
                local_idx = (d - 1) // 2 - h
            ih_python = ih_python * d + local_idx

        perm[ih_python] = ih_fortran

    return perm


# =============================================================================
# DensityHelper Class - Adapter for Valentin's DensityInterface
# =============================================================================


class DensityHelper:
    """Lightweight adapter to compute density matrices using Valentin's infrastructure.

    Instead of instantiating a full DensityInterface (which requires heavy initialization),
    this class:
    1. Copies needed attributes from an existing ReweightInterface
    2. Uses density-specific id_to_path and f2pylib (from standalone exports)
    3. Provides the methods that DensityInterface.calculate_matrix_element needs

    Usage:
        helper = DensityHelper(reweight_interface, density_prod_id_to_path, density_prod_f2pylib)
        matrix = helper.compute_density_matrix(event, target_pdgs=[6, -6], res_dims=[2, 2])
    """

    def __init__(self, reweight_interface, id_to_path, f2pylib):
        """Initialize the density helper.

        Args:
            reweight_interface: The existing ReweightInterface instance (for shared config)
            id_to_path: Density-specific mapping (e.g., density_prod_id_to_path)
            f2pylib: Density-specific F2PY modules (e.g., density_prod_f2pylib)
        """
        # Copy needed attributes from ReweightInterface
        self.keep_ordering = getattr(reweight_interface, "keep_ordering", True)
        self.helicity_reweighting = getattr(reweight_interface, "helicity_reweighting", False)
        self.options = getattr(
            reweight_interface, "options", {"identical_particle_in_prod_and_decay": "average"}
        )
        self.me_dir = getattr(reweight_interface, "me_dir", os.getcwd())

        # Use provided density-specific paths
        self.id_to_path = id_to_path
        self.f2pylib = f2pylib

        # Density-specific config (defaults - no boost/rotation for EW Sudakov)
        self.momenta_boost = ([0], "", [])  # No boost
        self.helicity_direction = ([0], "", [])  # No rotation
        self.axis_referential = [0]  # Default axis
        self.symmetrise_initial_state = False  # No symmetrization

        # These get set per-call in compute_density_matrix
        self.particle_in_density_matrix = None
        self.allowed_helicities = None
        self.number_changing_helicities = None
        self.number_combinations = None
        self.spins = None
        self.flag_particle_in_density_matrix = True
        self._initialized_modules = set()  # Track which F2PY modules have been initialized

    # =========================================================================
    # Methods needed by calculate_matrix_element
    # =========================================================================

    @staticmethod
    def invert_momenta(p):
        """Transpose momenta from Python to Fortran layout.

        Input from get_momenta() is ALWAYS (n_particles, 4) format:
        - Each row is one particle: (E, px, py, pz)

        F2PY expects P(0:3, NEXTERNAL) = (4, n_particles) in numpy:
        - Each row is one component: [E_1, E_2, ...], [px_1, px_2, ...], etc.

        So we ALWAYS transpose and ensure Fortran-contiguous memory order.
        """
        p_arr = np.asarray(p, dtype=np.float64)

        # get_momenta() always returns (n_particles, 4), so always transpose
        # The transpose gives us (4, n_particles) which is what Fortran expects
        return np.asfortranarray(p_arr.T, dtype=np.float64)

    def method_boost_event(self, event, all_p, orig_order, hypp_id, boost_corrected):
        """Boost event to CM frame. For EW Sudakov with boost_corrected=[-1], returns unchanged."""
        if boost_corrected == [-1]:
            return all_p
        return all_p

    def calculate_angles_rotation(self, position_particles, all_p, module):
        """Compute rotation angles. For EW Sudakov with position_particles=[-1], returns zeros."""
        if position_particles == [-1]:
            return [0] * len(all_p), [0] * len(all_p)
        return [0] * len(all_p), [0] * len(all_p)

    def rotation_density(self, module, all_p, phi, theta):
        """Apply rotation to momenta. For EW Sudakov, just converts to Fortran format."""
        for i in range(len(all_p)):
            all_p[i] = self.invert_momenta(all_p[i])
        return all_p

    # =========================================================================
    # Main density matrix computation
    # =========================================================================

    def compute_density_matrix(self, event, target_pdgs, res_dims):
        """Compute density matrix for given event and target particles.

        Args:
            event: LHE event object
            target_pdgs: List of PDG codes for particles in density matrix
            res_dims: Spin dimensions for each particle (2=fermion, 3=vector)

        Returns:
            numpy array: Full density matrix or None if computation fails
        """
        _dbg(
            f"[DensityHelper.compute_density_matrix] ENTER target_pdgs={target_pdgs}, res_dims={res_dims}"
        )
        import madgraph.various.Density_functions as dens

        # Configure for this specific calculation
        self._configure_for_pdgs(target_pdgs, res_dims)
        _dbg(
            f"[DensityHelper] Configured: n_changing={self.number_changing_helicities}, n_comb={self.number_combinations}"
        )

        # Get event tag and find matching module
        tag, order = event.get_tag_and_order()
        if self.keep_ordering:
            # Use original IS and FS ordering (not sorted) to match id_to_path keys
            tag = (tuple(order[0]), tuple(order[1]))

        if tag not in self.id_to_path:
            _dbg(f"[DensityHelper] Tag {tag} not in id_to_path")
            _dbg(f"[DensityHelper] Available tags: {list(self.id_to_path.keys())[:5]}...")
            return None

        orig_order, Pdir, hel_dict = self.id_to_path[tag]
        base = os.path.basename(os.path.dirname(Pdir))
        moduletag = (base, 2)
        _dbg(f"[DensityHelper] Tag found, moduletag={moduletag}")

        if moduletag not in self.f2pylib:
            _dbg(f"[DensityHelper] Module {moduletag} not in f2pylib")
            _dbg(f"[DensityHelper] Available modules: {list(self.f2pylib.keys())[:5]}...")
            return None

        module = self.f2pylib[moduletag]
        _dbg(f"[DensityHelper] Got module: {module}")

        # Get momenta
        # Debug: show event particles before extracting momenta
        _dbg(f"[DensityHelper] Event has {len(event)} particles:")
        for i, p in enumerate(event):
            _dbg(
                f"[DensityHelper]   [{i}] pid={p.pid}, status={p.status}, E={p.E:.4f}, px={p.px:.4f}, py={p.py:.4f}, pz={p.pz:.4f}"
            )

        if self.keep_ordering:
            all_p = [event.get_momenta(orig_order)]
        else:
            all_p = event.get_all_momenta(orig_order)
        _dbg(f"[DensityHelper] get_momenta returned: {all_p}")

        # Apply boost (skipped for EW Sudakov)
        boost_corrected = [-1]
        all_p = self.method_boost_event(event, all_p, orig_order, 0, boost_corrected)

        # Apply rotation (skipped for EW Sudakov)
        refChoice_corrected = [-1]
        phi, theta = self.calculate_angles_rotation(refChoice_corrected, all_p, module)
        all_p = self.rotation_density(module, all_p, phi, theta)

        # Get particle positions for density matrix
        pos_corrected = self._find_particle_positions(orig_order, target_pdgs)
        _dbg(
            f"[DensityHelper] Particle positions for {target_pdgs}: {pos_corrected} (orig_order={orig_order})"
        )

        # Get module prefix - must find the correct one for this specific tag
        # module.get_pdg_order() returns (all_pdgs, all_pids)
        # module.get_prefix() returns all prefixes
        # We need to find which index matches our tag
        all_pdgs_raw, all_pids = module.get_pdg_order()
        all_pdgs = [[pdg for pdg in pdgs if pdg != 0] for pdgs in all_pdgs_raw]
        PREFIX = module.get_prefix()
        all_prefixes = [bytes(j).decode(errors="ignore").strip().lower() for j in PREFIX]

        # Reconstruct the lookup tag from orig_order
        # For decay: nincoming=1, for production: nincoming=2
        nincoming = len(tag[0])
        lookup_incoming = sorted(orig_order[0])
        lookup_outgoing = sorted(orig_order[1])
        lookup_tag = (tuple(lookup_incoming), tuple(lookup_outgoing))

        # Find matching process index
        proc_idx = None
        for i, pdg in enumerate(all_pdgs):
            incoming = pdg[:nincoming]
            outgoing = pdg[nincoming:]
            incoming_sorted = sorted(incoming)
            outgoing_sorted = sorted(outgoing)
            cmp_tag = (tuple(incoming_sorted), tuple(outgoing_sorted))
            if cmp_tag == lookup_tag:
                proc_idx = i
                break

        if proc_idx is None:
            _dbg(f"[DensityHelper] Could not find matching process for tag {lookup_tag}")
            _dbg(
                f"[DensityHelper] Available: {[(tuple(sorted(p[:nincoming])), tuple(sorted(p[nincoming:]))) for p in all_pdgs[:5]]}..."
            )
            return None

        prefix = all_prefixes[proc_idx]
        _dbg(f"[DensityHelper] Found process index {proc_idx}, prefix={prefix}")

        # Initialize param card (once per module) - derive from Pdir
        # Pdir is: /path/to/rw_me_density_prod/SubProcesses
        # We need: /path/to/rw_me_density_prod/Cards/param_card.dat
        module_id = id(module)
        if module_id not in self._initialized_modules:
            density_me_dir = os.path.dirname(Pdir)  # Goes from SubProcesses to rw_me_density_prod
            Card_dir = os.path.join(density_me_dir, "Cards", "param_card.dat")
            _dbg(f"[DensityHelper] Using Card_dir={Card_dir}")
            if os.path.exists(Card_dir):
                Initialise = getattr(module, "initialise", None)
                _dbg(f"[DensityHelper] Initialise function exists: {Initialise is not None}")
                if Initialise:
                    Initialise(Card_dir)
                InitModel = getattr(module, prefix + "initialisemodel", None)
                _dbg(
                    f"[DensityHelper] {prefix}initialisemodel function exists: {InitModel is not None}"
                )
                if InitModel:
                    InitModel(Card_dir)
            else:
                _dbg("[DensityHelper] WARNING: Card_dir does not exist!")
            self._initialized_modules.add(module_id)

        # Call GET_DENSITY via F2PY
        func_name = prefix + "get_density"
        get_density_func = getattr(module, func_name, None)
        _dbg(f"[DensityHelper] Looking for function: {func_name} in module {module}")
        if get_density_func is None:
            _dbg(f"[DensityHelper] Function {func_name} not found in module")
            _dbg(
                f"[DensityHelper] Available functions: {[f for f in dir(module) if 'get_density' in f.lower()]}"
            )
            return None

        me_value = None
        _dbg(
            f"[DensityHelper] About to call get_density_func, n_changing={self.number_changing_helicities}, n_comb={self.number_combinations}, aqcd={event.aqcd}"
        )

        # Ensure arrays are proper numpy types for safe f2py calls
        # (defensive - source functions should already return correct types)
        pos_arr = np.asarray(pos_corrected, dtype=np.int32)
        hel_arr = np.asarray(self.allowed_helicities, dtype=np.int32)
        _dbg(f"[DensityHelper] pos_arr={pos_arr}, hel_arr={hel_arr[:min(10, len(hel_arr))]}")

        for i in range(len(all_p)):
            # NOTE: all_p[i] is ALREADY in Fortran layout (4, n_particles) from rotation_density
            # Do NOT call invert_momenta again - that would double-transpose back to Python layout!
            pinv = np.asfortranarray(all_p[i], dtype=np.float64)
            _dbg(
                f"[DensityHelper] Calling Fortran get_density... pinv.shape={pinv.shape}, pinv.dtype={pinv.dtype}"
            )
            _dbg(
                f"[DensityHelper] Momenta (E,px,py,pz per particle): {pinv.T if pinv.shape[0] == 4 else pinv}"
            )
            # Validate momenta for debugging NaN issues
            if np.any(np.isnan(pinv)) or np.any(np.isinf(pinv)):
                _dbg("[DensityHelper] WARNING: NaN/Inf in input momenta!")
            # Check energy positivity
            E_values = pinv[0, :] if pinv.shape[0] == 4 else pinv[:, 0]
            _dbg(f"[DensityHelper] Energies: {E_values}")
            # Check momentum conservation
            if pinv.shape[0] == 4:  # Fortran layout (4, n_particles)
                p_sum = np.sum(pinv[:, 1:], axis=1) - pinv[:, 0]  # FS - IS
                _dbg(
                    f"[DensityHelper] p_sum (FS-IS): E={p_sum[0]:.4f}, px={p_sum[1]:.4f}, py={p_sum[2]:.4f}, pz={p_sum[3]:.4f}"
                )
            production_matrix = get_density_func(
                pinv,
                pos_arr,
                self.number_changing_helicities,
                hel_arr,
                self.number_combinations,
                event.aqcd,
            )
            _dbg(
                f"[DensityHelper] Fortran returned, production_matrix type={type(production_matrix)}"
            )

            # Convert to full matrix using Valentin's utility
            # Use integer division (//) to ensure len_user_input is an int
            len_tri = self.number_combinations * (self.number_combinations + 1) // 2
            _dbg(
                f"[DensityHelper] Calling DensityMatrixObservables with len_tri={len_tri}, production_matrix shape={production_matrix.shape if hasattr(production_matrix, 'shape') else len(production_matrix)}"
            )
            # Debug: show raw Fortran output (first few elements)
            if hasattr(production_matrix, "__len__") and len(production_matrix) >= len_tri:
                _dbg(
                    f"[DensityHelper] Raw Fortran output (first {len_tri}): {production_matrix[:len_tri]}"
                )
            rho_instance = dens.DensityMatrixObservables(production_matrix, len_tri)
            # Use square_matrix() to convert triangular storage to full 2D matrix
            me_value = rho_instance.square_matrix()
            _dbg(f"[DensityHelper] square_matrix shape: {np.array(me_value).shape}")

        return np.array(me_value) if me_value is not None else None

    def _configure_for_pdgs(self, pdg_codes, res_dims):
        """Configure density attributes for given PDGs."""
        self.particle_in_density_matrix = (list(pdg_codes), "", [])
        self.number_changing_helicities = len(pdg_codes)
        self.spins = list(res_dims)

        # Compute number of helicity combinations
        self.number_combinations = 1
        for spin in self.spins:
            self.number_combinations *= spin

        # Build allowed_helicities array
        self.allowed_helicities = self._build_allowed_helicities(res_dims)

    def _build_allowed_helicities(self, res_dims):
        """Build flat array of allowed helicity combinations."""
        # Generate helicity values for each particle
        hel_lists = []
        for dim in res_dims:
            if dim == 2:  # Fermion: +1, -1
                hel_lists.append([+1, -1])
            elif dim == 3:  # Vector: +1, 0, -1
                hel_lists.append([+1, 0, -1])
            else:
                hel_lists.append(list(range(-(dim - 1) // 2, (dim - 1) // 2 + 1)))

        # Generate all combinations and flatten
        combos = list(product(*hel_lists))
        flat = []
        for combo in combos:
            flat.extend(combo)

        # Return numpy int32 array for safe f2py calls
        return np.asarray(flat, dtype=np.int32)

    def _find_particle_positions(self, orig_order, target_pdgs):
        """Find Fortran-indexed positions of target particles.

        Searches both initial and final state particles.
        For decay events, the resonance is in the initial state.
        For production events, resonances are in the final state.

        Returns numpy int32 array for safe f2py calls.
        """
        positions = []
        n_initial = len(orig_order[0])

        for pid in target_pdgs:
            found = False
            # First search initial state (for decay events)
            for idx, initial_pid in enumerate(orig_order[0]):
                if initial_pid == pid:
                    pos = idx + 1  # 1-based for Fortran
                    if pos not in positions:
                        positions.append(pos)
                        found = True
                        break
            if found:
                continue
            # Then search final state (for production events)
            for idx, final_pid in enumerate(orig_order[1]):
                if final_pid == pid:
                    pos = n_initial + idx + 1  # 1-based for Fortran
                    if pos not in positions:
                        positions.append(pos)
                        break

        # Return numpy int32 array for safe f2py calls
        return (
            np.asarray(positions, dtype=np.int32) if positions else np.array([-1], dtype=np.int32)
        )


# =============================================================================
# DensityMatrix Class
# =============================================================================


class DensityMatrix:
    """
    Spin density matrix for resonance system.

    Represents ρ[λ₁...λN; λ₁'...λN'] where λᵢ are helicity indices
    for each resonance in the system.

    The matrix is stored as a 2D numpy array of shape (D, D) where
    D = ∏ᵢ dᵢ is the total Hilbert space dimension.

    Attributes:
        pdgs: List of PDG codes for resonances in this system
        dims: List of spin dimensions [d₁, d₂, ...] for each resonance
        total_dim: Total dimension D = ∏ dᵢ
        matrix: Complex D×D numpy array storing the density matrix
    """

    def __init__(self, resonance_pdgs: List[int]):
        """
        Initialize density matrix for given resonance system.

        Args:
            resonance_pdgs: List of PDG codes for resonances
                           (e.g., [6, -6] for tt̄, [24, -24] for W+W-)
        """
        self.pdgs = list(resonance_pdgs)
        self.dims = [SPIN_DIMS[abs(pdg)] for pdg in self.pdgs]
        self.total_dim = _total_dimension(self.dims)
        self.matrix = np.zeros((self.total_dim, self.total_dim), dtype=complex)

    @classmethod
    def from_full_matrix(cls, matrix: np.ndarray, resonance_pdgs: List[int]) -> "DensityMatrix":
        """
        Create density matrix from full D×D matrix.

        Args:
            matrix: Full D×D complex numpy array
            resonance_pdgs: List of PDG codes for resonances

        Returns:
            DensityMatrix object
        """
        dm = cls(resonance_pdgs)
        if matrix.shape != (dm.total_dim, dm.total_dim):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"expected ({dm.total_dim}, {dm.total_dim})"
            )
        dm.matrix = matrix.astype(complex)
        return dm

    def copy(self) -> "DensityMatrix":
        """Create a deep copy of this density matrix."""
        dm = DensityMatrix(self.pdgs)
        dm.matrix = self.matrix.copy()
        return dm

    def apply_sudakov(self, delta: np.ndarray) -> "DensityMatrix":
        """
        Apply linearized (NLO) Sudakov correction to the density matrix.

        The density matrix encodes amplitude interference: B[h,h'] = M_h × M*_h'
        At the amplitude level, EW Sudakov correction gives: M_h → M_h × (1 + Δ_h)

        The density matrix transforms as:
            B[h,h'] → B[h,h'] × (1 + Δ_h) × (1 + Δ*_h')
                    = B[h,h'] × (1 + Δ_h + Δ*_h' + O(α²))

        For NLO consistency, we drop O(α²) terms:
            B^EWSL[h,h'] = B[h,h'] × (1 + Δ_h + Δ*_h')

        Args:
            delta: Array of shape (total_dim,) with complex Δ_h values

        Returns:
            New DensityMatrix with NLO Sudakov correction applied
        """
        if len(delta) != self.total_dim:
            raise ValueError(
                f"Delta length {len(delta)} doesn't match "
                f"density matrix dimension {self.total_dim}"
            )

        result = self.copy()

        for h in range(self.total_dim):
            for hp in range(self.total_dim):
                # Linearized NLO update: (1 + Δ_h + Δ*_h')
                factor = 1.0 + delta[h] + np.conj(delta[hp])
                result.matrix[h, hp] = self.matrix[h, hp] * factor

        return result

    def trace(self) -> float:
        """Compute trace of the density matrix."""
        return np.trace(self.matrix).real

    def contract(self, other: "DensityMatrix") -> float:
        """
        Compute trace contraction with another density matrix.

        Computes Tr[ρ₁ · ρ₂] = Σ_{h,h'} ρ₁[h,h'] × ρ₂[h',h]

        Args:
            other: Another DensityMatrix (typically decay matrix C)

        Returns:
            Tr[self · other] (real part)
        """
        if self.total_dim != other.total_dim:
            raise ValueError(f"Dimension mismatch: {self.total_dim} vs {other.total_dim}")

        return np.trace(self.matrix @ other.matrix).real

    def __repr__(self) -> str:
        return (
            f"DensityMatrix(pdgs={self.pdgs}, dims={self.dims}, "
            f"total_dim={self.total_dim}, trace={self.trace():.6f})"
        )

    def __str__(self) -> str:
        return f"DensityMatrix({self.total_dim}×{self.total_dim}) for {self.pdgs}"


# =============================================================================
# Tensor Product Operations
# =============================================================================


def _tensor_product(dm1: DensityMatrix, dm2: DensityMatrix) -> DensityMatrix:
    """
    Compute tensor product of two density matrices.

    Args:
        dm1: First density matrix
        dm2: Second density matrix

    Returns:
        Tensor product ρ₁ ⊗ ρ₂
    """
    combined_pdgs = dm1.pdgs + dm2.pdgs
    result = DensityMatrix(combined_pdgs)
    result.matrix = np.kron(dm1.matrix, dm2.matrix)
    return result


def build_decay_tensor_product(decay_matrices: List[DensityMatrix]) -> DensityMatrix:
    """
    Build tensor product of multiple decay density matrices.

    C = C₁ ⊗ C₂ ⊗ ... ⊗ Cₙ

    Args:
        decay_matrices: List of individual decay density matrices

    Returns:
        Combined decay density matrix
    """
    if not decay_matrices:
        raise ValueError("Need at least one decay matrix")

    result = decay_matrices[0].copy()
    for dm in decay_matrices[1:]:
        result = _tensor_product(result, dm)

    return result


# =============================================================================
# ResonanceIdentifier Class
# =============================================================================


class ResonanceIdentifier:
    """
    Utility class for identifying and tracking resonances in LHE events.
    """

    @staticmethod
    def find_resonances(event) -> List[Tuple[int, int]]:
        """
        Find resonance legs in an LHE event.

        Args:
            event: LHE event object with particle list

        Returns:
            List of (leg_index, pdg) for each resonance found
            Indices are 0-based into the event particle list
        """
        resonances = []
        for i, particle in enumerate(event):
            pdg = getattr(particle, "pdg", None) or getattr(particle, "pid", None)
            status = getattr(particle, "status", 1)

            if pdg is None:
                continue

            # Only consider final-state or intermediate resonances
            if abs(pdg) in RESONANCE_PDGS and status in (1, 2):
                resonances.append((i, int(pdg)))

        return resonances

    @staticmethod
    def _mother_matches(mother, resonance_idx: int) -> bool:
        """
        Check if a mother pointer matches a resonance index.

        Handles both cases:
        - mother is a Particle object (with event_id attribute) - LHE parser resolved pointers
        - mother is a number (float from parsing) - raw LHE 1-based numbering
        """
        if mother is None:
            return False
        if hasattr(mother, "event_id"):
            # Mother is a particle object - compare event_id (0-based)
            return mother.event_id == resonance_idx
        elif isinstance(mother, (int, float)):
            # Mother is a raw LHE number (1-based)
            return int(mother) == resonance_idx + 1
        return False

    @staticmethod
    def _identify_decay_products(event, resonance_idx: int) -> List[int]:
        """
        Find decay products of a resonance using mother pointers.

        Args:
            event: Full LHE event
            resonance_idx: Index of resonance particle in event

        Returns:
            List of indices of decay product particles
        """
        decay_products = []

        for i, particle in enumerate(event):
            m1 = getattr(particle, "mother1", None)
            m2 = getattr(particle, "mother2", None)

            if ResonanceIdentifier._mother_matches(
                m1, resonance_idx
            ) or ResonanceIdentifier._mother_matches(m2, resonance_idx):
                decay_products.append(i)

        return decay_products

    @staticmethod
    def find_decayed_resonances(event) -> List[Tuple[int, int, List[int]]]:
        """
        Find resonances and their decay products in an event.

        Handles both:
        - Case 1: Hard ME decays (resonance with status=2, daughters in event)
        - Case 2: MadSpin decays (identified via mother pointers)

        Args:
            event: Full LHE event

        Returns:
            List of (resonance_idx, pdg, [decay_product_indices])
        """
        results = []

        for i, particle in enumerate(event):
            pdg = getattr(particle, "pdg", None) or getattr(particle, "pid", None)
            status = getattr(particle, "status", 1)

            if pdg is None:
                continue

            if abs(pdg) in RESONANCE_PDGS:
                decay_products = ResonanceIdentifier._identify_decay_products(event, i)

                if decay_products:
                    results.append((i, int(pdg), decay_products))
                elif status == 2:
                    _dbg(
                        f"Intermediate resonance {pdg} at index {i} has no identified decay products"
                    )

        return results


# =============================================================================
# Weight Computation
# =============================================================================


def compute_density_weight(
    B: DensityMatrix, C: Optional[DensityMatrix], delta: np.ndarray
) -> float:
    """
    Compute the EW Sudakov reweighting factor using density matrices.

    Implements:
        w = Tr[B^EWSL · C] / Tr[B · C]

    where B^EWSL is the NLO Sudakov-corrected Born density matrix:
        B^EWSL[h,h'] = B[h,h'] × (1 + Δ_h + Δ*_h')

    Args:
        B: Born-like production density matrix
        C: Decay density matrix (None for spin-summed limit, uses C=Identity)
        delta: Complex Sudakov corrections Δ_h per helicity

    Returns:
        Event weight w (NLO accurate)
    """
    _dbg("compute_density_weight: start")
    _dbg(f"  B trace={B.trace():.6e}")
    if C is not None:
        _dbg(f"  C trace={C.trace():.6e}")
    _dbg(f"  delta={delta}")
    B_ewsl = B.apply_sudakov(delta)

    if C is not None:
        numerator = B_ewsl.contract(C)
        denominator = B.contract(C)
    else:
        numerator = B_ewsl.trace()
        denominator = B.trace()

    if abs(denominator) < 1e-15:
        _dbg("Denominator Tr[B·C] ≈ 0, returning weight=1")
        return 1.0

    _dbg(f"  numerator={numerator}")
    _dbg(f"  denominator={denominator}")
    return numerator / denominator


# =============================================================================
# Logging and Diagnostics
# =============================================================================


class DensitySudakovLogger:
    """
    Structured logging for density-matrix Sudakov reweighting.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.events_processed = 0

    def log_event(
        self,
        mode: str,
        resonances: List[Tuple[int, int]],
        dimension: int,
        weight: float,
        clustering_info: Optional[dict] = None,
    ):
        """Log information for a single event."""
        self.events_processed += 1

        if not self.verbose:
            return

        res_str = ", ".join(f"{pdg}@{idx}" for idx, pdg in resonances)
        msg = (
            f"Event {self.events_processed}: mode={mode}, "
            f"resonances=[{res_str}], D={dimension}, w={weight:.6f}"
        )

        if clustering_info:
            msg += f", clustering={clustering_info.get('type', 'unknown')}"

        _dbg(msg)


# =============================================================================
# FxFxEWSudakovMixin Class
# =============================================================================


class FxFxEWSudakovMixin:
    """
    Mixin class providing FxFx clustering and EW Sudakov reweighting methods.

    This mixin is designed to be used with ReweightInterface. It expects the
    following attributes to be present on self:
    - model: The MadGraph model object
    - banner: The event banner
    - lhe_input: The LHE event file
    - me_dir: The MadEvent directory path
    - keep_ordering: Boolean for momentum ordering
    - options: Dictionary of reweight options
    - Various density-related attributes (density_prod_id_to_path, etc.)
    """

    # =========================================================================
    # Pole Mass Initialization from param_card
    # =========================================================================

    def _init_pole_masses_from_banner(self):
        """Read pole masses from param_card in the banner, updating module globals.

        Called once per run. Falls back to hardcoded defaults if banner is
        unavailable or the MASS block cannot be read.
        """
        global MW_POLE, MZ_POLE, MT_POLE, MH_POLE, _POLE_MASSES_INITIALIZED
        if _POLE_MASSES_INITIALIZED:
            _dbg("[INIT] _init_pole_masses_from_banner: already initialized, skipping")
            return
        _POLE_MASSES_INITIALIZED = True

        banner = getattr(self, 'banner', None)
        if banner is None:
            _dbg("[INIT] _init_pole_masses_from_banner: no banner attached, using compile-time defaults for MW/MZ/MT/MH")
            return
        mass_map = {24: 'MW_POLE', 23: 'MZ_POLE', 6: 'MT_POLE', 25: 'MH_POLE'}
        for pdg, name in mass_map.items():
            try:
                val = float(banner.get_detail('param_card', 'mass', pdg).value)
                globals()[name] = val
                _dbg(f"[INIT] {name} = {val} (from param_card, PDG {pdg})")
            except Exception:
                _dbg(f"[INIT] {name} = {globals()[name]} (default, param_card read failed)")

    # =========================================================================
    # FxFx Clustering Methods
    # =========================================================================

    # -------------------------------------------------------------------------
    # MadSpin Forced Clustering (applied BEFORE FxFx clustering)
    # -------------------------------------------------------------------------

    def _build_madspin_decay_tree(self, event):
        """
        Build decay tree from MadSpin mother pointers.

        Scans the event for status=2 resonances and identifies their decay
        products using mother1/mother2 pointers.

        Args:
            event: LHE event object

        Returns:
            Dict mapping LHE index (1-based) to decay info:
            {lhe_idx: {'pdg': int, 'status': int, 'children': [lhe_idx, ...], 'depth': int}}
        """
        tree = {}

        # Pass 1: Create nodes for all particles
        for i, particle in enumerate(event):
            lhe_idx = i + 1  # 1-based LHE indexing
            pdg = int(getattr(particle, "pid", 0))
            status = int(getattr(particle, "status", 0))
            tree[lhe_idx] = {
                "pdg": pdg,
                "status": status,
                "children": [],
                "depth": 0,
                "mother": None,
            }

        # Pass 2: Link parent-child relationships for status=2 resonances
        for i, particle in enumerate(event):
            lhe_idx = i + 1
            m1 = _get_mother_lhe_idx(getattr(particle, "mother1", None))
            m2 = _get_mother_lhe_idx(getattr(particle, "mother2", None))

            # Check if mother is a status=2 resonance
            if m1 > 0 and m1 in tree and tree[m1]["status"] == 2:
                tree[m1]["children"].append(lhe_idx)
                tree[lhe_idx]["mother"] = m1
            elif m2 > 0 and m2 in tree and tree[m2]["status"] == 2:
                # Use m2 only if m1 didn't match (avoid double-counting for m1==m2)
                if m2 != m1:
                    tree[m2]["children"].append(lhe_idx)
                    tree[lhe_idx]["mother"] = m2

        # Pass 3: Compute depths (bottom-up from leaves)
        # Depth = maximum distance to a leaf descendant
        # Resonances with no children have depth 0; those with children have depth >= 1
        def compute_depth(lhe_idx):
            node = tree[lhe_idx]
            if not node["children"]:
                return 0
            max_child_depth = 0
            for child_idx in node["children"]:
                child_depth = compute_depth(child_idx)
                max_child_depth = max(max_child_depth, child_depth)
            node["depth"] = max_child_depth + 1
            return node["depth"]

        # Compute depth for all particles (resonances will get non-zero depths)
        for lhe_idx in tree:
            if tree[lhe_idx]["status"] == 2 and tree[lhe_idx]["children"]:
                compute_depth(lhe_idx)

        _dbg(f"[FORCED] Built decay tree with {len(tree)} nodes")
        for lhe_idx, node in tree.items():
            if node["children"]:
                _dbg(
                    f"[FORCED]   {lhe_idx} (pdg={node['pdg']}, depth={node['depth']}) -> children={node['children']}"
                )

        return tree

    def _generate_forced_cluster_steps(self, event, decay_tree):
        """
        Generate forced clustering steps from MadSpin decay tree.

        Steps are ordered by depth ascending (inner decays first).
        Depth = distance to leaves, so W (depth=1) < top (depth=2).
        This ensures nested decays are handled correctly:
        first cluster W -> l nu, then cluster t -> b W.

        Args:
            event: LHE event object
            decay_tree: Output from _build_madspin_decay_tree()

        Returns:
            List[ForcedClusterStep] ordered by depth ascending (inner first)
        """
        steps = []

        # Find all resonances with children (status=2 particles that decayed)
        resonances = [
            (lhe_idx, node)
            for lhe_idx, node in decay_tree.items()
            if node["status"] == 2 and node["children"]
        ]

        # Sort by depth ascending (inner decays first: W before t)
        # depth=1 (W, one step from leaves) must be clustered before depth=2 (top)
        resonances.sort(key=lambda x: x[1]["depth"])

        for lhe_idx, node in resonances:
            # Compute clustering scale = invariant mass of children
            children = node["children"]
            sum_E = sum_px = sum_py = sum_pz = 0.0

            for child_idx in children:
                p = event[child_idx - 1]  # Convert to 0-based
                sum_E += float(p.E)
                sum_px += float(p.px)
                sum_py += float(p.py)
                sum_pz += float(p.pz)

            m2 = sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2
            scale = math.sqrt(max(0, m2))

            step = ForcedClusterStep(
                children_lhe_idx=children,
                mother_lhe_idx=lhe_idx,
                mother_pdg=node["pdg"],
                scale=scale,
                depth=node["depth"],
            )
            steps.append(step)

            _dbg(
                f"[FORCED] Step: cluster {children} -> {lhe_idx} (pdg={node['pdg']}), scale={scale:.2f} GeV, depth={node['depth']}"
            )

        return steps

    def _apply_forced_clustering(self, event, steps):
        """
        Apply forced clustering steps with FKS kinematic reshuffling.

        For each step:
        1. Identify the two children to cluster (MadSpin always does 1->2)
        2. Call FKS FSR reshuffling to put mother on-shell
        3. Remove children, keep mother with updated momentum
        4. Update groups to track original LHE indices

        Args:
            event: LHE event object (will be modified)
            steps: List[ForcedClusterStep] from _generate_forced_cluster_steps()

        Returns:
            Tuple of (modified_event, groups)
            - modified_event: Event with resonances restored (hard process)
            - groups: List of sets mapping positions to original LHE indices
        """
        buff_event = copy.deepcopy(event)

        # Initialize groups: track which original LHE indices are at each position
        # Only track external particles (status = ±1), not intermediates (status=2)
        # But we need to include status=2 resonances that will become final-state
        groups = []
        pos_to_lhe = {}  # Current position -> original LHE index
        lhe_to_pos = {}  # Original LHE index -> current position

        for i, p in enumerate(buff_event):
            lhe_idx = i + 1
            pos_to_lhe[i] = lhe_idx
            lhe_to_pos[lhe_idx] = i
            groups.append(set([lhe_idx]))

        _dbg("")
        _dbg("=" * 70)
        _dbg("  STAGE 1: MADSPIN FORCED CLUSTERING (Decay Reversal)")
        _dbg("=" * 70)
        _dbg(f"[FORCED] Starting forced clustering with {len(steps)} steps")
        _dbg(f"[FORCED] Initial event has {len(buff_event)} particles")

        for step_num, step in enumerate(steps):
            _dbg(
                f"[FORCED] === Step {step_num + 1}: cluster {step.children_lhe_idx} -> {step.mother_lhe_idx} (pdg={step.mother_pdg}) ==="
            )

            # MadSpin always produces 1->2 decays
            if len(step.children_lhe_idx) != 2:
                _dbg(
                    f"[FORCED]   WARNING: Expected 2 children, got {len(step.children_lhe_idx)}, skipping"
                )
                continue

            child1_lhe, child2_lhe = step.children_lhe_idx
            mother_lhe = step.mother_lhe_idx

            # Find current positions
            if child1_lhe not in lhe_to_pos or child2_lhe not in lhe_to_pos:
                _dbg("[FORCED]   Children not found in current event, skipping")
                continue
            if mother_lhe not in lhe_to_pos:
                _dbg("[FORCED]   Mother not found in current event, skipping")
                continue

            child1_pos = lhe_to_pos[child1_lhe]
            child2_pos = lhe_to_pos[child2_lhe]
            mother_pos = lhe_to_pos[mother_lhe]

            _dbg(
                f"[FORCED]   Positions: child1={child1_pos}, child2={child2_pos}, mother={mother_pos}"
            )

            # =====================================================================
            # FKS RESHUFFLING FOR MADSPIN FORCED CLUSTERING
            # =====================================================================
            # When clustering decay products back to resonance, we must put the
            # mother ON-SHELL using FKS kinematic reshuffling. This redistributes
            # momentum to spectators (other final-state particles) or scales beams.
            #
            # Key difference from normal FxFx clustering:
            # - The mother already exists as status=2 intermediate
            # - We must exclude it from the momentum conservation participants
            # =====================================================================

            # Build list of particles participating in momentum conservation:
            # - Beams (positions 0, 1)
            # - Final-state particles (status=1 or -1), excluding status=2 intermediates
            active_positions = []  # positions in buff_event
            for pos in range(len(buff_event)):
                p = buff_event[pos]
                status = getattr(p, "status", 1)
                if pos <= 1:  # beams
                    active_positions.append(pos)
                elif abs(status) == 1:  # final-state
                    active_positions.append(pos)
                # Skip status=2 intermediates (like the mother before clustering)

            _dbg(f"[FORCED]   Active positions for FKS: {active_positions}")

            # Build orig_momenta for FKS (only active particles)
            # Create mapping: active_idx -> buff_event position
            active_to_buff = {i: pos for i, pos in enumerate(active_positions)}
            buff_to_active = {pos: i for i, pos in enumerate(active_positions)}

            orig_momenta = [FourMomentum(buff_event[pos]) for pos in active_positions]

            # Map child positions to active indices
            if child1_pos not in buff_to_active or child2_pos not in buff_to_active:
                _dbg("[FORCED]   WARNING: Children not in active list, skipping reshuffling")
                # Fallback: just sum momenta (off-shell)
                p1 = buff_event[child1_pos]
                p2 = buff_event[child2_pos]
                mother_E = float(p1.E) + float(p2.E)
                mother_px = float(p1.px) + float(p2.px)
                mother_py = float(p1.py) + float(p2.py)
                mother_pz = float(p1.pz) + float(p2.pz)
                mother_p = buff_event[mother_pos]
                mother_p.E = mother_E
                mother_p.px = mother_px
                mother_p.py = mother_py
                mother_p.pz = mother_pz
                mother_p.status = 1
            else:
                child1_active = buff_to_active[child1_pos]
                child2_active = buff_to_active[child2_pos]

                # Get child momenta for logging
                p1 = buff_event[child1_pos]
                p2 = buff_event[child2_pos]
                mother_E = float(p1.E) + float(p2.E)
                mother_px = float(p1.px) + float(p2.px)
                mother_py = float(p1.py) + float(p2.py)
                mother_pz = float(p1.pz) + float(p2.pz)
                off_shell_m2 = mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2
                off_shell_m = math.sqrt(max(0, off_shell_m2))

                _dbg(f"[FORCED]   Off-shell mother: E={mother_E:.4f}, m={off_shell_m:.4f} GeV")

                # Get target mass (pole mass for known resonances, else off-shell)
                target_mass = _get_mother_mass(abs(step.mother_pdg)) or off_shell_m

                # =================================================================
                # FKS IN CM FRAME: Transform to CM, apply FKS, transform back
                # =================================================================
                # The FKS constraint is more relaxed in CM frame. In lab frame,
                # asymmetric beam momenta can make the constraint unsatisfiable
                # even when there's enough phase space.
                # =================================================================

                # Compute CM boost: q = p1 + p2 (beams)
                q_lab = orig_momenta[0] + orig_momenta[1]
                sqrt_s = math.sqrt(max(0, _m2(q_lab)))

                # Beta for boost to CM frame (along z since beams are along z)
                if q_lab.E > 1e-10:
                    beta_cm = q_lab.pz / q_lab.E
                else:
                    beta_cm = 0.0

                # Transform all active momenta to CM frame (boost along z)
                _z_hat = (0, 0, 1)
                orig_momenta_cm = [_boost_along_direction(p, beta_cm, _z_hat) for p in orig_momenta]

                _dbg(f"[FORCED]   CM boost: β={beta_cm:.6f}, √s={sqrt_s:.2f} GeV")

                # Prepare moth argument for _fks_fsr_mapping
                moth = [{"number": child1_active + 1, "id": step.mother_pdg}]

                # Call FKS FSR mapping in CM frame
                fks_result = _fks_fsr_mapping(child1_active, child2_active, moth, orig_momenta_cm)

                _dbg(f"[FORCED]   FKS method: {fks_result['method']}")

                # Transform results back to lab frame (inverse boost: -beta)
                onshell_mom_lab = _boost_along_direction(fks_result["mother"], -beta_cm, _z_hat)
                beam0_lab = _boost_along_direction(fks_result["beams"][0], -beta_cm, _z_hat)
                beam1_lab = _boost_along_direction(fks_result["beams"][1], -beta_cm, _z_hat)
                spectators_lab = {
                    idx: _boost_along_direction(p, -beta_cm, _z_hat)
                    for idx, p in fks_result["spectators"].items()
                }

                # Apply results to buff_event
                mother_p = buff_event[mother_pos]
                mother_p.E = onshell_mom_lab.E
                mother_p.px = onshell_mom_lab.px
                mother_p.py = onshell_mom_lab.py
                mother_p.pz = onshell_mom_lab.pz
                mother_p.status = 1  # Now final-state

                onshell_m2 = _m2(onshell_mom_lab)
                onshell_m = math.sqrt(max(0, onshell_m2))

                # Set mass based on FKS result
                fks_method = fks_result.get("method", "")
                if fks_method in ("fks_fsr_no_solution", "fks_fsr_superluminal"):
                    mother_p.mass = onshell_m
                    _dbg("[FORCED]   WARNING: FKS failed even in CM frame, using invariant mass")
                else:
                    mother_p.mass = target_mass

                _dbg(
                    f"[FORCED]   On-shell mother: E={onshell_mom_lab.E:.4f}, m={onshell_m:.4f} GeV (target={target_mass:.4f})"
                )

                # Apply beam modifications
                buff_event[0].E = beam0_lab.E
                buff_event[0].px = beam0_lab.px
                buff_event[0].py = beam0_lab.py
                buff_event[0].pz = beam0_lab.pz
                buff_event[1].E = beam1_lab.E
                buff_event[1].px = beam1_lab.px
                buff_event[1].py = beam1_lab.py
                buff_event[1].pz = beam1_lab.pz

                # Apply spectator boosts
                for active_idx, boosted_p in spectators_lab.items():
                    buff_pos = active_to_buff[active_idx]
                    if buff_pos != child1_pos and buff_pos != child2_pos:
                        buff_event[buff_pos].E = boosted_p.E
                        buff_event[buff_pos].px = boosted_p.px
                        buff_event[buff_pos].py = boosted_p.py
                        buff_event[buff_pos].pz = boosted_p.pz
                        _dbg(f"[FORCED]   Boosted spectator at pos {buff_pos}: E={boosted_p.E:.4f}")

                _dbg(
                    f"[FORCED]   Mass shift: {off_shell_m:.4f} -> {onshell_m:.4f} GeV (Δ={onshell_m - off_shell_m:+.4f})"
                )

            # Merge groups: mother group now includes children's LHE indices
            mother_group = groups[mother_pos]
            child1_group = groups[child1_pos]
            child2_group = groups[child2_pos]
            mother_group.update(child1_group)
            mother_group.update(child2_group)

            _dbg(f"[FORCED]   Merged groups: mother now contains {mother_group}")

            # Remove children from event (higher index first to preserve lower indices)
            children_pos = sorted([child1_pos, child2_pos], reverse=True)
            for pos in children_pos:
                buff_event.pop(pos)
                groups.pop(pos)

            # Rebuild position mappings
            pos_to_lhe = {}
            lhe_to_pos = {}
            for i, grp in enumerate(groups):
                # The "representative" LHE index is the minimum in the group
                # (or we could track the mother's original LHE)
                rep_lhe = min(grp)
                pos_to_lhe[i] = rep_lhe
                for lhe_idx in grp:
                    lhe_to_pos[lhe_idx] = i

            _dbg(f"[FORCED]   After removal: {len(buff_event)} particles, {len(groups)} groups")

        _dbg(f"[FORCED] Forced clustering complete. Final event has {len(buff_event)} particles")
        _dbg("=" * 70)
        _dbg("  END STAGE 1: Hard process restored")
        _dbg("=" * 70)
        _dbg("")

        return buff_event, groups

    # -------------------------------------------------------------------------
    # Original FxFx Clustering Methods
    # -------------------------------------------------------------------------

    def _fxfx_parse_clustering(self, event):
        """Parse the FxFx <clustering ...> block from an event."""
        ev_lines = str(event).splitlines()
        for line in ev_lines:
            if "<clustering" not in line:
                continue
            attrs = dict(FXFX_CLUSTER_ATTR_RE.findall(line))
            steps = []
            for step, payload in FXFX_CLUSTER_STEP_RE.findall(line):
                parts = [p.strip() for p in payload.split(";")]
                idx_part = parts[0] if parts else ""
                mom_pdg = None
                for comp in parts[1:]:
                    if comp.startswith("mom="):
                        try:
                            mom_pdg = int(comp.split("=", 1)[1])
                        except Exception:
                            mom_pdg = None
                try:
                    step_int = int(step)
                except Exception:
                    continue
                idx_clean = idx_part.strip()
                if idx_clean.startswith("(") and ")" in idx_clean:
                    idx_clean = idx_clean[1 : idx_clean.find(")")]
                idxs = []
                for token in idx_clean.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        idxs.append(int(token))
                    except Exception:
                        continue
                steps.append({"n": step_int, "idxs": idxs, "mom": mom_pdg})
            steps.sort(key=lambda s: s["n"])
            attrs["steps"] = steps
            return attrs
        return None

    def _cluster_fxfx_event(self, event, record_tag=False, return_groups=False):
        """
        Main entry point for event clustering.

        This method handles both MadSpin forced clustering and FxFx clustering:
        1. If event has MadSpin decays (status=2 resonances with children),
           apply forced clustering first to restore the hard process
        2. Then apply FxFx clustering (if <clustering> tag exists)

        Args:
            event: LHE event object
            record_tag: If True, record the sorted_tag for later use
            return_groups: If True, return groups tracking original LHE indices

        Returns:
            If return_groups=False: (clustered_event, sorted_tag) or None
            If return_groups=True: (clustered_event, sorted_tag, groups) or None
        """
        _dbg(f"[CLUSTER] _cluster_fxfx_event ENTER: npart={len(event)}, record_tag={record_tag}, return_groups={return_groups}")

        # Step 1: Check for MadSpin decayed resonances
        decayed_resonances = ResonanceIdentifier.find_decayed_resonances(event)
        _dbg(f"[CLUSTER]   decayed_resonances detected: {len(decayed_resonances) if decayed_resonances else 0}")

        if decayed_resonances:
            _dbg(
                f"[CLUSTER] Found {len(decayed_resonances)} decayed resonances, applying forced clustering"
            )

            # Build decay tree from mother pointers
            decay_tree = self._build_madspin_decay_tree(event)
            _dbg(f"[CLUSTER]   built decay_tree with {len(decay_tree)} nodes")

            # Generate forced clustering steps (depth-first order)
            forced_steps = self._generate_forced_cluster_steps(event, decay_tree)
            _dbg(f"[CLUSTER]   forced_steps generated: {len(forced_steps) if forced_steps else 0}")

            if forced_steps:
                # Apply forced clustering to restore hard process
                hard_event, forced_groups = self._apply_forced_clustering(event, forced_steps)

                _dbg(
                    f"[CLUSTER] Forced clustering complete, hard process has {len(hard_event)} particles"
                )

                # Now apply FxFx clustering on the hard process
                # Pass the forced_groups to maintain LHE index tracking
                _dbg(f"[CLUSTER]   -> dispatching internal cluster on hard_event with forced groups (len={len(forced_groups) if forced_groups else 0})")
                return self._cluster_fxfx_event_internal(
                    hard_event,
                    original_event=event,
                    initial_groups=forced_groups,
                    record_tag=record_tag,
                    return_groups=return_groups,
                )
            else:
                _dbg("[CLUSTER] No forced steps generated, using standard FxFx")

        # No MadSpin decays or no forced steps: use standard FxFx clustering
        _dbg("[CLUSTER]   -> dispatching standard internal cluster (no forced clustering)")
        return self._cluster_fxfx_event_internal(
            event,
            original_event=event,
            initial_groups=None,
            record_tag=record_tag,
            return_groups=return_groups,
        )

    def _cluster_fxfx_event_internal(
        self, event, original_event=None, initial_groups=None, record_tag=False, return_groups=False
    ):
        """
        Internal FxFx clustering implementation.

        This method applies the FxFx clustering steps from the <clustering> tag.
        It can accept pre-initialized groups from forced clustering.

        Args:
            event: Event to cluster (may be output of forced clustering)
            original_event: The original LHE event (for accessing particle data)
            initial_groups: Pre-initialized groups from forced clustering, or None
            record_tag: If True, record the sorted_tag
            return_groups: If True, return groups

        Returns:
            Same as _cluster_fxfx_event
        """
        if original_event is None:
            original_event = event

        _dbg("")
        _dbg("=" * 70)
        _dbg("  STAGE 2: FXFX CLUSTERING (QCD/EW Jet Clustering)")
        _dbg("=" * 70)
        _dbg(
            f"[FXFX] _cluster_fxfx_event_internal: start (record_tag={record_tag}, return_groups={return_groups})"
        )
        _dbg(f"[FXFX]   input particles={len(event)}")
        _dbg(f"[FXFX]   initial_groups provided: {initial_groups is not None}")

        # Parse FxFx clustering tag from the original event (it's preserved through MadSpin)
        clustering = self._fxfx_parse_clustering(original_event)
        if not clustering:
            _dbg("[FXFX]   no <clustering> block found")
            if initial_groups is not None:
                # Forced clustering was done - return the hard process directly
                _dbg("[FXFX]   returning forced-clustered hard process (no further FxFx)")
                try:
                    sorted_tag = event.get_tag_and_order()[0]
                except Exception:
                    sorted_tag = None
                if record_tag and sorted_tag:
                    if not hasattr(self, "fxfx_clustered_final_states"):
                        self.fxfx_clustered_final_states = set()
                    self.fxfx_clustered_final_states.add(sorted_tag)
                if return_groups:
                    return (event, sorted_tag, initial_groups)
                return (event, sorted_tag)
            return None
        _dbg(
            f"[FXFX]   clustering attrs: type={clustering.get('type')}, steps={len(clustering.get('steps', []))}"
        )

        buff_event = copy.deepcopy(event)
        to_pop = [
            ip
            for ip, part in enumerate(buff_event)
            if (abs(getattr(part, "status", 0)) != 1) and (getattr(part, "status", 0) != -1)
        ]
        for ip in reversed(to_pop):
            buff_event.pop(ip)

        # Use initial_groups if provided (from forced clustering), otherwise create new
        if initial_groups is not None:
            groups = [set(g) for g in initial_groups]  # Deep copy
            # Build lhe_external_indices from the groups
            lhe_external_indices = []
            for g in groups:
                lhe_external_indices.append(min(g))  # Use minimum as representative
            ext_index_to_lhe = {
                i + 1: lhe_external_indices[i] for i in range(len(lhe_external_indices))
            }
            ext_index_max = len(lhe_external_indices)
            _dbg("[FXFX]   using initial_groups from forced clustering")
        else:
            lhe_external_indices = [
                i
                for i, p in enumerate(event, start=1)
                if (abs(getattr(p, "status", 0)) == 1) or (getattr(p, "status", 0) == -1)
            ]
            # Map 1-based external index to 1-based LHE index
            ext_index_to_lhe = {
                i + 1: lhe_external_indices[i] for i in range(len(lhe_external_indices))
            }
            ext_index_max = len(lhe_external_indices)
            groups = [set([lhe_external_indices[k]]) for k in range(len(buff_event))]

        _dbg(f"[FXFX]   external indices (LHE): {lhe_external_indices}")
        _dbg(f"[FXFX]   ext_index_to_lhe: {ext_index_to_lhe}")
        _dbg(f"[FXFX]   groups: {[sorted(list(g)) for g in groups]}")

        def _find_position_single(i1b):
            for k, g in enumerate(groups):
                if i1b in g:
                    return k
            return None

        def _s_groups_lhe_from_pos(pos_i, pos_j):
            # Use original_event for particle lookups since LHE indices refer to it
            Ei = pxi = pyi = pzi = 0.0
            Ej = pxj = pyj = pzj = 0.0
            for lab in groups[pos_i]:
                q = original_event[lab - 1]
                Ei += float(q.E)
                pxi += float(q.px)
                pyi += float(q.py)
                pzi += float(q.pz)
            for lab in groups[pos_j]:
                q = original_event[lab - 1]
                Ej += float(q.E)
                pxj += float(q.px)
                pyj += float(q.py)
                pzj += float(q.pz)
            E = Ei + Ej
            px = pxi + pxj
            py = pyi + pyj
            pz = pzi + pzj
            return E * E - px * px - py * py - pz * pz

        def _group_rep(pos):
            """Return representative pdg/state for a dynamic particle group.

            Uses buff_event[pos] for current (possibly clustered) PDG,
            not the original event particles.
            """
            if pos < 0 or pos >= len(buff_event):
                return None
            part = buff_event[pos]
            pdg = int(getattr(part, "pid", 0))
            state = int(getattr(part, "status", 0)) != -1
            return {"pdg": pdg, "state": state}

        mW2 = MW_POLE**2
        # Multiplier for clustering threshold: cluster if s_ij < mW2_cluster_scale * mW2
        # Default 1.5 allows clustering up to ~98 GeV (covers M_Z = 91.2 GeV)
        mW2_cluster_scale = getattr(self, "mW2_cluster_scale", 1.5)
        mW2_threshold = mW2_cluster_scale * mW2
        _dbg(f"[FXFX]   mW2_threshold={mW2_threshold:.3f} (scale={mW2_cluster_scale})")

        # =====================================================================
        # CLUSTERING ORDER: Use original LHE step order (scale-ordered)
        # The LHE clustering tag specifies steps in scale order (soft -> hard).
        # We follow this order to ensure proper kinematic reshuffling.
        # =====================================================================
        original_steps = clustering.get("steps", [])
        ordered_steps = sorted(original_steps, key=lambda t: t["n"])

        _dbg("[FXFX]   CLUSTERING STEPS (scale-ordered):")
        for s in ordered_steps:
            mom = s.get("mom")
            idxs = s.get("idxs", [])
            _dbg(f"[FXFX]       step {s['n']}: idxs={idxs}, mom={mom}")

        stop_clustering = False
        for _step in ordered_steps:
            _stepnum = _step["n"]
            idxs = _step["idxs"]
            _step_mom = _step.get("mom")
            _dbg(f"[FXFX]   step {_stepnum}: idxs={idxs}, mom={_step_mom}, groups={len(groups)}")

            while True:
                all_positions = list(range(len(groups)))
                if len(all_positions) < 2:
                    stop_clustering = True
                    break

                all_pairs_sij = []
                for ii in range(len(all_positions)):
                    for jj in range(ii + 1, len(all_positions)):
                        pos_i = all_positions[ii]
                        pos_j = all_positions[jj]
                        sij = _s_groups_lhe_from_pos(pos_i, pos_j)
                        all_pairs_sij.append((pos_i, pos_j, sij))

                if not all_pairs_sij:
                    stop_clustering = True
                    break

                # Debug: show all pair invariants vs threshold
                _dbg(
                    f"[FXFX]   INVARIANT CHECK (threshold={mW2_threshold:.1f} GeV² = {mW2_cluster_scale}×MW²):"
                )
                all_above = True
                for pos_i, pos_j, sij in all_pairs_sij:
                    above = sij >= mW2_threshold
                    mark = "✓ ABOVE" if above else "✗ below"
                    _dbg(f"[FXFX]     s({pos_i},{pos_j}) = {sij:.1f} GeV²  {mark} threshold")
                    if not above:
                        all_above = False

                if all_above:
                    _dbg("[FXFX]   -> ALL pairs above threshold, STOP clustering (resolved)")
                    stop_clustering = True
                    break

                # FxFx clustering uses external-only indices (skipping status=2 intermediates)
                # Validate and map to LHE indices
                idxs_ext = idxs  # These are external indices from the clustering tag
                valid = all((1 <= k <= ext_index_max) for k in idxs_ext)
                if not valid:
                    stop_clustering = True
                    break

                # Map external indices to LHE indices
                idxs_lhe = []
                for ext_idx in idxs_ext:
                    lhe_idx = ext_index_to_lhe.get(ext_idx)
                    if lhe_idx is None:
                        break
                    idxs_lhe.append(lhe_idx)
                if len(idxs_lhe) != len(idxs_ext):
                    stop_clustering = True
                    break

                parent_pos = []
                seen = set()
                for lab in idxs_lhe:
                    pos = _find_position_single(lab)
                    if pos is None:
                        continue
                    if pos not in seen:
                        parent_pos.append(pos)
                        seen.add(pos)

                if len(idxs_lhe) == 3 and len(parent_pos) != 2:
                    stop_clustering = True
                    break

                if len(parent_pos) < 2:
                    break

                if len(parent_pos) == 2:
                    ai, bj = parent_pos
                    s_candidate = _s_groups_lhe_from_pos(ai, bj)
                    pair = (ai, bj)
                else:
                    best = None
                    best_pair = None
                    for ii in range(len(parent_pos)):
                        for jj in range(ii + 1, len(parent_pos)):
                            sij_loc = _s_groups_lhe_from_pos(parent_pos[ii], parent_pos[jj])
                            if (best is None) or (sij_loc < best):
                                best = sij_loc
                                best_pair = (parent_pos[ii], parent_pos[jj])
                    if best_pair is None:
                        break
                    pair = best_pair
                    s_candidate = best

                # Debug: show candidate pair invariant (info only)
                # NOTE: We do NOT stop here based on single-pair threshold.
                # The ALL-pairs check above handles "everything resolved" case.
                sqrt_s_cand = math.sqrt(max(0, s_candidate))
                _dbg(
                    f"[FXFX]   CLUSTERING pair {pair}: s={s_candidate:.1f} GeV² (√s={sqrt_s_cand:.1f} GeV)"
                )

                a, b = sorted(pair)
                keep = _choose_keep(a, b)

                if _step_mom is not None and self.model:
                    rep_a = _group_rep(a)
                    rep_b = _group_rep(b)
                    if rep_a and rep_b:
                        pdg_a = rep_a["pdg"]
                        pdg_b = rep_b["pdg"]
                        abs_a = abs(pdg_a)
                        abs_b = abs(pdg_b)

                        # Flavor conservation for QCD/EW clusterings:
                        # 1. q + g -> q : mother must have same flavor, sign from tag
                        # 2. q + q~ -> g : quarks must have same flavor (gluon is flavor-neutral)
                        # Also handle leptons/neutrinos (11-16) for EW processes
                        is_quark_a = 1 <= abs_a <= 6
                        is_quark_b = 1 <= abs_b <= 6
                        is_lepton_a = 11 <= abs_a <= 16
                        is_lepton_b = 11 <= abs_b <= 16
                        is_fermion_a = is_quark_a or is_lepton_a
                        is_fermion_b = is_quark_b or is_lepton_b
                        is_gluon_a = abs_a == 21
                        is_gluon_b = abs_b == 21
                        is_photon_a = abs_a == 22
                        is_photon_b = abs_b == 22
                        is_gauge_boson_a = is_gluon_a or is_photon_a
                        is_gauge_boson_b = is_gluon_b or is_photon_b

                        # Fix flavor using 3-point interaction rules, then charge conservation fixes sign
                        if is_gauge_boson_a and is_fermion_b:
                            # V + f -> f : take flavor from fermion
                            old_step_mom = _step_mom
                            sign = 1 if _step_mom >= 0 else -1
                            _step_mom = (
                                sign * abs_b
                            )  # sign from tag (initial guess), flavor from fermion
                            _dbg(
                                f"FxFx clustering: flavor fix V+f->f: {old_step_mom} -> {_step_mom} (sign={sign}, flavor={abs_b})"
                            )
                        elif is_fermion_a and is_gauge_boson_b:
                            # f + V -> f : take flavor from fermion
                            old_step_mom = _step_mom
                            sign = 1 if _step_mom >= 0 else -1
                            _step_mom = (
                                sign * abs_a
                            )  # sign from tag (initial guess), flavor from fermion
                            _dbg(
                                f"FxFx clustering: flavor fix f+V->f: {old_step_mom} -> {_step_mom} (sign={sign}, flavor={abs_a})"
                            )
                        elif is_quark_a and is_quark_b:
                            # q + q~ clustering -> gluon (if same flavor, opposite sign)
                            if abs_a == abs_b and pdg_a * pdg_b < 0:
                                # Same flavor quark-antiquark -> gluon
                                if abs(_step_mom) != 21:
                                    old_step_mom = _step_mom
                                    _step_mom = 21
                                    _dbg(
                                        f"FxFx clustering: q+q~->g flavor fix: {old_step_mom} -> {_step_mom}"
                                    )
                                else:
                                    _dbg(f"FxFx clustering: q+q~->g, keeping mom={_step_mom}")
                            elif abs_a == abs_b and pdg_a * pdg_b > 0:
                                # Same flavor, same sign (qq or q~q~) - shouldn't happen in valid QCD
                                _dbg(f"FxFx clustering: same-sign q+q, keeping tag mom={_step_mom}")
                            else:
                                # Different flavors - flavor-changing clustering
                                _dbg(
                                    f"FxFx clustering: different flavor q+q', keeping tag mom={_step_mom} (pdg_a={pdg_a}, pdg_b={pdg_b})"
                                )
                        elif is_lepton_a and is_lepton_b:
                            # l + l~ clustering -> Z or photon (if same flavor)
                            if abs_a == abs_b and pdg_a * pdg_b < 0:
                                # Same flavor lepton-antilepton -> Z(23) or gamma(22)
                                _dbg(f"FxFx clustering: l+l~->V, keeping tag mom={_step_mom}")
                            else:
                                _dbg(
                                    f"FxFx clustering: lepton pair, keeping tag mom={_step_mom} (pdg_a={pdg_a}, pdg_b={pdg_b})"
                                )
                        else:
                            # Use the existing family-based logic for other cases
                            emitter_state = False if (keep <= 1) else True
                            suggestions = suggest_mothers_from_interactions(
                                self.model, ("QCD", "QED"), rep_a, rep_b, emitter_state
                            )
                            _dbg(
                                f"FxFx clustering step {_stepnum}: _step_mom={_step_mom}, rep_a={rep_a}, rep_b={rep_b}, suggestions={[(c['mother_pdg'], _flavor_family(c['mother_pdg'])) for c in suggestions] if suggestions else []}"
                            )
                            family = _flavor_family(_step_mom)
                            if family and suggestions:
                                for cand in suggestions:
                                    if _flavor_family(cand["mother_pdg"]) == family:
                                        sug_pdg = int(cand["mother_pdg"])
                                        sign = 1 if _step_mom >= 0 else -1
                                        old_step_mom = _step_mom
                                        _step_mom = sign * abs(sug_pdg)
                                        _dbg(
                                            f"FxFx clustering: updated _step_mom from {old_step_mom} to {_step_mom} (sug_pdg={sug_pdg}, sign={sign})"
                                        )
                                        break

                    # Enforce global charge conservation by possibly flipping the mother sign
                    # This determines particle vs antiparticle after flavor is fixed
                    if _step_mom is not None:
                        try:
                            old_mom = _step_mom
                            _step_mom, charge_changed = _fix_mother_sign_global(
                                self.model, buff_event, a, b, _step_mom
                            )
                            if charge_changed:
                                _dbg(
                                    f"FxFx global charge conservation fix (step {_stepnum}): {old_mom} -> {_step_mom}"
                                )
                        except RuntimeError as err:
                            _dbg(f"Global charge conservation failed in FxFx clustering: {err}")
                            raise

                if _step_mom is None:
                    break

                moth_id = int(_step_mom)

                # Diagnostic: show mother determination and path implication
                sqrt_s = math.sqrt(max(0, s_candidate)) if s_candidate > 0 else 0
                is_resonance = abs(moth_id) in (6, 23, 24)  # t, Z, W
                path = "DENSITY" if is_resonance else "SCALAR"
                boson_name = {
                    21: "gluon",
                    22: "photon",
                    23: "Z",
                    24: "W+",
                    -24: "W-",
                    6: "top",
                    -6: "anti-top",
                }.get(moth_id, str(moth_id))
                _dbg(
                    f"[FXFX]   MOTHER: pdg={moth_id} ({boson_name}), √s={sqrt_s:.2f} GeV, M_Z={MZ_POLE:.2f} GeV"
                )
                _dbg(
                    f"[FXFX]   -> {boson_name} {'∈' if is_resonance else '∉'} {{t,W,Z}} => {path} PATH"
                )

                # STOP if this clustering would result in 2→1 with only γ/g in FS
                # Compute what FS looks like after this clustering
                current_fs = [i for i in range(len(groups)) if i > 1]
                a_is_fs, b_is_fs = (a > 1), (b > 1)

                fs_after_pdgs = []
                if a_is_fs and b_is_fs:
                    # FSR: mother stays in FS
                    fs_after_pdgs.append(moth_id)
                    for i in current_fs:
                        if i != a and i != b:
                            rep = _group_rep(i)
                            if rep:
                                fs_after_pdgs.append(rep["pdg"])
                else:
                    # ISR: FS particle goes to IS, only spectators remain
                    for i in current_fs:
                        if i != a and i != b:
                            rep = _group_rep(i)
                            if rep:
                                fs_after_pdgs.append(rep["pdg"])

                # Check: would clustering result in 2→1 with massless FS?
                # This is kinematically forbidden (no rest frame for massless particle)
                if len(fs_after_pdgs) == 1:
                    fs_pdg = fs_after_pdgs[0]
                    fs_mass = _get_mother_mass(abs(fs_pdg))
                    if fs_mass < 1e-6:
                        _dbg(f"[FXFX]   STOP: 2→1 with massless FS (pdg={fs_pdg})")
                        stop_clustering = True
                        break

                # Create moth structure for fxfx_merge_particles_kinematics
                # moth[0]['number'] indicates which particle becomes the mother (1-based)
                moth_loc = [{"number": keep + 1, "id": moth_id}]

                # Do kinematic reshuffling using FxFx-specific Catani-Seymour method
                try:
                    fxfx_merge_particles_kinematics(buff_event, a, b, moth_loc)
                except ValueError as e:
                    if "light-like" in str(e):
                        # Light-like p_clustered = can't boost, stop clustering here
                        _dbg(f"[FXFX]   STOP: {e}")
                        stop_clustering = True
                        break
                    if "math domain error" in str(e):
                        _dbg(f"EVENT SKIPPED - kinematic reshuffling failed: {e}")
                        return None
                    raise

                # Update groups
                if keep == a:
                    groups[a] = groups[a] | groups[b]
                    del groups[b]
                else:
                    groups[b] = groups[a] | groups[b]
                    del groups[a]
                _dbg(f"[FXFX]   clustered pair ({a},{b}) -> keep={keep}, new groups={len(groups)}")
                continue

            if stop_clustering:
                break

        try:
            mapped_tag, mapped_order = buff_event.get_tag_and_order()
            sorted_tag = (tuple(mapped_order[0]), tuple(sorted(mapped_order[1])))
        except Exception:
            return None

        if record_tag:
            self.fxfx_clustered_final_states.add(sorted_tag)

        _dbg(f"[FXFX]   clustered sorted_tag={sorted_tag}")
        _dbg(f"[FXFX]   final groups: {[sorted(list(g)) for g in groups]}")
        _dbg("=" * 70)
        _dbg("  END STAGE 2: FxFx clustering complete")
        _dbg("=" * 70)
        _dbg("")
        if return_groups:
            return buff_event, sorted_tag, groups
        return buff_event, sorted_tag

    def _needs_fxfx_cluster_catalogue(self):
        if not self.inc_sudakov:
            return False
        try:
            ickkw = int(self.banner.get("run_card", "ickkw"))
        except Exception:
            return False
        return ickkw == 3

    def _ensure_fxfx_cluster_catalogue(self):
        if not self._needs_fxfx_cluster_catalogue():
            return
        if self._fxfx_cluster_catalog_ready:
            return
        if not getattr(self, "lhe_input", None):
            return
        lhe_path = self.lhe_input.path
        _dbg("Scanning events to collect FxFx clustered final states for Sudakov reweighting")
        self.lhe_input.seek(0)
        for event in self.lhe_input:
            cluster_result = self._cluster_fxfx_event(event, record_tag=True, return_groups=True)
            if not cluster_result:
                continue
            clustered_event, _, groups = cluster_result
            resonances = ResonanceIdentifier.find_resonances(clustered_event)
            if not resonances:
                continue
            for res_idx, res_pdg in resonances:
                if res_idx >= len(groups):
                    continue
                decay_pdgs = []
                for lhe_idx in sorted(groups[res_idx]):
                    if lhe_idx < 1 or lhe_idx > len(event):
                        continue
                    part = event[lhe_idx - 1]
                    if int(getattr(part, "status", 0)) != 1:
                        continue
                    decay_pdgs.append(int(getattr(part, "pid", 0)))
                if len(decay_pdgs) < 2:
                    continue
                decay_key = (int(res_pdg), tuple(sorted(decay_pdgs)))
                self.fxfx_clustered_decay_processes.add(decay_key)
        self._fxfx_cluster_catalog_ready = True
        self.lhe_input.close()
        self.lhe_input = lhe_parser.EventFile(lhe_path)
        _dbg(
            f"Identified {len(self.fxfx_clustered_final_states)} unique FxFx clustered final states"
        )
        if self.fxfx_clustered_decay_processes:
            _dbg(
                f"Identified {len(self.fxfx_clustered_decay_processes)} unique FxFx clustered decay topologies for density"
            )

    # =========================================================================
    # Process Generation Methods
    # =========================================================================

    def _pdg_to_model_name(self, pdg):
        """Return the MG5 name for a PDG id, keeping track of antiparticles."""
        if not self.model:
            return None
        particle_dict = self.model.get("particle_dict") or {}
        part = particle_dict.get(pdg)
        if not part:
            part = particle_dict.get(-pdg)
        if not part:
            return None
        if pdg < 0 and not part.get("self_antipart"):
            return part.get("antiname") or ("anti_" + (part.get("name") or str(abs(pdg))))
        return part.get("name")

    def _sorted_tag_to_process(self, sorted_tag):
        if not sorted_tag or not self.model:
            return None
        incoming, outgoing = sorted_tag
        in_names = [self._pdg_to_model_name(pdg) for pdg in incoming]
        out_names = [self._pdg_to_model_name(pdg) for pdg in outgoing]
        if any(name is None for name in in_names + out_names):
            return None
        return "%s > %s" % (" ".join(in_names), " ".join(out_names))

    def _decay_pdgs_to_process(self, res_pdg, decay_pdgs):
        """Return a 1->N decay process string from PDG codes."""
        if not self.model:
            return None
        res_name = self._pdg_to_model_name(res_pdg)
        out_names = [self._pdg_to_model_name(pdg) for pdg in decay_pdgs]
        if res_name is None or any(name is None for name in out_names):
            return None
        return "%s > %s" % (res_name, " ".join(out_names))

    def _get_fxfx_extra_processes(self):
        if not (
            self._needs_fxfx_cluster_catalogue() and self.fxfx_clustered_final_states and self.model
        ):
            return []
        extra = []
        seen = set()
        for sorted_tag in sorted(self.fxfx_clustered_final_states):
            proc = self._sorted_tag_to_process(sorted_tag)
            if not proc or proc in seen:
                continue
            seen.add(proc)
            extra.append(proc)
        return extra

    def _get_fxfx_density_production_processes(self):
        if not (
            self._needs_fxfx_cluster_catalogue() and self.fxfx_clustered_final_states and self.model
        ):
            return []
        # CRITICAL: Only generate density production processes if there are DECAYED resonances.
        # For undecayed events (e.g., stable tops in ttH), there's no C matrix,
        # so density path is meaningless and we fall back to scalar at runtime anyway.
        # Generating these MEs would be wasteful and confusing.
        if not self.fxfx_clustered_decay_processes:
            _dbg(
                "[DENSITY_PROD_GEN] No decayed resonances found, skipping density production processes"
            )
            return []
        # Build set of resonance PDGs that actually have decays
        decayed_resonance_pdgs = set()
        for res_pdg, decay_pdgs in self.fxfx_clustered_decay_processes:
            decayed_resonance_pdgs.add(abs(res_pdg))
        _dbg(f"[DENSITY_PROD_GEN] Decayed resonance PDGs: {decayed_resonance_pdgs}")
        extra = []
        seen = set()
        for sorted_tag in sorted(self.fxfx_clustered_final_states):
            outgoing = sorted_tag[1]
            # Only include if the final state contains resonances that are ACTUALLY DECAYED
            if not any(abs(pdg) in decayed_resonance_pdgs for pdg in outgoing):
                continue
            proc = self._sorted_tag_to_process(sorted_tag)
            if not proc or proc in seen:
                continue
            seen.add(proc)
            extra.append(proc)
        return extra

    def _get_fxfx_density_decay_processes(self):
        _dbg("[DENSITY_DECAY_GEN] _get_fxfx_density_decay_processes called")
        _dbg(
            f"[DENSITY_DECAY_GEN]   _needs_fxfx_cluster_catalogue={self._needs_fxfx_cluster_catalogue()}"
        )
        _dbg(
            f"[DENSITY_DECAY_GEN]   fxfx_clustered_decay_processes={self.fxfx_clustered_decay_processes}"
        )
        _dbg(f"[DENSITY_DECAY_GEN]   model={bool(self.model)}")
        if not (
            self._needs_fxfx_cluster_catalogue()
            and self.fxfx_clustered_decay_processes
            and self.model
        ):
            _dbg("[DENSITY_DECAY_GEN]   -> Returning empty (condition not met)")
            return []
        extra = []
        seen = set()
        for res_pdg, decay_pdgs in sorted(self.fxfx_clustered_decay_processes):
            proc = self._decay_pdgs_to_process(res_pdg, decay_pdgs)
            _dbg(f"[DENSITY_DECAY_GEN]   res_pdg={res_pdg}, decay_pdgs={decay_pdgs} -> proc={proc}")
            if not proc or proc in seen:
                continue
            seen.add(proc)
            extra.append(proc)
        _dbg(f"[DENSITY_DECAY_GEN]   -> Returning {len(extra)} processes: {extra}")
        return extra

    def _prepare_fxfx_sudakov_inputs(self, event, sud_mod, event_to_sud):
        """Prepare FxFx Sudakov inputs (kinematics + canonical ordering)."""
        _dbg("[FXFX] _prepare_fxfx_sudakov_inputs: start")
        _dbg(f"[FXFX]   event_to_sud particles={len(event_to_sud)}")
        # Boost to CM frame if needed
        p_in_sum = lhe_parser.FourMomentum()
        for part in event_to_sud:
            if getattr(part, "status", 0) == -1:
                p_in_sum += part
        needs_boost = not (
            (abs(p_in_sum.px) < 1e-6 * p_in_sum.E)
            and (abs(p_in_sum.py) < 1e-6 * p_in_sum.E)
            and (abs(p_in_sum.pz) < 1e-6 * p_in_sum.E)
        )
        _dbg(f"[FXFX]   boost needed={needs_boost} (p_in_sum={p_in_sum})")
        if needs_boost:
            event_to_sud.boost(p_in_sum)

        # Rotate so initial-state particle is along z-axis (matches ickkw=0 path)
        initial = copy.deepcopy(event_to_sud[0])
        if not ((abs(initial.px) < 1e-6 * initial.E) and (abs(initial.py) < 1e-6 * initial.E)):
            _dbg(f"[FXFX]   rotating to z-axis (initial px={initial.px}, py={initial.py})")
            for p in event_to_sud:
                p.set_momentum(
                    lhe_parser.FourMomentum(p).rotate_to_z(prot=lhe_parser.FourMomentum(initial))
                )

        _dbg("[FXFX]   normalizing kinematics (set_final_jet_mass_to_zero, set_initial_mass_to_zero, check_kinematics_only)")
        event_to_sud.set_final_jet_mass_to_zero()
        event_to_sud.set_initial_mass_to_zero()
        event_to_sud.check_kinematics_only()

        gstr = math.sqrt(4.0 * math.pi * event.aqcd)  # G = √(4πα_s)

        mapped_tag, mapped_order = event_to_sud.get_tag_and_order()
        sorted_tag = (tuple(mapped_order[0]), tuple(sorted(mapped_order[1])))
        _dbg(f"[FXFX]   mapped_tag={mapped_tag}")
        _dbg(f"[FXFX]   mapped_order={mapped_order}")
        _dbg(f"[FXFX]   sorted_tag={sorted_tag}")

        try:
            canon = list(sud_mod.original_pdg_list_dict[sorted_tag][1])
        except Exception as exc:
            raise KeyError(f"Unknown sorted_tag for Sudakov module: {sorted_tag}") from exc
        _dbg(f"[FXFX]   canonical order={canon}")

        # Build permutation: perm[i] = position in canon where mapped_order[1][i] should go
        perm = []
        used = set()
        for r in mapped_order[1]:
            for idx, c in enumerate(canon):
                if c == r and idx not in used:
                    perm.append(idx)
                    used.add(idx)
                    break

        # Reorder final-state particles: put element i into position perm[i]
        event_to_sud_order = copy.deepcopy(event_to_sud)
        event_to_sud_order[: len(mapped_tag[0])] = event_to_sud[: len(mapped_tag[0])]
        offset = len(mapped_tag[0])
        for i, target_pos in enumerate(perm):
            event_to_sud_order[target_pos + offset] = event_to_sud[i + offset]
        _dbg(f"[FXFX]   perm={perm}")

        p_in = np.zeros(shape=(len(event_to_sud_order), 4))
        for i, el in enumerate(event_to_sud_order):
            p_in[i] = [float(el.E), float(el.px), float(el.py), float(el.pz)]

        mapped_tag2, mapped_order2 = event_to_sud_order.get_tag_and_order()
        expected_order = list(sud_mod.original_pdg_list_dict[sorted_tag][1])
        actual_order = mapped_order2[1]
        if expected_order != actual_order:
            _dbg(
                f"Momentum order mismatch: expected={expected_order}, actual={actual_order}, sorted_tag={sorted_tag}, mapped_order={mapped_order}, perm={perm}"
            )
            raise RuntimeError("Order in particle momenta does not match MG convention")
        _dbg(f"[FXFX]   momentum order VALIDATED: expected==actual={expected_order}")

        incoming_pdgs = list(mapped_order2[0])
        outgoing_pdgs = list(mapped_order2[1])
        iflist = [-1] * len(incoming_pdgs) + [1] * len(outgoing_pdgs)
        pdg_order = incoming_pdgs + outgoing_pdgs
        _dbg(f"[FXFX]   incoming_pdgs={incoming_pdgs}")
        _dbg(f"[FXFX]   outgoing_pdgs={outgoing_pdgs}")
        _dbg(f"[FXFX]   iflist={iflist}")
        _dbg(f"[FXFX]   pdg_order={pdg_order}")
        _dbg(f"[FXFX]   gstr={gstr}")
        _dbg(f"[FXFX] _prepare_fxfx_sudakov_inputs EXIT: p_in.shape={p_in.shape}, sorted_tag={sorted_tag}, iflist={iflist}, gstr={gstr:.6f}")

        return {
            "sorted_tag": sorted_tag,
            "p_in": p_in,
            "gstr": gstr,
            "iflist": iflist,
            "pdg_order": pdg_order,
            "perm": perm,  # Permutation from LHE order to canonical MG5 order (final-state only)
        }

    def _get_fxfx_ewsudpy_module(self, sud_mod, sorted_tag):
        """Return the ewsudpy module for a given sorted_tag (if available)."""
        _dbg(f"[OVERRIDE] _get_fxfx_ewsudpy_module: sorted_tag={sorted_tag}")
        try:
            result = sud_mod.pdg2ewsud_dict.get(sorted_tag)
            _dbg(f"[OVERRIDE]   resolved module: {result.__name__ if result is not None else None}")
            return result
        except Exception as exc:
            _dbg(f"[OVERRIDE]   exception fetching module: {type(exc).__name__}: {exc} -> returning None")
            return None

    def _set_ewsud_rij_ge_mw(self, ewsud_mod, disable_clamp):
        """Disable invariant clamping on the ewsudpy module; return previous value if set."""
        _dbg(f"[OVERRIDE] _set_ewsud_rij_ge_mw: module={ewsud_mod.__name__ if ewsud_mod else None}, disable_clamp={disable_clamp}")
        if ewsud_mod is None:
            _dbg("[OVERRIDE]   module is None -> returning None (no-op)")
            return None
        if hasattr(ewsud_mod, "fxfx_ignore_invariant_checks"):
            prev = getattr(ewsud_mod, "fxfx_ignore_invariant_checks")
            _dbg(f"[OVERRIDE]   path=fxfx_ignore_invariant_checks, prev={prev}, target={disable_clamp}")
            try:
                ewsud_mod.fxfx_ignore_invariant_checks = disable_clamp
                _dbg(f"[OVERRIDE]   set via attribute assignment -> returning ('fxfx', {prev})")
                return ("fxfx", prev)
            except Exception as exc:
                _dbg(f"[OVERRIDE]   attribute assignment failed ({type(exc).__name__}: {exc}), retrying with [...]=")
                try:
                    ewsud_mod.fxfx_ignore_invariant_checks[...] = disable_clamp
                    _dbg(f"[OVERRIDE]   set via array assignment -> returning ('fxfx', {prev})")
                    return ("fxfx", prev)
                except Exception as exc2:
                    _dbg(f"[OVERRIDE]   array assignment also failed ({type(exc2).__name__}: {exc2}), returning None")
                    return None
        if not hasattr(ewsud_mod, "rij_ge_mw"):
            _dbg("[OVERRIDE]   module lacks BOTH fxfx_ignore_invariant_checks AND rij_ge_mw -> returning None")
            return None
        prev = getattr(ewsud_mod, "rij_ge_mw")
        value = not disable_clamp
        _dbg(f"[OVERRIDE]   path=rij_ge_mw, prev={prev}, target={value} (= not {disable_clamp})")
        try:
            ewsud_mod.rij_ge_mw = value
            _dbg(f"[OVERRIDE]   set via attribute assignment -> returning ('rij', {prev})")
            return ("rij", prev)
        except Exception as exc:
            _dbg(f"[OVERRIDE]   attribute assignment failed ({type(exc).__name__}: {exc}), retrying with [...]=")
            try:
                ewsud_mod.rij_ge_mw[...] = value
                _dbg(f"[OVERRIDE]   set via array assignment -> returning ('rij', {prev})")
                return ("rij", prev)
            except Exception as exc2:
                _dbg(f"[OVERRIDE]   array assignment also failed ({type(exc2).__name__}: {exc2}), returning None")
                return None

    def _restore_ewsud_rij_ge_mw(self, ewsud_mod, prev):
        """Restore invariant-clamp setting on the ewsudpy module if it was set."""
        _dbg(f"[OVERRIDE] _restore_ewsud_rij_ge_mw: module={ewsud_mod.__name__ if ewsud_mod else None}, prev={prev}")
        if prev is None or ewsud_mod is None:
            _dbg("[OVERRIDE]   prev is None or module is None -> nothing to restore")
            return
        kind = "rij"
        value = prev
        if isinstance(prev, tuple) and len(prev) == 2:
            kind, value = prev
        _dbg(f"[OVERRIDE]   kind={kind}, value to restore={value}")
        if kind == "fxfx" and hasattr(ewsud_mod, "fxfx_ignore_invariant_checks"):
            try:
                ewsud_mod.fxfx_ignore_invariant_checks = value
                _dbg(f"[OVERRIDE]   restored fxfx_ignore_invariant_checks={value} via attribute assignment")
            except Exception as exc:
                _dbg(f"[OVERRIDE]   attribute restore failed ({type(exc).__name__}: {exc}), retrying with [...]=")
                try:
                    ewsud_mod.fxfx_ignore_invariant_checks[...] = value
                    _dbg(f"[OVERRIDE]   restored fxfx_ignore_invariant_checks={value} via array assignment")
                except Exception as exc2:
                    _dbg(f"[OVERRIDE]   array restore also failed ({type(exc2).__name__}: {exc2}) -- leaving as-is")
            return
        if not hasattr(ewsud_mod, "rij_ge_mw"):
            _dbg("[OVERRIDE]   module lacks rij_ge_mw -> nothing to restore")
            return
        try:
            ewsud_mod.rij_ge_mw = value
            _dbg(f"[OVERRIDE]   restored rij_ge_mw={value} via attribute assignment")
        except Exception as exc:
            _dbg(f"[OVERRIDE]   attribute restore failed ({type(exc).__name__}: {exc}), retrying with [...]=")
            try:
                ewsud_mod.rij_ge_mw[...] = value
                _dbg(f"[OVERRIDE]   restored rij_ge_mw={value} via array assignment")
            except Exception as exc2:
                _dbg(f"[OVERRIDE]   array restore also failed ({type(exc2).__name__}: {exc2}) -- leaving as-is")

    def _is_2to1_topology(self, clustered_event):
        """Check if clustered event is 2→1 (2 initial, 1 final state particle).

        Events that cluster down to 2→1 are too soft for meaningful Sudakov
        corrections - the only scale is s = M² with no additional invariants.
        """
        n_initial = sum(1 for p in clustered_event if p.status == -1)
        n_final = sum(1 for p in clustered_event if p.status == 1)
        is_2to1 = n_initial == 2 and n_final == 1
        _dbg(f"[2TO1] _is_2to1_topology: n_init={n_initial}, n_final={n_final} -> is_2to1={is_2to1}")
        return is_2to1

    def _has_small_invariants(self, p_in, iflist, mw2=None):
        """Check if any pair invariant |s_ij| < MW².

        Events with small invariants are in a soft/collinear regime where
        EW Sudakov logs are not reliable. Return True to trigger pass-through.

        Uses the same convention as Fortran Source/kin_functions.f::SumDot,
        i.e. s_ij = (p_i + sign·p_j)² = m_i² + m_j² + sign·2·(p_i·p_j).
        The mass terms matter when any leg is on-shell at a non-light scale
        (e.g. the Z resonance in Z+jets: m_Z² ≈ 1.28·M_W²) — without them
        forward-Z events with Fortran-side r_ij≈0 sneak past this veto and
        the kernel's rij_ge_mw clamp engages on a clamped log argument.
        """
        if mw2 is None:
            mw2 = MW_POLE**2

        nlegs = len(iflist)
        _dbg(f"[SMALLINV] _has_small_invariants ENTER: nlegs={nlegs}, iflist={list(iflist)}, mw2={mw2:.3g}")
        for i in range(nlegs):
            for j in range(i + 1, nlegs):
                sign = float(iflist[i] * iflist[j])
                e  = p_in[i][0] + sign * p_in[j][0]
                px = p_in[i][1] + sign * p_in[j][1]
                py = p_in[i][2] + sign * p_in[j][2]
                pz = p_in[i][3] + sign * p_in[j][3]
                sij = e * e - px * px - py * py - pz * pz
                _dbg(f"[SMALLINV]   pair({i},{j}) sign={sign:+.0f}: s_ij={sij:.3g}, |s_ij|/MW²={abs(sij)/mw2:.3f}")
                if abs(sij) < mw2:
                    _dbg(f"[SMALLINV]   HIT: |s({i},{j})|={abs(sij):.1f} < MW²={mw2:.1f} -> returning True")
                    return True
        _dbg(f"[SMALLINV] _has_small_invariants EXIT: all {nlegs*(nlegs-1)//2} pairs above MW² -> returning False")
        return False

    # Sudakov variant order — must match the indexing used by both the scalar
    # ewsudakov() Fortran call (returns res[1..5]) and the banner-label decoder
    # in reweight_interface.py.
    #
    # Three variants emitted per ξ (down from five). The Fortran kernel still
    # returns res[1..5]; we just don't propagate res[4] (both_off) and res[5]
    # (rij_ge_mw_off) into weight columns, because the Python-side SMALL_INV
    # pre-filter (_has_small_invariants) drops any event with |s_ij| < M_W²
    # BEFORE the Fortran call, so the kernel's rij_ge_mw clamp never engages.
    # With the clamp inert, res[4] is algebraically forced to equal res[3]
    # (both_off ≡ s_to_rij_off) and res[5] ≡ res[2] (rij_ge_mw_off ≡ central).
    # Confirmed in the audit: 92/92 events show this exact degeneracy.
    #
    # Stride remains 5 in base_prefix to preserve legacy weight-IDs across ξ
    # indices (ξ=0 → 20/21/22XX, ξ=1 → 25/26/27XX, ξ=2 → 30/31/32XX). IDs at
    # (20+5k+3)XX and (20+5k+4)XX are no longer emitted (used to hold the
    # redundant both_off and rij_ge_mw_off columns).
    SUDAKOV_VARIANT_NAMES = (
        "central",         # res[2]: NLL s_to_rij=ON,  rij_ge_mw=ON   (legacy 20XX)
        "s_to_rij_off",    # res[3]: NLL s_to_rij=OFF, rij_ge_mw=ON   (legacy 21XX)
        "LL",              # res[1]: leading-log only                  (legacy 22XX)
    )

    def _build_sudakov_rwgt_dict(self, event, weights):
        """Build reweight dictionary with three Sudakov variants per ξ.

        Args:
            weights: iterable of up to 3 floats, in SUDAKOV_VARIANT_NAMES order
                     (central, s_to_rij_off, LL). Missing trailing entries are
                     padded with 1.0 (no-correction). Inputs longer than 3 are
                     truncated to the first 3 (backward-compat for callers that
                     still pass 5-element lists; the dropped entries used to
                     hold both_off and rij_ge_mw_off, both redundant — see
                     SUDAKOV_VARIANT_NAMES docstring).

        Maps existing weights for ξ-index k (driven by the ξ-scan loop):
            10XX -> (20+5k+v)XX  for v ∈ [0..2], one prefix per variant.
        Stride remains 5 (not 3) to preserve legacy weight-IDs across ξ-indices:
            ξ=0 → 20/21/22XX, ξ=1 → 25/26/27XX, ξ=2 → 30/31/32XX.
        IDs (20+5k+3)XX and (20+5k+4)XX are no longer emitted.
        Legacy compat: ξ=0, v=0 → 2001 (central NLL), v=1 → 2101 (s_to_rij OFF).
        """
        rwgt_dict = copy.deepcopy(event.parse_reweight())
        _dbg(f"[WEIGHT] _build_sudakov_rwgt_dict ENTER: existing rwgt keys={list(rwgt_dict.keys()) if rwgt_dict else 'EMPTY'}, event.wgt={event.wgt}")
        if rwgt_dict == {}:
            rwgt_dict["1001"] = event.wgt
            _dbg(f"[WEIGHT]   rwgt_dict was EMPTY -> seeded with '1001'={event.wgt}")

        weights = list(weights)
        _dbg(f"[WEIGHT] Input variant weights: {weights} (len={len(weights)})")
        if len(weights) < 3:
            weights = weights + [1.0] * (3 - len(weights))
            _dbg(f"[WEIGHT]   padded to 3 with trailing 1.0s -> {weights}")
        elif len(weights) > 3:
            _dropped = weights[3:]
            weights = weights[:3]
            _dbg(f"[WEIGHT]   truncated to first 3 -> {weights} (dropped trailing entries: {_dropped})")

        xi_idx = getattr(self, "_current_xi_idx", 0)
        # Stride 5 preserves legacy weight-IDs; only first 3 slots per ξ-block emitted.
        base_prefix = 20 + 5 * xi_idx
        _dbg(f"[XI] xi_idx={xi_idx} (from self._current_xi_idx), base_prefix={base_prefix} (= 20 + 5*{xi_idx})")
        _dbg(f"[XI]   variant IDs for this ξ will use prefixes {base_prefix}XX..{base_prefix+2}XX (3 variants; slots {base_prefix+3}XX and {base_prefix+4}XX intentionally unused)")

        # 'orig' only on the first ξ-iteration; dispatcher drops it on later merges.
        rwgt_dict_new = {"orig": event.wgt} if xi_idx == 0 else {}
        _dbg(f"[WEIGHT]   xi_idx={xi_idx} -> {'INCLUDING' if xi_idx == 0 else 'SKIPPING'} 'orig' key")
        _dbg(f"[WEIGHT] Expanding {len(rwgt_dict)} source keys × 3 variants:")
        for el in rwgt_dict:
            ending = el[-2:]
            _dbg(f"[WEIGHT]   source key '{el}' (ending='{ending}', base_value={rwgt_dict[el]:.6e})")
            for variant_idx, w in enumerate(weights):
                prefix = base_prefix + variant_idx
                new_id = "%d%s" % (prefix, ending)
                new_val = rwgt_dict[el] * w
                rwgt_dict_new[new_id] = new_val
                _dbg(f"[WEIGHT]     variant {variant_idx} ({self.SUDAKOV_VARIANT_NAMES[variant_idx]}): new_id='{new_id}' = {rwgt_dict[el]:.6e} × {w:.6f} = {new_val:.6e}")
        _dbg(f"[WEIGHT] _build_sudakov_rwgt_dict EXIT: wrote {len(rwgt_dict_new)} keys")
        return rwgt_dict_new

    def _compute_ewsudakov_fxfx_reweight(self, event, sud_mod):
        """FxFx-aware scalar Sudakov reweighting."""
        _dbg("=" * 70)
        _dbg(f"[SCALAR] _compute_ewsudakov_fxfx_reweight ENTER: event_id={CURRENT_EVENT_ID}, npart={len(event)}, event.wgt={event.wgt}")
        self._init_pole_masses_from_banner()
        # PASS-THROUGH: 2→1 raw topology (qq̄ → resonance, no real radiation) —
        # nothing to cluster, no Sudakov logs valid. Must fire BEFORE the cluster
        # call so we never fall back to the legacy ickkw==0 _compute_ewsudakov_reweight
        # (which lacks the 2→1 guard and the 3-NLL-variants / ξ-scan output schema).
        n_init  = sum(1 for p in event if p.status == -1)
        n_final = sum(1 for p in event if p.status ==  1)
        _dbg(f"[SCALAR] raw event topology: n_init={n_init}, n_final={n_final} (total={len(event)})")
        if n_init == 2 and n_final == 1:
            _dbg(f"[2TO1] -> RAW 2→1 PASSTHROUGH (no clustering needed, no Sudakov), event_id={CURRENT_EVENT_ID}")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        _dbg("[SCALAR] calling _cluster_fxfx_event(event, record_tag=True)...")
        cluster_result = self._cluster_fxfx_event(event, record_tag=True)
        _dbg(f"[SCALAR]   cluster_result = {'EMPTY/None' if not cluster_result else f'OK ({len(cluster_result[0])} particles in event_to_sud)'}")
        if not cluster_result:
            _dbg("[FALLBACK] NO CLUSTERING INFO -> falling back to legacy ickkw=0 _compute_ewsudakov_reweight (NB: legacy lacks 2→1 guard and 3-variant schema)")
            return self._compute_ewsudakov_reweight(event, sud_mod)

        event_to_sud, _ = cluster_result

        # PASS-THROUGH: 2→1 topology after clustering (e.g. a single hard parton
        # merged back into the initial state). Kept as belt-and-suspenders for
        # cases where clustering reduces a multi-parton event to 2→1.
        _dbg(f"[SCALAR] checking post-cluster topology on event_to_sud ({len(event_to_sud)} parts)")
        if self._is_2to1_topology(event_to_sud):
            _dbg(f"[2TO1] -> POST-CLUSTER 2→1 PASSTHROUGH, returning weight=1 for all 3 variants (event_id={CURRENT_EVENT_ID})")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        _dbg("[SCALAR] calling _prepare_fxfx_sudakov_inputs(event, sud_mod, event_to_sud)...")
        try:
            prep = self._prepare_fxfx_sudakov_inputs(event, sud_mod, event_to_sud)
        except KeyError as exc:
            _dbg(f"[FALLBACK] _prepare_fxfx_sudakov_inputs raised KeyError({exc}) -> passthrough, returning 3×1.0")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])
        except RuntimeError:
            _dbg("[FATAL] _prepare_fxfx_sudakov_inputs: order in particle momenta does not match MG convention -> exit(3)")
            sys.exit(3)

        _dbg(f"[SCALAR] _prepare returned OK: sorted_tag={prep['sorted_tag']}, iflist={list(prep['iflist'])}, p_in.shape={prep['p_in'].shape}, gstr={prep['gstr']:.6f}")

        # PASS-THROUGH: Small invariants (event too soft for Sudakov)
        # This SMALL_INV check is what makes the Fortran rij_ge_mw clamp inert,
        # which in turn is why we no longer emit the both_off / rij_ge_mw_off
        # variants — they would be algebraically forced to coincide with
        # s_to_rij_off / central respectively.
        _dbg("[SCALAR] checking small invariants on p_in...")
        if self._has_small_invariants(prep["p_in"], prep["iflist"]):
            _dbg(f"[FALLBACK] SMALL INVARIANT (s_ij < MW²) -> passthrough, returning 3×1.0 (event_id={CURRENT_EVENT_ID})")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        # Compute Sudakov and build reweight dictionary
        rij_override_mod = None
        rij_override_prev = None
        _ignore_invs = getattr(self, "fxfx_ignore_invariant_checks", False)
        _dbg(f"[OVERRIDE] self.fxfx_ignore_invariant_checks = {_ignore_invs}")
        if _ignore_invs:
            rij_override_mod = self._get_fxfx_ewsudpy_module(sud_mod, prep["sorted_tag"])
            rij_override_prev = self._set_ewsud_rij_ge_mw(rij_override_mod, False)
            _dbg(f"[OVERRIDE]   applied: mod={rij_override_mod.__name__ if rij_override_mod else None}, prev={rij_override_prev}")
        _dbg(f"[5SUD] calling sud_mod.ewsudakov(sorted_tag={prep['sorted_tag']}, p_in.shape={prep['p_in'].shape}, gstr={prep['gstr']:.6f})")
        try:
            res = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
        finally:
            if rij_override_prev is not None:
                _dbg(f"[OVERRIDE]   restoring rij_ge_mw via prev={rij_override_prev}")
                self._restore_ewsud_rij_ge_mw(rij_override_mod, rij_override_prev)

        _dbg(f"[5SUD] Fortran returned res[0..5] = [{res[0]:.6e}, {res[1]:.6e}, {res[2]:.6e}, {res[3]:.6e}, {res[4]:.6e}, {res[5]:.6e}]")
        _dbg(f"[5SUD] Born check: |res[0]| = {abs(res[0]):.3e} (threshold 1e-30)")
        # Three NLL variants now propagated (down from five). Fortran still returns
        # res[1..5]; we drop res[4] (both_off) and res[5] (rij_ge_mw_off) because
        # the SMALL_INV pre-filter forces res[4]==res[3] and res[5]==res[2].
        # res[0]=Born, res[1]=LL, res[2]=central, res[3]=s_to_rij_off.
        if abs(res[0]) > 1e-30:
            _dbg(f"[5SUD] Born is non-zero -> normalizing 3 variants by Born={res[0]:.6e}")
            sudrats = [
                1.0 + res[2] / res[0],   # central
                1.0 + res[3] / res[0],   # s_to_rij_off
                1.0 + res[1] / res[0],   # LL
            ]
            _dbg(f"[5SUD]   sudrats[0] central      = 1 + res[2]/res[0] = 1 + {res[2]:.3e}/{res[0]:.3e} = {sudrats[0]:.6f}")
            _dbg(f"[5SUD]   sudrats[1] s_to_rij_off = 1 + res[3]/res[0] = 1 + {res[3]:.3e}/{res[0]:.3e} = {sudrats[1]:.6f}")
            _dbg(f"[5SUD]   sudrats[2] LL           = 1 + res[1]/res[0] = 1 + {res[1]:.3e}/{res[0]:.3e} = {sudrats[2]:.6f}")
            # Sanity check that the dropped variants would have been degenerate
            # (visible only when the kernel actually clamped, which it shouldn't
            # given the SMALL_INV pre-filter — log if it ever isn't).
            _dropped_both_off      = 1.0 + res[4] / res[0]
            _dropped_rij_ge_mw_off = 1.0 + res[5] / res[0]
            if abs(_dropped_both_off - sudrats[1]) > 1e-12 or abs(_dropped_rij_ge_mw_off - sudrats[0]) > 1e-12:
                _dbg(f"[5SUD]   ⚠️  DROPPED VARIANTS WERE NOT DEGENERATE: dropped_both_off={_dropped_both_off:.6f} (expected ≡ sudrats[1]={sudrats[1]:.6f}), dropped_rij_ge_mw_off={_dropped_rij_ge_mw_off:.6f} (expected ≡ sudrats[0]={sudrats[0]:.6f}) — SMALL_INV filter may have a hole or rij_ge_mw clamp engaged unexpectedly")
        else:
            _dbg(f"[5SUD] WARNING: Born |res[0]|={abs(res[0]):.3e} < 1e-30 -> setting all 3 sudrats to 1.0 (no reweight)")
            sudrats = [1.0] * 3

        # Damp the central; if it runaway, neutralize the entire variant set so
        # downstream ratios across variants stay sane.
        _dbg(f"[5SUD] runaway-damping check: |sudrats[0] central|={abs(sudrats[0]):.3f} (threshold 200)")
        if abs(sudrats[0]) > 200:
            _dbg(f"[5SUD] RUNAWAY DAMPING engaged: |central|={abs(sudrats[0]):.3f} > 200; pre-damping sudrats={sudrats}; setting all 3 to 1.0")
            sudrats = [1.0] * 3

        # Final output (matching density path format)
        _dbg("-" * 70)
        _dbg(f"[SCALAR] FINAL WEIGHTS (scalar FxFx): " +
             ", ".join(f"{n}={v:.6f}" for n, v in zip(self.SUDAKOV_VARIANT_NAMES, sudrats)))
        _dbg(f"[SCALAR] event_id={CURRENT_EVENT_ID}, event.wgt={event.wgt}")
        _dbg(f"[SCALAR] reweighted (central) = {event.wgt} × {sudrats[0]:.6f} = {event.wgt * sudrats[0]:.6e}")
        _dbg("=" * 70)
        _dbg(f"[SCALAR] _compute_ewsudakov_fxfx_reweight EXIT: passing sudrats={sudrats} to _build_sudakov_rwgt_dict")
        return self._build_sudakov_rwgt_dict(event, sudrats)

    def _compute_density_ewsudakov_reweight(self, event, sud_mod):
        """
        Density-matrix aware EW Sudakov reweighting.

        This method preserves spin correlations when events contain resonances
        (t, W, Z) that undergo decay. It applies Sudakov corrections helicity-by-
        helicity on the Born-like density matrix, then contracts with the decay
        density matrix to obtain a scalar weight.

        IMPORTANT: Only runs for FxFx (ickkw=3) and only when:
        - Event contains resonances (t, W, Z)
        - Resonances are approximately on-shell, determined through FxFx clustering

        Flow:
        1. FxFx cluster event -> Born-like event
        2. Identify resonance legs in clustered event
        3. Get Born density matrix B from matrix element M and Sudakov corrections delta_h
        4. If decays present: get decay density matrix C
        5. Apply Sudakov per helicity: M^EWSL_h = M_h * (1 + delta_h)
        6. Compute weight: w = Tr[B^EWSL * C] / Tr[B * C]

        Options (set via self attributes):
        - density_verbose: Enable verbose logging (default: False)

        Args:
            event: LHE event object
            sud_mod: Sudakov module with amplitude evaluation

        Returns:
            Dict with reweight information including 'orig' and Sudakov-weighted entries
        """
        self._init_pole_masses_from_banner()
        _dbg("=" * 70)
        _dbg(f"EVENT NUMBER: {CURRENT_EVENT_ID}")
        _dbg("ENTERING _compute_density_ewsudakov_reweight")
        _dbg("Event weight:", event.wgt)
        _dbg("Event particles:")
        for i, p in enumerate(event):
            _dbg(
                f"  [{i}] pdg={p.pid:>4}, status={p.status:>2}, E={p.E:.4f}, px={p.px:.4f}, py={p.py:.4f}, pz={p.pz:.4f}"
            )
        _dbg("-" * 70)

        # Initialize logger for diagnostics
        density_logger = getattr(self, "_density_logger", None)
        if density_logger is None:
            density_logger = DensitySudakovLogger(verbose=getattr(self, "density_verbose", False))
            self._density_logger = density_logger

        # PASS-THROUGH: 2→1 raw topology (qq̄ → resonance, no real radiation) —
        # nothing to cluster, no Sudakov logs valid. Must fire BEFORE the cluster
        # call so we never fall back to the legacy ickkw==0 _compute_ewsudakov_reweight
        # (which lacks the 2→1 guard and the 5-NLL-variants / ξ-scan output schema).
        n_init  = sum(1 for p in event if p.status == -1)
        n_final = sum(1 for p in event if p.status ==  1)
        if n_init == 2 and n_final == 1:
            _dbg("  -> RAW 2→1 TOPOLOGY (pre-cluster), returning weight=1 pass-through")
            density_logger.log_event("passthrough_2to1_raw", [], 0, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        # Step 1: Cluster event (same as FxFx path)
        _dbg("Step 1: Clustering event (FxFx)...")
        cluster_result = self._cluster_fxfx_event(event, record_tag=True, return_groups=True)
        if not cluster_result:
            # No clustering info — fall back to the FxFx scalar path (which has the
            # ξ-scan + 5-NLL-variants schema). The legacy LO entry
            # _compute_ewsudakov_reweight is only for ickkw==0 and is no longer
            # reachable from the ickkw==3 chain after this redirect.
            _dbg("  -> NO CLUSTERING INFO, falling back to FxFx scalar path")
            density_logger.log_event("fallback_no_cluster", [], 0, 1.0)
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        clustered_event, cluster_sorted_tag, groups = cluster_result

        # PASS-THROUGH: 2→1 topology (event too soft for Sudakov)
        if self._is_2to1_topology(clustered_event):
            _dbg("  -> 2→1 TOPOLOGY, returning weight=1 pass-through")
            density_logger.log_event("passthrough_2to1", [], 0, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        _dbg("  -> Clustered event particles:")
        for i, p in enumerate(clustered_event):
            _dbg(f"     [{i}] pdg={p.pid:>4}, status={p.status:>2}, E={p.E:.4f}")

        # Diagnostic: check for unphysical kinematics
        incoming = [p for p in clustered_event if p.status < 0]
        if len(incoming) >= 2:
            E1, E2 = incoming[0].E, incoming[1].E
            pz1, pz2 = incoming[0].pz, incoming[1].pz
            s_hat = (E1 + E2) ** 2 - (pz1 + pz2) ** 2  # assuming px=py=0 for beams
            _dbg(
                f"  -> KINEMATICS CHECK: E1={E1:.2f}, E2={E2:.2f}, sqrt(s_hat)={abs(s_hat)**0.5:.2f} GeV"
            )
            if E1 < 0 or E2 < 0:
                _dbg(f"*** WARNING: NEGATIVE BEAM ENERGY! E1={E1:.2f}, E2={E2:.2f} ***")
            if s_hat < 0:
                _dbg(f"*** WARNING: SPACELIKE s_hat! s_hat={s_hat:.2f} GeV^2 ***")

            # FALLBACK: Degenerate kinematics - cannot boost to CM frame
            # This happens when clustering produces a light-like final state (single on-shell photon)
            # where one beam has zero energy. Cannot compute density matrices in this frame.
            if E1 < 1e-6 or E2 < 1e-6 or s_hat < 1.0:  # s_hat < 1 GeV² is effectively massless
                _dbg(
                    "  -> DEGENERATE KINEMATICS (light-like system), falling back to scalar FxFx path"
                )
                return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # Step 2: Identify resonances in clustered event
        _dbg("Step 2: Identifying resonances...")
        resonances = ResonanceIdentifier.find_resonances(clustered_event)
        _dbg(f"  -> Found {len(resonances)} candidate resonances: {resonances}")

        # Filter out "resonances" that weren't actually clustered (singleton groups)
        # A true resonance must have decay products that merged into it
        # Singleton groups (len=1) are just stable final-state particles, not resonances
        resonances = [
            (idx, pdg) for idx, pdg in resonances if idx < len(groups) and len(groups[idx]) > 1
        ]
        _dbg(
            f"  -> After filtering non-clustered: {len(resonances)} actual resonances: {resonances}"
        )

        res_indices = [r[0] for r in resonances]
        res_pdgs = [r[1] for r in resonances]
        res_dims = [SPIN_DIMS[abs(pdg)] for pdg in res_pdgs]
        total_dim = 1
        for d in res_dims:
            total_dim *= d
        if not resonances:
            total_dim = 0
        _dbg(
            f"  -> res_indices={res_indices}, res_pdgs={res_pdgs}, res_dims={res_dims}, total_dim={total_dim}"
        )

        # FALLBACK: If no resonances found, use scalar path
        # This happens when:
        # 1. Clustering produces non-resonance mothers (photon, gluon)
        # 2. Stable particles with resonance PDGs but no clustering (e.g., ttH with stable tops)
        # The density matrix formalism requires at least one true resonance with decay products
        if not resonances:
            _dbg("  -> NO RESONANCES found, falling back to scalar FxFx path")
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # Step 2b: Define B and C from clustering history (no density evaluation yet)
        _dbg("Step 2b: Defining B/C per event from clustering history...")

        def _pdg_label(pdg):
            name = self._pdg_to_model_name(pdg)
            return name if name else str(pdg)

        def _format_lhe_list(lhe_indices):
            formatted = []
            for lhe_idx in lhe_indices:
                if lhe_idx < 1 or lhe_idx > len(event):
                    continue
                particle = event[lhe_idx - 1]
                pdg_val = int(getattr(particle, "pid", 0))
                formatted.append(f"{lhe_idx}({_pdg_label(pdg_val)})")
            return ", ".join(formatted) if formatted else "none"

        B_def = {
            "cluster_sorted_tag": cluster_sorted_tag,
            "resonances": [],
        }
        C_def = []
        for res_idx, res_pdg in resonances:
            if res_idx >= len(groups):
                msg = f"Resonance index {res_idx} (pdg={res_pdg}) out of groups range (len={len(groups)}), skipping"
                warnings.warn(msg, RuntimeWarning)
                _dbg(f"  -> {msg}")
                continue
            group_lhe = sorted(groups[res_idx])
            B_def["resonances"].append(
                {
                    "cluster_idx": res_idx,
                    "pdg": res_pdg,
                    "group_lhe": group_lhe,
                }
            )

            decay_lhe = []
            decay_pdgs = []
            for lhe_idx in group_lhe:
                if lhe_idx < 1 or lhe_idx > len(event):
                    continue
                particle = event[lhe_idx - 1]
                if int(getattr(particle, "status", 0)) != 1:
                    continue
                decay_lhe.append(lhe_idx)
                decay_pdgs.append(int(getattr(particle, "pid", 0)))
            C_def.append(
                {
                    "res_cluster_idx": res_idx,
                    "res_pdg": res_pdg,
                    "decay_lhe": decay_lhe,
                    "decay_pdgs": decay_pdgs,
                }
            )

        # Print B definition ONCE (the full production process)
        incoming_names = [
            _pdg_label(pdg) for pdg in (cluster_sorted_tag[0] if cluster_sorted_tag else [])
        ]
        outgoing_names = [
            _pdg_label(pdg) for pdg in (cluster_sorted_tag[1] if cluster_sorted_tag else [])
        ]
        incoming_str = " ".join(incoming_names) if incoming_names else "?"
        outgoing_str = " ".join(outgoing_names) if outgoing_names else "?"
        res_indices_str = ", ".join(
            [f"{r['pdg']}@idx{r['cluster_idx']}" for r in B_def["resonances"]]
        )
        _dbg(f"  -> B (production): {incoming_str} -> {outgoing_str}")
        _dbg(f"     resonance legs: [{res_indices_str}]")

        # Print each C definition (one per resonance decay)
        for c_entry in C_def:
            res_name = _pdg_label(c_entry["res_pdg"])
            decay_str = _format_lhe_list(c_entry["decay_lhe"])
            decay_names = (
                " ".join([_pdg_label(pdg) for pdg in c_entry["decay_pdgs"]])
                if c_entry["decay_pdgs"]
                else "(none)"
            )
            group_lhe = sorted(groups[c_entry["res_cluster_idx"]])
            group_str = _format_lhe_list(group_lhe)
            _dbg(f"  -> C[{res_name}] (decay): {res_name} -> {decay_names}")
            _dbg(f"     group LHE=[{group_str}], decay LHE=[{decay_str}]")

        self._density_last_C_def = C_def

        # Step 3: Prepare kinematics (same standardization as FxFx)
        _dbg("Step 3: Preparing kinematics...")
        event_to_sud = clustered_event
        try:
            prep = self._prepare_fxfx_sudakov_inputs(event, sud_mod, event_to_sud)
            _dbg(f"  -> sorted_tag: {prep['sorted_tag']}")
            _dbg(f"  -> gstr (alpha_s): {prep['gstr']}")
            _dbg(
                f"  -> p_in shape: {prep['p_in'].shape if hasattr(prep['p_in'], 'shape') else 'N/A'}"
            )
            _dbg("  -> Momenta (p_in):")
            for i, mom in enumerate(prep["p_in"]):
                _dbg(
                    f"     p[{i}] = [{mom[0]:12.4f}, {mom[1]:12.4f}, {mom[2]:12.4f}, {mom[3]:12.4f}]"
                )
            if DEBUG and "iflist" in prep and "pdg_order" in prep:

                def _sumdot(pi, pj, sign):
                    # Match Fortran Source/kin_functions.f::SumDot = (pi + sign·pj)²
                    # (= m_i² + m_j² + sign·2·(pi·pj)). Including mass terms is
                    # required for consistency with the kernel's rij_ge_mw clamp.
                    e  = pi[0] + sign * pj[0]
                    px = pi[1] + sign * pj[1]
                    py = pi[2] + sign * pj[2]
                    pz = pi[3] + sign * pj[3]
                    return e * e - px * px - py * py - pz * pz

                mw = getattr(sud_mod, "mdl_mw", None)
                if mw is None and self.model:
                    param_dict = self.model.get("parameter_dict") or {}
                    mw = param_dict.get("MW", param_dict.get("mdl_MW"))
                try:
                    mw = float(mw) if mw is not None else None
                except (TypeError, ValueError):
                    mw = None
                mw2 = mw * mw if mw is not None else None

                iflist = prep["iflist"]
                pdg_order = prep["pdg_order"]
                nlegs = len(iflist)
                min_abs = None
                _dbg("  -> Sudakov invariants (sumdot):")
                for i in range(nlegs):
                    for j in range(i, nlegs):
                        sign = float(iflist[i] * iflist[j])
                        sij = _sumdot(prep["p_in"][i], prep["p_in"][j], sign)
                        abs_sij = abs(sij)
                        if min_abs is None or abs_sij < min_abs:
                            min_abs = abs_sij
                        tag = ""
                        if mw2 is not None and abs_sij < mw2:
                            tag = f" (< MW^2={mw2:.6e})"
                        _dbg(
                            f"     ({i+1},{j+1}) pdg=({pdg_order[i]},{pdg_order[j]}) s_ij={sij:.6e}{tag}"
                        )
                if mw2 is not None and min_abs is not None:
                    _dbg(f"     min|s_ij|={min_abs:.6e}, MW^2={mw2:.6e}")
        except KeyError as exc:
            _dbg(f"  -> KeyError: {exc}, returning weight=1 pass-through (nominal)")
            density_logger.log_event("passthrough_keyerror", [], 0, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])
        except RuntimeError:
            _dbg("  -> RuntimeError: momentum order mismatch!")
            sys.exit(3)

        # PASS-THROUGH: Small invariants (event too soft for Sudakov)
        if self._has_small_invariants(prep["p_in"], prep["iflist"]):
            _dbg("  -> SMALL INVARIANT (s_ij < MW²), returning weight=1 pass-through")
            density_logger.log_event("passthrough_small_sij", resonances, total_dim, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        # Step 4: Try to get density matrix and Sudakov corrections
        weight_var = None  # Initialize variation weight (computed in density path only)
        try:
            # Standard Sudakov evaluation on clustered event
            _dbg("Step 4-5: Evaluating Sudakov corrections...")
            _dbg(f"  -> has density_sudakov attr: {hasattr(sud_mod, 'density_sudakov')}")
            use_density_path = False

            if hasattr(sud_mod, "density_sudakov") and resonances:
                try:
                    # Map resonance indices from pre-reorder to canonical MG5 order
                    # resonances contains (idx, pdg) where idx is into clustered_event BEFORE reordering
                    # prep['perm'] contains the final-state permutation applied during reordering
                    # Fortran expects indices in canonical (post-reorder) order
                    res_indices_fortran = []
                    offset = len(prep["sorted_tag"][0])  # Number of initial-state particles
                    perm = prep["perm"]
                    for idx, pdg in resonances:
                        if idx < offset:
                            # Initial-state particle (shouldn't happen for resonances, but handle it)
                            fortran_idx = idx + 1
                        else:
                            # Final-state: apply permutation to map to canonical order
                            final_idx = idx - offset
                            canonical_final_idx = perm[final_idx]
                            canonical_idx = canonical_final_idx + offset
                            fortran_idx = canonical_idx + 1  # 1-based indexing for Fortran
                        res_indices_fortran.append(fortran_idx)

                    _dbg("  -> Calling density_sudakov with:")
                    _dbg(f"     sorted_tag: {prep['sorted_tag']}")
                    _dbg(f"     res_indices_fortran: {res_indices_fortran}")
                    _dbg(f"     res_dims: {res_dims}")

                    # Call density matrix Sudakov (FxFx override can disable MW^2 clamping)
                    rij_override_mod = None
                    rij_override_prev = None
                    if getattr(self, "fxfx_ignore_invariant_checks", False):
                        rij_override_mod = self._get_fxfx_ewsudpy_module(
                            sud_mod, prep["sorted_tag"]
                        )
                        rij_override_prev = self._set_ewsud_rij_ge_mw(rij_override_mod, False)
                        if rij_override_prev is not None:
                            _dbg("  -> FXFX override: rij_ge_mw disabled")
                        else:
                            _dbg("  -> FXFX override requested but rij_ge_mw not settable")
                    try:
                        density_result = sud_mod.density_sudakov(
                            prep["sorted_tag"],
                            prep["p_in"],
                            prep["gstr"],
                            res_indices_fortran,
                            res_dims,
                        )
                    finally:
                        if rij_override_prev is not None:
                            self._restore_ewsud_rij_ge_mw(rij_override_mod, rij_override_prev)

                    _dbg("  -> Fortran returned (from density_matrix_wrapper.f):")
                    _dbg(f"     density_delta (per-hel Sudakov): {density_result['density_delta']}")
                    _dbg(
                        f"     born_diag (for helicity order check): {density_result['born_diag']}"
                    )
                    _dbg(
                        f"     results[0] (scalar_born, IDEN-averaged): {density_result['results'][0] if len(density_result['results']) > 0 else 'N/A'}"
                    )
                    _dbg(
                        f"     total_dim: {density_result['total_dim']}, nres: {density_result['nres']}"
                    )

                    # =========================================================
                    # Compute B matrix via DensityHelper (Python/standalone)
                    # =========================================================
                    # The Fortran wrapper returns zeros for density_born (Phase 1 removed).
                    # B matrix is computed via DensityHelper using the standalone
                    # density_prod modules which have working GET_DENSITY.
                    # =========================================================
                    total_dim = density_result["total_dim"]
                    B = None

                    if getattr(self, "density_prod_id_to_path", None) and getattr(
                        self, "density_prod_f2pylib", None
                    ):
                        _dbg("  -> Computing B matrix via DensityHelper (standalone density_prod)")
                        try:
                            helper = DensityHelper(
                                self, self.density_prod_id_to_path, self.density_prod_f2pylib
                            )
                            B_matrix = helper.compute_density_matrix(
                                clustered_event, res_pdgs, res_dims
                            )
                            if B_matrix is not None:
                                B = DensityMatrix.from_full_matrix(B_matrix, res_pdgs)
                                _dbg(f"     DensityHelper returned B with shape {B_matrix.shape}")
                            else:
                                _dbg("     DensityHelper returned None for B matrix")
                        except Exception as exc:
                            _dbg(f"     DensityHelper FAILED: {exc}")
                            import traceback

                            _dbg(f"     Traceback: {traceback.format_exc()}")

                    if B is None:
                        raise ValueError(
                            "DensityHelper failed to compute B matrix. "
                            "Check that rw_me_density_prod modules are generated and compiled."
                        )
                    delta_raw = density_result["density_delta"]
                    # Variation delta (s_to_rij=False) for systematic uncertainty
                    delta_var_raw = density_result.get("density_delta_var", None)
                    # Per-helicity Born from Fortran (summed over spectators, /IDEN)
                    born_diag_fortran_raw = density_result["born_diag"]

                    # Reorder delta from Fortran convention to Python convention.
                    # Fortran uses little-endian ascending (-1,+1); Python uses
                    # big-endian descending (+1,-1) via itertools.product.
                    _hel_perm = _fortran_to_python_hel_permutation(res_dims)
                    delta = np.array([delta_raw[_hel_perm[i]] for i in range(len(delta_raw))])
                    delta_var = None
                    if delta_var_raw is not None:
                        delta_var = np.array(
                            [delta_var_raw[_hel_perm[i]] for i in range(len(delta_var_raw))]
                        )
                    # Reorder per-helicity Born from Fortran convention to Python convention
                    born_diag_fortran = np.array(
                        [born_diag_fortran_raw[_hel_perm[i]] for i in range(len(born_diag_fortran_raw))]
                    )
                    _dbg("=" * 70)
                    _dbg("===== DENSITY PATH =====")
                    _dbg("=" * 70)
                    _dbg("  -> DENSITY PATH active")
                    _dbg(f"     B matrix shape: {B.matrix.shape}, trace: {B.trace():.6e}")
                    scalar_born = (
                        density_result["results"][0] if len(density_result["results"]) > 0 else None
                    )
                    _dbg("     B matrix (from DensityHelper):")
                    # Use scientific notation to show small values properly
                    with np.printoptions(
                        precision=4,
                        linewidth=120,
                        formatter={"complex_kind": lambda x: f"{x.real:+.4e}{x.imag:+.4e}j"},
                    ):
                        for row_idx, row in enumerate(B.matrix):
                            _dbg(f"       [{row_idx}] {row}")

                    # B diagonal for delta normalization
                    born_diag_from_B = np.real(np.diag(B.matrix))
                    _dbg(f"     B diagonal (for delta normalization): {born_diag_from_B}")
                    _dbg(f"     delta_unnorm (Fortran raw ordering): {delta_raw}")
                    _dbg(f"     delta_unnorm (reordered to Python): {delta}")
                    _dbg(f"     hel_perm (Fortran->Python): {_hel_perm}")

                    # =========================================================
                    # Normalize Fortran delta using per-helicity Born from Fortran
                    # =========================================================
                    # Fortran returns UNNORMALIZED delta*B values and per-helicity Born.
                    # Both are /IDEN, so IDEN cancels in the ratio:
                    #   delta_normalized = density_delta[h] / born_diag[h]
                    # =========================================================
                    _dbg(f"     born_diag_fortran (per-hel Born, /IDEN): {born_diag_fortran}")
                    # =========================================================
                    # NORMALIZATION CROSS-CHECK
                    # =========================================================
                    # born_diag_fortran[h] = per-resonance-helicity Born summed
                    #   over spectator helicities, from sborn_onehel() in the
                    #   NLO Sudakov wrapper. Each element is |M|^2/IDEN for the
                    #   subset of full helicities matching resonance-helicity h.
                    # scalar_born = total Born from sborn(), also /IDEN, summed
                    #   over ALL helicities.
                    # Since the per-helicity Born partitions the total Born:
                    #   sum_h(born_diag_fortran[h]) == scalar_born
                    # A ratio != 1.0 would indicate a bug in the helicity
                    # partitioning or in the resonance-helicity selection filter.
                    # =========================================================
                    if scalar_born is not None and scalar_born != 0:
                        born_sum = sum(born_diag_fortran)
                        born_ratio = born_sum / scalar_born
                        _dbg(f"     NORMALIZATION CROSS-CHECK (born_diag_fortran vs scalar_born):")
                        _dbg(f"       sum(born_diag_fortran) = {born_sum:.6e}  "
                             f"(per-hel Born summed over resonance helicities)")
                        _dbg(f"       scalar_born            = {scalar_born:.6e}  "
                             f"(total Born from sborn(), same Fortran module)")
                        _dbg(f"       ratio = {born_ratio:.6f}  (expect 1.0: "
                             f"both from same sborn_onehel, IDEN cancels)")
                    if np.any(np.abs(delta) > 1e-30):
                        _dbg("     Normalizing delta using per-helicity Born from Fortran (IDEN-free)")
                        delta_normalized = np.zeros_like(delta)
                        for h in range(len(delta)):
                            if abs(born_diag_fortran[h]) > 1e-30:
                                delta_normalized[h] = delta[h] / born_diag_fortran[h]
                            else:
                                delta_normalized[h] = 0.0
                        delta = delta_normalized
                        _dbg(f"     Normalized delta: {delta}")
                        # Also normalize delta_var if present
                        if delta_var is not None and np.any(np.abs(delta_var) > 1e-30):
                            delta_var_normalized = np.zeros_like(delta_var)
                            for h in range(len(delta_var)):
                                if abs(born_diag_fortran[h]) > 1e-30:
                                    delta_var_normalized[h] = delta_var[h] / born_diag_fortran[h]
                                else:
                                    delta_var_normalized[h] = 0.0
                            delta_var = delta_var_normalized
                            _dbg(f"     Normalized delta_var: {delta_var}")
                    else:
                        _dbg("     NOTE: delta is all zeros from Fortran - no normalization needed")

                    # Build full delta matrix for debug visualization
                    dim = len(delta)
                    delta_full = np.zeros((dim, dim), dtype=complex)
                    for h in range(dim):
                        for hp in range(dim):
                            delta_full[h, hp] = delta[h] + np.conj(delta[hp])
                    _dbg(f"     delta_full ({dim}x{dim} matrix):")
                    with np.printoptions(
                        precision=4,
                        linewidth=120,
                        formatter={"complex_kind": lambda x: f"{x.real:+.4e}{x.imag:+.4e}j"},
                    ):
                        for row_idx, row in enumerate(delta_full):
                            _dbg(f"       [{row_idx}] {row}")
                    if not np.any(np.abs(delta) > 1e-30):
                        _dbg(
                            "     NOTE: delta is all zeros; verify density_matrix_wrapper and PDG list wiring."
                        )

                    # Get decay density matrix
                    C = self._get_decay_density_matrix(event, clustered_event, resonances)
                    if C is not None:
                        _dbg("=" * 70)
                        _dbg("     DECAY DENSITY MATRIX C")
                        _dbg("=" * 70)
                        _dbg(f"     C matrix shape: {C.matrix.shape}, trace: {C.trace():.6e}")
                        _dbg("     C matrix (complex):")
                        with np.printoptions(
                            precision=4,
                            linewidth=120,
                            formatter={"complex_kind": lambda x: f"{x.real:+.4e}{x.imag:+.4e}j"},
                        ):
                            for row_idx, row in enumerate(C.matrix):
                                _dbg(f"       [{row_idx}] {row}")
                        _dbg(f"     C diagonal: {np.diag(C.matrix)}")
                        _dbg(
                            f"     C off-diagonal norm: {np.linalg.norm(C.matrix - np.diag(np.diag(C.matrix))):.6e}"
                        )

                        # Detailed B and C analysis
                        _dbg("=" * 70)
                        _dbg("     DENSITY MATRIX ANALYSIS")
                        _dbg("=" * 70)
                        _dbg(f"     B diagonal elements: {np.real(np.diag(B.matrix))}")
                        _dbg(f"     C diagonal elements: {np.real(np.diag(C.matrix))}")
                        _dbg(f"     Tr[B] = {B.trace():.6e}")
                        _dbg(f"     Tr[C] = {C.trace():.6e}")

                        # Compute contractions explicitly
                        tr_BC = B.contract(C)
                        _dbg(f"     Tr[B*C] = {tr_BC:.6e}")

                        # B_EWSL construction
                        B_ewsl = B.apply_sudakov(delta)
                        _dbg(f"     B_EWSL diagonal: {np.real(np.diag(B_ewsl.matrix))}")
                        tr_BewslC = B_ewsl.contract(C)
                        _dbg(f"     Tr[B_EWSL*C] = {tr_BewslC:.6e}")

                        # Compute density weight (central)
                        weight = compute_density_weight(B, C, delta)
                        _dbg(f"     density_weight (central): {weight:.6f}")

                        # Compute variation weight (s_to_rij=False) if available
                        weight_var = None
                        if delta_var is not None:
                            weight_var = compute_density_weight(B, C, delta_var)
                            _dbg(f"     density_weight (s_to_rij=False): {weight_var:.6f}")

                        use_density_path = True
                    else:
                        _dbg("     C matrix: None (no decay info), falling back to scalar")
                        use_density_path = False

                except Exception as e:
                    warnings.warn(
                        f"Density matrix computation failed, falling back to scalar: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _dbg(f"  -> Density matrix call FAILED: {e}")
                    use_density_path = False

            # Fallback to scalar path if density matrix not available or failed
            if not use_density_path:
                _dbg("=" * 70)
                _dbg("===== SCALAR PATH (fallback) =====")
                _dbg("=" * 70)
                res = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
                born, sud_weak = res[0], res[2]
                weight = 1.0 + sud_weak / born if abs(born) > 1e-30 else 1.0
                weight_var = None  # No variation for scalar fallback
                _dbg(f"     born={born:.6e}, sud_weak={sud_weak:.6e}")
                _dbg(f"     weight = 1 + sud_weak/born = {weight:.6f}")

            # Log event processing
            density_logger.log_event(
                mode="density" if use_density_path else "scalar",
                resonances=resonances,
                dimension=total_dim,
                weight=weight,
            )

        except Exception as e:
            _dbg(f"  -> EXCEPTION in density Sudakov: {e}")
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # Damp overly large ratios
        if abs(weight) > 200:
            _dbg(f"  -> WARNING: weight {weight:.6f} > 200, capping to 1.0")
            weight = 1.0
        if weight_var is not None and abs(weight_var) > 200:
            _dbg(f"  -> WARNING: weight_var {weight_var:.6f} > 200, capping to 1.0")
            weight_var = 1.0

        # Compute the LL (leading-log only) variant via the scalar
        # ewsudakov() Fortran call. density_sudakov() returns only central
        # (NLL, s_to_rij=ON) and s_to_rij_off; it does not implement an
        # LL-only mode. Earlier versions padded the LL slot with 1.0, which
        # silently turned the downstream `ewsl_ll` curves into pure baseline
        # for every Z+jets event (the density path activates for every event
        # with a reconstructed on-shell Z, so ~100% of the sample). The LL
        # approximation is kinematics-only and doesn't depend on the helicity
        # density structure that distinguishes density vs scalar paths — so
        # the scalar Fortran result is the right value to use here.
        # On the scalar-fallback path (use_density_path=False) we already
        # called ewsudakov() at the fallback site above; we re-call here for
        # code uniformity rather than threading a result across branches.
        # Cost is microseconds per event; the alternative is a second control
        # path inside _build_sudakov_rwgt_dict.
        weight_ll = 1.0
        try:
            res_for_ll = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
            if abs(res_for_ll[0]) > 1e-30:
                weight_ll = 1.0 + res_for_ll[1] / res_for_ll[0]
                _dbg(
                    f"  -> LL extraction: born={res_for_ll[0]:.6e}, "
                    f"LL_delta={res_for_ll[1]:.6e}, weight_ll={weight_ll:.6f}"
                )
            else:
                _dbg(
                    f"  -> LL extraction: |born|={abs(res_for_ll[0]):.3e} < 1e-30, "
                    f"weight_ll=1.0 (no reweight)"
                )
        except Exception as exc:  # noqa: BLE001 — robustness: never let LL failure abort the event
            _dbg(f"  -> WARNING: LL extraction via scalar ewsudakov() failed ({exc}); using 1.0")
            weight_ll = 1.0
        if abs(weight_ll) > 200:
            _dbg(f"  -> WARNING: weight_ll {weight_ll:.6f} > 200, capping to 1.0")
            weight_ll = 1.0

        # Build output dictionary
        _dbg("-" * 70)
        _dbg(f"FINAL WEIGHT (central): {weight:.6f}")
        if weight_var is not None:
            _dbg(f"FINAL WEIGHT (s_to_rij=False): {weight_var:.6f}")
        _dbg(f"FINAL WEIGHT (LL only): {weight_ll:.6f}")
        _dbg(f"Event original weight: {event.wgt}")
        _dbg(f"Reweighted = {event.wgt} * {weight:.6f} = {event.wgt * weight:.6e}")
        _dbg("=" * 70)
        # Density path computes central (s_to_rij ON) and s_to_rij OFF via
        # density_sudakov(); LL is now computed by an additional scalar
        # ewsudakov() call above (it's a kinematics-only approximation).
        # The retired variants v=3 (both_off) and v=4 (rij_ge_mw_off) remain
        # algebraically degenerate with v=1 and v=0 after the small_inv
        # pre-filter — _build_sudakov_rwgt_dict drops them by truncating to
        # the first 3 entries (see SUDAKOV_VARIANT_NAMES at L3560).
        wv = weight_var if weight_var is not None else 1.0
        return self._build_sudakov_rwgt_dict(event, [weight, wv, weight_ll])

    # =========================================================================
    # Decay Density Matrix Methods
    # =========================================================================

    def _get_decay_density_matrix(self, full_event, clustered_event, resonances):
        """
        Build decay density matrix C from event information.

        Args:
            full_event: Original LHE event with all particles
            clustered_event: Event after FxFx clustering
            resonances: List of (index, pdg) for resonances

        Returns:
            DensityMatrix for decays, or None if no decays present
        """
        _dbg("  -> _get_decay_density_matrix: Building decay matrix C")
        c_defs = getattr(self, "_density_last_C_def", None)
        if not c_defs:
            _dbg("     No clustering-based C_def available, returning None")
            return None
        _dbg(f"     C_def entries={len(c_defs)}")

        decay_matrices = []

        for entry in c_defs:
            res_pdg = entry["res_pdg"]
            res_idx = entry["res_cluster_idx"]
            decay_lhe = entry["decay_lhe"]
            _dbg(f"     resonance pdg={res_pdg}, cluster_idx={res_idx}, decay_lhe={decay_lhe}")
            if not decay_lhe:
                _dbg(f"     Resonance {res_pdg} has no decay LHE indices, skipping")
                continue

            decay_event = self._build_decay_event_from_cluster(
                full_event, clustered_event, res_idx, res_pdg, decay_lhe
            )
            try:
                decay_tag, decay_order = decay_event.get_tag_and_order()
                _dbg(f"     Decay event tag/order: {decay_tag}, {decay_order}")
            except Exception as exc:
                _dbg(f"     Failed to get decay tag/order: {exc}")
            dim = SPIN_DIMS.get(abs(res_pdg), 2)
            C_r = self._compute_density_matrix_from_event(
                decay_event, [res_pdg], [dim], label=f"C:{res_pdg}"
            )
            if C_r is None:
                _dbg(f"     Decay density matrix failed for {res_pdg}, returning None")
                return None

            _dbg(f"     Decay density matrix for {res_pdg}: trace={C_r.trace():.6e}")
            decay_matrices.append(C_r)

        if not decay_matrices:
            _dbg("     No decay matrices computed, returning None")
            return None

        if len(decay_matrices) == 1:
            return decay_matrices[0]

        result = build_decay_tensor_product(decay_matrices)
        _dbg(f"     Tensor product shape: {result.matrix.shape}, trace: {result.trace():.6e}")
        return result

    def _build_decay_event_from_cluster(
        self, full_event, clustered_event, res_idx, res_pdg, decay_lhe
    ):
        """
        Build a minimal decay event (resonance -> clustered decay products).

        IMPORTANT: The decay products from the original event have momenta that sum
        to the original (off-shell) resonance momentum, not the on-shell momentum
        from FKS clustering. We transform the decay products to match the on-shell
        resonance while preserving decay angles in the resonance rest frame.

        This ensures the decay matrix C is evaluated in the same helicity basis
        as the production matrix B.
        """
        decay_event = lhe_parser.Event()
        decay_event.wgt = getattr(full_event, "wgt", 0.0)
        decay_event.aqcd = getattr(full_event, "aqcd", 0.0)

        # Get on-shell resonance from clustered event
        res_part = clustered_event[res_idx]
        onshell_res_p4 = FourMomentum([res_part.E, res_part.px, res_part.py, res_part.pz])
        _dbg(f"     [decay_build] res_idx={res_idx}, res_pdg={res_pdg}")
        _dbg(
            f"     [decay_build] on-shell res_p4=({res_part.E:.6e},{res_part.px:.6e},{res_part.py:.6e},{res_part.pz:.6e})"
        )

        # Collect original decay product momenta and metadata
        original_momenta = []
        decay_pids = []
        decay_masses = []
        sum_E = sum_px = sum_py = sum_pz = 0.0

        for lhe_idx in decay_lhe:
            if lhe_idx < 1 or lhe_idx > len(full_event):
                continue
            src = full_event[lhe_idx - 1]
            p4 = FourMomentum([float(src.E), float(src.px), float(src.py), float(src.pz)])
            original_momenta.append(p4)
            decay_pids.append(int(getattr(src, "pid", 0)))
            decay_masses.append(float(getattr(src, "mass", 0.0)))
            sum_E += p4.E
            sum_px += p4.px
            sum_py += p4.py
            sum_pz += p4.pz

        # Original resonance momentum (sum of original decay products)
        original_res_p4 = FourMomentum([sum_E, sum_px, sum_py, sum_pz])
        original_m2 = sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2
        original_m = math.sqrt(max(0, original_m2))
        _dbg(
            f"     [decay_build] original res_p4=({sum_E:.6e},{sum_px:.6e},{sum_py:.6e},{sum_pz:.6e})"
        )
        _dbg(f"     [decay_build] original invariant mass = {original_m:.4f} GeV")
        _dbg(
            f"     [decay_build] delta (before transform)=({res_part.E - sum_E:.6e},{res_part.px - sum_px:.6e},{res_part.py - sum_py:.6e},{res_part.pz - sum_pz:.6e})"
        )

        # Transform decay products to match on-shell resonance
        transformed_momenta = _transform_decay_products_to_onshell(
            original_momenta, original_res_p4, onshell_res_p4
        )

        # Build resonance particle (initial state for decay)
        res_particle = lhe_parser.Particle(event=decay_event)
        res_particle.pid = int(res_pdg)
        res_particle.status = -1
        res_particle.px = float(res_part.px)
        res_particle.py = float(res_part.py)
        res_particle.pz = float(res_part.pz)
        res_particle.E = float(res_part.E)
        res_particle.mass = float(getattr(res_part, "mass", 0.0))
        decay_event.append(res_particle)

        # Build transformed decay products
        sum_E_new = sum_px_new = sum_py_new = sum_pz_new = 0.0
        for i, p4 in enumerate(transformed_momenta):
            p = lhe_parser.Particle(event=decay_event)
            p.pid = decay_pids[i]
            p.status = 1
            p.px = float(p4.px)
            p.py = float(p4.py)
            p.pz = float(p4.pz)
            p.E = float(p4.E)
            p.mass = decay_masses[i]
            decay_event.append(p)
            sum_E_new += p4.E
            sum_px_new += p4.px
            sum_py_new += p4.py
            sum_pz_new += p4.pz

        _dbg(
            f"     [decay_build] transformed sum p4=({sum_E_new:.6e},{sum_px_new:.6e},{sum_py_new:.6e},{sum_pz_new:.6e})"
        )
        _dbg(
            f"     [decay_build] delta (after transform)=({res_part.E - sum_E_new:.6e},{res_part.px - sum_px_new:.6e},{res_part.py - sum_py_new:.6e},{res_part.pz - sum_pz_new:.6e})"
        )

        # Compute masses for verification
        res_m2 = res_part.E**2 - res_part.px**2 - res_part.py**2 - res_part.pz**2
        res_m = math.sqrt(max(0, res_m2))
        sum_m2 = sum_E_new**2 - sum_px_new**2 - sum_py_new**2 - sum_pz_new**2
        sum_m = math.sqrt(max(0, sum_m2))
        _dbg("     [decay_build] MASS CHECK:")
        _dbg(f"       Resonance mass (on-shell): {res_m:.4f} GeV")
        _dbg(f"       Sum of decay products mass: {sum_m:.4f} GeV")
        _dbg(f"       Original invariant mass: {original_m:.4f} GeV")
        _dbg(f"       Pole mass target: {_get_mother_mass(res_pdg):.4f} GeV")
        _dbg("     [decay_build] DECAY EVENT STRUCTURE:")
        for i, p in enumerate(decay_event):
            p_mag = math.sqrt(p.px**2 + p.py**2 + p.pz**2)
            m2 = p.E**2 - p.px**2 - p.py**2 - p.pz**2
            m = math.sqrt(max(0, m2))
            _dbg(
                f"       [{i}] pid={p.pid}, status={p.status}, E={p.E:.4f}, |p|={p_mag:.4f}, m={m:.4f}"
            )
        return decay_event

    def _compute_density_matrix_from_event(self, event, target_pdgs, res_dims, label="density"):
        """Compute density matrix from event using DensityHelper.

        This method uses DensityHelper to call GET_DENSITY from the standalone
        density modules (rw_me_density_prod or rw_me_density_decay).

        Args:
            event: LHE event object
            target_pdgs: PDG codes of particles in the density matrix
            res_dims: Spin dimensions (2=fermion, 3=vector)
            label: Debug label for logging

        Returns:
            DensityMatrix object or None if computation fails
        """
        _dbg(f"     [{label}] _compute_density_matrix_from_event: start")
        _dbg(f"     [{label}] target_pdgs={target_pdgs}, res_dims={res_dims}")
        _dbg(f"     [{label}] event particles={len(event)}")

        # Determine if this is decay (nincoming=1) or production (nincoming=2)
        try:
            tag_raw, order = event.get_tag_and_order()
            nincoming = len(tag_raw[0])
        except Exception:
            nincoming = 2  # Default to production

        _dbg(f"     [{label}] nincoming={nincoming}")

        # Select appropriate density module paths
        if nincoming == 1:
            # Decay density matrix (C matrix)
            if not getattr(self, "density_decay_id_to_path", None):
                _dbg(f"     [{label}] No density_decay_id_to_path available")
                return None
            id_to_path = self.density_decay_id_to_path
            f2pylib = self.density_decay_f2pylib
            _dbg(f"     [{label}] Using density_decay maps (keys={list(id_to_path.keys())[:3]}...)")
        else:
            # Production density matrix (B matrix)
            if not getattr(self, "density_prod_id_to_path", None):
                _dbg(f"     [{label}] No density_prod_id_to_path available")
                return None
            id_to_path = self.density_prod_id_to_path
            f2pylib = self.density_prod_f2pylib
            _dbg(f"     [{label}] Using density_prod maps (keys={list(id_to_path.keys())[:3]}...)")

        # Create DensityHelper with the appropriate module paths
        helper = DensityHelper(self, id_to_path, f2pylib)

        try:
            # Compute the density matrix
            density_matrix = helper.compute_density_matrix(event, target_pdgs, res_dims)

            if density_matrix is None:
                _dbg(f"     [{label}] DensityHelper returned None")
                return None

            _dbg(f"     [{label}] DensityHelper returned matrix shape: {density_matrix.shape}")
            return DensityMatrix.from_full_matrix(density_matrix, target_pdgs)

        except Exception as exc:
            _dbg(f"     [{label}] Density matrix computation FAILED: {exc}")
            import traceback

            _dbg(f"     [{label}] Traceback: {traceback.format_exc()}")
            return None
