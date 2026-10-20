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
import contextlib
import logging
import math
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from typing import List, Optional, Tuple
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr

import madgraph.various.lhe_parser as lhe_parser
import numpy as np

FourMomentum = lhe_parser.FourMomentum
logger = logging.getLogger("madgraph.various.fxfx_ewsudakov")


class MissingSudakovProcessError(Exception):
    """The compiled Sudakov library does not contain a required process."""

DEBUG = False

# Event counter set by reweight_interface.py during the event loop (1-based).
CURRENT_EVENT_ID = 0

_DIM_OVERFLOW_WARNED = False

# Fixed dimensions in the generated density dispatcher/Fortran wrapper.
MAX_DENSITY_DIM = 36
MAX_DENSITY_RESONANCES = 10
MAX_DENSITY_ALLOW_HEL = 300


def limit_sudakov_factors(factors, accounting_key=None):
    """Set all factors to unity when the central factor exceeds 200 in magnitude."""
    factors = list(factors)
    if abs(factors[0]) > 200:
        _dbg(f"Neutralizing all Sudakov factors: |central|={abs(factors[0]):g} exceeds 200")
        if accounting_key is not None:
            _count_fks_event(accounting_key)
        return [1.0] * len(factors)
    return factors


def density_dimensions_supported(total_dim, nres):
    """Return whether a density call fits every fixed generated buffer."""
    return (
        total_dim <= MAX_DENSITY_DIM
        and nres <= MAX_DENSITY_RESONANCES
        and total_dim * nres <= MAX_DENSITY_ALLOW_HEL
    )

# =============================================================================
# FKS-mapping accounting and fallback policy
# =============================================================================
# The FSR mapping can fail to reach the pole mass (see _fks_fsr_mapping): each
# outcome is identified by a method string. Successful methods leave the mother
# on its pole mass (within 1e-4 M, or 1e-3 √q² for a massless mother); every
# other method returns an off-pole mother.
FKS_SUCCESS_METHODS = frozenset(
    [
        "fks_fsr_massive_onshell",
        "fks_fsr_no_spectators_beam_scaled",
        "fks_isr_cm_boost",
    ]
)

# Per-launch counters, reset by fks_reset_accounting and reported by
# fks_accounting_summary.
FKS_METHOD_COUNTS = Counter()  # method string -> number of clusterings
FKS_QUALITY_COUNTS = Counter()  # coarse quality events (aborts, damping, ...)

# Policy for events whose clustering history contains any FSR fallback:
#   "accept"      : keep the event, compute weights on the off-pole Born-like
#                   object (mass label is set to the actual invariant mass so
#                   downstream kinematics checks remain consistent);
#   "unit_weight" : force all Sudakov variants to 1 for such events.
# An event whose clustering fails gets unit weights under either policy.
FKS_FALLBACK_POLICY = "accept"

# Mapping counters use the first clustering pass of each main-loop event.
# Event outcome counters cover the full driver chain and every xi evaluation.
# Both exclude the catalogue prescan (CURRENT_EVENT_ID == 0) and are per worker.
_FKS_COUNT_ACTIVE = False
_FKS_COUNTED_UPTO = 0
_FKS_QUALITY_EVENT_IDS = {}


def _count_fks_event(name):
    """Count an outcome once for the current main-loop event."""
    if CURRENT_EVENT_ID > 0 and _FKS_QUALITY_EVENT_IDS.get(name) != CURRENT_EVENT_ID:
        FKS_QUALITY_COUNTS[name] += 1
        _FKS_QUALITY_EVENT_IDS[name] = CURRENT_EVENT_ID


def fks_reset_accounting():
    """Reset the per-launch counters (called at the start of each event loop)."""
    global _FKS_COUNT_ACTIVE, _FKS_COUNTED_UPTO
    FKS_METHOD_COUNTS.clear()
    FKS_QUALITY_COUNTS.clear()
    _FKS_QUALITY_EVENT_IDS.clear()
    _FKS_COUNT_ACTIVE = False
    _FKS_COUNTED_UPTO = 0


def fks_accounting_summary():
    """Multi-line summary of FKS-mapping outcomes accumulated in this launch.

    Mapping counts use the first clustering pass; outcomes count each event
    once across driver fallbacks and xi evaluations. All counts are per worker.
    """
    total = sum(FKS_METHOD_COUNTS.values())
    n_fallback = sum(
        c for m, c in FKS_METHOD_COUNTS.items() if m not in FKS_SUCCESS_METHODS
    )
    lines = [
        "FKS mapping outcomes (once per event, per worker): "
        "%d clusterings, %d fallback(s), policy=%s"
        % (total, n_fallback, FKS_FALLBACK_POLICY)
    ]
    for method, count in sorted(FKS_METHOD_COUNTS.items()):
        marker = "" if method in FKS_SUCCESS_METHODS else "  <-- fallback"
        lines.append("    %-42s %d%s" % (method, count, marker))
    for name, count in sorted(FKS_QUALITY_COUNTS.items()):
        lines.append("    %-42s %d" % (name, count))
    return "\n".join(lines)


def _dbg(*args, **kwargs):
    """Debug output controlled by DEBUG flag."""
    if DEBUG:
        print("[DEBUG]", *args, flush=True, **kwargs)


# Pole masses in GeV: defaults, overridden from the param_card by
# _init_pole_masses_from_banner().
MW_POLE = 80.385
MZ_POLE = 91.1876
MT_POLE = 173.3
MH_POLE = 125.0
_POLE_MASSES_INITIALIZED = False


@dataclass
class ForcedClusterStep:
    """One forced clustering step reversing a MadSpin decay (e.g. W+ -> l+ nu).

    LHE indices are 1-based; scale is the children's invariant mass in GeV;
    depth is the distance to the leaves (inner decays have lower depth and
    are clustered first).
    """

    children_lhe_idx: List[int]
    mother_lhe_idx: int
    mother_pdg: int
    scale: float
    depth: int


FXFX_CLUSTER_STEP_RE = re.compile(r'step_(\d+)="([^"]+)"')
FXFX_CLUSTER_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# Flavour families that may be relabelled to restore charge conservation.
FXFX_FLAVOR_FAMILIES = {
    "quark": set(range(1, 7)),
    "lepton": {11, 13, 15},
    "neutrino": {12, 14, 16},
}


# =============================================================================
# Clustering helpers
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
    """1-based LHE index of a mother (Particle: event_id + 1, number: int, None: 0)."""
    if mother is None:
        return 0
    if hasattr(mother, "event_id"):
        return mother.event_id + 1
    elif isinstance(mother, (int, float)):
        return int(mother)
    return 0


def _fix_mother_sign_global(model, buff_event, pos_i, pos_j, mother_pdg, tol=1e-6):
    """Fix the mother PDG sign/flavour by global charge conservation.

    Tries mother_pdg, then -mother_pdg, then the other flavours of its family
    (the LHE coin flip can pick a different flavour than the FxFx clustering).
    Returns (pdg, changed); raises RuntimeError if no candidate conserves charge.
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

    original_family = _flavor_family(mother_pdg)
    if original_family:
        family_pdgs = list(FXFX_FLAVOR_FAMILIES.get(original_family, set()))
        candidates = []
        for pdg in family_pdgs:
            candidates.extend([pdg, -pdg])

        for cand_pdg in candidates:
            if cand_pdg == mother_pdg or cand_pdg == flipped:
                continue
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
# FxFx clustering kinematics (FKS mappings)
# =============================================================================


# Debug flag for verbose clustering output
_cluster_dbg = _dbg  # Alias for clustering debug output


def _m2(p):
    """Invariant mass squared."""
    return p.E**2 - p.px**2 - p.py**2 - p.pz**2


def _get_mother_mass(mother_pid):
    """Pole mass of a mother for t, Z, W, H; 0 for every other PDG."""
    abs_pid = abs(mother_pid)
    if abs_pid == 6:
        return MT_POLE
    elif abs_pid == 23:
        return MZ_POLE
    elif abs_pid == 24:
        return MW_POLE
    elif abs_pid == 25:
        return MH_POLE
    else:
        return 0.0


def _boost_along_direction(p, beta, direction):
    """Boost p with velocity beta along the unit vector n̂ = direction.

    This is the boost B of FKS Sec. 5.2 (Eqs. 5.25-5.27):
        E' = γ(E - β p·n̂),   p'_∥ = γ(p·n̂ - β E),
    i.e. to a frame moving with +β along n̂. For a momentum |p| n̂ and
    β = |p|/E, +β goes to its rest frame and -β back from it.
    """
    if abs(beta) < 1e-12:
        return FourMomentum(p)

    nx, ny, nz = direction
    gamma = 1.0 / math.sqrt(1.0 - beta**2)

    p_dot_n = p.px * nx + p.py * ny + p.pz * nz

    E_new = gamma * (p.E - beta * p_dot_n)
    p_parallel_new = gamma * (p_dot_n - beta * p.E)

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


def _apply_fxfx_frame_transform(momentum, frame_transform):
    """Apply ordered clustering and event-preparation Lorentz operations.

    Event.boost uses the HELAS boost convention: it negates the spatial part
    of the total incoming momentum before calling FourMomentum.boost.  Mirror
    that implementation exactly so decay products can follow the same Lorentz
    map as the clustered production event.
    """
    transformed = FourMomentum(momentum)
    for operation, reference in frame_transform or []:
        if operation == "boost_to_cm":
            pboost = FourMomentum(reference)
            pboost.px *= -1.0
            pboost.py *= -1.0
            pboost.pz *= -1.0
            transformed = transformed.boost(pboost)
        elif operation == "rotate_to_z":
            transformed = transformed.rotate_to_z(prot=FourMomentum(reference))
        elif operation == "pt_boost":
            transformed = transformed.pt_boost(pboost=FourMomentum(reference))
        elif operation == "boost_along_direction":
            beta, direction = reference
            transformed = _boost_along_direction(transformed, beta, direction)
        elif operation == "boost_from_rest":
            transformed = transformed.boost(FourMomentum(reference))
        elif operation == "rescale":
            # Rescaling: on a lightlike momentum, the boost along its own direction.
            transformed = FourMomentum(
                [
                    reference * transformed.E,
                    reference * transformed.px,
                    reference * transformed.py,
                    reference * transformed.pz,
                ]
            )
        else:
            raise ValueError("Unknown FxFx frame operation: %s" % operation)
    return transformed


def _transform_decay_products_to_onshell(
    decay_momenta, original_res_p4, onshell_res_p4, frame_transform=None,
):
    """Project decay products onto the resonance momentum onshell_res_p4.

    original_res_p4 is the sum of decay_momenta. frame_transform, if given,
    is the ordered list of Lorentz operations recorded while preparing the
    clustered production event; it is applied to the decay system first. The
    products are then boosted to the rest frame of original_res_p4, rebuilt
    with the mass of onshell_res_p4 keeping their rest-frame directions, and
    boosted so that they sum to onshell_res_p4. If M <= m1 + m2 (two-body),
    or the source has neither mass nor momentum, the transported inputs are
    returned unprojected. Returns momenta in input order.
    """
    if not decay_momenta:
        return []

    # Express the input decay system and the target parent in the same frame.
    if frame_transform:
        decay_momenta = [
            _apply_fxfx_frame_transform(p, frame_transform) for p in decay_momenta
        ]
        original_res_p4 = _apply_fxfx_frame_transform(
            original_res_p4, frame_transform
        )

    n_particles = len(decay_momenta)

    orig_m2 = _m2(original_res_p4)
    orig_m = math.sqrt(max(0, orig_m2))
    orig_p = math.sqrt(original_res_p4.px**2 + original_res_p4.py**2 + original_res_p4.pz**2)

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

    # Vanishing source momentum and mass: nothing to project.
    if orig_p < 1e-10 and orig_m < 1e-10:
        _cluster_dbg("  [decay_transform] WARNING: degenerate original resonance, returning unchanged")
        return [FourMomentum(p) for p in decay_momenta]

    if orig_p > 1e-10:
        orig_dir = (
            original_res_p4.px / orig_p,
            original_res_p4.py / orig_p,
            original_res_p4.pz / orig_p,
        )
        orig_beta = orig_p / original_res_p4.E
    else:
        orig_dir = (0, 0, 1)
        orig_beta = 0

    # +β = |p|/E along p̂: to the source rest frame.
    rest_momenta = []
    for p in decay_momenta:
        p_rest = _boost_along_direction(p, orig_beta, orig_dir)
        rest_momenta.append(p_rest)

    _cluster_dbg(f"  [decay_transform] Boosted {len(rest_momenta)} particles to rest frame")

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

    if len(decay_momenta) == 2:
        # Two-body: keep m1, m2 and the direction of product 1; back to back
        # with |p| = λ^{1/2}(M², m1², m2²) / (2M) in the on-shell rest frame.
        m1, m2 = rest_masses[0], rest_masses[1]
        if onshell_m > m1 + m2:
            lambda_val = (onshell_m2 - (m1 + m2) ** 2) * (onshell_m2 - (m1 - m2) ** 2)
            p_new = math.sqrt(max(0, lambda_val)) / (2 * onshell_m)
        else:
            # Below threshold M <= m1 + m2: return the (transported) inputs.
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
        # Multi-body: rescale all components uniformly so that sum E = M; this
        # keeps the rest-frame directions but not the product masses.
        sum_E_rest = sum(p.E for p in rest_momenta)
        scale = onshell_m / sum_E_rest if sum_E_rest > 0 else 1.0
        new_rest_momenta = []
        for p_rest in rest_momenta:
            p_new = FourMomentum(
                [p_rest.E * scale, p_rest.px * scale, p_rest.py * scale, p_rest.pz * scale]
            )
            new_rest_momenta.append(p_new)
        _cluster_dbg(f"  [decay_transform] multi-body: scale={scale:.6f}")

    if onshell_p > 1e-10:
        onshell_dir = (
            onshell_res_p4.px / onshell_p,
            onshell_res_p4.py / onshell_p,
            onshell_res_p4.pz / onshell_p,
        )
        onshell_beta = onshell_p / onshell_res_p4.E
    else:
        onshell_dir = (0, 0, 1)
        onshell_beta = 0

    # -β: from the on-shell rest frame to the frame of onshell_res_p4.
    lab_momenta = []
    for p_rest in new_rest_momenta:
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


def _transport_decay_products(products, frame_transform):
    """Transport leaf momenta while retaining their original LHE labels."""
    return {
        label: _apply_fxfx_frame_transform(momentum, frame_transform)
        for label, momentum in products.items()
    }


def _project_cluster_decay_products(children, momenta, mother):
    """Project two children onto a parent and transport each child's leaves.

    A timelike parent's two-body projection preserves child masses and
    rest-frame directions. Lightlike jets carry collinear constituents with
    the input energy fractions.
    Each composite child carries its leaves through the same canonical boosts,
    preserving the invariant masses of its internal decay systems.
    """
    source_parent = momenta[0] + momenta[1]
    if _m2(mother) <= 1e-12 * mother.E**2:
        leaves = dict(children[0])
        leaves.update(children[1])
        total_energy = sum(p.E for p in leaves.values())
        return {
            label: FourMomentum([component * p.E / total_energy
                                 for component in (mother.E, mother.px, mother.py, mother.pz)])
            for label, p in leaves.items()
        }
    projected = _transform_decay_products_to_onshell(
        momenta, source_parent, mother,
    )
    products = {}
    for leaves, source, target in zip(children, momenta, projected):
        if len(leaves) == 1:
            products[next(iter(leaves))] = FourMomentum(target)
        elif leaves and _m2(source) <= 1e-12 * source.E**2:
            products.update({
                label: FourMomentum([component * p.E / source.E
                                     for component in (target.E, target.px, target.py, target.pz)])
                for label, p in leaves.items()
            })
        elif leaves:
            products.update(_transport_decay_products(leaves, [
                ("boost_to_cm", source), ("boost_from_rest", target),
            ]))
    return products


def _fks_isr_mapping(fks_i, fks_j, moth, orig_momenta):
    """FKS initial-state mapping (FKS paper Sec. 5.1).

    The radiation k = orig_momenta[fks_j] is removed from the initial state,
    q̄ = q - k, √ŝ' = √(q̄²); the Born-like beams are massless along z with
    E = √ŝ'/2 in the q̄ rest frame. This changes x± and leaves the final state
    unchanged up to the transverse boost that puts q̄ along z.
    orig_momenta is [beam0, beam1, final state...]; fks_i and moth are unused.

    Returns {'beams': [b0, b1], 'final_state': {idx: p} without the
    radiation, 'sqrt_s_new', 'method', 'final_state_transform': the ordered
    Lorentz operations applied to the surviving final state}.
    """
    p_initial = orig_momenta[0] + orig_momenta[1]
    p_radiation = orig_momenta[fks_j]

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

    # A nearly lightlike q̄ has no usable rest frame: below √(q̄²) = 0.1 GeV,
    # γ > E/0.1 GeV. The caller stops clustering on this "light-like" error.
    if sqrt_s < 0.1:
        raise ValueError(f"ISR mapping: light-like p_clustered (√s={sqrt_s:.4f} GeV), cannot boost")

    result = {
        "final_state": {},
        "sqrt_s_new": sqrt_s,
        "method": "fks_isr_cm_boost",
        "final_state_transform": [],
    }

    p_clustered_pt = math.sqrt(p_clustered.px**2 + p_clustered.py**2)
    needs_pt_boost = p_clustered_pt > 1e-10

    # The pT boost is not undone: the output frame has q̄ along z, as a
    # Born-level LHE event needs (beams along z). The Sudakov computation
    # boosts to the CM frame and is unaffected.
    if needs_pt_boost:
        result["final_state_transform"] = [("pt_boost", FourMomentum(p_clustered))]
        after_pt = [p.pt_boost(pboost=p_clustered) for p in orig_momenta]
        p_clustered_after_pt = p_clustered.pt_boost(pboost=p_clustered)
    else:
        after_pt = list(orig_momenta)
        p_clustered_after_pt = FourMomentum(p_clustered)

    cm_momenta = [p.zboost(pboost=p_clustered_after_pt) for p in after_pt]

    # Beams in the q̄ rest frame, keeping the orientation of beam 0.
    E_beam_cm = sqrt_s / 2.0
    if cm_momenta[0].pz >= 0:
        beam0_new = FourMomentum([E_beam_cm, 0, 0, E_beam_cm])
        beam1_new = FourMomentum([E_beam_cm, 0, 0, -E_beam_cm])
    else:
        beam0_new = FourMomentum([E_beam_cm, 0, 0, -E_beam_cm])
        beam1_new = FourMomentum([E_beam_cm, 0, 0, E_beam_cm])

    # Longitudinal boost back to the frame of q̄ (after the pT boost).
    for ip, p_cm in enumerate(cm_momenta):
        if ip <= 1:
            continue
        if ip == fks_j:
            continue
        p_lab = p_cm.zboost_inv(pboost=p_clustered_after_pt)
        result["final_state"][ip] = p_lab

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
    """FKS final-state mapping (FKS paper Sec. 5.2) with an on-shell mother.

    The spectators (recoil system k_rec) are boosted along n̂ = k⃗_rec/|k⃗_rec|
    so that the mother q - B k_rec satisfies (q - B k_rec)² = M² instead of
    the FKS lightlike condition (q - B k_rec)² = 0. With q_n = q⃗·n̂,
    p_n = |k⃗_rec|,
        A = q_E E_rec - q_n p_n
        B = q_n E_rec - q_E p_n
        R = (q² + m²_rec - M²)/2 + q⃗_perp·k⃗_perp    (k⃗_perp ≡ 0 for the
                                                       total recoil system)
    the on-shell condition reads γ(A + β B) = R; squaring gives
        (B² + R²) β² + 2AB β + (A² - R²) = 0,
        β = (-AB ± R √(R² + B² - A²)) / (B² + R²).

    Spurious roots: A ± B = (q_E ± q_n)(E_rec ∓ p_n) ≥ 0 for timelike q and
    physical k_rec, hence A ≥ |B| and γ(A + βB) > 0 for all |β| < 1. For R > 0
    every root with |β| < 1 satisfies the unsquared (+R) equation. For R ≤ 0
    the target is unreachable (q·k̄ ≥ 0 for future-pointing momenta), yet the
    squared quadratic still has |β| < 1 roots, all on the spurious (-R)
    branch: R ≤ 0 is rejected ("fks_fsr_unreachable_target"), and the chosen
    root is checked against the unsquared constraint ("fks_fsr_spurious_root";
    reached in floating point for soft recoil systems, E_rec up to about 1 GeV
    at TeV-scale q²).

    A recoil system is lightlike when |m²_rec| < 1e-6 GeV² and
    |m²_rec| ≤ 1e-6 E²_rec. The boost along n̂ then acts on it as the rescaling
    k̄ = r k_rec with r = √((1-β)/(1+β)), applied directly (frame operation
    "rescale"); r solves (q - r k_rec)² = M² for the input momenta.

    A mapping is reported as on-shell only if the mother is within 1e-4 M of a
    massive pole (ten times inside the 0.1% mass gate of
    lhe_parser.Event.check_kinematics_only), or within 1e-3 √q² of zero mass;
    otherwise the mother is the naive pair momentum (method "fks_fsr_inexact").

    fks_i becomes the mother (PDG moth[0]['id']) and fks_j is removed;
    orig_momenta is [beam0, beam1, final state...]. Returns {'mother',
    'beams' (changed only without spectators), 'spectators': {idx: p},
    'spectator_transform': the ordered frame operations applied to the
    spectators, 'method'}.
    """
    mother_pid = abs(moth[0]["id"])
    mother_mass = _get_mother_mass(mother_pid)
    M2 = mother_mass**2

    q = orig_momenta[0] + orig_momenta[1]
    q2 = _m2(q)

    spectator_indices = [
        idx for idx in range(len(orig_momenta)) if idx >= 2 and idx != fks_i and idx != fks_j
    ]

    result = {
        "beams": [FourMomentum(orig_momenta[0]), FourMomentum(orig_momenta[1])],
        "spectators": {},
        "spectator_transform": [],
    }

    # Compute invariant mass of the clustering pair (before any reshuffling)
    mother_naive = orig_momenta[fks_i] + orig_momenta[fks_j]
    invariant_m2 = _m2(mother_naive)
    invariant_m = math.sqrt(max(0, invariant_m2))

    if DEBUG:
        _cluster_dbg(f"  FKS FSR: pid={mother_pid}, M={mother_mass:.4f}, m_inv={invariant_m:.4f}, spectators={spectator_indices}")

    if len(spectator_indices) == 0:
        # No spectators: the mother is q. Rescaling both beams by M/√(q²)
        # puts it on the pole and keeps its rapidity (E1/E2 fixed).
        actual_m = math.sqrt(max(0, q2))

        # Clustering stops before a 2 -> 1 configuration with a massless
        # mother (γ, g).
        if mother_mass < 1e-6:
            raise ValueError(
                f"FSR no-spectators: massless mother (M={mother_mass:.2e}) should have been "
                "prevented by clustering stop logic. This is a bug."
            )

        if actual_m > 1e-10:
            scale = mother_mass / actual_m
        else:
            scale = 1.0

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

        q_new = beam0_new + beam1_new
        result["mother"] = q_new
        result["method"] = "fks_fsr_no_spectators_beam_scaled"

        if DEBUG:
            mother_m_check = math.sqrt(max(0, _m2(q_new)))
            _cluster_dbg(f"  No-spectator: scale={scale:.6f}, m_check={mother_m_check:.4f}, Δ={abs(mother_m_check - mother_mass):.2e}")

        return result

    k_rec = FourMomentum()
    for idx in spectator_indices:
        k_rec += orig_momenta[idx]

    k_rec_E = k_rec.E
    k_rec_p = math.sqrt(k_rec.px**2 + k_rec.py**2 + k_rec.pz**2)
    k_rec_m2 = _m2(k_rec)

    _cluster_dbg(f"  k_rec: E={k_rec_E:.4f}, |p|={k_rec_p:.4f}, m²={k_rec_m2:.4f}")

    if k_rec_p < 1e-10:
        # |k⃗_rec| = 0 defines no boost direction: the mother is q - k_rec.
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

    # A naive pair mass below M is not rejected: a boost with βB < 0 (β > 0 in
    # the q rest frame, where B = -√q² |k⃗_rec|) lowers q·k̄ and raises the
    # mother mass; the quadratic decides whether such a |β| < 1 exists.
    n_rec = (k_rec.px / k_rec_p, k_rec.py / k_rec_p, k_rec.pz / k_rec_p)

    # Lightlike test: m²_rec negligible both absolutely and relative to E²_rec
    # (a soft wide-angle system is massive).
    rescale = None
    m2_negligible = abs(k_rec_m2) < 1e-6 and abs(k_rec_m2) <= 1e-6 * k_rec_E**2
    if m2_negligible:
        # For lightlike k̄ ∥ n̂: q·k̄ = k̄_E (q_E - q⃗·n̂) = (q² - M²)/2 fixes
        # k̄_E, which must be positive.
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

        # The boost of a lightlike k along n̂ is k̄ = r k with
        # r = (1 - β) / √(1 - β²) = √((1-β)/(1+β)), i.e. β = (1 - r²) / (1 + r²).
        # r solves (q - r k_rec)² = M² for the input momenta, whose m²_rec is
        # not exactly zero: m²_rec r² - 2 (q·k_rec) r + (q² - M²) = 0, on the
        # root continuous with the lightlike limit r = (q² - M²) / (2 q·k_rec).
        q_dot_k = q.E * k_rec.E - q.px * k_rec.px - q.py * k_rec.py - q.pz * k_rec.pz
        r = (q2 - M2) / (
            q_dot_k + math.sqrt(max(0.0, q_dot_k**2 - k_rec_m2 * (q2 - M2)))
        )
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

        # Applied as the rescaling k̄ = r k, which keeps the mother on its pole
        # for r far from 1 (soft recoil), where 1 - β² cancels.
        rescale = r

    else:
        # Massive recoil system: solve γ(A + βB) = R for β (see docstring).
        _cluster_dbg(f"  Massive spectator (m²={k_rec_m2:.4f}), using general approach")
        q_dot_n = q.px * n_rec[0] + q.py * n_rec[1] + q.pz * n_rec[2]
        p_n = k_rec.px * n_rec[0] + k_rec.py * n_rec[1] + k_rec.pz * n_rec[2]

        k_perp_x = k_rec.px - p_n * n_rec[0]
        k_perp_y = k_rec.py - p_n * n_rec[1]
        k_perp_z = k_rec.pz - p_n * n_rec[2]
        q_dot_k_perp = q.px * k_perp_x + q.py * k_perp_y + q.pz * k_perp_z

        target = (q2 + k_rec_m2 - M2) / 2.0

        A = q.E * k_rec_E - q_dot_n * p_n
        B = q_dot_n * k_rec_E - q.E * p_n
        rhs = target + q_dot_k_perp

        _cluster_dbg(f"  A={A:.4f}, B={B:.4f}, rhs={rhs:.4f}")

        # rhs = R ≤ 0 (q² + m²_rec ≤ M²: too light for an on-shell mother plus
        # this recoil system) is unreachable; the squared quadratic would still
        # return roots of γ(A + βB) = -R (see docstring).
        if rhs <= 0:
            _cluster_dbg(
                f"  WARNING: unreachable on-shell target (rhs={rhs:.4f} <= 0, "
                f"q²+m²_rec={q2 + k_rec_m2:.2f} < M²={M2:.2f})"
            )
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_unreachable_target"
            return result

        if abs(B) < 1e-8 * max(abs(A), abs(rhs), 1.0):
            _cluster_dbg(
                "  WARNING: B ≈ 0 (spectators collinear with beam), using simplified formula"
            )
            # B ≈ 0: the quadratic degenerates; γ A = R gives β² = 1 - A²/R².
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
            # The sign of β does not enter γ A = R; β ≥ 0 is taken.
            beta = math.sqrt(beta_sq) if beta_sq > 0 else 0.0
            _cluster_dbg(f"  Collinear limit: β = {beta:.6f}")
        else:
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

            # The smallest |β| among the roots with |β| < 1.
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

            # The root must solve the unsquared constraint γ(A + βB) = R.
            _gamma_chk = 1.0 / math.sqrt(1.0 - beta**2)
            if abs(_gamma_chk * (A + beta * B) - rhs) > 1e-6 * max(1.0, abs(rhs)):
                _cluster_dbg(
                    f"  WARNING: spurious quadratic root (γ(A+βB)={_gamma_chk * (A + beta * B):.4f} != rhs={rhs:.4f})"
                )
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_spurious_root"
                return result

    # A boost needs |β| < 1; the lightlike rescaling needs only r > 0.
    if rescale is None and abs(beta) >= 1.0:
        _cluster_dbg(f"  WARNING: |β| = {abs(beta):.6f} >= 1 (superluminal)")
        result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
        for idx in spectator_indices:
            result["spectators"][idx] = FourMomentum(orig_momenta[idx])
        result["method"] = "fks_fsr_superluminal"
        return result

    if DEBUG:
        _cluster_dbg(f"  [FKS FSR] β={beta:.6f}, n̂=({n_rec[0]:.4f}, {n_rec[1]:.4f}, {n_rec[2]:.4f})")

    if rescale is None:
        result["spectator_transform"] = [("boost_along_direction", (beta, n_rec))]
        k_rec_boosted = _boost_along_direction(k_rec, beta, n_rec)

        for idx in spectator_indices:
            p_orig = orig_momenta[idx]
            k_bar_i = _boost_along_direction(p_orig, beta, n_rec)
            result["spectators"][idx] = k_bar_i
    else:
        result["spectator_transform"] = [("rescale", rescale)]
        k_rec_boosted = _apply_fxfx_frame_transform(
            k_rec, result["spectator_transform"]
        )

        for idx in spectator_indices:
            result["spectators"][idx] = _apply_fxfx_frame_transform(
                orig_momenta[idx], result["spectator_transform"]
            )

    mother = FourMomentum(
        [
            q.E - k_rec_boosted.E,
            q.px - k_rec_boosted.px,
            q.py - k_rec_boosted.py,
            q.pz - k_rec_boosted.pz,
        ]
    )

    if mother_mass > 0.0:
        off_pole = abs(math.sqrt(max(0.0, _m2(mother))) - mother_mass)
        off_shell = off_pole > 1e-4 * mother_mass
    else:
        off_shell = abs(_m2(mother)) > 1e-6 * q2
    if off_shell:
        result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
        for idx in spectator_indices:
            result["spectators"][idx] = FourMomentum(orig_momenta[idx])
        result["spectator_transform"] = []
        result["method"] = "fks_fsr_inexact"
        return result

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


def _fks_fsr_mapping_cm(fks_i, fks_j, moth, orig_momenta):
    """_fks_fsr_mapping applied in the partonic CM frame and boosted back.

    The FSR map boosts the recoil system along its own direction, so it depends
    on the frame; it is defined with q at rest (FKS Sec. 5.2). The beams lie
    along z, so the CM boost is along z. Returns the mapping in the input frame,
    with 'spectator_transform' = to_cm + map + from_cm, together with 'to_cm',
    'from_cm', 'momenta_cm' and 'mother_cm' for projecting decay products in
    the CM frame.
    """
    q = orig_momenta[0] + orig_momenta[1]
    beta_cm = q.pz / q.E if q.E > 1e-10 else 0.0
    z_hat = (0, 0, 1)
    momenta_cm = [_boost_along_direction(p, beta_cm, z_hat) for p in orig_momenta]
    result = _fks_fsr_mapping(fks_i, fks_j, moth, momenta_cm)
    to_cm = [("boost_along_direction", (beta_cm, z_hat))]
    from_cm = [("boost_along_direction", (-beta_cm, z_hat))]
    return {
        "mother": _boost_along_direction(result["mother"], -beta_cm, z_hat),
        "beams": [_boost_along_direction(p, -beta_cm, z_hat) for p in result["beams"]],
        "spectators": {
            idx: _boost_along_direction(p, -beta_cm, z_hat)
            for idx, p in result["spectators"].items()
        },
        "spectator_transform": to_cm + result["spectator_transform"] + from_cm,
        "method": result["method"],
        "to_cm": to_cm,
        "from_cm": from_cm,
        "momenta_cm": momenta_cm,
        "mother_cm": result["mother"],
    }


def fxfx_merge_particles_kinematics(event, i, j, moth):
    """Cluster particles i and j of event in place with an FKS mapping.

    ISR (FKS Sec. 5.1) if the surviving particle is a beam, otherwise FSR
    (Sec. 5.2: spectators boosted along k⃗_rec, mother = q - B k_rec, with β
    fixed by the on-shell condition instead of the lightlike one; see
    _fks_fsr_mapping). moth[0] = {'number': 1-based index of the surviving
    particle, 'id': mother PDG}. Decay products attached as
    event._fxfx_decay_products follow the same Lorentz maps. Returns the FKS
    method string (FKS_SUCCESS_METHODS lists the successful outcomes: on-pole
    FSR and ISR).
    """
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

    orig_momenta = [FourMomentum(p) for p in event]
    decay_products = getattr(event, "_fxfx_decay_products", None)

    cluster_type = "ISR" if fks_i <= 1 else "FSR"
    _cluster_dbg(
        f"FxFx merge: {cluster_type} clustering [{fks_i}]+[{fks_j}] -> mother (pid={moth[0]['id']})"
    )

    if fks_i <= 1:
        isr_result = _fks_isr_mapping(fks_i, fks_j, moth, orig_momenta)
        method = isr_result.get("method", "fks_isr_unknown")

        if decay_products is not None:
            for idx in isr_result["final_state"]:
                decay_products[idx] = _transport_decay_products(
                    decay_products[idx], isr_result["final_state_transform"],
                )

        event[0].set_momentum(isr_result["beams"][0])
        event[1].set_momentum(isr_result["beams"][1])

        for idx, p in isr_result["final_state"].items():
            event[idx].set_momentum(p)

        event[fks_i].pid = moth[0]["id"]

    else:
        fks_result = _fks_fsr_mapping_cm(fks_i, fks_j, moth, orig_momenta)
        method = fks_result.get("method", "fks_fsr_unknown")

        if decay_products is not None:
            children = [
                _transport_decay_products(decay_products[idx], fks_result["to_cm"])
                for idx in (fks_i, fks_j)
            ]
            projected = _project_cluster_decay_products(
                children,
                [fks_result["momenta_cm"][fks_i], fks_result["momenta_cm"][fks_j]],
                fks_result["mother_cm"],
            )
            decay_products[fks_i] = _transport_decay_products(
                projected, fks_result["from_cm"]
            )
            for idx in fks_result["spectators"]:
                decay_products[idx] = _transport_decay_products(
                    decay_products[idx], fks_result["spectator_transform"],
                )

        event[0].set_momentum(fks_result["beams"][0])
        event[1].set_momentum(fks_result["beams"][1])
        event[fks_i].set_momentum(fks_result["mother"])
        event[fks_i].pid = moth[0]["id"]

        # The mass label is the pole mass only if the mapping reached the pole,
        # else the actual invariant mass: Event.check_kinematics_only (called
        # from _prepare_fxfx_sudakov_inputs) raises on a >0.1% mismatch
        # between the label and the four-momentum.
        if method in FKS_SUCCESS_METHODS:
            event[fks_i].mass = _get_mother_mass(abs(moth[0]["id"]))
        else:
            event[fks_i].mass = math.sqrt(max(0.0, _m2(fks_result["mother"])))

        for idx, p in fks_result["spectators"].items():
            event[idx].set_momentum(p)

    event.pop(to_remove)
    if decay_products is not None:
        decay_products.pop(to_remove)

    # Counted on the first clustering pass of each event only. Every mapping,
    # fallbacks included, conserves momentum, so a conservation violation
    # signals an error, not a fallback.
    if _FKS_COUNT_ACTIVE:
        FKS_METHOD_COUNTS[method] += 1
        p_in_sum = FourMomentum(event[0]) + FourMomentum(event[1])
        p_out_sum = FourMomentum()
        for idx in range(2, len(event)):
            p_out_sum += FourMomentum(event[idx])
        cons_scale = max(abs(p_in_sum.E), 1.0)
        if (
            max(
                abs(p_in_sum.E - p_out_sum.E),
                abs(p_in_sum.px - p_out_sum.px),
                abs(p_in_sum.py - p_out_sum.py),
                abs(p_in_sum.pz - p_out_sum.pz),
            )
            > 1e-6 * cons_scale
        ):
            FKS_QUALITY_COUNTS["merge_momentum_conservation_violation"] += 1

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

    return method


# =============================================================================
# Density matrices
# =============================================================================

# Helicity-space dimension per resonance, keyed by abs(PDG).
SPIN_DIMS = {
    6: 2,
    23: 3,
    24: 3,
}

# Resonances of the density-matrix path (particle and antiparticle codes).
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
        k = ih_fortran
        hels = []
        for d in res_dims:
            idx = k % d
            k //= d
            if d == 2:
                hels.append(2 * idx - 1)
            elif d == 3:
                hels.append(idx - 1)
            else:
                hels.append(idx - (d - 1) // 2)

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


def unpack_density_triangle(packed, len_tri=None):
    """Full density matrix from the packed triangle returned by GET_DENSITY.

    GET_ALL_INTER stores the upper triangle row by row,
    packed[i*n + k - i*(i+1)//2] = B[i,k] = M_i * conj(M_k) for i <= k
    (summed over colours and over the helicities not in the matrix).
    The upper triangle (diagonal included) is kept as stored and the lower
    triangle is filled with B[k,i] = conj(B[i,k]), giving B[h,h'] = M_h M*_h'
    as DensityMatrix.apply_sudakov and DensityMatrix.contract use it.
    Density_functions.DensityMatrixObservables.square_matrix returns the
    transpose of this matrix.

    Only the first len_tri entries are used (all if None). Returns a complex
    (n, n) array; raises ValueError unless they form a 1D line of n(n+1)/2
    entries with n >= 1.
    """
    line = np.asarray(packed, dtype=np.complex128)
    if line.ndim == 1 and len_tri is not None:
        line = line[:len_tri]
    n = (int(math.sqrt(8 * line.size + 1)) - 1) // 2
    if line.ndim != 1 or n < 1 or n * (n + 1) // 2 != line.size:
        raise ValueError(
            f"Packed density triangle of shape {line.shape} is not a 1D line of n(n+1)/2 entries"
        )
    matrix = np.zeros((n, n), dtype=np.complex128)
    matrix[np.triu_indices(n)] = line
    lower = np.tril_indices(n, -1)
    matrix[lower] = np.conj(matrix.T[lower])
    return matrix


# =============================================================================
# DensityHelper: density matrices through the DensityInterface machinery
# =============================================================================


class DensityHelper:
    """Density-matrix evaluation without a full DensityInterface.

    Copies the needed attributes from a ReweightInterface, uses the
    density-specific id_to_path and f2pylib (standalone exports), and provides
    the methods that DensityInterface.calculate_matrix_element needs:

        helper = DensityHelper(reweight_interface, density_prod_id_to_path, density_prod_f2pylib)
        matrix = helper.compute_density_matrix(event, target_pdgs=[6, -6], res_dims=[2, 2])
    """

    def __init__(self, reweight_interface, id_to_path, f2pylib):
        """id_to_path, f2pylib: the density-specific maps (e.g. density_prod_*)."""
        self.keep_ordering = getattr(reweight_interface, "keep_ordering", True)
        self.helicity_reweighting = getattr(reweight_interface, "helicity_reweighting", False)
        self.options = getattr(
            reweight_interface, "options", {"identical_particle_in_prod_and_decay": "average"}
        )
        self.me_dir = getattr(reweight_interface, "me_dir", os.getcwd())

        self.id_to_path = id_to_path
        self.f2pylib = f2pylib

        # No boost, rotation or initial-state symmetrisation for EW Sudakov.
        self.momenta_boost = ([0], "", [])
        self.helicity_direction = ([0], "", [])
        self.axis_referential = [0]
        self.symmetrise_initial_state = False

        # Set per call in compute_density_matrix.
        self.particle_in_density_matrix = None
        self.allowed_helicities = None
        self.number_changing_helicities = None
        self.number_combinations = None
        self.spins = None
        self.flag_particle_in_density_matrix = True
        self._initialized_modules = set()

    @staticmethod
    def invert_momenta(p):
        """(n, 4) momenta of get_momenta() -> Fortran-ordered (4, n) array.

        This is the F2PY layout of P(0:3, NEXTERNAL).
        """
        p_arr = np.asarray(p, dtype=np.float64)

        return np.asfortranarray(p_arr.T, dtype=np.float64)

    def method_boost_event(self, event, all_p, orig_order, hypp_id, boost_corrected):
        """No boost for EW Sudakov: returns all_p unchanged."""
        if boost_corrected == [-1]:
            return all_p
        return all_p

    def calculate_angles_rotation(self, position_particles, all_p, module):
        """No rotation for EW Sudakov: zero angles."""
        if position_particles == [-1]:
            return [0] * len(all_p), [0] * len(all_p)
        return [0] * len(all_p), [0] * len(all_p)

    def rotation_density(self, module, all_p, phi, theta):
        """No rotation for EW Sudakov: momentum sets converted to the Fortran layout."""
        for i in range(len(all_p)):
            all_p[i] = self.invert_momenta(all_p[i])
        return all_p

    def compute_density_matrix(self, event, target_pdgs, res_dims):
        """Density matrix (production B or decay C) of target_pdgs.

        res_dims are the spin dimensions (2 fermion, 3 vector).

        Returns None if no density module or subprocess matches the event.
        """
        _dbg(
            f"[DensityHelper.compute_density_matrix] ENTER target_pdgs={target_pdgs}, res_dims={res_dims}"
        )

        self._configure_for_pdgs(target_pdgs, res_dims)
        _dbg(
            f"[DensityHelper] Configured: n_changing={self.number_changing_helicities}, n_comb={self.number_combinations}"
        )

        tag, order = event.get_tag_and_order()
        if self.keep_ordering:
            # Unsorted IS and FS ordering, as in the id_to_path keys.
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

        boost_corrected = [-1]
        all_p = self.method_boost_event(event, all_p, orig_order, 0, boost_corrected)

        refChoice_corrected = [-1]
        phi, theta = self.calculate_angles_rotation(refChoice_corrected, all_p, module)
        all_p = self.rotation_density(module, all_p, phi, theta)

        pos_corrected = self._find_particle_positions(orig_order, target_pdgs)
        _dbg(
            f"[DensityHelper] Particle positions for {target_pdgs}: {pos_corrected} (orig_order={orig_order})"
        )

        # The module serves several subprocesses: take the prefix whose PDG
        # list matches this process (up to ordering within IS and FS).
        all_pdgs_raw, all_pids = module.get_pdg_order()
        all_pdgs = [[pdg for pdg in pdgs if pdg != 0] for pdgs in all_pdgs_raw]
        PREFIX = module.get_prefix()
        all_prefixes = [bytes(j).decode(errors="ignore").strip().lower() for j in PREFIX]

        # nincoming = 1 for a decay, 2 for production.
        nincoming = len(tag[0])
        lookup_incoming = sorted(orig_order[0])
        lookup_outgoing = sorted(orig_order[1])
        lookup_tag = (tuple(lookup_incoming), tuple(lookup_outgoing))

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

        # Load the param card once per module: Pdir is <me_dir>/SubProcesses,
        # the card <me_dir>/Cards/param_card.dat (me_dir: rw_me_density_prod or
        # rw_me_density_decay).
        module_id = id(module)
        if module_id not in self._initialized_modules:
            density_me_dir = os.path.dirname(Pdir)
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

        # GET_DENSITY via F2PY, in one of two conventions (tried in this order):
        #   <prefix>get_density(p, pos, n_changing, allow_hel, n_comb, alphas),
        #     per subprocess, exported by density modules such as ttHj;
        #   py_get_density(pdgs, procid, p, pos, allow_hel, alphas, scale2), the
        #     PDG dispatcher, the only GET_DENSITY entry point of the MG5
        #     standalone f2py wrapper (f2py_wrapper_all.inc). F77_DENSITY selects
        #     the subprocess by PDGS when procid <= 0. In this MG5 3.8.1 f2py
        #     signature n_comb, n_changing and npdg are hidden and inferred
        #     from len(allow_hel)/len(pos), len(pos), len(pdgs).
        func_name = prefix + "get_density"
        get_density_func = getattr(module, func_name, None)
        use_density_dispatcher = False
        if get_density_func is not None:
            _dbg(f"[DensityHelper] Using per-process density function: {func_name}")
        else:
            get_density_func = getattr(module, "py_get_density", None)
            use_density_dispatcher = True
            _dbg(
                f"[DensityHelper] {func_name} absent; using unified dispatcher py_get_density "
                f"(present={get_density_func is not None})"
            )
        if get_density_func is None:
            _dbg(
                f"[DensityHelper] No density entry point found (neither {func_name} nor py_get_density)"
            )
            _dbg(
                f"[DensityHelper] Available functions: {[f for f in dir(module) if 'get_density' in f.lower()]}"
            )
            return None

        # Dispatcher only: PDGS in the momentum order (orig_order = (incoming,
        # outgoing)) and procid = -1 (select by PDGS), as in the standard
        # reweight call smatrixhel(pdg, pid=-1, ...).
        density_pdgs = np.asarray(
            list(orig_order[0]) + list(orig_order[1]), dtype=np.int32
        )
        density_procid = -1

        me_value = None
        _dbg(
            f"[DensityHelper] About to call get_density_func, n_changing={self.number_changing_helicities}, n_comb={self.number_combinations}, aqcd={event.aqcd}"
        )

        pos_arr = np.asarray(pos_corrected, dtype=np.int32)
        hel_arr = np.asarray(self.allowed_helicities, dtype=np.int32)
        _dbg(f"[DensityHelper] pos_arr={pos_arr}, hel_arr={hel_arr[:min(10, len(hel_arr))]}")

        for i in range(len(all_p)):
            # all_p[i] is already in the Fortran layout (rotation_density).
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
            if use_density_dispatcher:
                # The inferred n_comb = len(allow_hel)/len(pos) is number_combinations.
                # scale2 is the event scale squared, as in reweight_interface's own
                # density call (a dummy argument for LO matrix elements).
                production_matrix = get_density_func(
                    density_pdgs,
                    density_procid,
                    pinv,
                    pos_arr,
                    hel_arr,
                    event.aqcd,
                    event.scale ** 2,
                )
            else:
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

            len_tri = self.number_combinations * (self.number_combinations + 1) // 2
            _dbg(
                f"[DensityHelper] Calling unpack_density_triangle with len_tri={len_tri}, production_matrix shape={production_matrix.shape if hasattr(production_matrix, 'shape') else len(production_matrix)}"
            )
            # Debug: show raw Fortran output (first few elements)
            if hasattr(production_matrix, "__len__") and len(production_matrix) >= len_tri:
                _dbg(
                    f"[DensityHelper] Raw Fortran output (first {len_tri}): {production_matrix[:len_tri]}"
                )
            me_value = unpack_density_triangle(production_matrix, len_tri)
            _dbg(f"[DensityHelper] unpacked matrix shape: {np.array(me_value).shape}")

        return np.array(me_value) if me_value is not None else None

    def _configure_for_pdgs(self, pdg_codes, res_dims):
        """Configure density attributes for given PDGs."""
        self.particle_in_density_matrix = (list(pdg_codes), "", [])
        self.number_changing_helicities = len(pdg_codes)
        self.spins = list(res_dims)

        self.number_combinations = 1
        for spin in self.spins:
            self.number_combinations *= spin

        self.allowed_helicities = self._build_allowed_helicities(res_dims)

    def _build_allowed_helicities(self, res_dims):
        """Flat int32 array of all helicity combinations, in itertools.product order.

        Values are [+1, -1] for fermions and [+1, 0, -1] for vectors.
        """
        hel_lists = []
        for dim in res_dims:
            if dim == 2:
                hel_lists.append([+1, -1])
            elif dim == 3:
                hel_lists.append([+1, 0, -1])
            else:
                hel_lists.append(list(range(-(dim - 1) // 2, (dim - 1) // 2 + 1)))

        combos = list(product(*hel_lists))
        flat = []
        for combo in combos:
            flat.extend(combo)

        return np.asarray(flat, dtype=np.int32)

    def _find_particle_positions(self, orig_order, target_pdgs):
        """1-based (Fortran) positions of target_pdgs as an int32 array ([-1] if none).

        The initial state is searched first (the resonance of a decay), then
        the final state (the resonances of a production process).
        """
        positions = []
        n_initial = len(orig_order[0])

        for pid in target_pdgs:
            found = False
            for idx, initial_pid in enumerate(orig_order[0]):
                if initial_pid == pid:
                    pos = idx + 1
                    if pos not in positions:
                        positions.append(pos)
                        found = True
                        break
            if found:
                continue
            for idx, final_pid in enumerate(orig_order[1]):
                if final_pid == pid:
                    pos = n_initial + idx + 1
                    if pos not in positions:
                        positions.append(pos)
                        break

        return (
            np.asarray(positions, dtype=np.int32) if positions else np.array([-1], dtype=np.int32)
        )


class DensityMatrix:
    """Spin density matrix ρ[λ₁…λN; λ₁'…λN'] of a system of resonances.

    matrix is a complex (D, D) array, D = ∏ᵢ dᵢ, with dims[i] the spin
    dimension of pdgs[i] (e.g. [6, -6] for tt̄, [24, -24] for W+W-).
    """

    def __init__(self, resonance_pdgs: List[int]):
        self.pdgs = list(resonance_pdgs)
        self.dims = [SPIN_DIMS[abs(pdg)] for pdg in self.pdgs]
        self.total_dim = _total_dimension(self.dims)
        self.matrix = np.zeros((self.total_dim, self.total_dim), dtype=complex)

    @classmethod
    def from_full_matrix(cls, matrix: np.ndarray, resonance_pdgs: List[int]) -> "DensityMatrix":
        """DensityMatrix from a full D×D matrix (ValueError on a shape mismatch)."""
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
        """Linearised (NLO) EW Sudakov correction of B[h,h'] = M_h M*_h'.

        M_h → M_h (1 + Δ_h) gives B[h,h'] (1 + Δ_h)(1 + Δ*_h'); dropping
        O(α²), B^EWSL[h,h'] = B[h,h'] (1 + Δ_h + Δ*_h'). delta has shape
        (total_dim,); returns a new DensityMatrix.
        """
        if len(delta) != self.total_dim:
            raise ValueError(
                f"Delta length {len(delta)} doesn't match "
                f"density matrix dimension {self.total_dim}"
            )

        result = self.copy()

        for h in range(self.total_dim):
            for hp in range(self.total_dim):
                factor = 1.0 + delta[h] + np.conj(delta[hp])
                result.matrix[h, hp] = self.matrix[h, hp] * factor

        return result

    def trace(self) -> float:
        """Real part of the trace."""
        return np.trace(self.matrix).real

    def contract(self, other: "DensityMatrix") -> float:
        """Real part of <ρ₁, ρ₂> = Σ_{h,h'} ρ₁[h,h'] ρ₂[h,h'] = Tr[ρ₁ · ρ₂ᵀ].

        Both matrices are stored as X[h,h'] = A_h A*_h': the production matrix
        from amplitudes with the resonances outgoing, the decay matrix from
        amplitudes with them incoming, in the same helicity ordering.  The
        coherent production-decay rate |Σ_h M_h D_h|² is then
        Σ_{h,h'} B[h,h'] C[h,h'].
        """
        if self.total_dim != other.total_dim:
            raise ValueError(f"Dimension mismatch: {self.total_dim} vs {other.total_dim}")

        return np.sum(self.matrix * other.matrix).real

    def __repr__(self) -> str:
        return (
            f"DensityMatrix(pdgs={self.pdgs}, dims={self.dims}, "
            f"total_dim={self.total_dim}, trace={self.trace():.6f})"
        )

    def __str__(self) -> str:
        return f"DensityMatrix({self.total_dim}×{self.total_dim}) for {self.pdgs}"


def _tensor_product(dm1: DensityMatrix, dm2: DensityMatrix) -> DensityMatrix:
    """Tensor product ρ₁ ⊗ ρ₂ (np.kron, pdgs concatenated)."""
    combined_pdgs = dm1.pdgs + dm2.pdgs
    result = DensityMatrix(combined_pdgs)
    result.matrix = np.kron(dm1.matrix, dm2.matrix)
    return result


def build_decay_tensor_product(decay_matrices: List[DensityMatrix]) -> DensityMatrix:
    """Decay density matrix C = C₁ ⊗ C₂ ⊗ ... ⊗ Cₙ in list order."""
    if not decay_matrices:
        raise ValueError("Need at least one decay matrix")

    result = decay_matrices[0].copy()
    for dm in decay_matrices[1:]:
        result = _tensor_product(result, dm)

    return result


class ResonanceIdentifier:
    """Resonances and their decay products in LHE events."""

    @staticmethod
    def find_resonances(event) -> List[Tuple[int, int]]:
        """(0-based index, pdg) of each resonance with status 1 or 2."""
        resonances = []
        for i, particle in enumerate(event):
            pdg = getattr(particle, "pdg", None) or getattr(particle, "pid", None)
            status = getattr(particle, "status", 1)

            if pdg is None:
                continue

            if abs(pdg) in RESONANCE_PDGS and status in (1, 2):
                resonances.append((i, int(pdg)))

        return resonances

    @staticmethod
    def _mother_matches(mother, resonance_idx: int) -> bool:
        """Whether a mother pointer is the particle at 0-based resonance_idx.

        The pointer is either a Particle (0-based event_id) or a raw 1-based
        LHE number.
        """
        if mother is None:
            return False
        if hasattr(mother, "event_id"):
            return mother.event_id == resonance_idx
        elif isinstance(mother, (int, float)):
            return int(mother) == resonance_idx + 1
        return False

    @staticmethod
    def _identify_decay_products(event, resonance_idx: int) -> List[int]:
        """Indices of the particles whose mother1 or mother2 is resonance_idx."""
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
        """(resonance_idx, pdg, [decay product indices]) for each decayed resonance.

        Daughters are found through the mother pointers, which covers decays
        in the hard matrix element and MadSpin decays alike.
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


def compute_density_weight(
    B: DensityMatrix, C: Optional[DensityMatrix], delta: np.ndarray
) -> float:
    """EW Sudakov weight w = <B^EWSL, C> / <B, C> with density matrices.

    <X, C> = Σ_{h,h'} X[h,h'] C[h,h'] (DensityMatrix.contract) and
    B^EWSL[h,h'] = B[h,h'] (1 + Δ_h + Δ*_h') is the NLO-corrected Born-like
    production matrix. C = None is the spin-summed limit (C = identity:
    traces). Returns 1 if <B, C> vanishes.
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
        _dbg("Denominator <B,C> ≈ 0, returning weight=1")
        return 1.0

    _dbg(f"  numerator={numerator}")
    _dbg(f"  denominator={denominator}")
    return numerator / denominator


class DensitySudakovLogger:
    """Structured logging for density-matrix Sudakov reweighting."""

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
# FxFxEWSudakovMixin
# =============================================================================


class FxFxEWSudakovMixin:
    """FxFx clustering and EW Sudakov reweighting methods for ReweightInterface.

    Uses model, banner, lhe_input, me_dir, keep_ordering, options and the
    density attributes (density_prod_id_to_path, ...) of self.
    """

    def _init_pole_masses_from_banner(self):
        """Set the module pole masses from the banner's param_card.

        Runs once per process; a default stays if the banner or its MASS
        entry is unavailable.
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

    # -------------------------------------------------------------------------
    # MadSpin forced clustering (applied before the FxFx clustering)
    # -------------------------------------------------------------------------

    def _build_madspin_decay_tree(self, event):
        """Decay tree of the status-2 resonances from the mother1/mother2 pointers.

        Returns {lhe_idx (1-based): {'pdg', 'status', 'children': [lhe_idx, ...],
        'depth', 'mother'}}; depth is the maximum distance to a leaf (0 for a
        leaf).
        """
        tree = {}

        for i, particle in enumerate(event):
            lhe_idx = i + 1
            pdg = int(getattr(particle, "pid", 0))
            status = int(getattr(particle, "status", 0))
            tree[lhe_idx] = {
                "pdg": pdg,
                "status": status,
                "children": [],
                "depth": 0,
                "mother": None,
            }

        for i, particle in enumerate(event):
            lhe_idx = i + 1
            m1 = _get_mother_lhe_idx(getattr(particle, "mother1", None))
            m2 = _get_mother_lhe_idx(getattr(particle, "mother2", None))

            if m1 > 0 and m1 in tree and tree[m1]["status"] == 2:
                tree[m1]["children"].append(lhe_idx)
                tree[lhe_idx]["mother"] = m1
            elif m2 > 0 and m2 in tree and tree[m2]["status"] == 2:
                if m2 != m1:
                    tree[m2]["children"].append(lhe_idx)
                    tree[lhe_idx]["mother"] = m2

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
        """ForcedClusterSteps of the decay tree, inner decays first.

        Steps are sorted by depth, so nested decays cluster W -> l nu
        (depth 1) before t -> b W (depth 2). The scale is the children's
        invariant mass.
        """
        steps = []

        resonances = [
            (lhe_idx, node)
            for lhe_idx, node in decay_tree.items()
            if node["status"] == 2 and node["children"]
        ]

        resonances.sort(key=lambda x: x[1]["depth"])

        for lhe_idx, node in resonances:
            children = node["children"]
            sum_E = sum_px = sum_py = sum_pz = 0.0

            for child_idx in children:
                p = event[child_idx - 1]
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

    def _apply_forced_clustering(self, event, steps, track_decays=False):
        """Reverse the MadSpin decays with the FKS FSR mapping (mother on its pole).

        Each 1 -> 2 step replaces the two children by their status-2 mother,
        which becomes final state; the other final-state particles recoil.
        track_decays carries the decay leaf momenta through every mapping.
        Returns (event, groups): the event with the hard process restored,
        with the per-step methods in event.fks_methods, and per position the
        set of original LHE indices it contains.
        """
        buff_event = copy.deepcopy(event)
        decay_products = None
        if track_decays:
            decay_products = [
                {i + 1: FourMomentum(p)} if p.status == 1 else {}
                for i, p in enumerate(event)
            ]
            buff_event._fxfx_decay_products = decay_products

        # _cluster_fxfx_event_internal seeds its method history from these.
        forced_methods = []

        # groups[pos]: original LHE indices at each position (every particle,
        # intermediates included).
        groups = []
        pos_to_lhe = {}
        lhe_to_pos = {}

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

            # Only 1 -> 2 decays are reversed. A skipped mother stays status 2
            # (stripped later), so the hard process is incomplete: the skip is
            # recorded for the fallback policy and the counters.
            if len(step.children_lhe_idx) != 2:
                _dbg(
                    f"[FORCED]   WARNING: Expected 2 children, got {len(step.children_lhe_idx)}, skipping"
                )
                forced_methods.append("forced_step_skipped")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_step_skipped"] += 1
                continue

            child1_lhe, child2_lhe = step.children_lhe_idx
            mother_lhe = step.mother_lhe_idx

            if child1_lhe not in lhe_to_pos or child2_lhe not in lhe_to_pos:
                _dbg("[FORCED]   Children not found in current event, skipping")
                forced_methods.append("forced_step_skipped")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_step_skipped"] += 1
                continue
            if mother_lhe not in lhe_to_pos:
                _dbg("[FORCED]   Mother not found in current event, skipping")
                forced_methods.append("forced_step_skipped")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_step_skipped"] += 1
                continue

            child1_pos = lhe_to_pos[child1_lhe]
            child2_pos = lhe_to_pos[child2_lhe]
            mother_pos = lhe_to_pos[mother_lhe]

            _dbg(
                f"[FORCED]   Positions: child1={child1_pos}, child2={child2_pos}, mother={mother_pos}"
            )

            # The FKS FSR mapping puts the mother on its pole, the recoil going
            # to the other final-state particles (or the beams). The mother is
            # still a status-2 intermediate, so only the beams and the status
            # ±1 particles ("active") enter momentum conservation.
            active_positions = []
            for pos in range(len(buff_event)):
                p = buff_event[pos]
                status = getattr(p, "status", 1)
                if pos <= 1:
                    active_positions.append(pos)
                elif abs(status) == 1:
                    active_positions.append(pos)

            _dbg(f"[FORCED]   Active positions for FKS: {active_positions}")

            active_to_buff = {i: pos for i, pos in enumerate(active_positions)}
            buff_to_active = {pos: i for i, pos in enumerate(active_positions)}

            orig_momenta = [FourMomentum(buff_event[pos]) for pos in active_positions]

            if child1_pos not in buff_to_active or child2_pos not in buff_to_active:
                _dbg("[FORCED]   WARNING: Children not in active list, skipping reshuffling")
                # A child is not active (an intermediate whose own decay was
                # not reversed): the mother is the off-shell sum of the children.
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
                # The mass label must match the momentum (check_kinematics_only).
                mother_p.mass = math.sqrt(
                    max(0.0, mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2)
                )
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_children_not_active"] += 1
                forced_methods.append("forced_children_not_active")
                if decay_products is not None:
                    decay_products[mother_pos] = dict(decay_products[child1_pos])
                    decay_products[mother_pos].update(decay_products[child2_pos])
            else:
                child1_active = buff_to_active[child1_pos]
                child2_active = buff_to_active[child2_pos]

                p1 = buff_event[child1_pos]
                p2 = buff_event[child2_pos]
                mother_E = float(p1.E) + float(p2.E)
                mother_px = float(p1.px) + float(p2.px)
                mother_py = float(p1.py) + float(p2.py)
                mother_pz = float(p1.pz) + float(p2.pz)
                off_shell_m2 = mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2
                off_shell_m = math.sqrt(max(0, off_shell_m2))

                _dbg(f"[FORCED]   Off-shell mother: E={mother_E:.4f}, m={off_shell_m:.4f} GeV")

                # Pole mass, or the pair mass for a mother without one.
                target_mass = _get_mother_mass(abs(step.mother_pdg)) or off_shell_m

                sqrt_s = math.sqrt(max(0, _m2(orig_momenta[0] + orig_momenta[1])))
                moth = [{"number": child1_active + 1, "id": step.mother_pdg}]

                # Mapped in the partonic CM frame, as every FSR clustering step.
                fks_result = _fks_fsr_mapping_cm(child1_active, child2_active, moth, orig_momenta)
                _dbg(f"[FORCED]   CM boost: β={fks_result['to_cm'][0][1][0]:.6f}, √s={sqrt_s:.2f} GeV")

                if decay_products is not None:
                    children = [
                        _transport_decay_products(decay_products[pos], fks_result["to_cm"])
                        for pos in (child1_pos, child2_pos)
                    ]
                    projected = _project_cluster_decay_products(
                        children,
                        [
                            fks_result["momenta_cm"][child1_active],
                            fks_result["momenta_cm"][child2_active],
                        ],
                        fks_result["mother_cm"],
                    )
                    decay_products[mother_pos] = _transport_decay_products(
                        projected, fks_result["from_cm"]
                    )
                    for active_idx in fks_result["spectators"]:
                        pos = active_to_buff[active_idx]
                        decay_products[pos] = _transport_decay_products(
                            decay_products[pos], fks_result["spectator_transform"],
                        )

                _dbg(f"[FORCED]   FKS method: {fks_result['method']}")

                onshell_mom_lab = fks_result["mother"]
                beam0_lab = fks_result["beams"][0]
                beam1_lab = fks_result["beams"][1]
                spectators_lab = fks_result["spectators"]

                mother_p = buff_event[mother_pos]
                mother_p.E = onshell_mom_lab.E
                mother_p.px = onshell_mom_lab.px
                mother_p.py = onshell_mom_lab.py
                mother_p.pz = onshell_mom_lab.pz
                mother_p.status = 1

                onshell_m2 = _m2(onshell_mom_lab)
                onshell_m = math.sqrt(max(0, onshell_m2))

                # Pole-mass label only for an on-pole mapping, else the actual
                # mass (see fxfx_merge_particles_kinematics).
                fks_method = fks_result.get("method", "")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS[fks_method] += 1
                forced_methods.append(fks_method)
                if fks_method in FKS_SUCCESS_METHODS:
                    mother_p.mass = target_mass
                else:
                    mother_p.mass = onshell_m
                    _dbg(
                        f"[FORCED]   WARNING: FKS mapping fell back ({fks_method}), labelling with invariant mass"
                    )

                _dbg(
                    f"[FORCED]   On-shell mother: E={onshell_mom_lab.E:.4f}, m={onshell_m:.4f} GeV (target={target_mass:.4f})"
                )

                buff_event[0].E = beam0_lab.E
                buff_event[0].px = beam0_lab.px
                buff_event[0].py = beam0_lab.py
                buff_event[0].pz = beam0_lab.pz
                buff_event[1].E = beam1_lab.E
                buff_event[1].px = beam1_lab.px
                buff_event[1].py = beam1_lab.py
                buff_event[1].pz = beam1_lab.pz

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

            mother_group = groups[mother_pos]
            child1_group = groups[child1_pos]
            child2_group = groups[child2_pos]
            mother_group.update(child1_group)
            mother_group.update(child2_group)

            _dbg(f"[FORCED]   Merged groups: mother now contains {mother_group}")

            children_pos = sorted([child1_pos, child2_pos], reverse=True)
            for pos in children_pos:
                buff_event.pop(pos)
                groups.pop(pos)
                if decay_products is not None:
                    decay_products.pop(pos)

            pos_to_lhe = {}
            lhe_to_pos = {}
            for i, grp in enumerate(groups):
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

        buff_event.fks_methods = forced_methods
        return buff_event, groups

    # -------------------------------------------------------------------------
    # FxFx clustering from the <clustering> history
    # -------------------------------------------------------------------------

    def _fxfx_parse_clustering(self, event):
        """Parse the FxFx <clustering ...> block: its attributes and 'steps'.

        'steps' is [{'n', 'idxs', 'mom'}] sorted by n; None without a block.
        """
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
        """Cluster an event to its Born-like hard process.

        MadSpin decays (status-2 resonances with children) are reversed first
        by forced clustering, then the FxFx <clustering> history is applied.
        Returns (clustered_event, sorted_tag), or with return_groups
        (clustered_event, sorted_tag, groups), groups[pos] being the original
        LHE indices at each position and the event carrying the decay leaf
        momenta; None on failure. record_tag adds sorted_tag to
        self.fxfx_clustered_final_states.
        """
        _dbg(f"[CLUSTER] _cluster_fxfx_event ENTER: npart={len(event)}, record_tag={record_tag}, return_groups={return_groups}")

        # Mapping counters include only the first clustering pass of each
        # main-loop event; event outcomes are counted by _count_fks_event.
        global _FKS_COUNT_ACTIVE, _FKS_COUNTED_UPTO
        if CURRENT_EVENT_ID > 0 and CURRENT_EVENT_ID > _FKS_COUNTED_UPTO:
            _FKS_COUNT_ACTIVE = True
            _FKS_COUNTED_UPTO = CURRENT_EVENT_ID
        else:
            _FKS_COUNT_ACTIVE = False

        decayed_resonances = ResonanceIdentifier.find_decayed_resonances(event)
        _dbg(f"[CLUSTER]   decayed_resonances detected: {len(decayed_resonances) if decayed_resonances else 0}")

        if decayed_resonances:
            _dbg(
                f"[CLUSTER] Found {len(decayed_resonances)} decayed resonances, applying forced clustering"
            )

            decay_tree = self._build_madspin_decay_tree(event)
            _dbg(f"[CLUSTER]   built decay_tree with {len(decay_tree)} nodes")

            forced_steps = self._generate_forced_cluster_steps(event, decay_tree)
            _dbg(f"[CLUSTER]   forced_steps generated: {len(forced_steps) if forced_steps else 0}")

            if forced_steps:
                hard_event, forced_groups = self._apply_forced_clustering(
                    event, forced_steps, track_decays=return_groups,
                )

                _dbg(
                    f"[CLUSTER] Forced clustering complete, hard process has {len(hard_event)} particles"
                )

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
        """Apply the FxFx <clustering> steps of original_event to event.

        event may be the forced-clustering output, with initial_groups its
        position -> LHE-index groups. Returns as _cluster_fxfx_event.
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

        # The <clustering> tag of the original event survives MadSpin.
        clustering = self._fxfx_parse_clustering(original_event)
        if not clustering:
            _dbg("[FXFX]   no <clustering> block found")
            if initial_groups is not None:
                # No FxFx history: the forced-clustered hard process is the result.
                _dbg("[FXFX]   returning forced-clustered hard process (no further FxFx)")
                try:
                    sorted_tag = event.get_tag_and_order()[0]
                except Exception:
                    sorted_tag = None
                if record_tag and sorted_tag:
                    if not hasattr(self, "fxfx_clustered_final_states"):
                        self.fxfx_clustered_final_states = set()
                    self.fxfx_clustered_final_states.add(sorted_tag)
                if not hasattr(event, "fks_methods"):
                    event.fks_methods = []
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
            if hasattr(buff_event, "_fxfx_decay_products"):
                buff_event._fxfx_decay_products.pop(ip)

        if return_groups and not hasattr(buff_event, "_fxfx_decay_products"):
            buff_event._fxfx_decay_products = [
                {i + 1: FourMomentum(p)} if p.status == 1 else {}
                for i, p in enumerate(event) if i not in to_pop
            ]

        # Method history: the forced-clustering methods, then one entry per
        # FxFx step.
        fks_methods = list(getattr(event, "fks_methods", []))

        if initial_groups is not None:
            groups = [set(g) for g in initial_groups]
            lhe_external_indices = []
            for g in groups:
                lhe_external_indices.append(min(g))
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
            # 1-based external index -> 1-based LHE index
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

        def _group_momentum_labels(grp):
            """LHE labels whose original momenta add up to this object's momentum.

            A forced-clustering group keeps the restored resonance's own LHE
            index next to its children (the density path uses it to identify
            the decay system), so summing every member would count the
            resonance twice (2 p_W for W -> l nu, 2 p_t + p_W for nested
            t -> b W(-> l nu)). Status-2 members are dropped whenever the group
            has a non-status-2 member, which also covers nested decays. A group
            of intermediates only (a decay that was not reversed) keeps its
            entries, and labels outside the original event are kept, so that a
            stale index fails in the momentum sum instead of being dropped.
            """
            leaves = []
            for lab in grp:
                if 1 <= lab <= len(original_event):
                    if int(getattr(original_event[lab - 1], "status", 1)) == 2:
                        continue
                leaves.append(lab)
            return leaves if leaves else list(grp)

        def _s_groups_lhe_from_pos(pos_i, pos_j):
            # LHE labels index original_event.
            Ei = pxi = pyi = pzi = 0.0
            Ej = pxj = pyj = pzj = 0.0
            for lab in _group_momentum_labels(groups[pos_i]):
                q = original_event[lab - 1]
                Ei += float(q.E)
                pxi += float(q.px)
                pyi += float(q.py)
                pzi += float(q.pz)
            for lab in _group_momentum_labels(groups[pos_j]):
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
            """PDG and state (False for status -1) of the current buff_event[pos]."""
            if pos < 0 or pos >= len(buff_event):
                return None
            part = buff_event[pos]
            pdg = int(getattr(part, "pid", 0))
            state = int(getattr(part, "status", 0)) != -1
            return {"pdg": pdg, "state": state}

        mW2 = MW_POLE**2
        # Clustering continues while some pair has s_ij < mW2_cluster_scale M_W²;
        # the default 1.5 reaches √s_ij ≈ 98 GeV and so covers M_Z.
        mW2_cluster_scale = getattr(self, "mW2_cluster_scale", 1.5)
        mW2_threshold = mW2_cluster_scale * mW2
        _dbg(f"[FXFX]   mW2_threshold={mW2_threshold:.3f} (scale={mW2_cluster_scale})")

        # Steps in the tag's order, which is scale ordered (soft to hard).
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

                # Tag indices count external particles only (no status-2
                # intermediates).
                idxs_ext = idxs
                valid = all((1 <= k <= ext_index_max) for k in idxs_ext)
                if not valid:
                    stop_clustering = True
                    break

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
                # The selected pair is clustered even above threshold: the
                # threshold enters only through the all-pairs check above.
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

                        # Mother flavour from the 3-point vertex (g/γ + f -> f
                        # takes the fermion flavour and the tag sign; same-flavour
                        # q + q~ -> g; other quark and lepton pairs keep the tag
                        # mother), then global charge conservation fixes the sign
                        # (or the flavour within the family).
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

                        if is_gauge_boson_a and is_fermion_b:
                            old_step_mom = _step_mom
                            sign = 1 if _step_mom >= 0 else -1
                            _step_mom = (
                                sign * abs_b
                            )
                            _dbg(
                                f"FxFx clustering: flavor fix V+f->f: {old_step_mom} -> {_step_mom} (sign={sign}, flavor={abs_b})"
                            )
                        elif is_fermion_a and is_gauge_boson_b:
                            old_step_mom = _step_mom
                            sign = 1 if _step_mom >= 0 else -1
                            _step_mom = (
                                sign * abs_a
                            )
                            _dbg(
                                f"FxFx clustering: flavor fix f+V->f: {old_step_mom} -> {_step_mom} (sign={sign}, flavor={abs_a})"
                            )
                        elif is_quark_a and is_quark_b:
                            if abs_a == abs_b and pdg_a * pdg_b < 0:
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
                            if abs_a == abs_b and pdg_a * pdg_b < 0:
                                # Same flavor lepton-antilepton -> Z(23) or gamma(22)
                                _dbg(f"FxFx clustering: l+l~->V, keeping tag mom={_step_mom}")
                            else:
                                _dbg(
                                    f"FxFx clustering: lepton pair, keeping tag mom={_step_mom} (pdg_a={pdg_a}, pdg_b={pdg_b})"
                                )
                        else:
                            # Other pairs: the first UFO 3-point mother in the
                            # flavour family of the tag mother.
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

                # Stop before a 2 -> 1 configuration with a massless final
                # state, which has no rest frame.
                current_fs = [i for i in range(len(groups)) if i > 1]
                a_is_fs, b_is_fs = (a > 1), (b > 1)

                fs_after_pdgs = []
                if a_is_fs and b_is_fs:
                    fs_after_pdgs.append(moth_id)
                    for i in current_fs:
                        if i != a and i != b:
                            rep = _group_rep(i)
                            if rep:
                                fs_after_pdgs.append(rep["pdg"])
                else:
                    # ISR: only the spectators stay in the final state.
                    for i in current_fs:
                        if i != a and i != b:
                            rep = _group_rep(i)
                            if rep:
                                fs_after_pdgs.append(rep["pdg"])

                if len(fs_after_pdgs) == 1:
                    fs_pdg = fs_after_pdgs[0]
                    fs_mass = _get_mother_mass(abs(fs_pdg))
                    if fs_mass < 1e-6:
                        _dbg(f"[FXFX]   STOP: 2→1 with massless FS (pdg={fs_pdg})")
                        stop_clustering = True
                        break

                moth_loc = [{"number": keep + 1, "id": moth_id}]

                try:
                    step_method = fxfx_merge_particles_kinematics(buff_event, a, b, moth_loc)
                except ValueError as e:
                    if "light-like" in str(e):
                        _dbg(f"[FXFX]   STOP: {e}")
                        if _FKS_COUNT_ACTIVE:
                            FKS_QUALITY_COUNTS["clustering_stopped_lightlike"] += 1
                        stop_clustering = True
                        break
                    if "math domain error" in str(e):
                        # The event is kept with unit Sudakov weights.
                        _dbg(f"[FXFX]   clustering aborted (reshuffling failed: {e}); event kept with unit Sudakov weights")
                        if _FKS_COUNT_ACTIVE:
                            FKS_QUALITY_COUNTS["clustering_aborted_math_domain"] += 1
                        return None
                    raise
                fks_methods.append(step_method)

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
        buff_event.fks_methods = fks_methods
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
        if getattr(self, "lhe_input", None) is None:
            return
        source = getattr(self, "fxfx_catalogue_source", None) or self.lhe_input.path
        source = os.path.realpath(source)
        stat = os.stat(source)
        scale = getattr(self, "mW2_cluster_scale", 1.5)
        xis = tuple(getattr(self, "sudakov_xi_scan", None) or [scale])
        catalogue_key = (source, stat.st_size, stat.st_mtime_ns, xis)
        if (getattr(self, "_fxfx_cluster_catalog_ready", False)
                and getattr(self, "_fxfx_cluster_catalog_key", None) == catalogue_key):
            return
        self._fxfx_cluster_catalog_ready = False
        self.fxfx_clustered_final_states = set()
        self.fxfx_clustered_decay_processes = set()
        self._sudakov_source_ids = set()
        self._init_pole_masses_from_banner()
        logger.info("Collecting FxFx processes from %s at xi=%s", source, xis)
        try:
            with contextlib.closing(lhe_parser.EventFile(source)) as events:
                for event in events:
                    self._sudakov_source_ids.update(event.parse_reweight() or {"1001": event.wgt})
                    for xi in xis:
                        self.mW2_cluster_scale = xi
                        cluster_result = self._cluster_fxfx_event(event, record_tag=True, return_groups=True)
                        if not cluster_result:
                            continue
                        clustered_event, _, groups = cluster_result
                        for res_idx, res_pdg in ResonanceIdentifier.find_resonances(clustered_event):
                            if res_idx >= len(groups):
                                continue
                            decay_pdgs = [
                                int(event[i - 1].pid) for i in sorted(groups[res_idx])
                                if 1 <= i <= len(event) and event[i - 1].status == 1
                            ]
                            if len(decay_pdgs) >= 2:
                                self.fxfx_clustered_decay_processes.add(
                                    (int(res_pdg), tuple(sorted(decay_pdgs))))
        finally:
            self.mW2_cluster_scale = scale
        self._fxfx_cluster_catalog_key = catalogue_key
        self._fxfx_cluster_catalog_ready = True
        logger.info("Collected %d FxFx production and %d decay processes",
                    len(self.fxfx_clustered_final_states), len(self.fxfx_clustered_decay_processes))

    def _validate_fxfx_sudakov_catalogue(self, sud_mod):
        """Require compiled coverage for each clustered multi-particle process."""
        if not self._needs_fxfx_cluster_catalogue():
            return
        missing = [tag for tag in sorted(self.fxfx_clustered_final_states)
                   if len(tag[1]) > 1 and tag not in sud_mod.original_pdg_list_dict]
        if missing:
            raise MissingSudakovProcessError(
                "Sudakov library is missing FxFx processes %s. "
                "Generate the library from the complete event input and xi scan." % missing)

    # -------------------------------------------------------------------------
    # Process generation
    # -------------------------------------------------------------------------

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
        # Undecayed resonances (e.g. stable tops in ttH) have no decay matrix C
        # and use the scalar path at runtime: only final states with a decayed
        # resonance need a density production process.
        if not self.fxfx_clustered_decay_processes:
            _dbg(
                "[DENSITY_PROD_GEN] No decayed resonances found, skipping density production processes"
            )
            return []
        decayed_resonance_pdgs = set()
        for res_pdg, decay_pdgs in self.fxfx_clustered_decay_processes:
            decayed_resonance_pdgs.add(abs(res_pdg))
        _dbg(f"[DENSITY_PROD_GEN] Decayed resonance PDGs: {decayed_resonance_pdgs}")
        extra = []
        seen = set()
        for sorted_tag in sorted(self.fxfx_clustered_final_states):
            outgoing = sorted_tag[1]
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
        """Sudakov inputs: CM-frame kinematics in the canonical library order."""
        _dbg("[FXFX] _prepare_fxfx_sudakov_inputs: start")
        _dbg(f"[FXFX]   event_to_sud particles={len(event_to_sud)}")
        frame_transform = []

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
            frame_transform.append(("boost_to_cm", FourMomentum(p_in_sum)))
            event_to_sud.boost(p_in_sum)

        # Initial state along z, as in the ickkw=0 path.
        initial = copy.deepcopy(event_to_sud[0])
        if not ((abs(initial.px) < 1e-6 * initial.E) and (abs(initial.py) < 1e-6 * initial.E)):
            _dbg(f"[FXFX]   rotating to z-axis (initial px={initial.px}, py={initial.py})")
            frame_transform.append(("rotate_to_z", FourMomentum(initial)))
            for p in event_to_sud:
                p.set_momentum(
                    lhe_parser.FourMomentum(p).rotate_to_z(prot=lhe_parser.FourMomentum(initial))
                )

        # Remove the ~1e-7 E round-off of rotate_to_z: set_initial_mass_to_zero
        # strictly requires px = py = 0 and pz0 = -pz1 for the two beams.
        p0, p1 = event_to_sud[0], event_to_sud[1]
        E_in   = abs(p0.E) + abs(p1.E)
        pz_abs = (abs(p0.pz) + abs(p1.pz)) / 2.0
        sign0  = +1.0 if p0.pz >= 0 else -1.0
        p0.set_momentum(lhe_parser.FourMomentum([E_in/2.0, 0.0, 0.0,  sign0 * pz_abs]))
        p1.set_momentum(lhe_parser.FourMomentum([E_in/2.0, 0.0, 0.0, -sign0 * pz_abs]))

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
        except KeyError as exc:
            raise MissingSudakovProcessError(
                "Sudakov library has no process %s. Generate the library from "
                "the complete event input and xi scan." % (sorted_tag,)) from exc
        _dbg(f"[FXFX]   canonical order={canon}")

        # perm[i]: position in canon of the final-state particle mapped_order[1][i].
        perm = []
        used = set()
        for r in mapped_order[1]:
            for idx, c in enumerate(canon):
                if c == r and idx not in used:
                    perm.append(idx)
                    used.add(idx)
                    break

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
            "perm": perm,  # LHE -> canonical MG5 order, final state only
            # Lorentz operations applied before the non-Lorentz mass
            # normalisations/reordering.  The decay system must follow these
            # operations before its on-shell adjustment.
            "frame_transform": frame_transform,
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
        """Set fxfx_ignore_invariant_checks = disable_clamp (else rij_ge_mw = not it).

        Returns (kind, previous value) for _restore_ewsud_rij_ge_mw, or None if
        nothing was set.
        """
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
        """Whether the clustered event is 2 -> 1.

        Its only scale is s = M², with no invariant for Sudakov logarithms.
        """
        n_initial = sum(1 for p in clustered_event if p.status == -1)
        n_final = sum(1 for p in clustered_event if p.status == 1)
        is_2to1 = n_initial == 2 and n_final == 1
        _dbg(f"[2TO1] _is_2to1_topology: n_init={n_initial}, n_final={n_final} -> is_2to1={is_2to1}")
        return is_2to1

    def _has_small_invariants(self, p_in, iflist, mw2=None):
        """Whether any pair has |s_ij| < M_W² (soft/collinear: pass-through).

        s_ij = (p_i + sign·p_j)² = m_i² + m_j² + sign·2·(p_i·p_j), the
        convention of Source/kin_functions.f::SumDot. The mass terms matter
        for legs on shell at a non-light scale (the Z in Z+jets:
        m_Z² ≈ 1.28·M_W²): without them forward-Z events with r_ij ≈ 0 in
        Fortran would pass and the kernel's rij_ge_mw clamp would act on the
        log argument.
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

    # FxFx applies the invariant cut before evaluating these three variants.
    SUDAKOV_VARIANT_NAMES = ("central", "s_to_rij_off", "sherpa_qed")
    SUDAKOV_LO_VARIANT_NAMES = SUDAKOV_VARIANT_NAMES + ("both_off", "rij_ge_mw_off")

    def _collect_sudakov_weight_sources(self):
        """Collect full source IDs from the input header and event stream."""
        header = self.banner["initrwgt"] if "initrwgt" in self.banner else ""
        tree = ElementTree.fromstring("<initrwgt>" + header + "</initrwgt>")
        for group in tree.findall("weightgroup"):
            if group.get("name") == "ew_sudakov":
                raise ValueError("Input declares EW Sudakov weights; supply source event weights.")
        declared = set()
        for entry in tree.iter("weight"):
            declared.add(entry.attrib["id"])
            if re.search(r"scale_\S+_sud(?:\s|$)", "".join(entry.itertext())):
                raise ValueError("Input declares EW Sudakov weights; supply source event weights.")
        self._sudakov_input_header = header
        self._sudakov_declared_source_ids = declared
        if not self._needs_fxfx_cluster_catalogue():
            self._sudakov_source_ids = set()
            source = getattr(self, "fxfx_catalogue_source", None) or self.lhe_input.path
            with contextlib.closing(lhe_parser.EventFile(source)) as events:
                for event in events:
                    self._sudakov_source_ids.update(event.parse_reweight() or {"1001": event.wgt})
        self._sudakov_source_ids = set(getattr(self, "_sudakov_source_ids", ())) | declared
        if not self._sudakov_source_ids:
            self._sudakov_source_ids.add("1001")

    def _configure_sudakov_weight_ids(self, source_ids, xis, variant_names=None):
        """Assign a unique numeric ID to each source, xi index, and variant."""
        sources = sorted(set(map(str, source_ids)),
                         key=lambda key: (0, int(key), key) if key.isdigit() else (1, 0, key))
        names = tuple(variant_names or self.SUDAKOV_VARIANT_NAMES)
        self._sudakov_xi_list = list(xis)
        self._sudakov_variant_names = names
        self._sudakov_source_ids = set(sources)
        self._sudakov_weight_map = {}
        self._sudakov_weight_metadata = {}
        reserved = set(sources) | {str(int(key)) for key in sources if key.isdigit()}
        slots = [(k, v, source) for k in range(len(xis))
                 for v in range(len(names)) for source in sources]
        for k, v, source in slots:
            if re.fullmatch(r"10[0-9]{2}", source):
                candidate = str((20 + 5*k + v)*100 + int(source) - 1000)
                if candidate not in reserved:
                    self._sudakov_weight_map[k, v, source] = candidate
                    reserved.add(candidate)
        candidate = 10000
        for k, v, source in slots:
            slot = (k, v, source)
            if slot not in self._sudakov_weight_map:
                while str(candidate) in reserved:
                    candidate += 1
                self._sudakov_weight_map[slot] = str(candidate)
                reserved.add(str(candidate))
                candidate += 1
            output_id = self._sudakov_weight_map[slot]
            self._sudakov_weight_metadata[output_id] = (source, k, xis[k], names[v])
        return list(self._sudakov_weight_metadata)

    def _prepare_sudakov_weight_ids(self):
        """Select the xi scan and variant layout for the configured event mode."""
        try:
            self._sudakov_ickkw = int(self.banner.get('run_card', 'ickkw'))
        except Exception:
            self._sudakov_ickkw = 0
        xis = [getattr(self, 'mW2_cluster_scale', 1.5)]
        names = self.SUDAKOV_LO_VARIANT_NAMES
        if self._sudakov_ickkw == 3:
            xis = self.sudakov_xi_scan or xis
            names = self.SUDAKOV_VARIANT_NAMES
        elif self.sudakov_xi_scan and len(self.sudakov_xi_scan) > 1:
            logger.warning("The xi scan requires ickkw=3; using one set of Sudakov variants.")
        return self._configure_sudakov_weight_ids(self._sudakov_source_ids, xis, names)

    def _sudakov_weight_header(self):
        """Describe every output ID by its full source ID, xi, and variant."""
        header = getattr(self, "_sudakov_input_header", "")
        missing = self._sudakov_source_ids - getattr(self, "_sudakov_declared_source_ids", set())
        if missing:
            header += '\n<weightgroup name="source_weights">\n'
            for source in sorted(missing):
                header += '<weight id=%s>Source weight %s</weight>\n' % (quoteattr(source), escape(source))
            header += '</weightgroup>\n'
        header += '\n<weightgroup name="ew_sudakov" weight_name_strategy="includeIdInWeightName">\n'
        for output_id, (source, k, xi, variant) in self._sudakov_weight_metadata.items():
            header += ('<weight id=%s source_id=%s xi_index="%d" xi="%s" variant=%s>'
                       'source=%s xi=%s variant=%s</weight>\n' %
                       (quoteattr(output_id), quoteattr(source), k, repr(float(xi)), quoteattr(variant),
                        escape(source), repr(float(xi)), escape(variant)))
        return header + '</weightgroup>\n'

    def _build_sudakov_rwgt_dict(self, event, weights, variant_names=None):
        """Multiply each full source weight by the selected Sudakov factors."""
        sources = event.parse_reweight() or {"1001": event.wgt}
        names = tuple(variant_names or getattr(self, "_sudakov_variant_names", self.SUDAKOV_VARIANT_NAMES))
        xi_idx = getattr(self, "_current_xi_idx", 0)
        if not hasattr(self, "_sudakov_weight_map"):
            xis = getattr(self, "sudakov_xi_scan", None) or [getattr(self, "mW2_cluster_scale", 1.5)]
            self._configure_sudakov_weight_ids(sources, xis, names)
        factors = list(weights)[:len(names)]
        factors += [1.0] * (len(names) - len(factors))
        result = {"orig": event.wgt} if xi_idx == 0 else {}
        for source, value in sources.items():
            for variant, factor in enumerate(factors):
                slot = (xi_idx, variant, str(source))
                if slot not in self._sudakov_weight_map:
                    raise ValueError("Sudakov weight layout does not contain source/xi/variant %s" % (slot,))
                result[self._sudakov_weight_map[slot]] = value * factor
        return result

    def _compute_ewsudakov_fxfx_reweight(self, event, sud_mod):
        """FxFx-aware scalar Sudakov reweighting."""
        _dbg("=" * 70)
        _dbg(f"[SCALAR] _compute_ewsudakov_fxfx_reweight ENTER: event_id={CURRENT_EVENT_ID}, npart={len(event)}, event.wgt={event.wgt}")
        self._init_pole_masses_from_banner()
        # A 2 -> 1 input (qq̄ -> resonance, no radiation) has no Sudakov logs:
        # unit weights, before clustering.
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
            _dbg("[FALLBACK] No clustering information: no Born-like object")
            # Without a Born-like object there is no Sudakov factor: unit weights.
            _count_fks_event("events_clustering_failed")
            _dbg("[FALLBACK] clustering failed -> unit weights 3×1.0")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        event_to_sud, _ = cluster_result

        # Policy "unit_weight": unit weights for any clustering history with a
        # non-success FKS method.
        _fks_hist = getattr(event_to_sud, "fks_methods", [])
        if any(m not in FKS_SUCCESS_METHODS for m in _fks_hist):
            _count_fks_event("events_with_fsr_fallback")
            if FKS_FALLBACK_POLICY == "unit_weight":
                _dbg(f"[POLICY] FSR fallback in clustering history {_fks_hist} and policy=unit_weight -> 3×1.0")
                _count_fks_event("events_unit_weight_by_policy")
                return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        _dbg(f"[SCALAR] checking post-cluster topology on event_to_sud ({len(event_to_sud)} parts)")
        if self._is_2to1_topology(event_to_sud):
            _dbg(f"[2TO1] -> POST-CLUSTER 2→1 PASSTHROUGH, returning weight=1 for all 3 variants (event_id={CURRENT_EVENT_ID})")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        _dbg("[SCALAR] calling _prepare_fxfx_sudakov_inputs(event, sud_mod, event_to_sud)...")
        try:
            prep = self._prepare_fxfx_sudakov_inputs(event, sud_mod, event_to_sud)
        except RuntimeError:
            _dbg("[FATAL] _prepare_fxfx_sudakov_inputs: order in particle momenta does not match MG convention -> exit(3)")
            sys.exit(3)

        _dbg(f"[SCALAR] _prepare returned OK: sorted_tag={prep['sorted_tag']}, iflist={list(prep['iflist'])}, p_in.shape={prep['p_in'].shape}, gstr={prep['gstr']:.6f}")

        # Unit weights for small invariants. This cut makes the Fortran
        # rij_ge_mw clamp inert, so both_off and rij_ge_mw_off would equal
        # s_to_rij_off and central; they are not emitted.
        _dbg("[SCALAR] checking small invariants on p_in...")
        if self._has_small_invariants(prep["p_in"], prep["iflist"]):
            _dbg(f"[FALLBACK] SMALL INVARIANT (s_ij < MW²) -> passthrough, returning 3×1.0 (event_id={CURRENT_EVENT_ID})")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

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
        # res[0] = Born, res[1] = sherpa_qed (sud_mod=0), res[2] = central,
        # res[3] = s_to_rij_off; res[4] (both_off) and res[5] (rij_ge_mw_off)
        # equal res[3] and res[2] after the small-invariant cut.
        if abs(res[0]) > 1e-30:
            _dbg(f"[5SUD] Born is non-zero -> normalizing 3 variants by Born={res[0]:.6e}")
            sudrats = [
                1.0 + res[2] / res[0],   # central
                1.0 + res[3] / res[0],   # s_to_rij_off
                1.0 + res[1] / res[0],   # sherpa_qed
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

        sudrats = limit_sudakov_factors(sudrats, "events_runaway_damped_scalar")

        _dbg("-" * 70)
        _dbg(f"[SCALAR] FINAL WEIGHTS (scalar FxFx): " +
             ", ".join(f"{n}={v:.6f}" for n, v in zip(self.SUDAKOV_VARIANT_NAMES, sudrats)))
        _dbg(f"[SCALAR] event_id={CURRENT_EVENT_ID}, event.wgt={event.wgt}")
        _dbg(f"[SCALAR] reweighted (central) = {event.wgt} × {sudrats[0]:.6f} = {event.wgt * sudrats[0]:.6e}")
        _dbg("=" * 70)
        _dbg(f"[SCALAR] _compute_ewsudakov_fxfx_reweight EXIT: passing sudrats={sudrats} to _build_sudakov_rwgt_dict")
        return self._build_sudakov_rwgt_dict(event, sudrats)

    def _compute_density_ewsudakov_reweight(self, event, sud_mod):
        """Density-matrix EW Sudakov reweighting (FxFx, ickkw=3).

        For events with decayed t, W, Z resonances: Δ_h is applied per
        resonance helicity on the Born-like density matrix B of the clustered
        event and contracted with the decay matrix C,
        w = <B^EWSL, C> / <B, C> (compute_density_weight). Events without a
        decayed resonance, or whose density evaluation fails, use the scalar
        FxFx path. Returns the rwgt dict of _build_sudakov_rwgt_dict.

        Options (set via self attributes):
        - density_verbose: Enable verbose logging (default: False)
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

        density_logger = getattr(self, "_density_logger", None)
        if density_logger is None:
            density_logger = DensitySudakovLogger(verbose=getattr(self, "density_verbose", True))  # default flipped True 2026-06-19 (full debug; revert with the file)
            self._density_logger = density_logger

        # 2 -> 1 input: unit weights before clustering (see
        # _compute_ewsudakov_fxfx_reweight).
        n_init  = sum(1 for p in event if p.status == -1)
        n_final = sum(1 for p in event if p.status ==  1)
        if n_init == 2 and n_final == 1:
            _dbg("  -> RAW 2→1 TOPOLOGY (pre-cluster), returning weight=1 pass-through")
            density_logger.log_event("passthrough_2to1_raw", [], 0, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        _dbg("Step 1: Clustering event (FxFx)...")
        cluster_result = self._cluster_fxfx_event(event, record_tag=True, return_groups=True)
        if not cluster_result:
            # Evaluate the scalar path when clustering information is unavailable.
            _dbg("  -> NO CLUSTERING INFO, falling back to FxFx scalar path")
            density_logger.log_event("fallback_no_cluster", [], 0, 1.0)
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        clustered_event, cluster_sorted_tag, groups = cluster_result

        # Fallback policy as in _compute_ewsudakov_fxfx_reweight.
        _fks_hist = getattr(clustered_event, "fks_methods", [])
        if any(m not in FKS_SUCCESS_METHODS for m in _fks_hist):
            _count_fks_event("events_with_fsr_fallback_density")
            if FKS_FALLBACK_POLICY == "unit_weight":
                _dbg(f"  -> FSR fallback in clustering history {_fks_hist} and policy=unit_weight -> 3×1.0")
                _count_fks_event("events_unit_weight_by_policy")
                return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        if self._is_2to1_topology(clustered_event):
            _dbg("  -> 2→1 TOPOLOGY, returning weight=1 pass-through")
            density_logger.log_event("passthrough_2to1", [], 0, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        _dbg("  -> Clustered event particles:")
        for i, p in enumerate(clustered_event):
            _dbg(f"     [{i}] pdg={p.pid:>4}, status={p.status:>2}, E={p.E:.4f}")

        incoming = [p for p in clustered_event if p.status < 0]
        if len(incoming) >= 2:
            E1, E2 = incoming[0].E, incoming[1].E
            pz1, pz2 = incoming[0].pz, incoming[1].pz
            s_hat = (E1 + E2) ** 2 - (pz1 + pz2) ** 2  # beams along z
            _dbg(
                f"  -> KINEMATICS CHECK: E1={E1:.2f}, E2={E2:.2f}, sqrt(s_hat)={abs(s_hat)**0.5:.2f} GeV"
            )
            if E1 < 0 or E2 < 0:
                _dbg(f"*** WARNING: NEGATIVE BEAM ENERGY! E1={E1:.2f}, E2={E2:.2f} ***")
            if s_hat < 0:
                _dbg(f"*** WARNING: SPACELIKE s_hat! s_hat={s_hat:.2f} GeV^2 ***")

            # Degenerate beams (a zero-energy beam or ŝ < 1 GeV², e.g. a
            # lightlike final state) give no CM frame for the density matrices.
            if E1 < 1e-6 or E2 < 1e-6 or s_hat < 1.0:
                _dbg(
                    "  -> DEGENERATE KINEMATICS (light-like system), falling back to scalar FxFx path"
                )
                return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        _dbg("Step 2: Identifying resonances...")
        resonances = ResonanceIdentifier.find_resonances(clustered_event)
        _dbg(f"  -> Found {len(resonances)} candidate resonances: {resonances}")

        # Only resonances rebuilt from decay products count: a singleton group
        # is a stable final-state particle.
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

        # No decayed resonance (only γ/g mothers, or stable resonances such as
        # the tops in ttH): scalar path.
        if not resonances:
            _dbg("  -> NO RESONANCES found, falling back to scalar FxFx path")
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # Fixed buffers of the generated density kernel: density_delta and
        # born_diag(36), res arrays(10), allow_hel(300). Beyond them the
        # assumed-size Fortran arrays overrun silently: scalar path.
        if not density_dimensions_supported(total_dim, len(resonances)):
            _count_fks_event("events_density_dim_overflow")
            global _DIM_OVERFLOW_WARNED
            if not _DIM_OVERFLOW_WARNED:
                logger.warning(
                    "Resonance-helicity space too large for the density kernel "
                    "(total_dim=%d, nres=%d; caps %d/%d/%d): falling back "
                    "to the scalar Sudakov path.",
                    total_dim,
                    len(resonances),
                    MAX_DENSITY_DIM,
                    MAX_DENSITY_RESONANCES,
                    MAX_DENSITY_ALLOW_HEL,
                )
                _DIM_OVERFLOW_WARNED = True
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

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
        # Decay systems of C from the clustering groups (evaluated later).
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
        except RuntimeError:
            _dbg("  -> RuntimeError: momentum order mismatch!")
            sys.exit(3)

        if self._has_small_invariants(prep["p_in"], prep["iflist"]):
            _dbg("  -> SMALL INVARIANT (s_ij < MW²), returning weight=1 pass-through")
            density_logger.log_event("passthrough_small_sij", resonances, total_dim, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        weight_var = None
        try:
            _dbg("Step 4-5: Evaluating Sudakov corrections...")
            _dbg(f"  -> has density_sudakov attr: {hasattr(sud_mod, 'density_sudakov')}")
            use_density_path = False

            if hasattr(sud_mod, "density_sudakov") and resonances:
                try:
                    # 1-based resonance positions in the canonical MG5 order
                    # (prep["perm"] maps the final state of clustered_event).
                    res_indices_fortran = []
                    offset = len(prep["sorted_tag"][0])
                    perm = prep["perm"]
                    for idx, pdg in resonances:
                        if idx < offset:
                            fortran_idx = idx + 1
                        else:
                            final_idx = idx - offset
                            canonical_final_idx = perm[final_idx]
                            canonical_idx = canonical_final_idx + offset
                            fortran_idx = canonical_idx + 1
                        res_indices_fortran.append(fortran_idx)

                    _dbg("  -> Calling density_sudakov with:")
                    _dbg(f"     sorted_tag: {prep['sorted_tag']}")
                    _dbg(f"     res_indices_fortran: {res_indices_fortran}")
                    _dbg(f"     res_dims: {res_dims}")

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

                    # The Fortran wrapper returns zeros for density_born; B comes
                    # from DensityHelper and the standalone density_prod modules
                    # (GET_DENSITY).
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
                    # s_to_rij=False variation
                    delta_var_raw = density_result.get("density_delta_var", None)
                    # Per-helicity Born from Fortran (summed over spectators, /IDEN)
                    born_diag_fortran_raw = density_result["born_diag"]

                    # Fortran -> Python helicity order.
                    _hel_perm = _fortran_to_python_hel_permutation(res_dims)
                    delta = np.array([delta_raw[_hel_perm[i]] for i in range(len(delta_raw))])
                    delta_var = None
                    if delta_var_raw is not None:
                        delta_var = np.array(
                            [delta_var_raw[_hel_perm[i]] for i in range(len(delta_var_raw))]
                        )
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

                    # Fortran returns Δ_h times the per-helicity Born, and that
                    # Born, both divided by IDEN, which cancels in
                    # Δ_h = density_delta[h] / born_diag[h].
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

                    C = self._get_decay_density_matrix(
                        event,
                        clustered_event,
                        resonances,
                        frame_transform=prep.get("frame_transform"),
                    )
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
                        contr_BC = B.contract(C)
                        _dbg(f"     <B,C> = {contr_BC:.6e}")

                        # B_EWSL construction
                        B_ewsl = B.apply_sudakov(delta)
                        _dbg(f"     B_EWSL diagonal: {np.real(np.diag(B_ewsl.matrix))}")
                        contr_BewslC = B_ewsl.contract(C)
                        _dbg(f"     <B_EWSL,C> = {contr_BewslC:.6e}")

                        weight = compute_density_weight(B, C, delta)
                        _dbg(f"     density_weight (central): {weight:.6f}")

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

            # Scalar path if the density evaluation is unavailable or failed.
            if not use_density_path:
                _dbg("=" * 70)
                _dbg("===== SCALAR PATH (fallback) =====")
                _dbg("=" * 70)
                res = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
                born, sud_weak = res[0], res[2]
                weight = 1.0 + sud_weak / born if abs(born) > 1e-30 else 1.0
                # s_to_rij_off variant res[3], as in the scalar FxFx path.
                weight_var = (1.0 + res[3] / born) if abs(born) > 1e-30 else 1.0
                _dbg(f"     born={born:.6e}, sud_weak={sud_weak:.6e}")
                _dbg(f"     weight = 1 + sud_weak/born = {weight:.6f}")
                _dbg(f"     weight_var = 1 + res[3]/born = {weight_var:.6f}  (s_to_rij_off)")

            density_logger.log_event(
                mode="density" if use_density_path else "scalar",
                resonances=resonances,
                dimension=total_dim,
                weight=weight,
            )

        except Exception as e:
            _dbg(f"  -> EXCEPTION in density Sudakov: {e}")
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # The scalar kernel supplies the Sherpa QED factor; the density kernel
        # supplies the central and s_to_rij_off factors.
        weight_sherpa_qed = 1.0
        try:
            res_for_sq = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
            if abs(res_for_sq[0]) > 1e-30:
                weight_sherpa_qed = 1.0 + res_for_sq[1] / res_for_sq[0]
                _dbg(
                    f"  -> sherpa_qed extraction: born={res_for_sq[0]:.6e}, "
                    f"sq_delta={res_for_sq[1]:.6e}, weight_sherpa_qed={weight_sherpa_qed:.6f}"
                )
            else:
                _dbg(
                    f"  -> sherpa_qed extraction: |born|={abs(res_for_sq[0]):.3e} < 1e-30, "
                    f"weight_sherpa_qed=1.0 (no reweight)"
                )
        except Exception as exc:  # noqa: BLE001 — robustness: never let extraction failure abort the event
            _dbg(f"  -> WARNING: sherpa_qed extraction via scalar ewsudakov() failed ({exc}); using 1.0")
            weight_sherpa_qed = 1.0
        weight, weight_var, weight_sherpa_qed = limit_sudakov_factors(
            [weight, weight_var if weight_var is not None else 1.0, weight_sherpa_qed],
            "events_runaway_damped_density")

        _dbg("-" * 70)
        _dbg(f"FINAL WEIGHT (central): {weight:.6f}")
        if weight_var is not None:
            _dbg(f"FINAL WEIGHT (s_to_rij=False): {weight_var:.6f}")
        _dbg(f"FINAL WEIGHT (sherpa_qed): {weight_sherpa_qed:.6f}")
        _dbg(f"Event original weight: {event.wgt}")
        _dbg(f"Reweighted = {event.wgt} * {weight:.6f} = {event.wgt * weight:.6e}")
        _dbg("=" * 70)
        return self._build_sudakov_rwgt_dict(event, [weight, weight_var, weight_sherpa_qed])

    # -------------------------------------------------------------------------
    # Decay density matrices
    # -------------------------------------------------------------------------

    def _get_decay_density_matrix(
        self, full_event, clustered_event, resonances, frame_transform=None
    ):
        """Decay density matrix C = C_1 ⊗ ... ⊗ C_n of self._density_last_C_def.

        None if there is no decay system or any C_r cannot be computed.
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
                full_event,
                clustered_event,
                res_idx,
                res_pdg,
                decay_lhe,
                frame_transform=frame_transform,
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
        self,
        full_event,
        clustered_event,
        res_idx,
        res_pdg,
        decay_lhe,
        frame_transform=None,
    ):
        """Minimal decay event: the clustered resonance -> its decay products.

        Clustering supplies leaf momenta transported with each recoil and parent
        projection. Event preparation supplies the final CM boost and rotation.
        Without stored transport, the LHE momenta define the decay projection.
        """
        decay_event = lhe_parser.Event()
        decay_event.wgt = getattr(full_event, "wgt", 0.0)
        decay_event.aqcd = getattr(full_event, "aqcd", 0.0)

        res_part = clustered_event[res_idx]
        onshell_res_p4 = FourMomentum([res_part.E, res_part.px, res_part.py, res_part.pz])
        _dbg(f"     [decay_build] res_idx={res_idx}, res_pdg={res_pdg}")
        _dbg(
            f"     [decay_build] on-shell res_p4=({res_part.E:.6e},{res_part.px:.6e},{res_part.py:.6e},{res_part.pz:.6e})"
        )

        # LHE labels identify the transported leaves and their particle metadata.
        transported = getattr(clustered_event, "_fxfx_decay_products", None)
        original_momenta = []
        decay_pids = []
        decay_masses = []
        sum_E = sum_px = sum_py = sum_pz = 0.0

        for lhe_idx in decay_lhe:
            if lhe_idx < 1 or lhe_idx > len(full_event):
                continue
            src = full_event[lhe_idx - 1]
            p4 = (FourMomentum(transported[res_idx][lhe_idx]) if transported is not None
                  else FourMomentum(src))
            original_momenta.append(p4)
            decay_pids.append(int(getattr(src, "pid", 0)))
            decay_masses.append(float(getattr(src, "mass", 0.0)))
            sum_E += p4.E
            sum_px += p4.px
            sum_py += p4.py
            sum_pz += p4.pz

        # Parent momentum carried by the input daughters.
        original_res_p4 = FourMomentum([sum_E, sum_px, sum_py, sum_pz])
        original_m2 = sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2
        original_m = math.sqrt(max(0, original_m2))
        _dbg(
            f"     [decay_build] input res_p4=({sum_E:.6e},{sum_px:.6e},{sum_py:.6e},{sum_pz:.6e})"
        )
        _dbg(f"     [decay_build] input invariant mass = {original_m:.4f} GeV")
        _dbg(
            f"     [decay_build] delta (before transform)=({res_part.E - sum_E:.6e},{res_part.px - sum_px:.6e},{res_part.py - sum_py:.6e},{res_part.pz - sum_pz:.6e})"
        )

        transformed_momenta = _transform_decay_products_to_onshell(
            original_momenta,
            original_res_p4,
            onshell_res_p4,
            frame_transform=frame_transform,
        )

        res_particle = lhe_parser.Particle(event=decay_event)
        res_particle.pid = int(res_pdg)
        res_particle.status = -1
        res_particle.px = float(res_part.px)
        res_particle.py = float(res_part.py)
        res_particle.pz = float(res_part.pz)
        res_particle.E = float(res_part.E)
        res_particle.mass = float(getattr(res_part, "mass", 0.0))
        decay_event.append(res_particle)

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
        _dbg(f"       Input invariant mass: {original_m:.4f} GeV")
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
        """DensityMatrix of target_pdgs from GET_DENSITY of the standalone modules.

        A decay event (one incoming particle) uses rw_me_density_decay (C), a
        production event rw_me_density_prod (B). Returns None on failure;
        label is the debug label for logging.
        """
        _dbg(f"     [{label}] _compute_density_matrix_from_event: start")
        _dbg(f"     [{label}] target_pdgs={target_pdgs}, res_dims={res_dims}")
        _dbg(f"     [{label}] event particles={len(event)}")

        try:
            tag_raw, order = event.get_tag_and_order()
            nincoming = len(tag_raw[0])
        except Exception:
            nincoming = 2

        _dbg(f"     [{label}] nincoming={nincoming}")

        if nincoming == 1:
            if not getattr(self, "density_decay_id_to_path", None):
                _dbg(f"     [{label}] No density_decay_id_to_path available")
                return None
            id_to_path = self.density_decay_id_to_path
            f2pylib = self.density_decay_f2pylib
            _dbg(f"     [{label}] Using density_decay maps (keys={list(id_to_path.keys())[:3]}...)")
        else:
            if not getattr(self, "density_prod_id_to_path", None):
                _dbg(f"     [{label}] No density_prod_id_to_path available")
                return None
            id_to_path = self.density_prod_id_to_path
            f2pylib = self.density_prod_f2pylib
            _dbg(f"     [{label}] Using density_prod maps (keys={list(id_to_path.keys())[:3]}...)")

        helper = DensityHelper(self, id_to_path, f2pylib)

        try:
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
