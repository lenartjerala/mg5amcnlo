################################################################################
#
# Unit tests for the FKS kinematic reshuffling used by the FxFx EW-Sudakov
# clustering (madgraph/various/fxfx_ewsudakov.py).
#
# Covers:
#   - exactness of the on-shell FSR mapping (massive and massless spectators)
#   - the ISR mapping (momentum conservation, massless beams, reduced sqrt(s))
#   - the decay-product on-shell transformation (2-body and multi-body)
#   - the reachable fallback branches of _fks_fsr_mapping via engineered
#     kinematics: zero_krec_momentum, negative_energy, negative_r,
#     unreachable_target (incl. the pre-fix spurious-root acceptance),
#     no_solution, degenerate, collinear_unphysical. NOT covered:
#     degenerate_collinear (vestigial: only reachable for 0 < R < 1e-10 after
#     the R <= 0 guard), spurious_root (belt-and-suspenders, unreachable if
#     the R <= 0 guard and A >= |B| hold), superluminal (boundary-only: all
#     real roots of the squared quadratic satisfy |beta| <= 1 identically)
#   - mass-label consistency and method accounting in
#     fxfx_merge_particles_kinematics (label == sqrt(p^2) for fallbacks,
#     pole mass for successes; check_kinematics_only passes in both cases)
#   - the A >= |B| spurious-root safety property of the quadratic solver
#   - fks_methods threading and skip sentinels in _apply_forced_clustering
#   - the counting gate (_FKS_COUNT_ACTIVE) and fks_reset_accounting()
#
# NOT covered here (requires compiled f2py Sudakov modules): the unit_weight
# policy branch inside the reweight drivers and the end-of-run summary hook in
# reweight_interface.
################################################################################
from __future__ import division

import math
import os
import random
import sys
import unittest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import madgraph.various.fxfx_ewsudakov as fx  # noqa: E402  (path bootstrap above)
import madgraph.various.lhe_parser as lhe_parser  # noqa: E402

FourMomentum = fx.FourMomentum


def _m2(p):
    return p.E**2 - p.px**2 - p.py**2 - p.pz**2


def _mk_particle(event, pid, status, E, px, py, pz, mass=None):
    part = lhe_parser.Particle(event=event)
    part.pid = pid
    part.status = status
    part.E = E
    part.px = px
    part.py = py
    part.pz = pz
    if mass is None:
        mass = math.sqrt(max(0.0, E**2 - px**2 - py**2 - pz**2))
        if mass < 1e-9:
            mass = 0.0
    part.mass = mass
    part.mother1 = 0
    part.mother2 = 0
    part.color1 = 0
    part.color2 = 0
    part.vtim = 0
    part.helicity = 9
    return part


def _mk_event(entries):
    """entries: list of (pid, status, E, px, py, pz)."""
    event = lhe_parser.Event()
    for pid, status, E, px, py, pz in entries:
        event.append(_mk_particle(event, pid, status, E, px, py, pz))
    return event


def _fs_sum(event):
    total = FourMomentum()
    for idx in range(2, len(event)):
        total += FourMomentum(event[idx])
    return total


def _split_massless_z(E, pz):
    """Split (E,0,0,pz) into two massless momenta along +/- z (light-cone)."""
    plus = (E + pz) / 2.0
    minus = (E - pz) / 2.0
    return (plus, 0.0, 0.0, plus), (minus, 0.0, 0.0, -minus)


class TestFSRSuccessPath(unittest.TestCase):
    """The mapping must place the mother exactly on the pole mass."""

    def test_massless_spectator_onshell(self):
        # CM 500 GeV; single massless spectator along +x; pair -> W
        beams = [FourMomentum([250, 0, 0, 250]), FourMomentum([250, 0, 0, -250])]
        spec = FourMomentum([100, 100, 0, 0])
        p_a = FourMomentum([150, 150, 0, 0])
        p_b = FourMomentum([250, -250, 0, 0])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self.assertEqual(res["method"], "fks_fsr_massive_onshell")
        mother = res["mother"]
        self.assertAlmostEqual(math.sqrt(_m2(mother)), fx.MW_POLE, places=6)
        # exact momentum conservation
        total = FourMomentum(mother)
        for p in res["spectators"].values():
            total += p
        q = beams[0] + beams[1]
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(getattr(total, comp), getattr(q, comp), places=6)

    def test_massive_spectator_onshell_random(self):
        random.seed(7)
        n_ok = 0
        for _ in range(200):
            ecm = random.uniform(300.0, 2000.0)
            beams = [
                FourMomentum([ecm / 2, 0, 0, ecm / 2]),
                FourMomentum([ecm / 2, 0, 0, -ecm / 2]),
            ]
            q = beams[0] + beams[1]
            parts = []
            for _n in range(3):
                px, py, pz = (random.uniform(-ecm / 6, ecm / 6) for _c in range(3))
                E = math.sqrt(px**2 + py**2 + pz**2)
                parts.append(FourMomentum([E, px, py, pz]))
            tot = parts[0] + parts[1] + parts[2]
            last = FourMomentum([q.E - tot.E, q.px - tot.px, q.py - tot.py, q.pz - tot.pz])
            if last.E <= 0 or _m2(last) <= 0:
                continue
            momenta = beams + parts + [last]
            res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 23}], momenta)
            if res["method"] != "fks_fsr_massive_onshell":
                continue
            n_ok += 1
            self.assertAlmostEqual(
                math.sqrt(_m2(res["mother"])) / fx.MZ_POLE, 1.0, places=6
            )
            # spectator-system invariant mass is preserved by the boost
            krec = FourMomentum()
            krec_new = FourMomentum()
            for idx, p in res["spectators"].items():
                krec += momenta[idx]
                krec_new += p
            self.assertAlmostEqual(_m2(krec_new) / _m2(krec), 1.0, places=6)
        self.assertGreater(n_ok, 50)

    def test_no_spectators_beam_scaling(self):
        beams = [FourMomentum([50, 0, 0, 50]), FourMomentum([50, 0, 0, -50])]
        p_a = FourMomentum([50, 50, 0, 0])
        p_b = FourMomentum([50, -50, 0, 0])
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 23}], beams + [p_a, p_b])
        self.assertEqual(res["method"], "fks_fsr_no_spectators_beam_scaled")
        self.assertAlmostEqual(math.sqrt(_m2(res["mother"])), fx.MZ_POLE, places=6)
        # beam rapidity (E ratio) is preserved
        self.assertAlmostEqual(res["beams"][0].E / res["beams"][1].E, 1.0, places=9)


class TestFSRFallbackBranches(unittest.TestCase):
    """One engineered kinematic configuration per fallback branch."""

    def _naive(self, momenta, i, j):
        return momenta[i] + momenta[j]

    def _assert_fallback(self, res, momenta, method):
        self.assertEqual(res["method"], method)
        naive = self._naive(momenta, 2, 3)
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(
                getattr(res["mother"], comp), getattr(naive, comp), places=9
            )

    def test_zero_krec_momentum(self):
        beams = [FourMomentum([100, 0, 0, 100]), FourMomentum([100, 0, 0, -100])]
        p_a = FourMomentum([50, 50, 0, 0])
        p_b = FourMomentum([50, -50, 0, 0])
        spec1 = FourMomentum([50, 0, 0, 50])
        spec2 = FourMomentum([50, 0, 0, -50])
        momenta = beams + [p_a, p_b, spec1, spec2]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 23}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_zero_krec_momentum")

    def test_negative_energy(self):
        # sqrt(s) = 60 GeV < M_W: no on-shell W + spectator fits in the event
        beams = [FourMomentum([30, 0, 0, 30]), FourMomentum([30, 0, 0, -30])]
        spec = FourMomentum([10, 10, 0, 0])  # massless
        p_a = FourMomentum([20, -20, 0, 0])
        p_b = FourMomentum([30, 10, 0, 0])
        pair = p_a + p_b
        self.assertAlmostEqual(pair.E, 50.0)
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_negative_energy")

    def test_negative_r(self):
        # sqrt(s) = M_W exactly: target boosted spectator energy is exactly zero
        M = fx.MW_POLE
        beams = [FourMomentum([M / 2, 0, 0, M / 2]), FourMomentum([M / 2, 0, 0, -M / 2])]
        spec = FourMomentum([10, 10, 0, 0])
        p_a = FourMomentum([(M - 20) / 2, (M - 20) / 2, 0, 0])
        p_b = FourMomentum([M / 2, -M / 2, 0, 0])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_negative_r")

    def test_unreachable_target(self):
        # sqrt(s)=60, light massive spectator system (m_rec=5): R < 0, i.e. the
        # event cannot host an on-shell W plus this spectator system. The
        # squared quadratic HAS |beta|<1 roots here, but they all sit on the
        # spurious -R branch (gamma(A+beta B) = -R): before the explicit
        # R <= 0 guard the mapping accepted one, reported
        # "fks_fsr_massive_onshell", and delivered a mother at a WRONG mass
        # (~28 GeV for this configuration). This test pins the guard.
        beams = [FourMomentum([30, 0, 0, 30]), FourMomentum([30, 0, 0, -30])]
        E_r, p_r = math.sqrt(34.0), 3.0  # m_rec^2 = 25
        (k1E, _x, _y, k1z), (k2E, _x2, _y2, k2z) = _split_massless_z(E_r, p_r)
        spec1 = FourMomentum([k1E, 0, 0, k1z])
        spec2 = FourMomentum([k2E, 0, 0, k2z])
        pair = FourMomentum([60 - E_r, 0, 0, -p_r])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        p_a = FourMomentum([paE, 0, 0, paz])
        p_b = FourMomentum([pbE, 0, 0, pbz])
        momenta = beams + [p_a, p_b, spec1, spec2]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_unreachable_target")
        # and the wrong-mass acceptance must NOT happen: the naive mother is
        # far below the pole, and no branch may relabel it as on-shell
        self.assertLess(math.sqrt(max(0.0, _m2(res["mother"]))), fx.MW_POLE - 10.0)

    def test_no_solution(self):
        # sqrt(s)=100, m_rec=30: 0 < R but R^2 < A^2 - B^2 -> negative discriminant
        beams = [FourMomentum([50, 0, 0, 50]), FourMomentum([50, 0, 0, -50])]
        E_r = 30.4
        p_r = math.sqrt(E_r**2 - 900.0)
        (k1E, _x, _y, k1z), (k2E, _x2, _y2, k2z) = _split_massless_z(E_r, p_r)
        spec1 = FourMomentum([k1E, 0, 0, k1z])
        spec2 = FourMomentum([k2E, 0, 0, k2z])
        pair = FourMomentum([100 - E_r, 0, 0, -p_r])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        p_a = FourMomentum([paE, 0, 0, paz])
        p_b = FourMomentum([pbE, 0, 0, pbz])
        momenta = beams + [p_a, p_b, spec1, spec2]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_no_solution")

    def test_degenerate_lightlike_q(self):
        # q lightlike along the (massless) spectator direction: denominator = 0
        beams = [FourMomentum([50, 0, 0, 50]), FourMomentum([30, 0, 0, 30])]
        spec = FourMomentum([10, 0, 0, 10])
        p_a = FourMomentum([35, 0, 0, 35])
        p_b = FourMomentum([35, 0, 0, 35])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_degenerate")

    def test_collinear_unphysical(self):
        # Massive spectator collinear with q direction (B = 0), gamma^2 < 1
        beams = [FourMomentum([60, 0, 0, 60]), FourMomentum([40, 0, 0, -40])]
        q = beams[0] + beams[1]  # (100, 0, 0, 20)
        k_E = 20.0
        k_z = k_E * q.pz / q.E  # 4.0 -> B = 0 exactly, m_rec^2 = 384
        spec = FourMomentum([k_E, 0, 0, k_z])
        pair = FourMomentum([q.E - k_E, 0, 0, q.pz - k_z])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        p_a = FourMomentum([paE, 0, 0, paz])
        p_b = FourMomentum([pbE, 0, 0, pbz])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_collinear_unphysical")

    def test_threshold_collinear_is_unreachable(self):
        # B = 0 and R = 0: s + m_rec^2 = M^2 with a q-collinear massive
        # spectator. The event sits exactly AT threshold (beta = +-1 would be
        # needed), so the R <= 0 unreachable-target guard fires before the
        # legacy B~0 "degenerate_collinear" branch, which is now reachable only
        # for 0 < R < 1e-10 and kept as a vestigial safety net.
        M2 = fx.MW_POLE**2
        beams = [FourMomentum([40, 0, 0, 40]), FourMomentum([24, 0, 0, -24])]
        q = beams[0] + beams[1]  # (64, 0, 0, 16)
        s = _m2(q)
        m_rec2 = M2 - s
        self.assertGreater(m_rec2, 0.0)
        ratio = q.pz / q.E
        k_E = math.sqrt(m_rec2 / (1.0 - ratio**2))
        k_z = k_E * ratio
        spec = FourMomentum([k_E, 0, 0, k_z])
        pair = FourMomentum([q.E - k_E, 0, 0, q.pz - k_z])
        p_a = FourMomentum([pair.E / 2, pair.px / 2, 0, pair.pz / 2])
        p_b = FourMomentum([pair.E / 2, pair.px / 2, 0, pair.pz / 2])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        # rhs is an exact float 0 for this construction; do not depend on which
        # non-success branch classifies it — the physics statement is only that
        # an at-threshold configuration must never be labelled a success.
        self.assertNotIn(res["method"], fx.FKS_SUCCESS_METHODS)
        naive = self._naive(momenta, 2, 3)
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(
                getattr(res["mother"], comp), getattr(naive, comp), places=9
            )


class TestMergeLabelConsistency(unittest.TestCase):
    """fxfx_merge_particles_kinematics: mass label always matches sqrt(p^2),
    the method string is returned and counted, and check_kinematics_only
    passes for successes AND fallbacks."""

    def _merge(self, entries, i, j, mother_pdg):
        event = _mk_event(entries)
        before = dict(fx.FKS_METHOD_COUNTS)
        cons_before = fx.FKS_QUALITY_COUNTS["merge_momentum_conservation_violation"]
        # Enable the counting gate (normally set by _cluster_fxfx_event for the
        # first clustering pass of each event of the main loop).
        prev_gate = fx._FKS_COUNT_ACTIVE
        fx._FKS_COUNT_ACTIVE = True
        try:
            method = fx.fxfx_merge_particles_kinematics(
                event, i, j, [{"number": i + 1, "id": mother_pdg}]
            )
        finally:
            fx._FKS_COUNT_ACTIVE = prev_gate
        self.assertEqual(
            fx.FKS_METHOD_COUNTS[method], before.get(method, 0) + 1,
            "method counter did not increment",
        )
        self.assertEqual(
            fx.FKS_QUALITY_COUNTS["merge_momentum_conservation_violation"],
            cons_before,
            "momentum conservation violated by merge",
        )
        return event, method

    def test_success_label_is_pole_mass(self):
        entries = [
            (21, -1, 250, 0, 0, 250),
            (21, -1, 250, 0, 0, -250),
            (13, 1, 150, 150, 0, 0),
            (-14, 1, 250, -250, 0, 0),
            (21, 1, 100, 100, 0, 0),
        ]
        event, method = self._merge(entries, 2, 3, -24)
        self.assertEqual(method, "fks_fsr_massive_onshell")
        mother = event[2]
        self.assertAlmostEqual(mother.mass, fx.MW_POLE, places=9)
        self.assertAlmostEqual(
            math.sqrt(max(0.0, _m2(FourMomentum(mother)))), fx.MW_POLE, places=6
        )
        event.check_kinematics_only()

    def test_fallback_label_is_invariant_mass(self):
        # no_solution configuration (see TestFSRFallbackBranches.test_no_solution)
        E_r = 30.4
        p_r = math.sqrt(E_r**2 - 900.0)
        (k1E, _x, _y, k1z), (k2E, _x2, _y2, k2z) = _split_massless_z(E_r, p_r)
        pair = FourMomentum([100 - E_r, 0, 0, -p_r])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        entries = [
            (21, -1, 50, 0, 0, 50),
            (21, -1, 50, 0, 0, -50),
            (13, 1, paE, 0, 0, paz),
            (-14, 1, pbE, 0, 0, pbz),
            (21, 1, k1E, 0, 0, k1z),
            (21, 1, k2E, 0, 0, k2z),
        ]
        event, method = self._merge(entries, 2, 3, -24)
        self.assertEqual(method, "fks_fsr_no_solution")
        mother = event[2]
        m_kin = math.sqrt(max(0.0, _m2(FourMomentum(mother))))
        self.assertAlmostEqual(mother.mass, m_kin, places=9)
        self.assertNotAlmostEqual(mother.mass, fx.MW_POLE, places=1)
        # the >0.1% label/kinematics gate must NOT fire on the fallback
        event.check_kinematics_only()

    def test_zero_krec_label_is_invariant_mass(self):
        entries = [
            (21, -1, 100, 0, 0, 100),
            (21, -1, 100, 0, 0, -100),
            (13, 1, 50, 50, 0, 0),
            (-14, 1, 50, -50, 0, 0),
            (21, 1, 50, 0, 0, 50),
            (21, 1, 50, 0, 0, -50),
        ]
        event, method = self._merge(entries, 2, 3, 23)
        self.assertEqual(method, "fks_fsr_zero_krec_momentum")
        mother = event[2]
        m_kin = math.sqrt(max(0.0, _m2(FourMomentum(mother))))
        self.assertAlmostEqual(mother.mass, m_kin, places=9)
        event.check_kinematics_only()

    def test_isr_returns_method(self):
        entries = [
            (21, -1, 500, 0, 0, 500),
            (1, -1, 500, 0, 0, -500),
            (23, 1, 500, 100, 0, 20),
            (21, 1, 500, -100, 0, -20),
        ]
        event = _mk_event(entries)
        method = fx.fxfx_merge_particles_kinematics(
            event, 0, 3, [{"number": 1, "id": 1}]
        )
        self.assertEqual(method, "fks_isr_cm_boost")
        self.assertEqual(len(event), 3)


class TestISRMapping(unittest.TestCase):
    def test_conservation_and_beams(self):
        beams = [FourMomentum([400, 0, 0, 400]), FourMomentum([300, 0, 0, -300])]
        rad = FourMomentum([50, 30, -20, 10])
        q = beams[0] + beams[1]
        rest = FourMomentum([q.E - rad.E, q.px - rad.px, q.py - rad.py, q.pz - rad.pz])
        p_a = FourMomentum([rest.E / 2, rest.px / 2 + 5, rest.py / 2, rest.pz / 2 - 5])
        p_b = FourMomentum(
            [rest.E - p_a.E, rest.px - p_a.px, rest.py - p_a.py, rest.pz - p_a.pz]
        )
        momenta = beams + [p_a, p_b, rad]
        res = fx._fks_isr_mapping(0, 4, [{"number": 1, "id": 1}], momenta)
        b0, b1 = res["beams"]
        self.assertAlmostEqual(_m2(b0), 0.0, places=5)
        self.assertAlmostEqual(_m2(b1), 0.0, places=5)
        pin = b0 + b1
        self.assertAlmostEqual(_m2(pin) / _m2(rest), 1.0, places=9)
        pfs = FourMomentum()
        for p in res["final_state"].values():
            pfs += p
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(getattr(pin, comp), getattr(pfs, comp), places=5)

    def test_lightlike_guard(self):
        beams = [FourMomentum([100, 0, 0, 100]), FourMomentum([1, 0, 0, -1])]
        rad = FourMomentum([1, 0, 0, -1])
        p_a = FourMomentum([100, 0, 0, 100])
        momenta = beams + [p_a, rad]
        with self.assertRaises(ValueError):
            fx._fks_isr_mapping(0, 3, [{"number": 1, "id": 1}], momenta)


class TestDecayTransform(unittest.TestCase):
    def test_two_body_conservation_and_masses(self):
        # off-shell W (70 GeV) -> two massless; retarget to on-shell moving W
        minv = 70.0
        pstar = minv / 2
        d1 = FourMomentum([pstar, pstar, 0, 0])
        d2 = FourMomentum([pstar, -pstar, 0, 0])
        boost_dir = (0.6, 0.0, 0.8)
        d1lab = fx._boost_along_direction(d1, -0.7, boost_dir)
        d2lab = fx._boost_along_direction(d2, -0.7, boost_dir)
        orig = d1lab + d2lab
        wrest = FourMomentum([fx.MW_POLE, 0, 0, 0])
        target = fx._boost_along_direction(wrest, -0.5, (0.0, 1.0, 0.0))
        out = fx._transform_decay_products_to_onshell([d1lab, d2lab], orig, target)
        tot = out[0] + out[1]
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(getattr(tot, comp), getattr(target, comp), places=6)
        self.assertAlmostEqual(_m2(out[0]), 0.0, places=5)
        self.assertAlmostEqual(_m2(out[1]), 0.0, places=5)

    def test_forbidden_returns_original(self):
        # target mass below the sum of (massive) decay-product masses
        d1 = FourMomentum([60, 0, 0, 10])  # m ~ 59.2
        d2 = FourMomentum([60, 0, 0, -10])
        orig = d1 + d2
        target = FourMomentum([fx.MW_POLE, 0, 0, 0])
        out = fx._transform_decay_products_to_onshell([d1, d2], orig, target)
        self.assertAlmostEqual(out[0].E, d1.E, places=9)
        self.assertAlmostEqual(out[1].pz, d2.pz, places=9)

    def test_decay_follows_prepared_frame_without_wigner_rotation(self):
        # A longitudinal event-frame boost is non-collinear with this W.  With
        # zero mass shift the decay transport must reduce exactly to that same
        # frame boost; mixing lab daughters with the prepared mother instead
        # produces the spurious Wigner rotation fixed by B2.
        mass = fx.MW_POLE
        w_lab = FourMomentum(
            [math.sqrt(mass**2 + 60.0**2 + 150.0**2), 60.0, 0.0, 150.0]
        )
        direction = (0.3, 0.8, math.sqrt(1.0 - 0.3**2 - 0.8**2))
        d1_rest = FourMomentum(
            [
                mass / 2.0,
                mass / 2.0 * direction[0],
                mass / 2.0 * direction[1],
                mass / 2.0 * direction[2],
            ]
        )
        d2_rest = FourMomentum(
            [
                mass / 2.0,
                -mass / 2.0 * direction[0],
                -mass / 2.0 * direction[1],
                -mass / 2.0 * direction[2],
            ]
        )
        w_p = math.sqrt(w_lab.px**2 + w_lab.py**2 + w_lab.pz**2)
        w_dir = (w_lab.px / w_p, w_lab.py / w_p, w_lab.pz / w_p)
        daughters_lab = [
            fx._boost_along_direction(d1_rest, -w_p / w_lab.E, w_dir),
            fx._boost_along_direction(d2_rest, -w_p / w_lab.E, w_dir),
        ]

        # beta_z = 0.55, in the exact Event.boost convention recorded by the
        # preparation routine.
        frame_transform = [
            ("boost_to_cm", FourMomentum([1000.0, 0.0, 0.0, 550.0]))
        ]
        w_prepared = fx._apply_fxfx_frame_transform(w_lab, frame_transform)
        expected = [
            fx._apply_fxfx_frame_transform(p, frame_transform)
            for p in daughters_lab
        ]
        transformed = fx._transform_decay_products_to_onshell(
            daughters_lab,
            w_lab,
            w_prepared,
            frame_transform=frame_transform,
        )

        for got, want in zip(transformed, expected):
            for component in ("E", "px", "py", "pz"):
                self.assertAlmostEqual(
                    getattr(got, component), getattr(want, component), places=10
                )

        # Pin the regression: the old mixed-frame call is measurably different
        # despite conserving the total four-momentum.
        mixed_frame = fx._transform_decay_products_to_onshell(
            daughters_lab, w_lab, w_prepared
        )
        self.assertGreater(
            max(
                abs(getattr(mixed_frame[0], component) - getattr(expected[0], component))
                for component in ("E", "px", "py", "pz")
            ),
            1.0e-3,
        )


class TestDensityKernelLimits(unittest.TestCase):
    def test_fixed_buffer_limits(self):
        self.assertTrue(fx.density_dimensions_supported(9, 2))
        self.assertTrue(fx.density_dimensions_supported(36, 4))
        self.assertFalse(fx.density_dimensions_supported(81, 4))
        self.assertFalse(fx.density_dimensions_supported(36, 9))
        self.assertFalse(fx.density_dimensions_supported(1, 11))


class TestSudakovSourceFiltering(unittest.TestCase):
    def test_prior_sudakov_columns_are_not_compounded(self):
        class FakeEvent(object):
            wgt = 2.0

            @staticmethod
            def parse_reweight():
                return {
                    "1001": 2.0,
                    "2001": 200.0,
                    "2101": 210.0,
                    "2201": 220.0,
                }

        dummy = type(
            "DummyMixin",
            (),
            {
                "_current_xi_idx": 0,
                "SUDAKOV_VARIANT_NAMES":
                    fx.FxFxEWSudakovMixin.SUDAKOV_VARIANT_NAMES,
            },
        )()
        weights = [1.1, 0.9, 1.05]
        result = fx.FxFxEWSudakovMixin._build_sudakov_rwgt_dict(
            dummy, FakeEvent(), weights
        )

        self.assertEqual(set(result), {"orig", "2001", "2101", "2201"})
        self.assertAlmostEqual(result["2001"], 2.0 * weights[0])
        self.assertAlmostEqual(result["2101"], 2.0 * weights[1])
        self.assertAlmostEqual(result["2201"], 2.0 * weights[2])
        self.assertTrue(fx.is_prior_sudakov_id("2001"))
        self.assertFalse(fx.is_prior_sudakov_id("1001"))


class TestForcedClusteringPlumbing(unittest.TestCase):
    """_apply_forced_clustering: fks_methods threading, conditional mass rule,
    and the skip-path sentinel."""

    def _dummy_self(self):
        return type("DummyMixin", (), {})()

    def _madspin_event(self):
        # LHE 1-based: 1,2 beams; 3: mu+; 4: vm; 5: W+ mother (status 2);
        # 6: recoil parton. FS total = q = (500,0,0,0).
        event = _mk_event(
            [
                (21, -1, 250, 0, 0, 250),
                (21, -1, 250, 0, 0, -250),
                (-13, 1, 150, 150, 0, 0),
                (14, 1, 250, -250, 0, 0),
                (24, 2, 400, -100, 0, 0),
                (21, 1, 100, 100, 0, 0),
            ]
        )
        return event

    def test_success_threading_and_mass(self):
        event = self._madspin_event()
        step = fx.ForcedClusterStep(
            children_lhe_idx=[3, 4], mother_lhe_idx=5, mother_pdg=24,
            scale=100.0, depth=0,
        )
        out, groups = fx.FxFxEWSudakovMixin._apply_forced_clustering(
            self._dummy_self(), event, [step]
        )
        self.assertEqual(out.fks_methods, ["fks_fsr_massive_onshell"])
        mothers = [p for p in out if abs(getattr(p, "pid", 0)) == 24]
        self.assertEqual(len(mothers), 1)
        self.assertEqual(mothers[0].status, 1)
        self.assertAlmostEqual(mothers[0].mass, fx.MW_POLE, places=9)
        m_kin = math.sqrt(max(0.0, _m2(FourMomentum(mothers[0]))))
        self.assertAlmostEqual(m_kin, fx.MW_POLE, places=6)

    def test_skipped_step_records_sentinel(self):
        event = self._madspin_event()
        # 1->3 step: not a MadSpin 1->2 topology -> must be skipped AND recorded
        step = fx.ForcedClusterStep(
            children_lhe_idx=[3, 4, 6], mother_lhe_idx=5, mother_pdg=24,
            scale=100.0, depth=0,
        )
        out, groups = fx.FxFxEWSudakovMixin._apply_forced_clustering(
            self._dummy_self(), event, [step]
        )
        self.assertEqual(out.fks_methods, ["forced_step_skipped"])
        self.assertNotIn("forced_step_skipped", fx.FKS_SUCCESS_METHODS)
        # mother left un-restored (still status 2)
        mothers = [p for p in out if abs(getattr(p, "pid", 0)) == 24]
        self.assertEqual(mothers[0].status, 2)

    def test_missing_children_records_sentinel(self):
        event = self._madspin_event()
        step = fx.ForcedClusterStep(
            children_lhe_idx=[3, 99], mother_lhe_idx=5, mother_pdg=24,
            scale=100.0, depth=0,
        )
        out, groups = fx.FxFxEWSudakovMixin._apply_forced_clustering(
            self._dummy_self(), event, [step]
        )
        self.assertEqual(out.fks_methods, ["forced_step_skipped"])


class TestAccountingGateAndReset(unittest.TestCase):
    """Counting gate semantics and per-launch reset."""

    def test_gate_off_means_no_counting(self):
        entries = [
            (21, -1, 250, 0, 0, 250),
            (21, -1, 250, 0, 0, -250),
            (13, 1, 150, 150, 0, 0),
            (-14, 1, 250, -250, 0, 0),
            (21, 1, 100, 100, 0, 0),
        ]
        event = _mk_event(entries)
        prev_gate = fx._FKS_COUNT_ACTIVE
        fx._FKS_COUNT_ACTIVE = False
        before = dict(fx.FKS_METHOD_COUNTS)
        try:
            method = fx.fxfx_merge_particles_kinematics(
                event, 2, 3, [{"number": 3, "id": -24}]
            )
        finally:
            fx._FKS_COUNT_ACTIVE = prev_gate
        self.assertEqual(method, "fks_fsr_massive_onshell")
        self.assertEqual(
            fx.FKS_METHOD_COUNTS.get(method, 0), before.get(method, 0),
            "counter incremented although the gate was off",
        )

    def test_reset_clears_counters_and_gate_state(self):
        fx.FKS_METHOD_COUNTS["fks_fsr_massive_onshell"] += 3
        fx.FKS_QUALITY_COUNTS["events_legacy_fallback"] += 1
        fx._FKS_COUNT_ACTIVE = True
        fx._FKS_COUNTED_UPTO = 42
        fx.fks_reset_accounting()
        self.assertEqual(sum(fx.FKS_METHOD_COUNTS.values()), 0)
        self.assertEqual(sum(fx.FKS_QUALITY_COUNTS.values()), 0)
        self.assertFalse(fx._FKS_COUNT_ACTIVE)
        self.assertEqual(fx._FKS_COUNTED_UPTO, 0)

    def test_summary_reports_fallbacks(self):
        fx.fks_reset_accounting()
        fx.FKS_METHOD_COUNTS["fks_fsr_massive_onshell"] += 5
        fx.FKS_METHOD_COUNTS["fks_fsr_unreachable_target"] += 2
        text = fx.fks_accounting_summary()
        self.assertIn("7 clusterings", text)
        self.assertIn("2 fallback(s)", text)
        self.assertIn("fks_fsr_unreachable_target", text)
        self.assertIn("<-- fallback", text)
        fx.fks_reset_accounting()


class TestSpuriousRootSafety(unittest.TestCase):
    def test_A_geq_absB(self):
        """A±B = (q_E±q_n)(E_rec∓p_n) >= 0 for timelike q and physical k_rec,
        so the min-|beta| root of the squared equation always satisfies the
        unsquared +R constraint (see _fks_fsr_mapping docstring)."""
        random.seed(11)
        for _ in range(2000):
            qE = random.uniform(50, 3000)
            qvec = [random.uniform(-0.9, 0.9) * qE / math.sqrt(3.0) for _c in range(3)]
            if qE**2 - sum(x * x for x in qvec) <= 0:
                continue
            kE = random.uniform(1, 500)
            p_n = random.uniform(0, 1) * kE
            n = [random.gauss(0, 1) for _c in range(3)]
            norm = math.sqrt(sum(x * x for x in n))
            q_n = sum(a * b / norm for a, b in zip(qvec, n))
            A = qE * kE - q_n * p_n
            B = q_n * kE - qE * p_n
            self.assertGreaterEqual(A, abs(B) - 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
