C     =================================================================
C     =========
C     Density Matrix Delta-Only Wrapper for EW Sudakov
C     =================================================================
C     =========
C     
C     This wrapper ONLY computes per-resonance-helicity Sudakov
C      corrections.
C     The B[h,h'] density matrix is computed by Python via
C      DensityHelper
C     (using Valentin's GET_DENSITY from standalone rw_me_density_prod
C      modules).
C     
C     Architecture:
C     - Fortran: Sudakov kernels via sudakov_wrapper (Phase 2)
C     - Python:  Density matrices via DensityHelper (replaces Phase 1)
C     - Python:  Normalization delta = delta*B / B[h,h] (replaces
C      Phase 3)
C     
C     Physics: B^EWSL[h,h'] = B[h,h'] * (1 + delta_h + delta*_h')
C     
C     =================================================================
C     =========

      SUBROUTINE DENSITY_MATRIX_WRAPPER(P, NRES, RES_INDICES, RES_DIMS,
     $     density_born, density_delta, born_diag)
C     -----------------------------------------------------------------
C     ---------
C     Compute per-helicity Sudakov corrections delta_h
C     
C     Arguments:
C     p(0:3, nexternal-1)  - external momenta
C     nres                 - number of resonances to track
C     res_indices(nres)    - which legs are resonances (1-based)
C     res_dims(nres)       - spin dimension per resonance (2=fermion,
C      3=vector)
C     density_born(*)      - output: ZEROS (Python computes via
C      DensityHelper)
C     density_delta(*)     - output: unnormalized Sudakov corrections
C      delta*B
C     born_diag(*)         - output: per-helicity Born summed over
C      spectators (/IDEN)
C     
C     Note: density_born returns zeros (Python uses DensityHelper for
C      B).
C     born_diag carries per-helicity Born for IDEN-free normalization.
C     -----------------------------------------------------------------
C     ---------
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'

C     Arguments
      DOUBLE PRECISION P(0:3, NEXTERNAL-1)
      INTEGER NRES
      INTEGER RES_INDICES(NRES)
      INTEGER RES_DIMS(NRES)
      DOUBLE COMPLEX DENSITY_BORN(*)
      DOUBLE COMPLEX DENSITY_DELTA(*)
      DOUBLE PRECISION BORN_DIAG(*)

C     Local variables
      INTEGER I, K, N
      INTEGER IH
      INTEGER TOTAL_DIM
      INTEGER H_INDICES(10)  ! Individual resonance hel indices (0-based)
      INTEGER HEL_VALUES(10)  ! Actual helicity values (-1, 0, +1)

C     For resonance-helicity selection
      INTEGER ALLOW_HEL(300)  ! Max 10 res * 3 states each = 30 combos * 10 = 300

C     For Phase 2: per-helicity Sudakov via scalar wrapper
      DOUBLE COMPLEX DELTA_TOTAL_THIS_HEL
      DOUBLE PRECISION P_SAVE(0:3, NEXTERNAL-1)
      INTEGER ILEG, IMU
      LOGICAL SAVED_FILTER_HEL

      DOUBLE COMPLEX CZERO
      PARAMETER (CZERO = (0D0, 0D0))

C     Common blocks for reading scalar wrapper per-helicity results
      INCLUDE 'orders.inc'
      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_LSC/AMP_SPLIT_EWSUD_LSC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_SSC/AMP_SPLIT_EWSUD_SSC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_XXC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_XXC/AMP_SPLIT_EWSUD_XXC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PAR/AMP_SPLIT_EWSUD_PAR
      DOUBLE PRECISION AMP_BORN_SUMHEL
      COMMON /TO_AMP_BORN_SUMHEL/AMP_BORN_SUMHEL

C     Resonance-helicity selection for sudakov_wrapper
      INTEGER EWSUD_NRES
      INTEGER EWSUD_RES_POS(10)
      INTEGER EWSUD_RES_HEL(10)
      COMMON/TO_EWSUD_RES_SEL/EWSUD_NRES, EWSUD_RES_POS, EWSUD_RES_HEL

C     Filter helicities flag
      LOGICAL SUD_FILTER_HEL
      COMMON /TO_FILTER_HEL/ SUD_FILTER_HEL

      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN


C     -----------------------------------------------------------------
C     ---------
C     BEGIN CODE
C     -----------------------------------------------------------------
C     ---------

C     Compute total dimension D = product of res_dims
      TOTAL_DIM = 1
      DO I = 1, NRES
        TOTAL_DIM = TOTAL_DIM * RES_DIMS(I)
      ENDDO

C     Initialize outputs
C     density_born and born_diag return zeros - Python computes B via
C      DensityHelper
      DO I = 1, TOTAL_DIM * (TOTAL_DIM + 1) / 2
        DENSITY_BORN(I) = CZERO
      ENDDO
      DO I = 1, TOTAL_DIM
        DENSITY_DELTA(I) = CZERO
        BORN_DIAG(I) = 0D0
      ENDDO

C     =================================================================
C     =========
C     Build ALLOW_HEL array for resonance-helicity selection
C     =================================================================
C     =========
C     ALLOW_HEL layout: for each combination ih=1..total_dim,
C     ALLOW_HEL((ih-1)*nres+1 : ih*nres) = helicity values for
C      resonances
C     =================================================================
C     =========

      DO IH = 1, TOTAL_DIM
C       Convert 1-based ih to 0-based flat index, then to individual
C        indices
        K = IH - 1
        DO N = 1, NRES
          H_INDICES(N) = MOD(K, RES_DIMS(N))
          K = K / RES_DIMS(N)
        ENDDO

C       Convert 0-based indices to actual helicity values
        DO N = 1, NRES
          IF (RES_DIMS(N) .EQ. 2) THEN
C           Fermion: 0 -> -1, 1 -> +1
            HEL_VALUES(N) = 2 * H_INDICES(N) - 1
          ELSE IF (RES_DIMS(N) .EQ. 3) THEN
C           Vector: 0 -> -1, 1 -> 0, 2 -> +1
            HEL_VALUES(N) = H_INDICES(N) - 1
          ELSE
C           Unknown spin - use index directly
            HEL_VALUES(N) = H_INDICES(N)
          ENDIF
        ENDDO

C       Store in ALLOW_HEL
        DO N = 1, NRES
          ALLOW_HEL((IH-1)*NRES + N) = HEL_VALUES(N)
        ENDDO
      ENDDO

C     =================================================================
C     =========
C     PHASE 1: SKIPPED - Python computes B via DensityHelper
C     =================================================================
C     =========
C     The broken GET_DENSITY call has been removed.
C     Python uses DensityHelper (which calls Valentin's GET_DENSITY
C      from
C     the standalone rw_me_density_prod modules) to compute B[h,h'].
C     =================================================================
C     =========

C     =================================================================
C     =========
C     PHASE 2: Per-resonance-helicity Sudakov corrections
C     Use resonance-helicity selection (ewsud_nres) to sum over
C      spectators
C     =================================================================
C     =========

C     Save and disable helicity filtering for density calls
      SAVED_FILTER_HEL = SUD_FILTER_HEL
      SUD_FILTER_HEL = .FALSE.

C     Save momenta (PAR numerical derivatives may modify p)
      DO ILEG = 1, NEXTERNAL-1
        DO IMU = 0, 3
          P_SAVE(IMU, ILEG) = P(IMU, ILEG)
        ENDDO
      ENDDO

      DO IH = 1, TOTAL_DIM
C       Restore momenta before each call (PAR may corrupt p)
        DO ILEG = 1, NEXTERNAL-1
          DO IMU = 0, 3
            P(IMU, ILEG) = P_SAVE(IMU, ILEG)
          ENDDO
        ENDDO

C       Set resonance-helicity selection
C       This makes sudakov_wrapper sum over all full helicities where
C       the resonance legs have the specified helicity values
        EWSUD_NRES = NRES
        DO N = 1, NRES
          EWSUD_RES_POS(N) = RES_INDICES(N)
          EWSUD_RES_HEL(N) = ALLOW_HEL((IH-1)*NRES + N)
        ENDDO

        CALCULATEDBORN = .FALSE.

C       Call scalar wrapper - it sums over spectator helicities
        CALL SUDAKOV_WRAPPER(P)

C       Read Born-weighted results from common blocks
        DELTA_TOTAL_THIS_HEL = AMP_SPLIT_EWSUD_LSC(1)
     $       + amp_split_ewsud_ssc(1)
     $       + amp_split_ewsud_xxc(1)
     $       + amp_split_ewsud_par(1)

C       Store unnormalized (Born-weighted) delta
C       Python will normalize: delta[h] = density_delta[h] /
C        born_diag[h]
        DENSITY_DELTA(IH) = DELTA_TOTAL_THIS_HEL
C       Per-helicity Born (summed over spectators, /IDEN).
C       IDEN cancels in delta/born ratio, so no Python IDEN needed.
        BORN_DIAG(IH) = AMP_BORN_SUMHEL
      ENDDO

C     Reset resonance-helicity selection
      EWSUD_NRES = 0

C     Restore filter and momenta
      SUD_FILTER_HEL = SAVED_FILTER_HEL
      CALCULATEDBORN = .FALSE.
      DO ILEG = 1, NEXTERNAL-1
        DO IMU = 0, 3
          P(IMU, ILEG) = P_SAVE(IMU, ILEG)
        ENDDO
      ENDDO

C     =================================================================
C     =========
C     PHASE 3: SKIPPED - Python normalizes using DensityHelper's B
C      diagonal
C     =================================================================
C     =========
C     delta[h] is returned unnormalized (= delta*B[h]).
C     Python will compute: delta[h] = delta*B[h] / born_diag[h]
C     where born_diag[h] is returned alongside delta (IDEN cancels).
C     =================================================================
C     =========

      RETURN
      END

