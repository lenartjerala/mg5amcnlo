C ==========================================================================
C     Density Matrix Delta-Only Wrapper for EW Sudakov
C ==========================================================================
C
C     This wrapper ONLY computes per-resonance-helicity Sudakov corrections.
C     The B[h,h'] density matrix is computed by Python via DensityHelper
C     (using Valentin's GET_DENSITY from standalone rw_me_density_prod modules).
C
C     Architecture:
C       - Fortran: Sudakov kernels via sudakov_wrapper (Phase 2)
C       - Python:  Density matrices via DensityHelper (replaces Phase 1)
C       - Python:  Normalization delta = delta*B / B[h,h] (replaces Phase 3)
C
C     Physics: B^EWSL[h,h'] = B[h,h'] * (1 + delta_h + delta*_h')
C
C ==========================================================================

      subroutine density_matrix_wrapper(p, nres, res_indices, res_dims,
     $     density_born, density_delta, born_diag)
C --------------------------------------------------------------------------
C     Compute per-helicity Sudakov corrections delta_h
C
C     Arguments:
C       p(0:3, nexternal-1)  - external momenta
C       nres                 - number of resonances to track
C       res_indices(nres)    - which legs are resonances (1-based)
C       res_dims(nres)       - spin dimension per resonance (2=fermion, 3=vector)
C       density_born(*)      - output: ZEROS (Python computes via DensityHelper)
C       density_delta(*)     - output: unnormalized Sudakov corrections delta*B
C       born_diag(*)         - output: per-helicity Born summed over spectators (/IDEN)
C
C     Note: density_born returns zeros (Python uses DensityHelper for B).
C           born_diag carries per-helicity Born for IDEN-free normalization.
C --------------------------------------------------------------------------
      implicit none
      include 'nexternal.inc'

C     Arguments
      double precision p(0:3, nexternal-1)
      integer nres
      integer res_indices(nres)
      integer res_dims(nres)
      double complex density_born(*)
      double complex density_delta(*)
      double precision born_diag(*)

C     Local variables
      integer i, k, n
      integer ih
      integer total_dim
      integer h_indices(10)        ! Individual resonance hel indices (0-based)
      integer hel_values(10)       ! Actual helicity values (-1, 0, +1)

C     For resonance-helicity selection
      integer allow_hel(300)       ! Max 10 res * 3 states each = 30 combos * 10 = 300

C     For Phase 2: per-helicity Sudakov via scalar wrapper
      double complex delta_total_this_hel
      double precision p_save(0:3, nexternal-1)
      integer ileg, imu
      logical saved_filter_hel

      double complex CZERO
      parameter (CZERO = (0d0, 0d0))

C     Common blocks for reading scalar wrapper per-helicity results
      include 'orders.inc'
      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_XXC(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_par/amp_split_ewsud_par
      DOUBLE PRECISION AMP_BORN_SUMHEL
      common /to_amp_born_sumhel/amp_born_sumhel

C     Resonance-helicity selection for sudakov_wrapper
      integer ewsud_nres
      integer ewsud_res_pos(10)
      integer ewsud_res_hel(10)
      common/to_ewsud_res_sel/ewsud_nres, ewsud_res_pos, ewsud_res_hel

C     Filter helicities flag
      logical sud_filter_hel
      COMMON /to_filter_hel/ sud_filter_hel

      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER DENSITY_DBG
      COMMON /DENSITY_DBG/ DENSITY_DBG


C --------------------------------------------------------------------------
C     BEGIN CODE
C --------------------------------------------------------------------------

C     Compute total dimension D = product of res_dims
      total_dim = 1
      do i = 1, nres
        total_dim = total_dim * res_dims(i)
      enddo
      if (density_dbg .ne. 0) then
        write(*,*)
        write(*,*) '+============================================================+'
        write(*,*) '|  DENSITY MATRIX WRAPPER (DELTA-ONLY MODE)                  |'
        write(*,*) '+============================================================+'
        write(*,*)
        write(*,*) '  Python computes B[h,h''] via DensityHelper'
        write(*,*) '  Fortran computes delta (per-helicity Sudakov)'
        write(*,*)
        write(*,'(A,I2,A)') '  Number of resonances: ', nres, ''
        do i = 1, nres
          write(*,'(A,I1,A,I2,A,I1,A)') '    Resonance ', i,
     $        ': particle #', res_indices(i),
     $        ' with ', res_dims(i), ' spin states'
        enddo
        write(*,'(A,I3)') '  Total helicity dimension (n_comb): ', total_dim
        write(*,*)
      endif

C     Initialize outputs
C     density_born and born_diag return zeros - Python computes B via DensityHelper
      do i = 1, total_dim * (total_dim + 1) / 2
        density_born(i) = CZERO
      enddo
      do i = 1, total_dim
        density_delta(i) = CZERO
        born_diag(i) = 0d0
      enddo

C ==========================================================================
C     Build ALLOW_HEL array for resonance-helicity selection
C ==========================================================================
C     ALLOW_HEL layout: for each combination ih=1..total_dim,
C       ALLOW_HEL((ih-1)*nres+1 : ih*nres) = helicity values for resonances
C ==========================================================================
      if (density_dbg .ne. 0) then
        write(*,*) '------------------------------------------------------------'
        write(*,*) '  Building ALLOW_HEL array for resonance-helicity selection'
        write(*,*) '------------------------------------------------------------'
      endif

      do ih = 1, total_dim
C       Convert 1-based ih to 0-based flat index, then to individual indices
        k = ih - 1
        do n = 1, nres
          h_indices(n) = mod(k, res_dims(n))
          k = k / res_dims(n)
        enddo

C       Convert 0-based indices to actual helicity values
        do n = 1, nres
          if (res_dims(n) .eq. 2) then
C           Fermion: 0 -> -1, 1 -> +1
            hel_values(n) = 2 * h_indices(n) - 1
          else if (res_dims(n) .eq. 3) then
C           Vector: 0 -> -1, 1 -> 0, 2 -> +1
            hel_values(n) = h_indices(n) - 1
          else
C           Unknown spin - use index directly
            hel_values(n) = h_indices(n)
          endif
        enddo

C       Store in ALLOW_HEL
        do n = 1, nres
          allow_hel((ih-1)*nres + n) = hel_values(n)
        enddo

        if (density_dbg .ne. 0) then
          write(*,'(A,I3,A)', advance='no') '    ih=', ih, ' hels=('
          do n = 1, nres
            write(*,'(I2)', advance='no') hel_values(n)
            if (n .lt. nres) write(*,'(A)', advance='no') ','
          enddo
          write(*,'(A)') ')'
        endif
      enddo

C ==========================================================================
C     PHASE 1: SKIPPED - Python computes B via DensityHelper
C ==========================================================================
C     The broken GET_DENSITY call has been removed.
C     Python uses DensityHelper (which calls Valentin's GET_DENSITY from
C     the standalone rw_me_density_prod modules) to compute B[h,h'].
C ==========================================================================

C ==========================================================================
C     PHASE 2: Per-resonance-helicity Sudakov corrections
C     Use resonance-helicity selection (ewsud_nres) to sum over spectators
C ==========================================================================
      if (density_dbg .ne. 0) then
        write(*,*)
        write(*,*) '------------------------------------------------------------'
        write(*,*) '  PHASE 2: PER-RESONANCE-HELICITY SUDAKOV'
        write(*,*) '           (using ewsud_nres spectator summation)'
        write(*,*) '------------------------------------------------------------'
      endif

C     Save and disable helicity filtering for density calls
      saved_filter_hel = sud_filter_hel
      sud_filter_hel = .false.

C     Save momenta (PAR numerical derivatives may modify p)
      do ileg = 1, nexternal-1
        do imu = 0, 3
          p_save(imu, ileg) = p(imu, ileg)
        enddo
      enddo

      do ih = 1, total_dim
C       Restore momenta before each call (PAR may corrupt p)
        do ileg = 1, nexternal-1
          do imu = 0, 3
            p(imu, ileg) = p_save(imu, ileg)
          enddo
        enddo

C       Set resonance-helicity selection
C       This makes sudakov_wrapper sum over all full helicities where
C       the resonance legs have the specified helicity values
        ewsud_nres = nres
        do n = 1, nres
          ewsud_res_pos(n) = res_indices(n)
          ewsud_res_hel(n) = allow_hel((ih-1)*nres + n)
        enddo

        CALCULATEDBORN = .FALSE.

C       Call scalar wrapper - it sums over spectator helicities
        call sudakov_wrapper(p)

C       Read Born-weighted results from common blocks
        delta_total_this_hel = amp_split_ewsud_lsc(1)
     $       + amp_split_ewsud_ssc(1)
     $       + amp_split_ewsud_xxc(1)
     $       + amp_split_ewsud_par(1)

C       Store unnormalized (Born-weighted) delta
C       Python will normalize: delta[h] = density_delta[h] / born_diag[h]
        density_delta(ih) = delta_total_this_hel
C       Per-helicity Born (summed over spectators, /IDEN).
C       IDEN cancels in delta/born ratio, so no Python IDEN needed.
        born_diag(ih) = amp_born_sumhel

        if (density_dbg .ne. 0) then
          write(*,'(A,I3,A)', advance='no') '  ih=', ih, '  res_hel=('
          do n = 1, nres
            write(*,'(I2)', advance='no') ewsud_res_hel(n)
            if (n .lt. nres) write(*,'(A)', advance='no') ','
          enddo
          write(*,'(A,ES12.4,A,ES12.4)')
     $        ')  delta*B=', dble(delta_total_this_hel),
     $        ' + i*', aimag(delta_total_this_hel)
        endif
      enddo

C     Reset resonance-helicity selection
      ewsud_nres = 0

C     Restore filter and momenta
      sud_filter_hel = saved_filter_hel
      CALCULATEDBORN = .FALSE.
      do ileg = 1, nexternal-1
        do imu = 0, 3
          p(imu, ileg) = p_save(imu, ileg)
        enddo
      enddo

C ==========================================================================
C     PHASE 3: SKIPPED - Python normalizes using DensityHelper's B diagonal
C ==========================================================================
C     delta[h] is returned unnormalized (= delta*B[h]).
C     Python will compute: delta[h] = delta*B[h] / born_diag[h]
C     where born_diag[h] is returned alongside delta (IDEN cancels).
C ==========================================================================

      if (density_dbg .ne. 0) then
        write(*,*)
        write(*,*) '------------------------------------------------------------'
        write(*,*) '  DENSITY MATRIX WRAPPER COMPLETE'
        write(*,*) '------------------------------------------------------------'
        write(*,*)
        write(*,'(A)') '  Returning unnormalized delta*B values:'
        write(*,'(A)') '  +--------+-------------------------------+'
        write(*,'(A)') '  |   h    |      delta*B (unnormalized)   |'
        write(*,'(A)') '  +--------+-------------------------------+'
        do ih = 1, total_dim
          write(*,'(A,I3,A,ES12.4,A,ES12.4,A)')
     $      '  | ', ih-1, '    | ',
     $      dble(density_delta(ih)), ' + i*', aimag(density_delta(ih)),
     $      ' |'
        enddo
        write(*,'(A)') '  +--------+-------------------------------+'
        write(*,*)
        write(*,*) '  Python will normalize: delta[h] = delta*B[h] / born_diag[h]'
        write(*,*) '  where born_diag[h] is returned alongside delta (IDEN cancels)'
        write(*,*)
        write(*,*) '============================================================'
        write(*,*) '  END DENSITY MATRIX WRAPPER (DELTA-ONLY)'
        write(*,*) '============================================================'
        write(*,*)
      endif

      return
      end
