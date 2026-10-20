C     EW Sudakov corrections per resonance-helicity configuration h for
C     the spin-density reweighting
C     B^EWSL[h,h'] = B[h,h'] * (1 + delta_h + delta*_h'),
C     with the amplitude-level delta_h averaged over spectator
C      helicities
C     (exact for h = h' only). The density matrix B[h,h'] is computed
C     separately (GET_DENSITY of the standalone density-matrix output).

      SUBROUTINE DENSITY_MATRIX_WRAPPER(P, NRES, RES_INDICES, RES_DIMS,
     $     density_born, density_delta, born_diag)
C     -----------------------------------------------------------------
C     ---------
C     Arguments:
C     p(0:3, nexternal-1)  - external momenta (restored on exit)
C     nres                 - number of resonances
C     res_indices(nres)    - resonance leg positions (1-based)
C     res_dims(nres)       - spin dimension per resonance (2=fermion,
C      3=vector)
C     density_born(*)      - output: zeros, D*(D+1)/2 entries
C     density_delta(*)     - output: sum of ME*delta, D entries
C     born_diag(*)         - output: Born, D entries
C     D = product of res_dims. density_delta and born_diag are summed
C      over
C     spectator helicities and both carry 1/IDEN;
C     delta_h = density_delta(h)/born_diag(h). ME is the Born, or the
C     Goldstone-boson ME for configurations with longitudinal vector
C      legs;
C     for those delta_h includes the ratio of the Goldstone ME to the
C      Born.
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

      INTEGER ALLOW_HEL(300)  ! D*nres entries, size checked by the caller

      DOUBLE COMPLEX DELTA_TOTAL_THIS_HEL
      DOUBLE PRECISION P_SAVE(0:3, NEXTERNAL-1)
      INTEGER ILEG, IMU
      LOGICAL SAVED_FILTER_HEL

      DOUBLE COMPLEX CZERO
      PARAMETER (CZERO = (0D0, 0D0))

C     Results of sudakov_wrapper
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

      LOGICAL SUD_FILTER_HEL
      COMMON /TO_FILTER_HEL/ SUD_FILTER_HEL

      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER DENSITY_DBG
      COMMON /DENSITY_DBG/ DENSITY_DBG


C     -----------------------------------------------------------------
C     ---------
C     BEGIN CODE
C     -----------------------------------------------------------------
C     ---------

      TOTAL_DIM = 1
      DO I = 1, NRES
        TOTAL_DIM = TOTAL_DIM * RES_DIMS(I)
      ENDDO
      IF (DENSITY_DBG .NE. 0) THEN
        WRITE(*,*)
        WRITE(*,*) '+=================================================='
     $   //'==========+'
        WRITE(*,*) '|  DENSITY MATRIX WRAPPER (DELTA-ONLY MODE)       '
     $   //'           |'
        WRITE(*,*) '+=================================================='
     $   //'==========+'
        WRITE(*,*)
        WRITE(*,*) '  Python computes B[h,h''] via DensityHelper'
        WRITE(*,*) '  Fortran computes delta (per-helicity Sudakov)'
        WRITE(*,*)
        WRITE(*,'(A,I2,A)') '  Number of resonances: ', NRES, ''
        DO I = 1, NRES
          WRITE(*,'(A,I1,A,I2,A,I1,A)') '    Resonance ', I,
     $        ': particle #', res_indices(i),
     $        ' with ', res_dims(i), ' spin states'
        ENDDO
        WRITE(*,'(A,I3)') '  Total helicity dimension (n_comb): ',
     $    TOTAL_DIM
        WRITE(*,*)
      ENDIF

      DO I = 1, TOTAL_DIM * (TOTAL_DIM + 1) / 2
        DENSITY_BORN(I) = CZERO
      ENDDO
      DO I = 1, TOTAL_DIM
        DENSITY_DELTA(I) = CZERO
        BORN_DIAG(I) = 0D0
      ENDDO

C     allow_hel((ih-1)*nres+n) = helicity of resonance n in
C      configuration
C     ih = 1..D, with ih-1 = h_1 + res_dims(1)*(h_2 +
C      res_dims(2)*(...))
C     for 0-based indices h_n (first resonance runs fastest).
      IF (DENSITY_DBG .NE. 0) THEN
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
        WRITE(*,*) '  Building ALLOW_HEL array for resonance-helicity'
     $   //' selection'
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
      ENDIF

      DO IH = 1, TOTAL_DIM
        K = IH - 1
        DO N = 1, NRES
          H_INDICES(N) = MOD(K, RES_DIMS(N))
          K = K / RES_DIMS(N)
        ENDDO

        DO N = 1, NRES
          IF (RES_DIMS(N) .EQ. 2) THEN
C           Fermion: 0 -> -1, 1 -> +1
            HEL_VALUES(N) = 2 * H_INDICES(N) - 1
          ELSE IF (RES_DIMS(N) .EQ. 3) THEN
C           Vector: 0 -> -1, 1 -> 0, 2 -> +1
            HEL_VALUES(N) = H_INDICES(N) - 1
          ELSE
            HEL_VALUES(N) = H_INDICES(N)
          ENDIF
        ENDDO

        DO N = 1, NRES
          ALLOW_HEL((IH-1)*NRES + N) = HEL_VALUES(N)
        ENDDO

        IF (DENSITY_DBG .NE. 0) THEN
          WRITE(*,'(A,I3,A)', ADVANCE='no') '    ih=', IH, ' hels=('
          DO N = 1, NRES
            WRITE(*,'(I2)', ADVANCE='no') HEL_VALUES(N)
            IF (N .LT. NRES) WRITE(*,'(A)', ADVANCE='no') ','
          ENDDO
          WRITE(*,'(A)') ')'
        ENDIF
      ENDDO

C     For each configuration ih, sudakov_wrapper restricted to the
C     resonance helicities allow_hel(ih) sums over spectator
C      helicities.
      IF (DENSITY_DBG .NE. 0) THEN
        WRITE(*,*)
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
        WRITE(*,*) '  PHASE 2: PER-RESONANCE-HELICITY SUDAKOV'
        WRITE(*,*) '           (using ewsud_nres spectator summation)'
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
      ENDIF

      SAVED_FILTER_HEL = SUD_FILTER_HEL
      SUD_FILTER_HEL = .FALSE.

C     Save momenta (PAR numerical derivatives may modify p)
      DO ILEG = 1, NEXTERNAL-1
        DO IMU = 0, 3
          P_SAVE(IMU, ILEG) = P(IMU, ILEG)
        ENDDO
      ENDDO

      DO IH = 1, TOTAL_DIM
        DO ILEG = 1, NEXTERNAL-1
          DO IMU = 0, 3
            P(IMU, ILEG) = P_SAVE(IMU, ILEG)
          ENDDO
        ENDDO

        EWSUD_NRES = NRES
        DO N = 1, NRES
          EWSUD_RES_POS(N) = RES_INDICES(N)
          EWSUD_RES_HEL(N) = ALLOW_HEL((IH-1)*NRES + N)
        ENDDO

        CALCULATEDBORN = .FALSE.

        CALL SUDAKOV_WRAPPER(P)

        DELTA_TOTAL_THIS_HEL = AMP_SPLIT_EWSUD_LSC(1)
     $       + amp_split_ewsud_ssc(1)
     $       + amp_split_ewsud_xxc(1)
     $       + amp_split_ewsud_par(1)

        DENSITY_DELTA(IH) = DELTA_TOTAL_THIS_HEL
        BORN_DIAG(IH) = AMP_BORN_SUMHEL

        IF (DENSITY_DBG .NE. 0) THEN
          WRITE(*,'(A,I3,A)', ADVANCE='no') '  ih=', IH, '  res_hel=('
          DO N = 1, NRES
            WRITE(*,'(I2)', ADVANCE='no') EWSUD_RES_HEL(N)
            IF (N .LT. NRES) WRITE(*,'(A)', ADVANCE='no') ','
          ENDDO
          WRITE(*,'(A,ES12.4,A,ES12.4)')
     $        ')  delta*B=', dble(delta_total_this_hel),
     $        ' + i*', aimag(delta_total_this_hel)
        ENDIF
      ENDDO

      EWSUD_NRES = 0

      SUD_FILTER_HEL = SAVED_FILTER_HEL
      CALCULATEDBORN = .FALSE.
      DO ILEG = 1, NEXTERNAL-1
        DO IMU = 0, 3
          P(IMU, ILEG) = P_SAVE(IMU, ILEG)
        ENDDO
      ENDDO

      IF (DENSITY_DBG .NE. 0) THEN
        WRITE(*,*)
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
        WRITE(*,*) '  DENSITY MATRIX WRAPPER COMPLETE'
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
        WRITE(*,*)
        WRITE(*,'(A)') '  Returning unnormalized delta*B values:'
        WRITE(*,'(A)') '  +--------+-------------------------------+'
        WRITE(*,'(A)') '  |   h    |      delta*B (unnormalized)   |'
        WRITE(*,'(A)') '  +--------+-------------------------------+'
        DO IH = 1, TOTAL_DIM
          WRITE(*,'(A,I3,A,ES12.4,A,ES12.4,A)')
     $      '  | ', ih-1, '    | ',
     $      dble(density_delta(ih)), ' + i*', aimag(density_delta(ih)),
     $      ' |'
        ENDDO
        WRITE(*,'(A)') '  +--------+-------------------------------+'
        WRITE(*,*)
        WRITE(*,*) '  Python will normalize: delta[h] = delta*B[h] /'
     $   //' born_diag[h]'
        WRITE(*,*) '  where born_diag[h] is returned alongside delta'
     $   //' (IDEN cancels)'
        WRITE(*,*)
        WRITE(*,*) '==================================================='
     $   //'========='
        WRITE(*,*) '  END DENSITY MATRIX WRAPPER (DELTA-ONLY)'
        WRITE(*,*) '==================================================='
     $   //'========='
        WRITE(*,*)
      ENDIF

      RETURN
      END

