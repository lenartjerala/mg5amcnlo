      SUBROUTINE SUDAKOV_WRAPPER(P)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
      DOUBLE PRECISION P(0:3, NEXTERNAL-1)

      INTEGER I
      INTEGER IHEL
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  12 )
      INTEGER NHEL(NEXTERNAL-1,NCOMB)
      DATA (NHEL(I,   1),I=1,3) /-1, 1,-1/
      DATA (NHEL(I,   2),I=1,3) /-1, 1, 0/
      DATA (NHEL(I,   3),I=1,3) /-1, 1, 1/
      DATA (NHEL(I,   4),I=1,3) /-1,-1,-1/
      DATA (NHEL(I,   5),I=1,3) /-1,-1, 0/
      DATA (NHEL(I,   6),I=1,3) /-1,-1, 1/
      DATA (NHEL(I,   7),I=1,3) / 1, 1,-1/
      DATA (NHEL(I,   8),I=1,3) / 1, 1, 0/
      DATA (NHEL(I,   9),I=1,3) / 1, 1, 1/
      DATA (NHEL(I,  10),I=1,3) / 1,-1,-1/
      DATA (NHEL(I,  11),I=1,3) / 1,-1, 0/
      DATA (NHEL(I,  12),I=1,3) / 1,-1, 1/
      DOUBLE COMPLEX ANS_SUMMED

      INCLUDE 'orders.inc'
      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_LO2(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD_LO2/ AMP_SPLIT_EWSUD_LO2

      DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_LSC/AMP_SPLIT_EWSUD_LSC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_SSC/AMP_SPLIT_EWSUD_SSC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_XXC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_XXC/AMP_SPLIT_EWSUD_XXC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PAR/AMP_SPLIT_EWSUD_PAR
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_QCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_QCD/AMP_SPLIT_EWSUD_QCD
      DOUBLE COMPLEX AMP_SPLIT_BORN_ONEHEL(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_BORN_ONEHEL/AMP_SPLIT_BORN_ONEHEL
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PARQCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PARQCD/AMP_SPLIT_EWSUD_PARQCD
      DOUBLE PRECISION AMP_BORN_SUMHEL
      COMMON /TO_AMP_BORN_SUMHEL/AMP_BORN_SUMHEL

      INTEGER IFLIST(NEXTERNAL-1), PDGLIST(NEXTERNAL-1),
     $  PDGLIST_OTH(NEXTERNAL-1)

      DOUBLE PRECISION INVARIANTS(NEXTERNAL-1, NEXTERNAL-1)

      DOUBLE COMPLEX GET_LSC_DIAG, GET_LSC_NONDIAG
      EXTERNAL GET_LSC_DIAG, GET_LSC_NONDIAG
      DOUBLE COMPLEX GET_SSC_N_DIAG, GET_SSC_N_NONDIAG_1,
     $  GET_SSC_N_NONDIAG_2, GET_SSC_C
      EXTERNAL GET_SSC_N_DIAG, GET_SSC_N_NONDIAG_1,
     $  GET_SSC_N_NONDIAG_2, GET_SSC_C
      DOUBLE COMPLEX GET_XXC_DIAG, GET_XXC_NONDIAG
      EXTERNAL GET_XXC_DIAG, GET_XXC_NONDIAG
      DOUBLE COMPLEX GET_QCD_LO2
      EXTERNAL GET_QCD_LO2

      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER DENSITY_DBG
      COMMON /DENSITY_DBG/ DENSITY_DBG

      INTEGER EWSUD_HELSELECT
      COMMON/TO_EWSUD_HELSELECT/EWSUD_HELSELECT

C     ============================================================
C     Resonance helicity selection (sums over spectators)
C     ============================================================
C     When ewsud_nres > 0, only compute for helicity configs where
C     the resonance legs (at positions ewsud_res_pos) have helicities
C     matching ewsud_res_hel. This sums over spectator helicities.
C     ============================================================
      INTEGER EWSUD_NRES
      INTEGER EWSUD_RES_POS(10)
      INTEGER EWSUD_RES_HEL(10)
      COMMON/TO_EWSUD_RES_SEL/EWSUD_NRES, EWSUD_RES_POS, EWSUD_RES_HEL
      DATA EWSUD_NRES /0/
      INTEGER IRES
      LOGICAL RES_HEL_MATCH

      DOUBLE PRECISION COMPENSATE_IDENTICAL_FACTOR
      EXTERNAL COMPENSATE_IDENTICAL_FACTOR
      DOUBLE PRECISION COMP_IDFAC

      LOGICAL GOOD_HEL(NCOMB), FIRSTTIME_HEL(NCOMB)
      DATA GOOD_HEL  /NCOMB*.TRUE./
      DATA FIRSTTIME_HEL  /NCOMB*.TRUE./
      INTEGER IAMP
      LOGICAL SUD_FILTER_HEL
      COMMON /TO_FILTER_HEL/ SUD_FILTER_HEL
C     stuff for the MC over helicities
      LOGICAL SUD_MC_HEL
      COMMON /TO_MC_HEL/ SUD_MC_HEL
      INTEGER IGOOD_HEL(NCOMB), HEL(0:NCOMB)
      DATA IGOOD_HEL  /NCOMB * 0/
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
      INTEGER HEL_MIN, HEL_MAX, HEL_PICK
      DOUBLE PRECISION VOLH

C     For per-helicity debug output (to compare with density path)
      DOUBLE COMPLEX LSC_PREV, SSC_PREV, XXC_PREV, PAR_PREV
      DOUBLE COMPLEX LSC_THIS, SSC_THIS, XXC_THIS, PAR_THIS

      INCLUDE 'ewsudakov_haslo.inc'


      AMP_SPLIT_EWSUD_LSC(:) = DCMPLX(0D0,0D0)
      AMP_SPLIT_EWSUD_SSC(:) = DCMPLX(0D0,0D0)
      AMP_SPLIT_EWSUD_XXC(:) = DCMPLX(0D0,0D0)
      AMP_SPLIT_EWSUD_PAR(:) = DCMPLX(0D0,0D0)

      AMP_SPLIT_EWSUD_QCD(:) = DCMPLX(0D0,0D0)
      AMP_SPLIT_EWSUD_PARQCD(:) = DCMPLX(0D0,0D0)
      AMP_BORN_SUMHEL = 0D0


      IFLIST = (/-1,-1,1/)
      IF (DENSITY_DBG.NE.0) THEN
        IF (EWSUD_NRES.GT.0) THEN
C         Resonance-helicity mode: summing over spectators
          WRITE(*,*)
          WRITE(*,*) 
     $     '+=========================================================='
     $     //'==+'
          WRITE(*,*) '|    SCALAR SUDAKOV WRAPPER (RESONANCE-HEL MODE)'
     $     //'             |'
          WRITE(*,*) 
     $     '+=========================================================='
     $     //'==+'
          WRITE(*,'(A,I2)') '  nres = ', EWSUD_NRES
          WRITE(*,'(A)',ADVANCE='no') '  res_pos = ['
          DO IRES=1,EWSUD_NRES
            WRITE(*,'(I3)',ADVANCE='no') EWSUD_RES_POS(IRES)
          ENDDO
          WRITE(*,'(A)') ' ]'
          WRITE(*,'(A)',ADVANCE='no') '  res_hel = ['
          DO IRES=1,EWSUD_NRES
            WRITE(*,'(I3)',ADVANCE='no') EWSUD_RES_HEL(IRES)
          ENDDO
          WRITE(*,'(A)') ' ]'
          WRITE(*,*) '  Summing over spectator helicities...'
          WRITE(*,*) '------------------------------------------------'
     $     //'------------'
        ELSE IF (EWSUD_HELSELECT.EQ.0) THEN
C         Full banner for standalone calls
          WRITE(*,*)
          WRITE(*,*) 
     $     '+=========================================================='
     $     //'==+'
          WRITE(*,*) '|           SCALAR SUDAKOV WRAPPER              '
     $     //'             |'
          WRITE(*,*) 
     $     '+=========================================================='
     $     //'==+'
          WRITE(*,*)
          WRITE(*,*) '  NOTE: Scalar path gives Born-WEIGHTED Sudakov
     $      corrections  !'
          WRITE(*,*) '        LSC,SSC,XXC,PAR here = Born *'
     $     //' delta_kernel'
          WRITE(*,*) '        Divide by Born to compare with density'
     $     //' path kernels.'
          WRITE(*,*)
          WRITE(*,'(A)',ADVANCE='no') '  Particle types (in=-1, out=+1'
     $     //'): '
          DO I=1,NEXTERNAL-1
            WRITE(*,'(I4)',ADVANCE='no') IFLIST(I)
          ENDDO
          WRITE(*,*)
          WRITE(*,'(A,L2,A,I3,A,L2)') '  MC hel=', SUD_MC_HEL,
     $   '  helselect=', ewsud_helselect, '  filter=', sud_filter_hel
          WRITE(*,*)
          WRITE(*,*) '------------------------------------------------'
     $     //'------------'
          WRITE(*,*) '  HELICITY LOOP'
          WRITE(*,*) '------------------------------------------------'
     $     //'------------'
        ELSE
C         Indented header for nested calls (from density wrapper)
          WRITE(*,'(A,I2,A)') '      +======== SCALAR WRAPPER [h=',
     $    ewsud_helselect, '] (called from DENSITY WRAPPER) ========+'
          WRITE(*,'(A)',ADVANCE='no') '        nhel = ['
          DO I=1,NEXTERNAL-1
            WRITE(*,'(I3)',ADVANCE='no') NHEL(I,EWSUD_HELSELECT)
          ENDDO
          WRITE(*,'(A)') ' ]'
        ENDIF
      ENDIF

      CALL SDK_GET_INVARIANTS(P, IFLIST, INVARIANTS)

      IF (FIRSTTIME.OR..NOT.SUD_MC_HEL) THEN
        HEL_MIN = 1
        HEL_MAX = NCOMB
        HEL(:)=0
        VOLH=1D0
        FIRSTTIME=.FALSE.
      ELSE
        CALL PICKHELICITYMC(P,IGOOD_HEL,HEL,HEL_PICK,VOLH)
        HEL_MIN = HEL(HEL_PICK)
        HEL_MAX = HEL_MIN
      ENDIF

      CALCULATEDBORN=.FALSE.

C     Main loop over the helicities
      DO IHEL=HEL_MIN, HEL_MAX
C       if an helicity is selected, then filter it
        IF (EWSUD_HELSELECT.NE.0.AND.EWSUD_HELSELECT.NE.IHEL) CYCLE

C       Resonance-helicity filter: skip if resonance hel doesn't match
C       This enables spectator summation while fixing resonance
C        helicities
        IF (EWSUD_NRES.GT.0) THEN
          RES_HEL_MATCH = .TRUE.
          DO IRES = 1, EWSUD_NRES
            IF (NHEL(EWSUD_RES_POS(IRES), IHEL) .NE.
     $        EWSUD_RES_HEL(IRES)) THEN
              RES_HEL_MATCH = .FALSE.
              EXIT
            ENDIF
          ENDDO
          IF (.NOT.RES_HEL_MATCH) CYCLE
        ENDIF

        IF (.NOT.GOOD_HEL(IHEL).AND.SUD_FILTER_HEL) CYCLE

C       call the born amplitude and store the result
        CALL SBORN_ONEHEL(P,NHEL(1,IHEL),IHEL,ANS_SUMMED)
        AMP_SPLIT_BORN_ONEHEL(:) = AMP_SPLIT_EWSUD(:)
        AMP_BORN_SUMHEL = AMP_BORN_SUMHEL +
     $    DBLE(AMP_SPLIT_BORN_ONEHEL(1))

C       Print helicity header with Born value
        IF (DENSITY_DBG.NE.0) THEN
C         Only print if Born is non-zero (interesting helicity)
          IF (DBLE(AMP_SPLIT_BORN_ONEHEL(1)).GT.1D-30) THEN
            IF (EWSUD_HELSELECT.EQ.0) THEN
C             Full header for standalone calls
              WRITE(*,*)
              WRITE(*,'(A,I2,A)') '  +--- HELICITY ', IHEL, ' (NON'
     $         //'-ZERO) ---+'
              WRITE(*,'(A)',ADVANCE='no') '  |  nhel = ['
              DO I=1,NEXTERNAL-1
                WRITE(*,'(I3)',ADVANCE='no') NHEL(I,IHEL)
              ENDDO
              WRITE(*,'(A)') ' ]'
              WRITE(*,'(A,ES12.4,A,ES12.4)') '  |  Born |M|^2 = ',
     $    dble(amp_split_born_onehel(1)), ' + i*', aimag(amp_split_born_onehel(1))
            ELSE
C             Indented Born value for nested calls (helicity config in
C              header)
              WRITE(*,'(A,ES12.4,A,ES12.4)') '        Born |M|^2 = ',
     $    dble(amp_split_born_onehel(1)), ' + i*', aimag(amp_split_born_onehel(1))
            ENDIF
          ELSE IF (EWSUD_HELSELECT.NE.0) THEN
C           For nested calls with zero Born, still print a note
            WRITE(*,'(A)') '        Born |M|^2 = 0 (helicity skipped)'
          ENDIF
        ENDIF

        IF (FIRSTTIME_HEL(IHEL)) THEN
          GOOD_HEL(IHEL) = .FALSE.
          DO IAMP = 1, AMP_SPLIT_SIZE
            GOOD_HEL(IHEL) = GOOD_HEL(IHEL)
     $       .OR.AMP_SPLIT_BORN_ONEHEL(IAMP)
     $       .NE.0D0.OR.AMP_SPLIT_EWSUD_LO2(IAMP).NE.0D0
          ENDDO

          IF (GOOD_HEL(IHEL)) THEN
            HEL(0) = HEL(0)+1
            HEL(HEL(0)) = IHEL
            IGOOD_HEL(HEL(0)) = 1
          ENDIF
C         write(*,*) 'EW SUDAKOV WRAPPER HEL FILTER', ihel,
C          nhel(:,ihel), good_hel(ihel)
          FIRSTTIME_HEL(IHEL) = .FALSE.
        ENDIF

C       Snapshot accumulated components for the helicity comparison
        LSC_PREV = AMP_SPLIT_EWSUD_LSC(1)
        SSC_PREV = AMP_SPLIT_EWSUD_SSC(1)
        XXC_PREV = AMP_SPLIT_EWSUD_XXC(1)
        PAR_PREV = AMP_SPLIT_EWSUD_PAR(1)

C       call the amplitudes with the goldstones for the
C        longitudinally-polarised legs and the born for all-transverse


C       Compute per-helicity increment (for comparison with density
C        path)
        LSC_THIS = AMP_SPLIT_EWSUD_LSC(1) - LSC_PREV
        SSC_THIS = AMP_SPLIT_EWSUD_SSC(1) - SSC_PREV
        XXC_THIS = AMP_SPLIT_EWSUD_XXC(1) - XXC_PREV
        PAR_THIS = AMP_SPLIT_EWSUD_PAR(1) - PAR_PREV

        IF (DENSITY_DBG.NE.0) THEN
          IF (DBLE(AMP_SPLIT_BORN_ONEHEL(1)).GT.1D-30) THEN
C           Full verbose output for standalone calls
            IF (EWSUD_HELSELECT.EQ.0) THEN
              WRITE(*,'(A)') '  |'
              WRITE(*,'(A)') '  |  THIS helicity Born-weighted'
     $         //' Sudakov:'
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |  LSC = ',
     $    dble(lsc_this), ' + i*', aimag(lsc_this)
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |  SSC = ',
     $    dble(ssc_this), ' + i*', aimag(ssc_this)
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |  XXC = ',
     $    dble(xxc_this), ' + i*', aimag(xxc_this)
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |  PAR = ',
     $    dble(par_this), ' + i*', aimag(par_this)
              WRITE(*,'(A)') '  |'
              WRITE(*,'(A)') '  |  Per-helicity delta (divide by Born'
     $         //'):'
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |    LSC/Born = ',
     $    dble(lsc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(lsc_this)/dble(amp_split_born_onehel(1))
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |    SSC/Born = ',
     $    dble(ssc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(ssc_this)/dble(amp_split_born_onehel(1))
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |    XXC/Born = ',
     $    dble(xxc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(xxc_this)/dble(amp_split_born_onehel(1))
              WRITE(*,'(A,ES11.3,A,ES11.3)') '  |    PAR/Born = ',
     $    dble(par_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(par_this)/dble(amp_split_born_onehel(1))
              WRITE(*,*) '  +-------------------------------+'
            ELSE
C             Verbose indented output for nested (per-helicity) calls
C              from density wrapper
              WRITE(*,'(A)') '        |'
              WRITE(*,'(A)') '        |  THIS helicity Born-weighted'
     $         //' Sudakov:'
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |  LSC = ',
     $    dble(lsc_this), ' + i*', aimag(lsc_this)
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |  SSC = ',
     $    dble(ssc_this), ' + i*', aimag(ssc_this)
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |  XXC = ',
     $    dble(xxc_this), ' + i*', aimag(xxc_this)
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |  PAR = ',
     $    dble(par_this), ' + i*', aimag(par_this)
              WRITE(*,'(A)') '        |'
              WRITE(*,'(A)') '        |  Per-helicity delta (divide by'
     $         //' Born):'
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |    LSC/Born ='
     $         //' ',
     $    dble(lsc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(lsc_this)/dble(amp_split_born_onehel(1))
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |    SSC/Born ='
     $         //' ',
     $    dble(ssc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(ssc_this)/dble(amp_split_born_onehel(1))
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |    XXC/Born ='
     $         //' ',
     $    dble(xxc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(xxc_this)/dble(amp_split_born_onehel(1))
              WRITE(*,'(A,ES11.3,A,ES11.3)') '        |    PAR/Born ='
     $         //' ',
     $    dble(par_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(par_this)/dble(amp_split_born_onehel(1))
              WRITE(*,'(A,I2,A)') '        +-- END [h='
     $         ,EWSUD_HELSELECT,'] --+'
            ENDIF
C           Skip zero-Born helicities silently for nested calls (too
C            noisy)
          ENDIF
        ENDIF
      ENDDO

      AMP_SPLIT_EWSUD_LSC(:) = AMP_SPLIT_EWSUD_LSC(:)/VOLH
      AMP_SPLIT_EWSUD_SSC(:) = AMP_SPLIT_EWSUD_SSC(:)/VOLH
      AMP_SPLIT_EWSUD_XXC(:) = AMP_SPLIT_EWSUD_XXC(:)/VOLH
      AMP_SPLIT_EWSUD_PAR(:) = AMP_SPLIT_EWSUD_PAR(:)/VOLH

      AMP_SPLIT_EWSUD_QCD(:) = AMP_SPLIT_EWSUD_QCD(:)/VOLH
      AMP_SPLIT_EWSUD_PARQCD(:) = AMP_SPLIT_EWSUD_PAR(:)
      AMP_BORN_SUMHEL = AMP_BORN_SUMHEL / VOLH
C     Only print summary for standalone calls
      IF (DENSITY_DBG.NE.0.AND.EWSUD_HELSELECT.EQ.0) THEN
        WRITE(*,*)
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
        WRITE(*,*) '  SCALAR SUDAKOV SUMMARY (after volh normalization'
     $   //')'
        WRITE(*,*) '--------------------------------------------------'
     $   //'----------'
        WRITE(*,'(A,F8.4)') '  volh = ', VOLH
        WRITE(*,*)
        WRITE(*,'(A)') '  TOTAL Sudakov contributions (summed over'
     $   //' helicities):'
        WRITE(*,'(A,ES12.4,A,ES12.4)') '    LSC = ',
     $    DBLE(AMP_SPLIT_EWSUD_LSC(1)),
     $   ' + i*', aimag(amp_split_ewsud_lsc(1))
        WRITE(*,'(A,ES12.4,A,ES12.4)') '    SSC = ',
     $    DBLE(AMP_SPLIT_EWSUD_SSC(1)),
     $   ' + i*', aimag(amp_split_ewsud_ssc(1))
        WRITE(*,'(A,ES12.4,A,ES12.4)') '    XXC = ',
     $    DBLE(AMP_SPLIT_EWSUD_XXC(1)),
     $   ' + i*', aimag(amp_split_ewsud_xxc(1))
        WRITE(*,'(A,ES12.4,A,ES12.4)') '    PAR = ',
     $    DBLE(AMP_SPLIT_EWSUD_PAR(1)),
     $   ' + i*', aimag(amp_split_ewsud_par(1))
        WRITE(*,*)
      ENDIF

      DO IAMP = 1, AMP_SPLIT_SIZE

        IF (HAS_LO1) THEN
          IF(IAMP.NE.LO1_POS) AMP_SPLIT_EWSUD_PAR(IAMP)=(0D0,0D0)
        ELSE
          AMP_SPLIT_EWSUD_PAR(IAMP)=(0D0,0D0)
        ENDIF

        IF (HAS_LO2) THEN
          IF(IAMP.NE.LO2_POS) AMP_SPLIT_EWSUD_PARQCD(IAMP)=(0D0,0D0)
        ELSE
          AMP_SPLIT_EWSUD_PARQCD(IAMP)=(0D0,0D0)
        ENDIF

      ENDDO



      CALCULATEDBORN=.FALSE.

C     Only print END banner for standalone calls
      IF (DENSITY_DBG.NE.0.AND.EWSUD_HELSELECT.EQ.0) THEN
        WRITE(*,*) '==================================================='
     $   //'========='
        WRITE(*,*) '  END SCALAR SUDAKOV WRAPPER'
        WRITE(*,*) '==================================================='
     $   //'========='
        WRITE(*,*)
      ENDIF
      RETURN
      END


      DOUBLE PRECISION FUNCTION COMPENSATE_IDENTICAL_FACTOR(IGOLD_ME)
C     provides the multiplicative factor for the result of the
C     sudakov MEs so that they have the same identical-particle
C     factor as the Born in the specific FKS configuration
      IMPLICIT NONE
      INTEGER IGOLD_ME
C     
      INTEGER SDK_IDENT_GOLDSTONE(1)
      DATA SDK_IDENT_GOLDSTONE / 0 /
C     
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IDEN_VALUES(4)
      DATA IDEN_VALUES /36, 36, 36, 36/
      INTEGER BORNSPINCOL
      DATA BORNSPINCOL / 36 /

      COMPENSATE_IDENTICAL_FACTOR = DBLE(SDK_IDENT_GOLDSTONE(IGOLD_ME))
     $ /DBLE(IDEN_VALUES(NFKSPROCESS)/BORNSPINCOL)

      RETURN
      END

