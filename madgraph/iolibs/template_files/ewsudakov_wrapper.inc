subroutine sudakov_wrapper(p)
implicit none
include 'nexternal.inc'
include "born_nhel.inc"
double precision p(0:3, nexternal-1)

integer i
INTEGER IHEL
INTEGER     NCOMB
PARAMETER ( NCOMB=  %(ncomb)d )
INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
double complex ans_summed

include 'orders.inc'
DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD
DOUBLE COMPLEX AMP_SPLIT_EWSUD_LO2(AMP_SPLIT_SIZE)
COMMON /TO_AMP_SPLIT_EWSUD_LO2/ AMP_SPLIT_EWSUD_LO2

DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
DOUBLE COMPLEX AMP_SPLIT_EWSUD_XXC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
common /to_amp_ewsud_PAR/amp_split_ewsud_PAR
DOUBLE COMPLEX AMP_SPLIT_EWSUD_QCD(AMP_SPLIT_SIZE)
common /to_amp_ewsud_QCD/amp_split_ewsud_QCD
DOUBLE COMPLEX AMP_SPLIT_BORN_ONEHEL(AMP_SPLIT_SIZE)
common /to_amp_born_onehel/amp_split_born_onehel
DOUBLE COMPLEX AMP_SPLIT_EWSUD_PARQCD(AMP_SPLIT_SIZE)
COMMON /TO_AMP_EWSUD_PARQCD/AMP_SPLIT_EWSUD_PARQCD
DOUBLE PRECISION AMP_BORN_SUMHEL
common /to_amp_born_sumhel/amp_born_sumhel

integer iflist(nexternal-1), pdglist(nexternal-1), pdglist_oth(nexternal-1)

double precision invariants(nexternal-1, nexternal-1)

double complex get_lsc_diag, get_lsc_nondiag
external get_lsc_diag, get_lsc_nondiag
double complex get_ssc_n_diag, get_ssc_n_nondiag_1, get_ssc_n_nondiag_2, get_ssc_c
external get_ssc_n_diag, get_ssc_n_nondiag_1, get_ssc_n_nondiag_2, get_ssc_c
double complex get_xxc_diag, get_xxc_nondiag
external get_xxc_diag, get_xxc_nondiag
double complex GET_QCD_LO2
external GET_QCD_LO2

LOGICAL CALCULATEDBORN
COMMON/CCALCULATEDBORN/CALCULATEDBORN
INTEGER DENSITY_DBG
COMMON /DENSITY_DBG/ DENSITY_DBG

integer ewsud_helselect
common/to_ewsud_helselect/ewsud_helselect

C ============================================================
C Resonance helicity selection (sums over spectators)
C ============================================================
C When ewsud_nres > 0, only compute for helicity configs where
C the resonance legs (at positions ewsud_res_pos) have helicities
C matching ewsud_res_hel. This sums over spectator helicities.
C ============================================================
      integer ewsud_nres
      integer ewsud_res_pos(10)
      integer ewsud_res_hel(10)
      common/to_ewsud_res_sel/ewsud_nres, ewsud_res_pos, ewsud_res_hel
      data ewsud_nres /0/
      integer ires
      logical res_hel_match

double precision compensate_identical_factor
external compensate_identical_factor
double precision comp_idfac

logical good_hel(ncomb), firsttime_hel(ncomb)
data good_hel  /ncomb*.true./
data firsttime_hel  /ncomb*.true./
integer iamp
logical sud_filter_hel
COMMON /to_filter_hel/ sud_filter_hel
C stuff for the MC over helicities
LOGICAL SUD_MC_HEL
COMMON /TO_MC_HEL/ SUD_MC_HEL
integer igood_hel(ncomb), HEL(0:ncomb)
DATA IGOOD_HEL  /NCOMB * 0/
logical firsttime
data firsttime /.true./
integer hel_min, hel_max, hel_pick
double precision volh

C For per-helicity debug output (to compare with density path)
double complex lsc_prev, ssc_prev, xxc_prev, par_prev
double complex lsc_this, ssc_this, xxc_this, par_this

include 'ewsudakov_haslo.inc'


AMP_SPLIT_EWSUD_LSC(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_SSC(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_XXC(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_PAR(:) = dcmplx(0d0,0d0)

AMP_SPLIT_EWSUD_QCD(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_PARQCD(:) = dcmplx(0d0,0d0)
AMP_BORN_SUMHEL = 0d0


%(iflist)s
if (density_dbg.ne.0) then
  if (ewsud_nres.gt.0) then
C   Resonance-helicity mode: summing over spectators
    write(*,*)
    write(*,*) '+============================================================+'
    write(*,*) '|    SCALAR SUDAKOV WRAPPER (RESONANCE-HEL MODE)             |'
    write(*,*) '+============================================================+'
    write(*,'(A,I2)') '  nres = ', ewsud_nres
    write(*,'(A)',advance='no') '  res_pos = ['
    do ires=1,ewsud_nres
      write(*,'(I3)',advance='no') ewsud_res_pos(ires)
    enddo
    write(*,'(A)') ' ]'
    write(*,'(A)',advance='no') '  res_hel = ['
    do ires=1,ewsud_nres
      write(*,'(I3)',advance='no') ewsud_res_hel(ires)
    enddo
    write(*,'(A)') ' ]'
    write(*,*) '  Summing over spectator helicities...'
    write(*,*) '------------------------------------------------------------'
  else if (ewsud_helselect.eq.0) then
C   Full banner for standalone calls
    write(*,*)
    write(*,*) '+============================================================+'
    write(*,*) '|           SCALAR SUDAKOV WRAPPER                           |'
    write(*,*) '+============================================================+'
    write(*,*)
    write(*,*) '  NOTE: Scalar path gives Born-WEIGHTED Sudakov corrections!'
    write(*,*) '        LSC,SSC,XXC,PAR here = Born * delta_kernel'
    write(*,*) '        Divide by Born to compare with density path kernels.'
    write(*,*)
    write(*,'(A)',advance='no') '  Particle types (in=-1, out=+1): '
    do i=1,nexternal-1
      write(*,'(I4)',advance='no') iflist(i)
    enddo
    write(*,*)
    write(*,'(A,L2,A,I3,A,L2)') '  MC hel=', sud_mc_hel,
     $   '  helselect=', ewsud_helselect, '  filter=', sud_filter_hel
    write(*,*)
    write(*,*) '------------------------------------------------------------'
    write(*,*) '  HELICITY LOOP'
    write(*,*) '------------------------------------------------------------'
  else
C   Indented header for nested calls (from density wrapper)
    write(*,'(A,I2,A)') '      +======== SCALAR WRAPPER [h=',
     $    ewsud_helselect, '] (called from DENSITY WRAPPER) ========+'
    write(*,'(A)',advance='no') '        nhel = ['
    do i=1,nexternal-1
      write(*,'(I3)',advance='no') nhel(i,ewsud_helselect)
    enddo
    write(*,'(A)') ' ]'
  endif
endif

call sdk_get_invariants(p, iflist, invariants)

if (firsttime.or..not.sud_mc_hel) then
 hel_min = 1
 hel_max = ncomb
 hel(:)=0
 volh=1d0
 firsttime=.false.
else
 call pickhelicitymc(p,igood_hel,hel,hel_pick,volh)
 hel_min = hel(hel_pick)
 hel_max = hel_min
endif

CALCULATEDBORN=.FALSE.

C Main loop over the helicities
DO IHEL=hel_min, hel_max
C if an helicity is selected, then filter it
if (ewsud_helselect.ne.0.and.ewsud_helselect.ne.ihel) cycle

C Resonance-helicity filter: skip if resonance hel doesn't match
C This enables spectator summation while fixing resonance helicities
if (ewsud_nres.gt.0) then
  res_hel_match = .true.
  do ires = 1, ewsud_nres
    if (nhel(ewsud_res_pos(ires), ihel) .ne. ewsud_res_hel(ires)) then
      res_hel_match = .false.
      exit
    endif
  enddo
  if (.not.res_hel_match) cycle
endif

if (.not.good_hel(ihel).and.sud_filter_hel) cycle

C call the born amplitude and store the result
call sborn_onehel(p,nhel(1,ihel),ihel,ans_summed)
amp_split_born_onehel(:) = amp_split_ewsud(:)
AMP_BORN_SUMHEL = AMP_BORN_SUMHEL + dble(amp_split_born_onehel(1))

C Print helicity header with Born value
if (density_dbg.ne.0) then
C   Only print if Born is non-zero (interesting helicity)
  if (dble(amp_split_born_onehel(1)).gt.1d-30) then
    if (ewsud_helselect.eq.0) then
C     Full header for standalone calls
      write(*,*)
      write(*,'(A,I2,A)') '  +--- HELICITY ', ihel, ' (NON-ZERO) ---+'
      write(*,'(A)',advance='no') '  |  nhel = ['
      do i=1,nexternal-1
        write(*,'(I3)',advance='no') nhel(i,ihel)
      enddo
      write(*,'(A)') ' ]'
      write(*,'(A,ES12.4,A,ES12.4)') '  |  Born |M|^2 = ',
     $    dble(amp_split_born_onehel(1)), ' + i*', aimag(amp_split_born_onehel(1))
    else
C     Indented Born value for nested calls (helicity config in header)
      write(*,'(A,ES12.4,A,ES12.4)') '        Born |M|^2 = ',
     $    dble(amp_split_born_onehel(1)), ' + i*', aimag(amp_split_born_onehel(1))
    endif
  else if (ewsud_helselect.ne.0) then
C   For nested calls with zero Born, still print a note
    write(*,'(A)') '        Born |M|^2 = 0 (helicity skipped)'
  endif
endif

if (firsttime_hel(ihel)) then
    good_hel(ihel) = .false.
    do iamp = 1, amp_split_size
      good_hel(ihel) = good_hel(ihel).or.amp_split_born_onehel(iamp).ne.0d0.or.amp_split_ewsud_lo2(iamp).ne.0d0
    enddo

    if (good_hel(ihel)) then
      hel(0) = hel(0)+1
      hel(hel(0)) = ihel
      igood_hel(hel(0)) = 1
    endif
    !write(*,*) 'EW SUDAKOV WRAPPER HEL FILTER', ihel, nhel(:,ihel), good_hel(ihel)
    firsttime_hel(ihel) = .false.
endif

C Save previous accumulated values for per-helicity comparison
lsc_prev = amp_split_ewsud_lsc(1)
ssc_prev = amp_split_ewsud_ssc(1)
xxc_prev = amp_split_ewsud_xxc(1)
par_prev = amp_split_ewsud_par(1)

C call the amplitudes with the goldstones for the longitudinally-polarised legs and the born for all-transverse
%(calls_to_me)s

C Compute per-helicity increment (for comparison with density path)
lsc_this = amp_split_ewsud_lsc(1) - lsc_prev
ssc_this = amp_split_ewsud_ssc(1) - ssc_prev
xxc_this = amp_split_ewsud_xxc(1) - xxc_prev
par_this = amp_split_ewsud_par(1) - par_prev

if (density_dbg.ne.0) then
  if (dble(amp_split_born_onehel(1)).gt.1d-30) then
C   Full verbose output for standalone calls
    if (ewsud_helselect.eq.0) then
      write(*,'(A)') '  |'
      write(*,'(A)') '  |  THIS helicity Born-weighted Sudakov:'
      write(*,'(A,ES11.3,A,ES11.3)') '  |  LSC = ',
     $    dble(lsc_this), ' + i*', aimag(lsc_this)
      write(*,'(A,ES11.3,A,ES11.3)') '  |  SSC = ',
     $    dble(ssc_this), ' + i*', aimag(ssc_this)
      write(*,'(A,ES11.3,A,ES11.3)') '  |  XXC = ',
     $    dble(xxc_this), ' + i*', aimag(xxc_this)
      write(*,'(A,ES11.3,A,ES11.3)') '  |  PAR = ',
     $    dble(par_this), ' + i*', aimag(par_this)
      write(*,'(A)') '  |'
      write(*,'(A)') '  |  Per-helicity delta (divide by Born):'
      write(*,'(A,ES11.3,A,ES11.3)') '  |    LSC/Born = ',
     $    dble(lsc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(lsc_this)/dble(amp_split_born_onehel(1))
      write(*,'(A,ES11.3,A,ES11.3)') '  |    SSC/Born = ',
     $    dble(ssc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(ssc_this)/dble(amp_split_born_onehel(1))
      write(*,'(A,ES11.3,A,ES11.3)') '  |    XXC/Born = ',
     $    dble(xxc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(xxc_this)/dble(amp_split_born_onehel(1))
      write(*,'(A,ES11.3,A,ES11.3)') '  |    PAR/Born = ',
     $    dble(par_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(par_this)/dble(amp_split_born_onehel(1))
      write(*,*) '  +-------------------------------+'
    else
C   Verbose indented output for nested (per-helicity) calls from density wrapper
      write(*,'(A)') '        |'
      write(*,'(A)') '        |  THIS helicity Born-weighted Sudakov:'
      write(*,'(A,ES11.3,A,ES11.3)') '        |  LSC = ',
     $    dble(lsc_this), ' + i*', aimag(lsc_this)
      write(*,'(A,ES11.3,A,ES11.3)') '        |  SSC = ',
     $    dble(ssc_this), ' + i*', aimag(ssc_this)
      write(*,'(A,ES11.3,A,ES11.3)') '        |  XXC = ',
     $    dble(xxc_this), ' + i*', aimag(xxc_this)
      write(*,'(A,ES11.3,A,ES11.3)') '        |  PAR = ',
     $    dble(par_this), ' + i*', aimag(par_this)
      write(*,'(A)') '        |'
      write(*,'(A)') '        |  Per-helicity delta (divide by Born):'
      write(*,'(A,ES11.3,A,ES11.3)') '        |    LSC/Born = ',
     $    dble(lsc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(lsc_this)/dble(amp_split_born_onehel(1))
      write(*,'(A,ES11.3,A,ES11.3)') '        |    SSC/Born = ',
     $    dble(ssc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(ssc_this)/dble(amp_split_born_onehel(1))
      write(*,'(A,ES11.3,A,ES11.3)') '        |    XXC/Born = ',
     $    dble(xxc_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(xxc_this)/dble(amp_split_born_onehel(1))
      write(*,'(A,ES11.3,A,ES11.3)') '        |    PAR/Born = ',
     $    dble(par_this)/dble(amp_split_born_onehel(1)),
     $    ' + i*', aimag(par_this)/dble(amp_split_born_onehel(1))
      write(*,'(A,I2,A)') '        +-- END [h=',ewsud_helselect,'] --+'
    endif
C   Skip zero-Born helicities silently for nested calls (too noisy)
  endif
endif
enddo

AMP_SPLIT_EWSUD_LSC(:) = AMP_SPLIT_EWSUD_LSC(:)/volh
AMP_SPLIT_EWSUD_SSC(:) = AMP_SPLIT_EWSUD_SSC(:)/volh
AMP_SPLIT_EWSUD_XXC(:) = AMP_SPLIT_EWSUD_XXC(:)/volh
AMP_SPLIT_EWSUD_PAR(:) = AMP_SPLIT_EWSUD_PAR(:)/volh

AMP_SPLIT_EWSUD_QCD(:) = AMP_SPLIT_EWSUD_QCD(:)/VOLH
AMP_SPLIT_EWSUD_PARQCD(:) = AMP_SPLIT_EWSUD_PAR(:)
AMP_BORN_SUMHEL = AMP_BORN_SUMHEL / volh
C Only print summary for standalone calls
if (density_dbg.ne.0.and.ewsud_helselect.eq.0) then
  write(*,*)
  write(*,*) '------------------------------------------------------------'
  write(*,*) '  SCALAR SUDAKOV SUMMARY (after volh normalization)'
  write(*,*) '------------------------------------------------------------'
  write(*,'(A,F8.4)') '  volh = ', volh
  write(*,*)
  write(*,'(A)') '  TOTAL Sudakov contributions (summed over helicities):'
  write(*,'(A,ES12.4,A,ES12.4)') '    LSC = ', dble(amp_split_ewsud_lsc(1)),
     $   ' + i*', aimag(amp_split_ewsud_lsc(1))
  write(*,'(A,ES12.4,A,ES12.4)') '    SSC = ', dble(amp_split_ewsud_ssc(1)),
     $   ' + i*', aimag(amp_split_ewsud_ssc(1))
  write(*,'(A,ES12.4,A,ES12.4)') '    XXC = ', dble(amp_split_ewsud_xxc(1)),
     $   ' + i*', aimag(amp_split_ewsud_xxc(1))
  write(*,'(A,ES12.4,A,ES12.4)') '    PAR = ', dble(amp_split_ewsud_par(1)),
     $   ' + i*', aimag(amp_split_ewsud_par(1))
  write(*,*)
endif

DO IAMP = 1, AMP_SPLIT_SIZE

  IF (HAS_LO1) THEN
    IF(IAMP.ne.LO1_POS) AMP_SPLIT_EWSUD_PAR(IAMP)=(0d0,0d0)
  ELSE
    AMP_SPLIT_EWSUD_PAR(IAMP)=(0d0,0d0)
  ENDIF

  IF (HAS_LO2) THEN
    IF(IAMP.ne.LO2_POS) AMP_SPLIT_EWSUD_PARQCD(IAMP)=(0d0,0d0)
  ELSE
    AMP_SPLIT_EWSUD_PARQCD(IAMP)=(0d0,0d0)
  ENDIF

ENDDO



CALCULATEDBORN=.FALSE.

C Only print END banner for standalone calls
if (density_dbg.ne.0.and.ewsud_helselect.eq.0) then
  write(*,*) '============================================================'
  write(*,*) '  END SCALAR SUDAKOV WRAPPER'
  write(*,*) '============================================================'
  write(*,*)
endif
return
end


double precision function compensate_identical_factor(igold_me)
C     provides the multiplicative factor for the result of the
C      sudakov MEs so that they have the same identical-particle
C      factor as the Born in the specific FKS configuration
implicit none
integer igold_me
C
integer sdk_ident_goldstone(%(ngoldstone_me)d)
data sdk_ident_goldstone / %(sdk_ident_goldstone)s /
C
integer nfksprocess
common/c_nfksprocess/nfksprocess
%(den_factor_lines)s
integer bornspincol
data bornspincol / %(bornspincol)d /

  compensate_identical_factor = dble(sdk_ident_goldstone(igold_me))/dble(iden_values(nfksprocess)/bornspincol)

return
end
