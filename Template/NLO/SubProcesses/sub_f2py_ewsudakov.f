      subroutine ewsudakov_py(p_born_in, gstr_in, results) 
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      implicit none
C arguments

      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1)
      double precision gstr_in, results(6)
      ! results contain (born, sud0, sud1)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
cc
      include 'coupl.inc'
      include 'orders.inc'

      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud

      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double precision amp_split_born(amp_split_size)
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PAR/AMP_SPLIT_EWSUD_PAR
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_QCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_QCD/AMP_SPLIT_EWSUD_QCD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PARQCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PARQCD/AMP_SPLIT_EWSUD_PARQCD

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

      logical sud_mc_hel
      COMMON /to_mc_hel/ sud_mc_hel

      double precision wgt_sud, wgt_born, born

      logical firsttime
      data firsttime/.true./

      logical s_to_rij
      COMMON /to_s_to_rij/ s_to_rij
      logical rij_ge_mw
      COMMON /rij_ge_mw/ rij_ge_mw
C     Debug flag that must be 0 for Sudakov functions to return non-zero
      integer deb_settozero
      common /to_deb_settozero/deb_settozero
C-----
C  BEGIN CODE
C-----

C     Initialize debug flag (CRITICAL: uninitialized causes Sudakov=0)
      deb_settozero = 0

      nfksprocess=1

      ! let us explicitly sum over the helicities
      sud_mc_hel=.false.

      if (firsttime) then
       call setpara('param_card.dat')   !Sets up couplings and masses
       firsttime = .false.
      endif
     
      g = gstr_in
      call update_as_param()
      p_born(:,:) = p_born_in(:,:)

      s_to_rij = .true.
      rij_ge_mw = .true.
      do sud_mod = 0,1
        ! call the born
        call sborn(p_born, born)
        amp_split_born(:) = amp_split(:)
        wgt_born = amp_split_born(1)

        ! call the EWsudakov
        call sudakov_wrapper(p_born) 
        wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
        results(1) = wgt_born
        results(2+sud_mod) = wgt_sud
      enddo
      !! MZ to be extended to LO_2 etc 

      !! TV: add the various sudakov outputs
      sud_mod = 1
      s_to_rij = .false.
      rij_ge_mw = .true.
      ! call the born
      call sborn(p_born, born)
      amp_split_born(:) = amp_split(:)
      wgt_born = amp_split_born(1)

      ! call the EWsudakov
      call sudakov_wrapper(p_born)
      wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
      results(4) = wgt_sud

      sud_mod = 1
      s_to_rij = .false.
      rij_ge_mw = .false.
      ! call the born
      call sborn(p_born, born)
      amp_split_born(:) = amp_split(:)
      wgt_born = amp_split_born(1)
      ! call the EWsudakov
      call sudakov_wrapper(p_born)
      wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
      results(5) = wgt_sud

      sud_mod = 1
      s_to_rij = .true.
      rij_ge_mw = .false.
      ! call the born
      call sborn(p_born, born)
      amp_split_born(:) = amp_split(:)
      wgt_born = amp_split_born(1)
      ! call the EWsudakov
      call sudakov_wrapper(p_born)
      wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
      results(6) = wgt_sud
      return

      end


      subroutine density_sudakov_py(p_born_in, gstr_in, nres,
     $     res_indices, res_dims, density_born, density_delta,
     $     density_delta_var, born_diag, results)
c**************************************************************************
c     Density matrix EW Sudakov computation for spin-correlated reweighting
c     with resonances. Returns the Born density matrix B[h,h'] and
c     diagonal Sudakov corrections delta_h for each helicity configuration.
c
c     Returns TWO delta arrays:
c       density_delta     - central (s_to_rij=.true., with higher-order corrections)
c       density_delta_var - variation (s_to_rij=.false., without corrections)
c
c     Note: rij_ge_mw is always .true. for FxFx (clustering ensures s_ij > MW^2)
c**************************************************************************
      implicit none

      include 'nexternal.inc'
      include 'coupl.inc'
      include 'orders.inc'

C     Maximum density matrix dimension (for ttWW: 2*2*3*3 = 36)
      integer MAX_DIM
      parameter (MAX_DIM = 36)
C     Triangular storage size: MAX_DIM*(MAX_DIM+1)/2 = 666
      integer MAX_DIM_TRI
      parameter (MAX_DIM_TRI = 666)
      integer MAX_RES
      parameter (MAX_RES = 10)

C     Arguments
      double precision p_born_in(0:3,nexternal-1)
      double precision gstr_in
      integer nres
      integer res_indices(MAX_RES)
      integer res_dims(MAX_RES)
      double complex density_born(MAX_DIM_TRI)
      double complex density_delta(MAX_DIM)
      double complex density_delta_var(MAX_DIM)
      double precision born_diag(MAX_DIM)
      double precision results(6)

CF2PY intent(in) :: p_born_in, gstr_in, nres, res_indices, res_dims
CF2PY intent(out) :: density_born, density_delta, density_delta_var
CF2PY intent(out) :: born_diag, results

C     Local variables
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      integer i, total_dim

      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud

      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double complex amp_split_ewsud_par(amp_split_size)
      common /to_amp_ewsud_par/amp_split_ewsud_par

      integer sud_mod
      common /to_sud_mod/ sud_mod
      integer nfksprocess
      common/c_nfksprocess/nfksprocess

      logical sud_mc_hel
      common /to_mc_hel/ sud_mc_hel

      double precision wgt_born, born

      logical firsttime
      data firsttime/.true./

      logical s_to_rij
      common /to_s_to_rij/ s_to_rij
      logical rij_ge_mw
      common /rij_ge_mw/ rij_ge_mw
C     Debug flag that must be 0 for Sudakov functions to return non-zero
      integer deb_settozero
      common /to_deb_settozero/deb_settozero
      integer density_dbg
      common /density_dbg/ density_dbg
      data density_dbg /0/

C     Helselect mechanism (density wrapper uses this per-helicity)
      integer ewsud_helselect
      common/to_ewsud_helselect/ewsud_helselect
C     Filter helicities flag
      logical sud_filter_hel
      COMMON /to_filter_hel/ sud_filter_hel

C-----
C  BEGIN CODE
C-----

C     Initialize debug flag (CRITICAL: uninitialized causes Sudakov=0)
      deb_settozero = 0
      if (density_dbg .ne. 0) then
        write(*,*)
        write(*,*) '+============================================================+'
        write(*,*) '|         DENSITY SUDAKOV - F2PY ENTRY POINT                 |'
        write(*,*) '+============================================================+'
        write(*,*)
        write(*,'(A,I2)') '  Number of resonances: ', nres
        if (nres .gt. 0) then
          do i = 1, nres
            write(*,'(A,I1,A,I2,A,I1,A)') '    Resonance ', i,
     $        ': particle #', res_indices(i),
     $        ' with ', res_dims(i), ' spin states'
          enddo
        endif
        write(*,'(A,F10.6)') '  Strong coupling (g): ', gstr_in
        write(*,*)
      endif

      nfksprocess = 1
      sud_mc_hel = .false.

      if (firsttime) then
        call setpara('param_card.dat')
        firsttime = .false.
      endif

      g = gstr_in
      call update_as_param()
      p_born(:,:) = p_born_in(:,:)

C     Initialize outputs
      total_dim = 1
      do i = 1, nres
        total_dim = total_dim * res_dims(i)
      enddo

      do i = 1, MAX_DIM * (MAX_DIM + 1) / 2
        density_born(i) = dcmplx(0d0, 0d0)
      enddo
      do i = 1, MAX_DIM
        density_delta(i) = dcmplx(0d0, 0d0)
        density_delta_var(i) = dcmplx(0d0, 0d0)
        born_diag(i) = 0d0
      enddo
      do i = 1, 6
        results(i) = 0d0
      enddo

C     Set standard Sudakov mode (rij_ge_mw always true for FxFx - clustering ensures s_ij > MW^2)
      sud_mod = 1
      rij_ge_mw = .true.

C     Just compute Born for normalization check (Sudakov done in density wrapper)
      call sborn(p_born, born)
      wgt_born = amp_split(1)
      results(1) = wgt_born
      if (density_dbg .ne. 0) then
        write(*,'(A,ES14.6)') '  Born (IDEN-averaged): ', wgt_born
      endif

C     Compute density matrix B[h,h'] and Sudakov corrections delta[h]
C     Phase 1: JAMP + color contraction -> B[h,h']
C     Phase 2: Per-helicity Sudakov via scalar wrapper (handles goldstone,
C              comp_idfac, SSC_C, nondiag, PAR automatically)
C     Phase 3: Normalize delta by diagonal Born
C
C     Initialize helselect and disable filtering for density computation
      ewsud_helselect = 0
      sud_filter_hel = .false.

C     CENTRAL: s_to_rij = .true. (with higher-order s->rij corrections)
      s_to_rij = .true.
      if (density_dbg .ne. 0) then
        write(*,*)
        write(*,*) '  === CENTRAL (s_to_rij=TRUE) ==='
      endif
      call density_matrix_wrapper(p_born, nres, res_indices, res_dims,
     $       density_born, density_delta, born_diag)
      results(2) = 0d0

C     VARIATION: s_to_rij = .false. (without higher-order corrections)
C     Recompute only the delta, reusing density_born structure
      s_to_rij = .false.
      if (density_dbg .ne. 0) then
        write(*,*)
        write(*,*) '  === VARIATION (s_to_rij=FALSE) ==='
      endif
C     Call wrapper again with variation setting, store in density_delta_var
C     Note: density_born and born_diag are recomputed but should be identical
      call density_matrix_wrapper(p_born, nres, res_indices, res_dims,
     $       density_born, density_delta_var, born_diag)
      results(3) = 0d0

      if (density_dbg .ne. 0) then
        write(*,*)
        write(*,*) '============================================================'
        write(*,*) '  END DENSITY SUDAKOV F2PY'
        write(*,*) '============================================================'
        write(*,*)
      endif

      return
      end

