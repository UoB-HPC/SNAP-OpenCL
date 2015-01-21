!-----------------------------------------------------------------------
!
! MODULE: octsweep_module
!> @brief
!> This module controls the setup and calls for sweeping a single octant
!> pair. It calls for the actual sweep logic depending on the spatial
!> dimensionality of the problem.
!
!-----------------------------------------------------------------------

MODULE octsweep_module

  USE global_module, ONLY: i_knd, r_knd, zero

  USE geom_module, ONLY: nc, ndimen, dinv, nx, ny, nz, hi, hj, hk

  USE sn_module, ONLY: ec, nang, wmu, weta, wxi, cmom, noct, mu

  USE data_module, ONLY: vdelt, ng

  USE control_module, ONLY: timedep

  USE solvar_module, ONLY: psii, qtot, ptr_in, ptr_out, flux, fluxm,   &
    psij, psik, jb_in, jb_out, kb_in, kb_out, flkx, flky, flkz, t_xs

  USE dim1_sweep_module, ONLY: dim1_sweep

  USE dim3_sweep_module, ONLY: dim3_sweep

  USE plib_module, ONLY: ichunk

  IMPLICIT NONE

  PUBLIC :: octsweep


  CONTAINS


  SUBROUTINE octsweep ( g, iop, jd, kd, jlo, jhi, jst, klo, khi, kst )

!-----------------------------------------------------------------------
!
! Call for the appropriate sweeper depending on the dimensionality. Let
! the actual sweep routine know the octant info to get the order right.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: g, iop, jd, kd, jlo, jhi, jst, klo,  &
      khi, kst
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: id, oct, ich, d1, d2, d3, d4, i1, i2

!_______________________________________________________________________
!
!   Local memory for the OpenCL sweep result
!_______________________________________________________________________

    REAL(r_knd), DIMENSION(:,:,:,:,:,:), POINTER :: ocl_flux
    INTEGER(i_knd) :: a, i, j, k, gg
    ALLOCATE( ocl_flux(nang,nx,ny,nz,noct,ng) )

!_______________________________________________________________________
!
!   Determine octant and chunk index.
!_______________________________________________________________________

    id = 1 + (iop-1)/nc
    oct = id + 2*(jd - 1) + 4*(kd - 1)

    IF ( id == 1 ) THEN
      ich = nc - iop + 1
    ELSE
      ich = iop - nc
    END IF
!_______________________________________________________________________
!
!   Send ptr_in and ptr_out dimensions dependent on timedep because they
!   are not allocated if static problem
!_______________________________________________________________________

    d1 = 0; d2 = 0; d3 = 0; d4 = 0
    i1 = 0; i2 = 0
    IF ( timedep == 1 ) THEN
      d1 = nang; d2 = nx; d3 = ny; d4 = nz
      i1 = oct; i2 = g
    END IF
!_______________________________________________________________________
!
!   As the groups are batched in SNAP, we do the sweep on the OpenCL
!   device many times, so we need to zero out the edge flux arrays
!_______________________________________________________________________


    CALL zero_edge_flux_buffers

!_______________________________________________________________________
!
!   Do one octant sweep on the OpenCL device
!_______________________________________________________________________

    CALL sweep_octant

!_______________________________________________________________________
!
!   Call for the actual sweeper. Ensure proper size/bounds of time-dep
!   arrays is given to avoid errors.
!_______________________________________________________________________

    IF ( ndimen == 1 ) THEN
      CALL dim1_sweep ( id, d1, d2, d3, d4, oct, g, psii(:,1,1,g),     &
        qtot(:,:,1,1,g), ec(:,:,oct), vdelt(g), ptr_in(:,:,:,:,i1,i2), &
        ptr_out(:,:,:,:,i1,i2), dinv(:,:,1,1,g), flux(:,1,1,g),        &
        fluxm(:,:,1,1,g), wmu, flkx(id,1,1,g), t_xs(:,1,1,g) )
    ELSE
      CALL dim3_sweep ( ich, id, d1, d2, d3, d4, jd, kd, jlo, klo,     &
        oct, g, jhi, khi, jst, kst, psii(:,:,:,g), psij(:,:,:,g),      &
        psik(:,:,:,g), qtot(:,:,:,:,g), ec(:,:,oct), vdelt(g),         &
        ptr_in(:,:,:,:,i1,i2), ptr_out(:,:,:,:,i1,i2),                 &
        dinv(:,:,:,:,g), flux(:,:,:,g), fluxm(:,:,:,:,g),              &
        jb_in(:,:,:,g), jb_out(:,:,:,g), kb_in(:,:,:,g),               &
        kb_out(:,:,:,g), wmu, weta, wxi, flkx(:,:,:,g), flky(:,:,:,g), &
        flkz(:,:,:,g), t_xs(:,:,:,g) )
    END IF

!_______________________________________________________________________
!
!   TODO
!   Check that the OpenCL sweep of the octant matches the original
!_______________________________________________________________________

  CALL get_output_flux ( ocl_flux )

  DO gg = 1, ng
    DO k = 1, nz
      DO j = 1, ny
        DO i = 1, ichunk
          DO a = 1, nang
            IF ( ABS( ocl_flux(a,i,j,k,1,g) - ptr_out(a,i,j,k,1,g) ) > 1.0E-12_r_knd ) THEN
              PRINT *, "FLUX INCORRECT", a, i, j, k, gg
              STOP
            END IF
          END DO
        END DO
      END DO
    END DO
  END DO

  DEALLOCATE( ocl_flux )
  STOP

!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE octsweep


END MODULE octsweep_module
