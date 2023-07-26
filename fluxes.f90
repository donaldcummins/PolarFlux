subroutine fluxes(n, z, u, theta, thetas, q, qs, Ci, ustar2, ustarthetastar, ustarqstar)
    ! This subroutine applies the neural network models developed in Cummins et al. (2023) to estimate bulk turbulent
    ! fluxes of momentum, sensible heat and latent heat above ocean and/or sea ice. It takes as inputs:
    ! - the geometric height z above the surface (in m),
    ! - the mean wind speed u at height z (in m.s-1),
    ! - the potential temperature theta at height z (in K),
    ! - the potential temperature thetas at the surface (in K),
    ! - the specific humidity q at height z (in kg.kg-1),
    ! - the specific humidity qs at the surface (in kg.kg-1),
    ! - the sea-ice concentration Ci.
    ! Outputs are the turbulent fluxes in kinematic units:
    ! - the momentum flux ustar2 (in m2.s-2),
    ! - the sensible heat flux ustarthetastar (in K.m.s-1),
    ! - the latent heat flux ustarqstar (in kg.kg-1.m.s-1).
    !
    ! Author: Donald P. Cummins - July 2023
    implicit none
    integer :: n, i
    double precision :: z(n), u(n), theta(n), thetas(n), q(n), qs(n), Ci(n)
    double precision :: ustar2(n), ustarthetastar(n), ustarqstar(n)
    double precision :: wts_ustar2(37), wts_ustarthetastar(37), wts_ustarqstar(37)
    double precision :: mu_ustar2(8), mu_ustarthetastar(8), mu_ustarqstar(8)
    double precision :: sigma_ustar2(8), sigma_ustarthetastar(8), sigma_ustarqstar(8)
    double precision :: h(5), x(8), x_ustar2(8), x_ustarthetastar(8), x_ustarqstar(8)
    
    ! network weights
    wts_ustar2 = (/ -2.698457, -0.2082982, 1.008843, -0.8962571, 0.6442579, -0.1815384, 0.545542, 0.04231869,&
                    0.1696185, -0.06591652, -1.417991, -0.597792, -0.2792875, 0.07032999, 0.7267943, -0.269739,&
                    -23.60562, -34.14375, 0.396079, -2.124988, -7.347982, 13.97365, 8.838685, -1.129445,&
                    -23.30324, -15.90844, 1.326914, -0.0006393174, 9.953861, -25.96599, 2.319868, -20.03327,&
                    -0.1386934, 6.542533, -0.9495974, 0.1208329, 0.7864153 /)
    wts_ustarthetastar = (/ 3.475624, -0.1737433, 0.3949561, -6.588252, 7.205673, 0.9138388, -0.2880016, -9.168941,&
                            -2.615316, 0.05753981, 0.5605595, -0.2962, -0.1596267, 0.6874786, -0.1090963, -0.04523122,&
                            -2.770014, 0.141199, -0.1122901, 6.042091, -6.831117, -0.5461082, -0.04397706, 8.080274,&
                            -2.461576, 0.04062876, 0.5459518, -1.416529, 0.9620431, 0.6131366, -0.1132944, -0.05117435,&
                            -12.98045, 13.03039, 54.13867, 12.56163, -57.19279 /)
    wts_ustarqstar = (/ -0.224414, -0.004587161, -0.01043204, -0.1156849, 0.1072338, 0.1426352, -0.1533125, -0.05399822,&
                        -6.964993, 4.441017, 0.6855671, 7.896905, -26.15913, -1.775534, 10.26618, 0.6300847,&
                        -1.680098, -0.4692678, 0.7209771, -16.6795, 17.88832, -25.11565, 25.56412, 0.2437814,&
                        -7.099287, 4.325113, 0.6564013, 9.566135, -26.91579, -5.725448, 12.94051, -0.06444113,&
                        -7.856596, 18.33054, 12.00217, -0.7311946, -12.08933 /)
    
    ! variable means
    mu_ustar2 = (/ 0.0707286, 5.316002, 5.209305, 254.7479, 253.9415, 0.001176814, 0.001172242, 0.9831443 /)
    mu_ustarthetastar = (/ 0.0004041111, 5.162423, 5.178461, 254.5794, 253.7612, 0.001153563, 0.001146864, 0.9891372 /)
    mu_ustarqstar = (/ -3.057404e-07, 6.103072, 5.471349, 258.04, 257.5672, 0.001464863, 0.001495832, 0.9819678 /)
    
    ! variable standard deviations
    sigma_ustar2 = (/ 0.08140814, 3.705363, 2.698133, 11.15911, 11.7273, 0.001234222, 0.001273011, 0.09355911 /)
    sigma_ustarthetastar = (/ 0.008042439, 3.388587, 2.687863, 11.05918, 11.62066, 0.001214216, 0.001249259, 0.06262574 /)
    sigma_ustarqstar = (/ 1.511262e-06, 4.200521, 2.916774, 11.05718, 11.46966, 0.001273526, 0.001324679, 0.09148071 /)
    
    do i = 1, n
        ! input layer nodes
        x = (/ 1.d0, z(i), u(i), theta(i), thetas(i), q(i), qs(i), Ci(i) /)
        
        ! hidden layer bias node
        h(1) = 1
        
        ! compute ustar2
        x_ustar2(1) = x(1)
        x_ustar2(2:8) = (x(2:8) - mu_ustar2(2:8))/sigma_ustar2(2:8)
        h(2) = 1/(1 + exp(-dot_product(x_ustar2,wts_ustar2(1:8))))
        h(3) = 1/(1 + exp(-dot_product(x_ustar2,wts_ustar2(9:16))))
        h(4) = 1/(1 + exp(-dot_product(x_ustar2,wts_ustar2(17:24))))
        h(5) = 1/(1 + exp(-dot_product(x_ustar2,wts_ustar2(25:32))))
        ustar2(i) = dot_product(h,wts_ustar2(33:37))
        ustar2(i) = ustar2(i)*sigma_ustar2(1) + mu_ustar2(1)
        ustar2(i) = max(ustar2(i), 0.d0)
        
        ! compute ustarthetastar
        x_ustarthetastar(1) = x(1)
        x_ustarthetastar(2:8) = (x(2:8) - mu_ustarthetastar(2:8))/sigma_ustarthetastar(2:8)
        h(2) = 1/(1 + exp(-dot_product(x_ustarthetastar,wts_ustarthetastar(1:8))))
        h(3) = 1/(1 + exp(-dot_product(x_ustarthetastar,wts_ustarthetastar(9:16))))
        h(4) = 1/(1 + exp(-dot_product(x_ustarthetastar,wts_ustarthetastar(17:24))))
        h(5) = 1/(1 + exp(-dot_product(x_ustarthetastar,wts_ustarthetastar(25:32))))
        ustarthetastar(i) = dot_product(h,wts_ustarthetastar(33:37))
        ustarthetastar(i) = ustarthetastar(i)*sigma_ustarthetastar(1) + mu_ustarthetastar(1)
        
        ! compute ustarqstar
        x_ustarqstar(1) = x(1)
        x_ustarqstar(2:8) = (x(2:8) - mu_ustarqstar(2:8))/sigma_ustarqstar(2:8)
        h(2) = 1/(1 + exp(-dot_product(x_ustarqstar,wts_ustarqstar(1:8))))
        h(3) = 1/(1 + exp(-dot_product(x_ustarqstar,wts_ustarqstar(9:16))))
        h(4) = 1/(1 + exp(-dot_product(x_ustarqstar,wts_ustarqstar(17:24))))
        h(5) = 1/(1 + exp(-dot_product(x_ustarqstar,wts_ustarqstar(25:32))))
        ustarqstar(i) = dot_product(h,wts_ustarqstar(33:37))
        ustarqstar(i) = ustarqstar(i)*sigma_ustarqstar(1) + mu_ustarqstar(1)
    end do
end subroutine fluxes
