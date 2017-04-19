"""
Exact solver for the Riemann problem for the 1D shallow water equations with a step
in the bathymetry.  The implementation here is based on the paper by Rosatti & Bugnanelli.

The solution consists of two genuinely nonlinear waves as well as a stationary wave
at x=0.  The ordering of these waves depends on the initial data; for subcritical flow
the stationary wave is in the middle, so we refer to the other two as 1-waves and
3-waves.  If the 1-wave or 3-wave is transonic, then it manifests as two waves, one
of which is a rarefaction; the other may be a shock or rarefaction.  In this sense
there may be as many as 4 waves in the solution.
"""

import numpy as np
from scipy.optimize import fsolve

def predictor(h_l, h_r, u_l, u_r, b_l, b_r, g=1.):
    "Generalized Roe solver used for initial guess."
    # Roe averages
    h_ave = 0.5*(h_l+h_r)
    u_ave = (u_l*np.sqrt(h_l)+u_r*np.sqrt(h_r))/(np.sqrt(h_l)+np.sqrt(h_r))
    c_ave = np.sqrt(g*h_ave)

    # Wave speeds
    lambda1 = u_ave-c_ave
    lambda2 = 0.
    lambda3 = u_ave+c_ave

    # Right eigenvectors of flux Jacobian
    if b_r >= b_l:
        a23 = g*(h_l - 0.5*np.abs(b_r-b_l))
    else:
        a23 = g*(h_r - 0.5*np.abs(b_r-b_l))
    r1 = np.array([1.,lambda1,0.])
    r2 = np.array([a23,0,u_ave**2 - c_ave**2])
    r3 = np.array([1.,lambda3,0.])

    # Delta q
    db = b_r - b_l
    dh = h_r - h_l
    duh = h_r*u_r - h_l*u_l

    # Wave strengths
    mu1 =  (dh*lambda3 - duh)/(2*c_ave) - a23*db/(2*c_ave*lambda1)
    mu2 = db/(lambda1*lambda3)
    mu3 = -(dh*lambda1 - duh)/(2*c_ave) + a23*db/(2*c_ave*lambda3)

    states = np.zeros((2,4))
    l, lstar, rstar, r = 0, 1, 2, 3
    states[0,l] = h_l
    states[1,l] = u_l*h_l
    states[0,r] = h_r
    states[1,r] = u_r*h_r

    wave_types = ['','','']

    if lambda1 < lambda2 < lambda3:
        states[:,1] = states[:,0] + mu1*r1[:2]
        states[:,2] = states[:,1] + mu2*r2[:2]
        states[:,3] = states[:,2] + mu3*r3[:2]
        wave_types[1] = 'contact'
    elif lambda2 < lambda1 < lambda3:
        states[:,1] = states[:,0] + mu2*r2[:2]
        states[:,2] = states[:,1] + mu1*r1[:2]
        states[:,3] = states[:,2] + mu3*r3[:2]
        wave_types[0] = 'contact'
    elif lambda1 < lambda3 < lambda2:
        states[:,1] = states[:,0] + mu1*r1[:2]
        states[:,2] = states[:,1] + mu3*r3[:2]
        states[:,3] = states[:,2] + mu2*r2[:2]
        wave_types[2] = 'contact'
    else:
        raise Exception()

    depth = 0
    mom   = 1
    c_l = u_l - np.sqrt(g*h_l)
    c_r = u_r + np.sqrt(g*h_r)
    c1 = states[mom,1]/states[depth,1] - np.sqrt(g*states[depth,1])
    c3 = states[mom,2]/states[depth,2] + np.sqrt(g*states[depth,2])

    if lambda1 < lambda2 < lambda3:
        if c1 > c_l:
            wave_types[0] = 'raref'
        else:
            wave_types[0] = 'shock'
        if c3 < c_r:
            wave_types[2] = 'raref'
        else:
            wave_types[2] = 'shock'

    speeds = (lambda1, lambda2, lambda3)

    return states, speeds, wave_types


def corrector(states, speeds, b_l, b_r, g=1.):
    # For now just handle the subcritical cases

    h_l = states[0,0]
    h_r = states[0,3]
    u_l = states[1,0]/h_l
    u_r = states[1,3]/h_r

    def phi_l(hlstar,ulstar,hrstar,urstar):
        if hlstar < h_l:
            # Rarefaction; use Riemann invariant
            return u_l + 2*np.sqrt(g*h_l) - (ulstar + 2*np.sqrt(g*hlstar))
        else:
            # Shock; use Hugoniot locus
            alpha = h_l-hlstar
            return hlstar*ulstar-h_l*u_l + \
                    alpha*(ulstar-np.sqrt(g*hlstar*
                           (1+alpha/hlstar)*(1+alpha/(2*hlstar))))

    def phi_r(hlstar,ulstar,hrstar,urstar):
        if hrstar < h_r:
            return u_r - 2*np.sqrt(g*h_r) - (urstar - 2*np.sqrt(g*hrstar))
        else:
            alpha = h_r-hrstar
            return hrstar*urstar-h_r*u_r + \
                    alpha*(urstar+np.sqrt(g*hrstar*
                           (1+alpha/hrstar)*(1+alpha/(2*hrstar))))
    def phi_m(hlstar,ulstar,hrstar,urstar):
        if b_r > b_l:
            h_s = hlstar - (b_r-b_l)
            resid = hrstar*urstar**2-hlstar*ulstar**2 + 0.5*g*(hrstar**2-h_s**2)
        else:
            h_s = hrstar - (b_l-b_r)
            resid = hrstar*urstar**2-hlstar*ulstar**2 + 0.5*g*(h_s**2-hlstar**2)
        return (hlstar*ulstar-hrstar*urstar,resid)

    def phi(args):
        x = phi_m(*args)
        return np.array([phi_l(*args),x[0],x[1],phi_r(*args)])

    guess = np.array([states[0,1],states[1,1]/states[0,1],states[0,2],states[1,2]/states[0,2]])
    middle_states,info, ier, msg = fsolve(phi, guess, full_output=True, xtol=1.e-14)
    states[0,1] = middle_states[0]
    states[1,1] = middle_states[1]*middle_states[0]
    states[0,2] = middle_states[2]
    states[1,2] = middle_states[2]*middle_states[3]

    h_l    = states[0,0]; hu_l    = states[1,0]
    hlstar = states[0,1]; hulstar = states[1,1]; ulstar = hulstar/hlstar
    hrstar = states[0,2]; hurstar = states[1,2]; urstar = hurstar/hrstar
    h_r    = states[0,3]; hu_r    = states[1,3]

    wave_types = ['', 'contact', '']
    if hlstar < h_l:
        wave_types[0] = 'raref'
    else:
        wave_types[0] = 'shock'
    if hrstar < h_r:
        wave_types[2] = 'raref'
    else:
        wave_types[2] = 'shock'

    ws = np.zeros(5)
    ws[2] = 0.
    speeds = ['',0,'']
    if wave_types[0] == 'shock':
        speeds[0] = (hulstar-hu_l)/(hlstar-h_l)
        ws[0] = speeds[0]
        ws[1] = speeds[0]
    else:  # 1-rarefaction
        ws[0] = u_l - np.sqrt(g*h_l)
        ws[1] = ulstar - np.sqrt(g*hlstar)
        speeds[0] = (ws[0],ws[1])

    if wave_types[2] == 'shock':
        speeds[2] = (hu_r-hurstar)/(h_r-hrstar)
        ws[3] = speeds[2]
        ws[4] = speeds[2]
    else:  # 2-rarefaction
        ws[3] = urstar + np.sqrt(g*hrstar)
        ws[4] = u_r + np.sqrt(g*h_r)
        speeds[2] = (ws[3],ws[4])
    def raref1(xi):
        RiemannInvariant = u_l + 2*np.sqrt(g*h_l)
        h = ((RiemannInvariant - xi)**2 / (9*g))
        u = (xi + np.sqrt(g*h))
        hu = h*u
        return h, hu

    def raref2(xi):
        RiemannInvariant = u_r - 2*np.sqrt(g*h_r)
        h = ((RiemannInvariant - xi)**2 / (9*g))
        u = (xi - np.sqrt(g*h))
        hu = h*u
        return h, hu

    def reval(xi):
        """
        Evaluate the Riemann solution for arbitrary xi = x/t.
        """
        rar1 = raref1(xi)
        rar2 = raref2(xi)
        h_out = (xi<=ws[0])*h_l + \
            (xi>ws[0])*(xi<=ws[1])*rar1[0] + \
            (xi>ws[1])*(xi<=ws[2])*hlstar +  \
            (xi>ws[2])*(xi<=ws[3])*hrstar + \
            (xi>ws[3])*(xi<=ws[4])*rar2[0] +  \
            (xi>ws[4])*h_r
        h_out[h_out>1e8] = np.nan
        hu_out = (xi<=ws[0])*hu_l + \
            (xi>ws[0])*(xi<=ws[1])*rar1[1] + \
            (xi>ws[1])*(xi<=ws[2])*hlstar +  \
            (xi>ws[2])*(xi<=ws[3])*hrstar +  \
            (xi>ws[3])*(xi<=ws[4])*rar2[1] +  \
            (xi>ws[4])*hu_r
        hu_out[hu_out>1e8] = np.nan
        return h_out, hu_out
    return states, speeds, reval, wave_types
