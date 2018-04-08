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
from scipy.optimize import fsolve, root

conserved_variables = ('Depth', 'Momentum')
primitive_variables = ('Depth', 'Velocity')
depth = 0
mom   = 1

def pospart(x):
    return np.maximum(1.e-15,x)

def primitive_to_conservative(h, u):
    hu = h*u
    return h, hu

def conservative_to_primitive(h, hu):
    assert np.all(h>=0)
    # We should instead check that hu is zero everywhere that h is
    u = hu/pospart(h)
    return h, u

def cons_to_prim(q):
    return conservative_to_primitive(*q)


def c1(h,u,g):
    return u - np.sqrt(g*h)

def c3(h,u,g):
    return u + np.sqrt(g*h)

def ri1(h,u,g):
    return u + 2*np.sqrt(g*h)

def ri3(h,u,g):
    return u - 2*np.sqrt(g*h)

def integral_curve_1(h1, u1, h2, u2, g):
    return ri1(h1,u1,g) - ri1(h2,u2,g)

def hugoniot_locus_1(h1, u1, h2, u2, g):
    alpha = h1-h2
    return h2*u2-h1*u1 + alpha*(u2-np.sqrt(g*h2*(1+alpha/h2)*(1+alpha/(2*h2))))

def integral_curve_3(h1, u1, h2, u2, g):
    return ri3(h1,u1,g) - ri3(h2,u2,g)

def hugoniot_locus_3(h1, u1, h2, u2, g):
    alpha = h1-h2
    return h2*u2-h1*u1 + alpha*(u2+np.sqrt(g*h2*(1+alpha/h2)*(1+alpha/(2*h2))))

def raref1(xi,href,uref,g):
    h = ((ri1(href,uref,g)- xi)**2 / (9*g))
    u = (xi + np.sqrt(g*h))
    hu = h*u
    return h, hu

def raref2(xi,href,uref,g):
    h = ((ri3(href,uref,g) - xi)**2 / (9*g))
    u = (xi - np.sqrt(g*h))
    hu = h*u
    return h, hu

def shock_speed(left, right):
    "Inputs are in primitive variables (h, u)."
    return (right[depth]*right[1]-left[depth]*left[1])/(right[depth]-left[depth])

def exact_riemann_solution(q_l, q_r, b_l, b_r, g=1., which='hydrostatic', primitive_inputs=False):
    if primitive_inputs:
        h_l, u_l = q_l
        h_r, u_r = q_r
    else:
        h_l, u_l = conservative_to_primitive(*q_l)
        h_r, u_r = conservative_to_primitive(*q_r)

    states, speeds, wave_types = predictor(h_l, h_r, u_l, u_r, b_l, b_r, g=1.)
    states, speeds, reval, wave_types = corrector(states,b_l,b_r, g=1., which=which)
    return states, speeds, reval, wave_types

def discharge_condition(left, right, b_l, b_r, g, which='hydrostatic'):
    if which == 'hydrostatic':
        if b_r > b_l:
            h_s = left[depth] - (b_r-b_l)
            return right[depth]*right[1]**2-left[depth]*left[1]**2 + 0.5*g*(right[depth]**2-h_s**2)
        else:
            h_s = right[depth] - (b_l-b_r)
            return right[depth]*right[1]**2-left[depth]*left[1]**2 + 0.5*g*(h_s**2-left[depth]**2)
    elif which == 'alcrudo':
        return right[depth] - left[depth] + (right[1]**2-left[1]**2)/(2.*g) + b_r - b_l
    elif which == 'dgeorge':
        h_mean = (right[depth]+left[depth])/2.
        lamtilde = max(0., right[1]*left[1])-g*h_mean
        lambar = 0.25*(right[1]+left[1])**2 - g*h_mean
        htilde = h_mean*lamtilde/lambar
        return right[depth]*right[1]**2-left[depth]*left[1]**2 + 0.5*g*(right[depth]**2-left[depth]**2) + g*htilde*(b_r-b_l)


def predictor(h_l, h_r, u_l, u_r, b_l, b_r, g=1.):
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

    # Delta q (jumps in conserved quantities)
    db = b_r - b_l
    dh = h_r - h_l
    duh = h_r*u_r - h_l*u_l

    # Wave strengths
    mu1 =  (dh*lambda3 - duh)/(2*c_ave) - a23*db/(2*c_ave*lambda1)
    mu2 = db/(lambda1*lambda3)
    mu3 = -(dh*lambda1 - duh)/(2*c_ave) + a23*db/(2*c_ave*lambda3)

    states = np.zeros((2,4))
    l=0
    r=3

    states[depth,l] = h_l
    states[mom,l] = u_l*h_l
    states[depth,r] = h_r
    states[mom,r] = u_r*h_r

    wave_types = ['','','']

    c_l = u_l - np.sqrt(g*h_l)
    c_r = u_r + np.sqrt(g*h_r)

    # Compute predicted states
    if lambda1 < lambda2 < lambda3:  # Subcritical flow
        states[:,1] = states[:,0] + mu1*r1[:2]
        states[:,2] = states[:,1] + mu2*r2[:2]
        states[:,3] = states[:,2] + mu3*r3[:2]
        wave_types[1] = 'contact'
        speeds = (lambda1, lambda2, lambda3)

        c1 = states[mom,1]/states[depth,1] - np.sqrt(g*states[depth,1])
        c3 = states[mom,2]/states[depth,2] + np.sqrt(g*states[depth,2])
        # Note that (69)-(70) in the paper seem to have typos
        if c1 > c_l:
            wave_types[0] = 'raref'
        else:
            wave_types[0] = 'shock'
        if c3 < c_r:
            wave_types[2] = 'raref'
        else:
            wave_types[2] = 'shock'

    elif lambda2 < lambda1 < lambda3:  # Supercritical flow to right
        states[:,1] = states[:,0] + mu2*r2[:2]
        states[:,2] = states[:,1] + mu1*r1[:2]
        states[:,3] = states[:,2] + mu3*r3[:2]
        wave_types[0] = 'contact'
        speeds = (lambda2, lambda1, lambda3)

        c1 = states[mom,1]/states[depth,1] - np.sqrt(g*states[depth,1])
        c3 = states[mom,2]/states[depth,2] + np.sqrt(g*states[depth,2])
        if c3 > c1:
            wave_types[1] = 'raref'
        else:
            wave_types[1] = 'shock'
        if c3 < c_r:
            wave_types[2] = 'raref'
        else:
            wave_types[2] = 'shock'

    elif lambda1 < lambda3 < lambda2:  # Supercritical flow to left
        states[:,1] = states[:,0] + mu1*r1[:2]
        states[:,2] = states[:,1] + mu3*r3[:2]
        states[:,3] = states[:,2] + mu2*r2[:2]
        wave_types[2] = 'contact'
        speeds = (lambda1, lambda3, lambda2)

        c1 = states[mom,1]/states[depth,1] - np.sqrt(g*states[depth,1])
        c3 = states[mom,2]/states[depth,2] + np.sqrt(g*states[depth,2])
        if c1 > c_l:
            wave_types[0] = 'raref'
        else:
            wave_types[0] = 'shock'
        if c1 < c3:
            wave_types[2] = 'raref'
        else:
            wave_types[2] = 'shock'

    else:
        raise Exception("Unexpected ordering of wavespeeds in predictor.")
    return states, speeds, wave_types

def corrector(states, b_l, b_r, g=1., which='hydrostatic'):
    l = 0
    r = 3

    h_l = states[depth,l]
    h_r = states[depth,r]
    u_l = states[mom,l]/h_l
    u_r = states[mom,r]/h_r

    c1_l = c1(h_l, u_l, g)
    c3_r = c3(h_r, u_r, g)

    def phi(intermediate_states):
        "Takes the 2 intermediate states and returns the residual of the relevant conditions."
        hlstar,ulstar,hrstar,urstar = intermediate_states  # Unpack
        c1_lstar = c1(hlstar, ulstar, g)
        c3_rstar = c3(hrstar, urstar, g)
        residuals = np.zeros(4)

        # Resonant cases
        if ((c1_lstar < 0) and (c1(hrstar,urstar,g)>0)) or ((c1_l<0) and (c1_lstar>0)):  # Resonant 1-wave
            if b_r > b_l:
                # Flow is critical on right side
                u_plus = urstar/3. + 2./3. * np.sqrt(g*hrstar)
                h_plus = u_plus**2/g

                if c1_lstar > c1_l:  # 1-rarefaction
                    residuals[0] = integral_curve_1(h_l, u_l, hlstar, ulstar, g)
                else:  # 1-shock
                    residuals[0] = hugoniot_locus_1(h_l, u_l, hlstar, ulstar, g)

                residuals[1] = hlstar*ulstar - h_plus*u_plus
                residuals[2] = discharge_condition([hlstar,ulstar],[h_plus,u_plus],b_l,b_r,g,which)

                if (c3_rstar < c3_r):  # 3-rarefaction
                    residuals[3] = integral_curve_3(h_r, u_r, hrstar, urstar, g)
                else:  # 3-shock
                    residuals[3] = hugoniot_locus_3(h_r, u_r, hrstar, urstar, g)

            else:
                # Flow is critical on left side
                raise NotImplementedError

        elif (c3_r > 0) and (c3_rstar < 0):  # Resonant 3-wave
            if b_r > b_l:
                raise NotImplementedError
            else:
                raise NotImplementedError

        # Non-resonant cases
        elif (c1_lstar < 0) and (c3_rstar > 0):  # Subcritical flow
            if c1_lstar > c1_l:  # 1-rarefaction
                residuals[0] = integral_curve_1(h_l, u_l, hlstar, ulstar, g)
            else:  # 1-shock
                residuals[0] = hugoniot_locus_1(h_l, u_l, hlstar, ulstar, g)

            residuals[1] = hlstar*ulstar - hrstar*urstar
            residuals[2] = discharge_condition([hlstar,ulstar],[hrstar,urstar],b_l,b_r,g,which)

            if (c3_rstar < c3_r):  # 3-rarefaction
                residuals[3] = integral_curve_3(h_r, u_r, hrstar, urstar, g)
            else:  # 3-shock
                residuals[3] = hugoniot_locus_3(h_r, u_r, hrstar, urstar, g)

        elif (c1_lstar > 0):  # Supercritical flow to the right
            if c1(hrstar, urstar, g=g) > c1(hlstar, ulstar, g=g):  # 1-rarefaction
                residuals[0] = integral_curve_1(hlstar, ulstar, hrstar, urstar, g)
            else:  # 1-shock
                residuals[0] = hugoniot_locus_1(hlstar, ulstar, hrstar, urstar, g)

            residuals[1] = hlstar*ulstar - h_l*u_l
            residuals[2] = discharge_condition([h_l,u_l],[hlstar,ulstar],b_l,b_r,g,which)

            if (c3_rstar < c3_r):  # 3-rarefaction
                residuals[3] = integral_curve_3(h_r, u_r, hrstar, urstar, g)
            else:  # 3-shock
                residuals[3] = hugoniot_locus_3(h_r, u_r, hrstar, urstar, g)

        elif (c3_rstar<0):  # Supercritical flow to the left
            if c1_lstar > c1_l:  # 1-rarefaction
                residuals[0] = integral_curve_1(h_l, u_l, hlstar, ulstar, g)
            else:  # 1-shock
                residuals[0] = hugoniot_locus_1(h_l, u_l, hlstar, ulstar, g)

            residuals[1] = hrstar*urstar - h_r*u_r
            residuals[2] = discharge_condition([hrstar,urstar],[h_r,u_r],b_l,b_r,g,which)

            if (c3(hlstar,ulstar,g=g) < c3(hrstar,urstar,g=g)):  # 3-rarefaction
                residuals[3] = integral_curve_3(hrstar, urstar, hlstar, ulstar, g)
            else:  # 3-shock
                residuals[3] = hugoniot_locus_3(hrstar, urstar, hlstar, ulstar, g)
        else:
            raise NotImplementedError
        return residuals

    guess = np.array([states[0,1],states[1,1]/states[0,1],states[0,2],states[1,2]/states[0,2]])
    use_fsolve = True
    if use_fsolve:
        middle_states,info, ier, msg = fsolve(phi, guess, full_output=True, xtol=1.e-14)

    else:
        soln = root(phi, guess, tol=1.e-14, method='lm',options={'xtol':1.e-14})
        middle_states = soln.x

    states[0,1] = middle_states[0]
    states[1,1] = middle_states[1]*middle_states[0]
    states[0,2] = middle_states[2]
    states[1,2] = middle_states[2]*middle_states[3]

    h_l    = states[0,0]; hu_l    = states[1,0]
    hlstar = states[0,1]; hulstar = states[1,1]; ulstar = hulstar/hlstar
    hrstar = states[0,2]; hurstar = states[1,2]; urstar = hurstar/hrstar
    h_r    = states[0,3]; hu_r    = states[1,3]

    c1_lstar = c1(hlstar, ulstar, g)
    c3_rstar = c3(hrstar, urstar, g)

    if (c1_lstar < 0) and (c1(hrstar,urstar,g)>0):  # Resonant 1-wave
        wave_types = ['','contact','raref','']  # Correct only if b_r > b_l
        ws = np.zeros(7)
        speeds = ['',0,'','']
        full_states = np.zeros((2,5))
        full_states[:,:2] = states[:,:2]
        full_states[:,3:] = states[:,2:]
        full_states[1,2] = urstar/3. + 2./3. * np.sqrt(g*hrstar)
        full_states[0,2] = full_states[1,2]**2/g

        left = (h_l, u_l)
        right = (hlstar, ulstar)
        i = 0

        if c1(*left,g=g) < c1(*right,g=g):  # 1-rarefaction
            wave_types[i] = 'raref'
            ws[i] = c1(*left, g=g)
            ws[i+1] = c1(*right, g=g)
            speeds[i] = (ws[i],ws[i+1])
        else:  # 1-shock
            wave_types[i] = 'shock'
            speeds[i] = shock_speed(left, right)
            ws[i] = speeds[i]
            ws[i+1] = speeds[i]

        speeds[2] = (0,c1(hrstar,urstar,g))
        ws[3] = 0
        ws[4] = c1(hrstar,urstar,g)

        left = (hrstar, urstar)
        right = (h_r, u_r)
        i = 3
        if c3(*left,g=g) < c3(*right,g=g):  # 3-rarefaction
            wave_types[i] = 'raref'
            ws[i+2] = c3(*left, g=g)
            ws[i+3] = c3(*right, g=g)
            speeds[i] = (ws[i+2],ws[i+3])
        else:  # 3-shock
            wave_types[i] = 'shock'
            speeds[i] = shock_speed(left, right)
            ws[i+2] = speeds[i]
            ws[i+3] = speeds[i]

        def reval(xi):
            rar1a = raref1(xi,hlstar,ulstar,g)
            rar1b = raref1(xi,hrstar,urstar,g)
            rar2 = raref2(xi,hrstar,urstar,g)
            h_out = (xi<=ws[0])*h_l + \
                    (xi>ws[0])*(xi<=ws[1])*rar1a[0] + \
                    (xi>ws[1])*(xi<=ws[2])*hlstar +  \
                    (xi>ws[3])*(xi<=ws[4])*rar1b[0] + \
                    (xi>ws[4])*(xi<=ws[5])*hrstar + \
                    (xi>ws[5])*(xi<=ws[6])*rar2[0] +  \
                    (xi>ws[6])*h_r
            hu_out = (xi<=ws[0])*hu_l + \
                    (xi>ws[0])*(xi<=ws[1])*rar1a[1] + \
                    (xi>ws[1])*(xi<=ws[2])*hlstar*ulstar +  \
                    (xi>ws[3])*(xi<=ws[4])*rar1b[1] +  \
                    (xi>ws[4])*(xi<=ws[4])*hrstar*urstar +  \
                    (xi>ws[5])*(xi<=ws[6])*rar2[1] +  \
                    (xi>ws[6])*hu_r
            h_out[h_out>1e8] = np.nan
            hu_out[hu_out>1e8] = np.nan
            return h_out, hu_out

    else:
        # Non-resonant cases
        # Assume subsonic case at first
        wave_types = ['','contact','']
        ws = np.zeros(5)
        speeds = ['',0,'']
        if c1(hlstar,ulstar,g=g) < 0:  # 1-wave goes left
            left = (h_l, u_l)
            right = (hlstar, ulstar)
            i = 0
        else:  # 1-wave is supersonic (both waves go right)
            left = (hlstar, ulstar)
            right = (hrstar, urstar)
            i = 1
            wave_types[0] = 'contact'
            ws[0] = 0.
            speeds[0] = 0.

        if c1(*left,g=g) < c1(*right,g=g):  # 1-rarefaction
            wave_types[i] = 'raref'
            ws[i] = c1(*left, g=g)
            ws[i+1] = c1(*right, g=g)
            speeds[i] = (ws[i],ws[i+1])
        else:  # 1-shock
            wave_types[i] = 'shock'
            speeds[i] = shock_speed(left, right)
            ws[i] = speeds[i]
            ws[i+1] = speeds[i]

        if c3(hrstar, urstar,g=g)>0:  # 3-wave goes right
            left = (hrstar, urstar)
            right = (h_r, u_r)
            i = 2
        else:  # 3-wave goes left
            left = (hlstar, ulstar)
            right = (hrstar, urstar)
            i = 1
            wave_types[2] = 'contact'
            ws[4] = 0.
            speeds[2] = 0.
        if c3(*left,g=g) < c3(*right,g=g):  # 3-rarefaction
            wave_types[i] = 'raref'
            ws[i+1] = c3(*left, g=g)
            ws[i+2] = c3(*right, g=g)
            speeds[i] = (ws[i+1],ws[i+2])
        else:  # 3-shock
            wave_types[i] = 'shock'
            speeds[i] = shock_speed(left, right)
            ws[i+1] = speeds[i]
            ws[i+2] = speeds[i]

        def reval(xi):
            """
            Evaluate the Riemann solution for arbitrary xi = x/t.
            """
            rar1 = raref1(xi,hlstar,ulstar,g)
            rar2 = raref2(xi,hrstar,urstar,g)
            if wave_types[1] == 'contact':  # Subsonic
                h_out = (xi<=ws[0])*h_l + \
                    (xi>ws[0])*(xi<=ws[1])*rar1[0] + \
                    (xi>ws[1])*(xi<=ws[2])*hlstar +  \
                    (xi>ws[2])*(xi<=ws[3])*hrstar + \
                    (xi>ws[3])*(xi<=ws[4])*rar2[0] +  \
                    (xi>ws[4])*h_r
                hu_out = (xi<=ws[0])*hu_l + \
                    (xi>ws[0])*(xi<=ws[1])*rar1[1] + \
                    (xi>ws[1])*(xi<=ws[2])*hlstar*ulstar +  \
                    (xi>ws[2])*(xi<=ws[3])*hrstar*urstar +  \
                    (xi>ws[3])*(xi<=ws[4])*rar2[1] +  \
                    (xi>ws[4])*hu_r
            elif wave_types[0] == 'contact':  # Supersonic to right
                h_out = (xi<=ws[0])*h_l + \
                    (xi>ws[0])*(xi<=ws[1])*hlstar + \
                    (xi>ws[1])*(xi<=ws[2])*rar1[0] +  \
                    (xi>ws[2])*(xi<=ws[3])*hrstar + \
                    (xi>ws[3])*(xi<=ws[4])*rar2[0] +  \
                    (xi>ws[4])*h_r
                hu_out = (xi<=ws[0])*hu_l + \
                    (xi>ws[0])*(xi<=ws[1])*hlstar*ulstar + \
                    (xi>ws[1])*(xi<=ws[2])*rar1[1] +  \
                    (xi>ws[2])*(xi<=ws[3])*hrstar*urstar +  \
                    (xi>ws[3])*(xi<=ws[4])*rar2[1] +  \
                    (xi>ws[4])*hu_r
            elif wave_types[2] == 'contact':  # Supersonic to left
                h_out = (xi<=ws[0])*h_l + \
                    (xi>ws[0])*(xi<=ws[1])*rar1[0] + \
                    (xi>ws[1])*(xi<=ws[2])*hlstar +  \
                    (xi>ws[2])*(xi<=ws[3])*rar2[0] + \
                    (xi>ws[3])*(xi<=ws[4])*hrstar +  \
                    (xi>ws[4])*h_r
                hu_out = (xi<=ws[0])*hu_l + \
                    (xi>ws[0])*(xi<=ws[1])*rar1[1] + \
                    (xi>ws[1])*(xi<=ws[2])*hlstar*ulstar +  \
                    (xi>ws[2])*(xi<=ws[3])*rar2[1] +  \
                    (xi>ws[3])*(xi<=ws[4])*hrstar*urstar +  \
                    (xi>ws[4])*hu_r

            h_out[h_out>1e8] = np.nan
            hu_out[hu_out>1e8] = np.nan
            return h_out, hu_out
    return states, speeds, reval, wave_types
