import numpy as np
from scipy.special import comb
from scipy.interpolate import interp1d

class ADSRBezier():
    def __init__(self, cfg, sampling_rate):
        self.cfg = cfg
        self.sampling_rate = sampling_rate
    
    def get_envelope(self, sustain_msec, gain = 1., verbose = False):
        ''' TODO bezier is based on interpolation between curves, so the X values are not uniformly sampled
            need to re-sample to a uniform time grid
            params:
                sustain_msec - the sustain part of the ADSR envelope, in milliseconds
        '''
        adsr_cfg = self.cfg.copy() #--- TODO find a better way of input args..
        adsr_cfg['s_t_msec'] = sustain_msec
        sr = self.sampling_rate
        
        #--- segment durations in samples
        nA, nD, nS, nR = [np.round(adsr_cfg[key] * sr / 1000).astype(int) for key in ['a_t_msec', 'd_t_msec', 's_t_msec', 'r_t_msec']]
        n0, n1, n2, n3, n4 = np.cumsum([0] + [nA, nD, nS, nR])
        #--- segment amplitudes
        e0, e1, e4 = 0., gain, 0.
        e2, e3 = [adsr_cfg[key] * gain for key in ['d_lvl', 's_lvl']]
    
        #--- control points params - these are made symmetric around the control points, so the bezier curve is "nice"
        #c1, c2, c3, c4, c5, c6 = .8, .5, .4, .1, 1.2, 0.
        # DEBUG: 
        c1, c2, c3, c4, c5, c6 = .8, .5, .4, .1, 0.8, 0.
        
        #=== Attack
        p_attack = [[n0, e0],  [c1 * n1, e0],      [c2 * n1, e1],          [n1, e1]]
        Ax, Ay =  bezier_curve(p_attack, nA)    
        
        #=== Decay
        de1 = (e1 - e2)
        p_decay = [[n1, e1], [n1 + c2 * nD, e1], [n2 - c2 * nD, e2 + c3 * de1], [n2, e2]]
        Dx, Dy =  bezier_curve(p_decay, nD)    
        
        #--- Sustain
        de2 = (e2 - e3)
        dx, dy = c3 * nS, c4 * de2
        slope3 = dy / dx #--- slope = dy / dx
        p_sustain = [[n2, e2], [n2 + c2 * nD, e2 - c3 * de1], [n3 - dx, e3 + slope3 * dx], [n3, e3]]
        Sx, Sy =  bezier_curve(p_sustain, nS)    
        
        #--- Release (make sure the sustain point 3 and release point 2 are on the same line (with slope given by "slope3")
        de3 = (e3 - e4)
        dx = min(c3 * nS, c5 * nR)
        p_release = [[n3, e3], [n3 + dx, e3 - slope3 * dx], [n4 - c5 * nR, e4 + c6 * de3], [n4, e4]]
        Rx, Ry =  bezier_curve(p_release, nR)  

        if verbose:
            print(f'attack:  {p_attack}')
            print(f'decay:   {p_decay}')
            print(f'sustain: {p_sustain}')
            print(f'release: {p_release}')

        #--- prepare an interpolation function
        x = np.r_[Ax, Dx, Sx, Rx]
        y = np.r_[Ay, Dy, Sy, Ry]
        ind = np.argsort(x)
        x = x[ind]
        y = y[ind]
        x, ind = np.unique(x, return_index = True)
        y = y[ind]
        f_intrp = interp1d(x, y, kind = 'cubic', assume_sorted = True)

        #--- interpolate to uniform grid, first sample index is 0, last is n4
        x_env = np.arange(n0, n4 + 1)
        env = f_intrp(x_env)
        
        return env, [np.r_[p] for p in [p_attack, p_decay, p_sustain, p_release]]
        
def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final

#--- functions copied from: https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals