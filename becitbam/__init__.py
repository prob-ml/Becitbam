import numpy as np
import scipy as sp
import scipy.optimize

def _check_arguments(s,a,mu=None,alpha=None):
    A=np.sum(a)
    assert (a>=0).all()
    assert 0 <= s <=A
    if mu is not None:
        assert 0 < mu < A
    if alpha is not None:
        assert 0 < alpha < 1

r'''

      _               _                            _ _
  ___| | __ _ ___ ___(_) ___   _ __ ___  ___ _   _| | |_ ___
 / __| |/ _` / __/ __| |/ __| | '__/ _ \/ __| | | | | __/ __|
| (__| | (_| \__ \__ \ | (__  | | |  __/\__ \ |_| | | |_\__ \
 \___|_|\__,_|___/___/_|\___| |_|  \___||___/\__,_|_|\__|___/

'''

def hoeffding_thm1(s,n,mu):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,1]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s).
    '''

    if s<=mu:
        return 0.0
    elif s>n:
        return -np.inf

    hoeffding_mu = mu/n
    hoeffding_t = s/n - mu/n

    assert hoeffding_t>=0
    assert s<=n
    assert mu>=0
    assert mu<=n

    T1 = (hoeffding_mu/(hoeffding_mu+hoeffding_t))**(hoeffding_mu+hoeffding_t)

    if 1-hoeffding_mu-hoeffding_t!=0:
        T2 = ((1-hoeffding_mu)/(1-hoeffding_mu-hoeffding_t))**(1-hoeffding_mu-hoeffding_t)
    else:
        T2 = 1.0

    return n*np.log(T1*T2)

def hoeffding_thm1_rescaled(s,n,mu,a):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s)
    '''

    '''
        P(S > s) = P(sum_i X_i > s)  subject to sum E[X_i] = mu
                 = P(sum_i X_i/a > s/a) subject to sum E[X_i/a] = mu/a
    '''

    a=float(a)

    if a<0:
        raise ValueError("a should be positive")

    return hoeffding_thm1(s/a,n,mu/a)


def hoeffding_thm2(s,mu,a):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s)
    '''

    _check_arguments(s,a,mu=mu)

    hoeffding_t = s - mu

    if s<=mu:
        return 0.0
    else:
        hoeffding_t = s-mu
        return -2*hoeffding_t*hoeffding_t / np.sum(a**2)



r'''
 _   _       _     _          _                            __  __
| |_(_) __ _| |__ | |_    ___| |__   ___ _ __ _ __   ___  / _|/ _|
| __| |/ _` | '_ \| __|  / __| '_ \ / _ \ '__| '_ \ / _ \| |_| |_
| |_| | (_| | | | | |_  | (__| | | |  __/ |  | | | | (_) |  _|  _|
 \__|_|\__, |_| |_|\__|  \___|_| |_|\___|_|  |_| |_|\___/|_| |_|
       |___/
'''

def uniqify(a):
    a=a[a!=0]
    return np.unique(a,return_counts=True)

def _lamstar(mu,asrt,w,t):
    b=(np.exp(asrt*t) -1) / asrt

    def meandiff(lam):
        return np.sum(_taustar(asrt,b,lam)*w) - mu

    LB=np.min((np.exp(asrt*t)-1)/(np.exp(asrt*t)-1+asrt))
    UB=np.max(b)

    return sp.optimize.bisect(meandiff,LB,UB)

def _taustar(asrt,b,lam):
    tau = (b - lam)/(b*lam)
    tau = np.clip(tau,0,asrt)
    return tau

def _g_weighted(s,mu,asrt,w,t,lam):
    assert t>=0 and lam>=0

    if t==0:
        return lam*mu - s*t
    if lam==0:
        return (np.sum(asrt*w)-s)*t

    b=(np.exp(asrt*t) -1) / asrt
    tau = (b - lam)/(b*lam)
    tau = np.clip(tau,0,asrt)

    rez=np.sum(np.log(1+b*tau)*w) + lam*(mu - np.sum(tau*w)) - s*t

    return rez

def sharp_chernoff(s,mu,a):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s)
    '''
    _check_arguments(s,a,mu=mu)
    asrt,w=uniqify(a)

     # keep a in check to maintain numerical stability
    mx=asrt.max()
    s=s/mx
    mu=mu/mx
    asrt=asrt/mx

    return _sharp_chernoff_weighted(s,mu,asrt,w).fun

def _sharp_chernoff_weighted(s,mu,asrt,w):
    # find reasonable initialization
    t_init=.5
    lam_init=_lamstar(mu,asrt,w,t_init)

    rez=sp.optimize.minimize(
        lambda tlam: _g_weighted(s,mu,asrt,w,tlam[0],tlam[1]),
        (t_init,lam_init),
        bounds=[(0,np.inf),(0,np.inf)],
        method='nelder-mead'
    )
    if not rez.success:
        raise Exception("Minimization failed!")

    return rez

r'''

     _           _
 ___| |__   __ _| |_ ___
/ __| '_ \ / _` | __/ __|
\__ \ | | | (_| | |_\__ \
|___/_| |_|\__,_|\__|___/

'''

def shat_from_sharp_chernoff(mu,a,alpha):
    '''
    Returns biggest s so that inf_t E_mu[exp(St-st)]<alpha
    '''

    asrt,w=uniqify(a)

     # keep a in check to maintain numerical stability
    mx=asrt.max()
    mu=mu/mx
    asrt=asrt/mx
    A=np.sum(asrt*w)

    def f(s):
        return _sharp_chernoff_weighted(s,mu,asrt,w).fun-np.log(alpha)

    if f(A)>0.0:
        return A

    UB=A-(A-mu)*.5
    while f(UB)>0:
        UB=A-(A-UB)*.5

    return sp.optimize.bisect(f,mu,UB)*mx

def shat_from_hoeffding_thm1(n,mu,alim,alpha):
    '''
    Returns biggest s so that inf_t E_mu[exp(St-st)]<alpha
    '''

    def f(s):
        return hoeffding_thm1_rescaled(s,n,mu,alim)-np.log(alpha)

    A=alim*n

    if f(A)>0.0:
        return A

    UB=A-(A-mu)*.5
    while f(UB)>0:
        UB=A-(A-UB)*.5

    return sp.optimize.bisect(f,mu,UB)


r'''
                  __ _     _
  ___ ___  _ __  / _(_) __| | ___ _ __   ___ ___
 / __/ _ \| '_ \| |_| |/ _` |/ _ \ '_ \ / __/ _ \
| (_| (_) | | | |  _| | (_| |  __/ | | | (_|  __/
 \___\___/|_| |_|_| |_|\__,_|\___|_| |_|\___\___|

 _       _                       _
(_)_ __ | |_ ___ _ ____   ____ _| |___
| | '_ \| __/ _ \ '__\ \ / / _` | / __|
| | | | | ||  __/ |   \ V / (_| | \__ \
|_|_| |_|\__\___|_|    \_/ \__,_|_|___/

'''


def confidence_from_hoeffding_thm2(s,a,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.
    '''

    _check_arguments(s,a,alpha=alpha)
    return s- np.sqrt(-.5*np.log(alpha)*np.sum(a**2))

def confidence_from_hoeffding_thm1(s,n,alim,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,alim]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.
    '''

    def f(mu):
        return hoeffding_thm1_rescaled(s,n,mu,alim)-np.log(alpha)

    LB=s*.5
    while f(LB)>0:
        LB=LB*.5

    return sp.optimize.bisect(f,LB,s)


def confidence_from_sharp_chernoff(s,a,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.

    Calculated by directly inverting sharp_chernoff.
    '''

    _check_arguments(s,a,alpha=alpha)

    asrt,w=uniqify(a)

     # keep a in check to maintain numerical stability
    mx=asrt.max()
    s=s/mx
    asrt=asrt/mx

    def f(mu):
        return _sharp_chernoff_weighted(s,mu,asrt,w).fun-np.log(alpha)

    LB=s*.5
    while f(LB)>0:
        LB=LB*.5

    return sp.optimize.bisect(f,LB,s)*mx



r'''
      _                                        __ _     _
  ___| | _____   _____ _ __    ___ ___  _ __  / _(_) __| | ___ _ __   ___ ___
 / __| |/ _ \ \ / / _ \ '__|  / __/ _ \| '_ \| |_| |/ _` |/ _ \ '_ \ / __/ _ \
| (__| |  __/\ V /  __/ |    | (_| (_) | | | |  _| | (_| |  __/ | | | (_|  __/
 \___|_|\___| \_/ \___|_|     \___\___/|_| |_|_| |_|\__,_|\___|_| |_|\___\___|

 _       _                       _
(_)_ __ | |_ ___ _ ____   ____ _| |___
| | '_ \| __/ _ \ '__\ \ / / _` | / __|
| | | | | ||  __/ |   \ V / (_| | \__ \
|_|_| |_|\__\___|_|    \_/ \__,_|_|___/

(using the dual problem)

'''

def _h_weighted(s,alpha,asrt,w,t,gam):

    last =np.log(alpha) + s*t

    if last<=0:
        return 0.0

    b=(np.exp(asrt*t) -1) / asrt
    nu = np.clip(np.log(gam*b),0,t*asrt)

    return np.sum(w*(np.exp(nu)-1)/b) + gam*(last-np.sum(w*nu))

def _nustar(s,asrt,last,b,gam,t):
    b=(np.exp(asrt*t) -1) / asrt
    return np.clip(np.log(gam*b),0,t*asrt)

def _gamstar(s,asrt,w,last,b,t):
    def meandiff(gam):
        return np.sum(_nustar(s,asrt,last,b,gam,t)*w) - last

    LB=np.min(1/b)
    UB=np.max(np.exp(t*asrt)/b)

    return sp.optimize.bisect(meandiff,LB,UB)

def confidence_from_inverted_sharp_chernoff(s,a,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.
    '''

    _check_arguments(s,a,alpha=alpha)

    ####
    # TODO: if set(a) has one elment, could use
    # confidence_from_hoeffding_thm1(s,n,alim,alpha) instead
    # (would be faster)
    ###

    # keep a in check to maintain numerical stability
    mx=a.max()
    s=s/mx
    a=a/mx
    asrt,w=uniqify(a)

    # if something goes weird, zero is always an acceptable lower bound
    if s<=1e-9: # if s is really small
        return 0.0

    # a good initial choice for t requires that log alpha + st >=0
    t_init = np.max([1,1e-9-np.log(alpha) / s])
    last =np.log(alpha) + s*t_init
    b=(np.exp(asrt*t_init) -1) / asrt
    gam_init = np.clip(_gamstar(s,asrt,w,last,b,t_init),1e-9,np.inf)

    rez=sp.optimize.minimize(
        lambda tgam: -_h_weighted(s,alpha,asrt,w,tgam[0],tgam[1]),
        (t_init,gam_init),
        bounds=[(0,np.inf),(1e-9,np.inf)],
        method='nelder-mead'
    )

    if not rez.success:
        raise Exception("Minimization failed!")

    return -rez.fun*mx

