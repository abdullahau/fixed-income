import numpy as np
from numpy import array, log, exp, where, vectorize

def duration(cf, rate, cf_freq=1, comp_freq=1, cf_t=None,
             immediate_start=False, modified=False):
    r"""Duration of arbitrary sequence of cash flows

    Parameters
    ----------
    cf : sequence of floats
         array of cash flows
    rate : float or sequence of floats
         discount rate
    cf_freq : float or sequence of floats, optional
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats, optional
         compounding frequency (for example, 2 for semi-annual)
    cf_t : float or sequence of floats or None, optional
         The timing of cash flows.
         If None, equally spaced cash flows are assumed
    immediate_start : bool or sequence of bool, optional
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    modified : bool or sequence of bool, optional
         If True, modified duration is returned

    Returns
    -------
    float or array of floats
       The duration of the cash flows

    Examples
    --------
    >>> duration(cf=[100, 50, 75, 25], rate=10e-2).item()
    1.9980073065426769

    >>> duration(cf=[100, 50, 75, 25], rate=10e-2,
    ...          immediate_start=[True, False])
    array([0.99800731, 1.99800731])

    """

    def one_duration(rate, cf_freq, comp_freq, immediate_start):
        if cf_t is None:
            start = 0 if immediate_start else 1/cf_freq
            stop = start + len(cf) / cf_freq
            cf_ta = np.arange(start=start, step=1/cf_freq, stop=stop)
        else:
            cf_ta = cf_t
        cc_rate = equiv_rate(rate, from_freq=comp_freq, to_freq=np.inf)
        df = exp(-cc_rate * cf_ta)
        return np.dot(cf*df, cf_ta) / np.dot(cf, df)

    D = vectorize(one_duration)(
        rate=rate, cf_freq=cf_freq, comp_freq=comp_freq,
        immediate_start=immediate_start)
    D /= where(modified, 1 + rate/comp_freq, 1)
    return D[()]
  
  
def npv(cf, rate, cf_freq=1, comp_freq=1, cf_t=None,
        immediate_start=False):
    r"""NPV of a sequence of cash flows

    Parameters
    ----------
    cf : float or sequence of floats
         array of cash flows
    rate : float or sequence of floats
         discount rate
    cf_freq : float or sequence of floats, optional
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats, optional
         compounding frequency (for example, 2 for semi-annual)
    cf_t : float or sequence of floats or None, optional
         The timing of cash flows.
         If None, equally spaced cash flows are assumed
    immediate_start : bool or sequence of bool, optional
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.

    Returns
    -------
    float or array of floats
       The net present value of the cash flows

    Examples
    --------
    >>> npv(cf=[-100, 150, -50, 75], rate=5e-2).item()
    59.327132213429586

    >>> npv(cf=[-100, 150, -50, 75], rate=5e-2, comp_freq=[1, 2])
    array([59.32713221, 59.15230661])

    >>> npv(cf=[-100, 150, -50, 75], rate=5e-2,
    ...     immediate_start=[False, True])
    array([59.32713221, 62.29348882])

    >>> npv(cf=[-100, 150, -50, 75], cf_t=[0, 2, 5, 7], rate=[5e-2, 8e-2])
    array([50.17921321, 38.33344284])

    """

    def one_npv(rate, cf_freq, comp_freq, immediate_start):
        if cf_t is None:
            start = 0 if immediate_start else 1/cf_freq
            stop = start + len(cf) / cf_freq
            cf_ta = np.arange(start=start, step=1/cf_freq, stop=stop)
        else:
            cf_ta = array(cf_t)
        cc_rate = equiv_rate(rate, from_freq=comp_freq, to_freq=np.inf)
        df = exp(-cc_rate * cf_ta)
        return np.dot(cf, df)

    cf = array(cf)
    return vectorize(one_npv)(
        rate=rate, cf_freq=cf_freq, comp_freq=comp_freq,
        immediate_start=immediate_start)[()]


def equiv_rate(rate, from_freq=1, to_freq=1):
    r"""Convert interest rate from one compounding frequency to another

    Parameters
    ----------
    rate : float or sequence of floats
           discount rate in decimal
    from_freq : float or sequence of floats
                compounding frequency of input rate
    to_freq : float or sequence of floats
              compounding frequency of output rate

    Returns
    -------
    float or array of floats
       The discount rate for the desired compounding frequency

    Examples
    --------
    >>> equiv_rate(
    ...    rate=10e-2, from_freq=1, to_freq=[1, 2, 12, 365, np.inf])
    array([0.1       , 0.0976177 , 0.09568969, 0.09532262, 0.09531018])

    >>> equiv_rate(
    ...    rate=10e-2, from_freq=[1, 2, 12, 365, np.inf], to_freq=1)
    array([0.1       , 0.1025    , 0.10471307, 0.10515578, 0.10517092])

    """
    rate, from_freq, to_freq = array(rate), array(from_freq), array(to_freq)
    old_settings = np.seterr(invalid='ignore')
    cc_rate = where(from_freq == np.inf, rate,
                    log(1 + np.divide(rate, from_freq)) * from_freq)
    res = where(from_freq == to_freq,
                rate,
                where(to_freq == np.inf,
                      cc_rate,
                      (exp(np.divide(cc_rate, to_freq)) - 1) * to_freq))[()]
    np.seterr(**old_settings)
    return res