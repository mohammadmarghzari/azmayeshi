from scipy.stats import norm
import numpy as np

def greeks(
    S,
    K,
    T,
    r,
    sigma
):

    d1 = (
        np.log(S / K)
        +
        (
            r +
            sigma**2 / 2
        )
        * T
    ) / (
        sigma
        *
        np.sqrt(T)
    )

    d2 = (
        d1
        -
        sigma
        *
        np.sqrt(T)
    )

    delta = norm.cdf(d1)

    gamma = (
        norm.pdf(d1)
        /
        (
            S
            *
            sigma
            *
            np.sqrt(T)
        )
    )

    theta = (
        -
        (
            S
            *
            norm.pdf(d1)
            *
            sigma
        )
        /
        (
            2
            *
            np.sqrt(T)
        )
    )

    vega = (
        S
        *
        norm.pdf(d1)
        *
        np.sqrt(T)
    )

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega
    }