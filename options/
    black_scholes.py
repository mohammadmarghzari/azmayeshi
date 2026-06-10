from scipy.stats import norm
import numpy as np

def d1(
    S,
    K,
    T,
    r,
    sigma
):

    return (
        np.log(S / K)
        +
        (
            r +
            sigma**2 / 2
        ) * T
    ) / (
        sigma * np.sqrt(T)
    )


def d2(
    S,
    K,
    T,
    r,
    sigma
):

    return (
        d1(
            S,
            K,
            T,
            r,
            sigma
        )
        -
        sigma
        *
        np.sqrt(T)
    )


def call_price(
    S,
    K,
    T,
    r,
    sigma
):

    D1 = d1(
        S,
        K,
        T,
        r,
        sigma
    )

    D2 = d2(
        S,
        K,
        T,
        r,
        sigma
    )

    return (
        S * norm.cdf(D1)
        -
        K
        *
        np.exp(-r * T)
        *
        norm.cdf(D2)
    )