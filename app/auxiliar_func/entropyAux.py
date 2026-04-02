import numpy as np


def int_cp_T(T, components):
    """
    Returns the integral of Cp/T from T0 to T for each component (numpy, no Pyomo).

    Integral = R*(a*ln(T/T0) + b*(T-T0) + c/2*(T^2-T0^2) + d/2*(1/T0^2 - 1/T^2))

    Returns:
        results : list of float — integral values for all components
        DeltaH  : list of float — standard enthalpy of formation [J/mol]
        DeltaG  : list of float — standard Gibbs free energy of formation [J/mol]
    """
    R = 8.314462
    T0 = 298.15

    results, DeltaH, DeltaG = [], [], []
    for component in components.values():
        deltaH = component.get('\u2206Hf298', 0)
        deltaG = component.get('\u2206Gf298', 0)
        a = component.get('a', 0)
        b = component.get('b', 0)
        c = component.get('c', 0)
        d = component.get('d', 0)

        integral_value = R * (
            a * np.log(T / T0)
            + b * (T - T0)
            + (c / 2.0) * (T**2 - T0**2)
            + (-d / 2.0) * (1.0 / T**2 - 1.0 / T0**2)
        )
        results.append(integral_value)
        DeltaH.append(deltaH)
        DeltaG.append(deltaG)

    return results, DeltaH, DeltaG


def enthalpy_T(T, components):
    """
    Returns the total enthalpy H_i(T) = deltaHf298 + integral_T0^T Cp dT for each component.

    H(T) = deltaH + R*(a*(T-T0) + b/2*(T^2-T0^2) + c/3*(T^3-T0^3) - d*(1/T - 1/T0))
    """
    R = 8.314462
    T0 = 298.15

    results = []
    for component in components.values():
        deltaH = component.get('\u2206Hf298', 0)
        a = component.get('a', 0)
        b = component.get('b', 0)
        c = component.get('c', 0)
        d = component.get('d', 0)

        integral_value = R * (
            a * (T - T0)
            + (b / 2.0) * (T**2 - T0**2)
            + (c / 3.0) * (T**3 - T0**3)
            - d * (1.0 / T - 1.0 / T0)
        )
        results.append(deltaH + integral_value)

    return results
