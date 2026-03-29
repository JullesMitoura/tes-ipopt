import numpy as np


def _compute_kij(gas_comp_names, gas_components):
    """
    Chueh-Prausnitz combining rule for binary interaction parameters.

        kij = 1 - 8*sqrt(Vc_i * Vc_j) / (Vc_i^(1/3) + Vc_j^(1/3))^3

    Vc units cancel in the ratio, so cm^3/mol can be used directly.
    kij[i,i] = 0 by definition.

    Returns:
        kij : np.ndarray of shape (n, n)
    """
    n = len(gas_comp_names)
    Vc = np.array([gas_components[name].get('Vc', 1.0) for name in gas_comp_names],
                  dtype=float)
    kij = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and Vc[i] > 0 and Vc[j] > 0:
                num = 8.0 * np.sqrt(Vc[i] * Vc[j])
                denom = (Vc[i] ** (1.0 / 3.0) + Vc[j] ** (1.0 / 3.0)) ** 3
                kij[i, j] = 1.0 - num / denom
    return kij


def fug(T, P, eq, n, components):
    """
    Fugacity coefficient phi_i for each component.

    Parameters
    ----------
    T          : float         — temperature [K]
    P          : float         — pressure [bar]
    eq         : str           — EOS name: 'Ideal Gas', 'Virial', 'Peng-Robinson',
                                 'Soave-Redlich-Kwong', 'Redlich-Kwong'
    n          : array-like    — molar amounts for all components
    components : dict          — component data from ReadData

    Returns
    -------
    list of float — phi_i for each component (1.0 for solids, 1.0 for ideal gas)
    """
    R = 8.314462
    P_pa = P * 1e5

    comp_names = list(components.keys())
    n_arr = np.asarray(n, dtype=float)
    total_n = n_arr.sum()

    if total_n < 1e-300:
        return [np.nan] * len(comp_names)

    resultados = [0.0] * len(comp_names)

    gas_names = [name for name, d in components.items() if d.get('Phase', 'g').lower() != 's']
    solid_names = [name for name, d in components.items() if d.get('Phase', 'g').lower() == 's']

    for name in solid_names:
        resultados[comp_names.index(name)] = 1.0

    if not gas_names:
        return resultados

    if eq == 'Ideal Gas':
        for name in gas_names:
            resultados[comp_names.index(name)] = 1.0
        return resultados

    gas_idx = [comp_names.index(name) for name in gas_names]
    n_gas = n_arr[gas_idx]
    n_total_gas = n_gas.sum()
    if n_total_gas < 1e-300:
        for name in gas_names:
            resultados[comp_names.index(name)] = 1.0
        return resultados

    y = n_gas / n_total_gas
    kij = _compute_kij(gas_names, {name: components[name] for name in gas_names})

    # --- Virial (truncated at 2nd coefficient) ---
    if eq == 'Virial':
        Tc = np.array([components[name]['Tc'] for name in gas_names])
        omega = np.array([components[name]['omega'] for name in gas_names])
        Zc = np.array([components[name]['Zc'] for name in gas_names])
        Vc_cm3 = np.array([components[name]['Vc'] for name in gas_names])
        Vc = Vc_cm3 / 1e6

        num_g = len(gas_names)
        B_matrix = np.zeros((num_g, num_g))
        for i in range(num_g):
            for j in range(num_g):
                Tcij = np.sqrt(Tc[i] * Tc[j]) * (1.0 - kij[i, j])
                wij = (omega[i] + omega[j]) / 2.0
                Vcij = ((Vc[i] ** (1.0/3) + Vc[j] ** (1.0/3)) / 2.0) ** 3
                Zcij = (Zc[i] + Zc[j]) / 2.0
                Pcij_pa = Zcij * R * Tcij / Vcij
                Tr_ij = T / Tcij
                B0 = 0.083 - 0.422 / (Tr_ij ** 1.6)
                B1 = 0.139 - 0.172 / (Tr_ij ** 4.2)
                B_matrix[i, j] = (R * Tcij / Pcij_pa) * (B0 + wij * B1)

        B_mix = y @ B_matrix @ y
        sum_yB = B_matrix @ y
        ln_phi = (2.0 * sum_yB - B_mix) * P_pa / (R * T)
        phi = np.exp(ln_phi)
        for i, name in enumerate(gas_names):
            resultados[comp_names.index(name)] = phi[i]
        return resultados

    # --- Cubic EOS (PR / SRK / RK) ---
    eos_params = {
        'Peng-Robinson': {
            'Omega_a': 0.45724, 'Omega_b': 0.07780,
            'm_func': lambda w: 0.37464 + 1.54226 * w - 0.26992 * w**2,
            'alpha_func': lambda Tr, m: (1.0 + m * (1.0 - np.sqrt(Tr))) ** 2,
            'Z_coeffs': lambda A, B: [1, B - 1, A - 2*B - 3*B**2, -A*B + B**2 + B**3],
            'ln_phi_term': lambda Z, B: (1.0 / (2.0 * np.sqrt(2.0))) *
                                        np.log((Z + (1.0 + np.sqrt(2.0)) * B) /
                                               (Z + (1.0 - np.sqrt(2.0)) * B)),
        },
        'Soave-Redlich-Kwong': {
            'Omega_a': 0.42748, 'Omega_b': 0.08664,
            'm_func': lambda w: 0.480 + 1.574 * w - 0.176 * w**2,
            'alpha_func': lambda Tr, m: (1.0 + m * (1.0 - np.sqrt(Tr))) ** 2,
            'Z_coeffs': lambda A, B: [1, -1, A - B - B**2, -A*B],
            'ln_phi_term': lambda Z, B: np.log(1.0 + B / Z),
        },
        'Redlich-Kwong': {
            'Omega_a': 0.42748, 'Omega_b': 0.08664,
            'm_func': lambda w: np.zeros_like(w),
            'alpha_func': lambda Tr, m: 1.0 / np.sqrt(Tr),
            'Z_coeffs': lambda A, B: [1, -1, A - B - B**2, -A*B],
            'ln_phi_term': lambda Z, B: np.log(1.0 + B / Z),
        },
    }

    if eq not in eos_params:
        raise ValueError(f"Equação de estado '{eq}' não suportada.")

    params = eos_params[eq]
    Tc = np.array([components[name]['Tc'] for name in gas_names])
    Pc = np.array([components[name]['Pc'] * 1e5 for name in gas_names])
    omega = np.array([components[name]['omega'] for name in gas_names])

    m = params['m_func'](omega)
    Tr = T / Tc
    alpha = np.array([params['alpha_func'](Tr[i], m[i]) for i in range(len(gas_names))])

    a_i = params['Omega_a'] * (R**2 * Tc**2 / Pc) * alpha
    b_i = params['Omega_b'] * (R * Tc / Pc)

    a_ij = (1.0 - kij) * np.sqrt(np.outer(a_i, a_i))
    a_mix = np.sum(np.outer(y, y) * a_ij)
    b_mix = np.dot(y, b_i)

    A = a_mix * P_pa / (R**2 * T**2)
    B = b_mix * P_pa / (R * T)

    Z_roots = np.roots(params['Z_coeffs'](A, B))
    real_roots = Z_roots[np.isreal(Z_roots)].real
    pos_roots = real_roots[real_roots > B]

    if len(pos_roots) == 0:
        for name in gas_names:
            resultados[comp_names.index(name)] = np.nan
        return resultados

    Z = pos_roots.max()

    term1 = b_i / b_mix * (Z - 1.0)
    term2 = -np.log(Z - B)
    sum_y_aij = a_ij @ y
    term3_dyn = (2.0 * sum_y_aij / a_mix) - (b_i / b_mix)
    term3_log = params['ln_phi_term'](Z, B)

    ln_phi = term1 + term2 - (A / B) * term3_dyn * term3_log
    phi = np.exp(ln_phi)

    for i, name in enumerate(gas_names):
        resultados[comp_names.index(name)] = float(phi[i])

    return resultados
