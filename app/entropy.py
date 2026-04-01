import numpy as np
import cyipopt
from app.auxiliar_func.entropyAux import int_cp_T, enthalpy_T


class Entropy:
    def __init__(self, data, species, components, inhibited_component, equation='Ideal Gas'):
        self.data = data
        self.species = species
        self.components = components
        self.total_components = len(components)
        self.total_species = len(species)
        self.A = np.array([[component[specie] for specie in species]
                           for component in data.values()])
        self.inhibited_component = inhibited_component
        self.equation = equation

    def identify_phases(self, phase_type):
        return [i for i, comp in enumerate(self.data)
                if self.data[comp].get('Phase') == phase_type]

    def bnds_values(self, initial):
        max_species = np.dot(initial, self.A)
        epsilon = 1e-5
        bnds = []

        aux_idx = None
        if self.inhibited_component and self.inhibited_component != '---':
            try:
                aux_idx = next(
                    idx for idx, (key, val) in enumerate(self.data.items())
                    if val['Component'] == self.inhibited_component
                )
            except StopIteration:
                pass

        for i in range(self.total_components):
            if aux_idx is not None and i == aux_idx:
                bnds.append((1e-8, epsilon))
            else:
                a_row = self.A[i]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(a_row != 0, max_species / a_row, np.inf)
                pos = ratios[ratios > 0]
                upper = float(np.min(pos)) if pos.size > 0 else epsilon
                bnds.append((1e-8, max(upper, epsilon)))

        return bnds

    def solve_entropy(self, initial, Tinit, P):
        initial = np.asarray(initial, dtype=float)
        bnds = self.bnds_values(initial)
        nc = self.total_components
        T0 = 298.15
        R = 8.314

        gases = self.identify_phases('g')

        # Pre-compute initial enthalpy (constant RHS for enthalpy balance)
        H_init_list = enthalpy_T(Tinit, self.data)
        H_init_sum = sum(initial[j] * H_init_list[j] for j in range(nc))

        # Pre-compute element balance RHS
        b_elem = self.A.T @ initial  # shape (total_species,)

        # x = [n[0..nc-1], T]
        x0 = np.append(initial, Tinit)
        lb = [b[0] for b in bnds] + [100.0]
        ub = [b[1] for b in bnds] + [10000.0]

        def objective(x):
            n = x[:nc]
            T = float(x[nc])
            T = max(T, 1.0)  # guard against non-positive T

            int_cp_vals, deltaH_list, deltaG_list = int_cp_T(T, self.data)
            n_gas_total = max(sum(n[gi] for gi in gases), 1e-300)

            S = 0.0
            for i in range(len(gases)):
                gi = gases[i]
                ni = max(n[gi], 1e-300)
                yi = max(ni / n_gas_total, 1e-300)
                s_i = (
                    (deltaH_list[gi] - deltaG_list[gi]) / T0
                    - R * np.log(P)
                    - R * np.log(yi)
                    + int_cp_vals[gi]
                )
                S += s_i * ni
            return -(S + 1e-6)  # minimise negative entropy

        def gradient(x):
            # Analytical gradient: d(-S)/dn_k = -s_k  (for gas k);  d(-S)/dT = -sum_i n_i*Cp_i/T
            n = x[:nc]
            T = float(x[nc])
            T_g = max(T, 1.0)
            int_cp_vals, deltaH_list, deltaG_list = int_cp_T(T_g, self.data)
            n_safe = np.maximum(n, 1e-300)
            n_gas_total = max(sum(n_safe[gi] for gi in gases), 1e-300)
            comp_list = list(self.data.values())

            grad = np.zeros(len(x))
            dS_dT = 0.0
            for gi in gases:
                ni = max(n_safe[gi], 1e-300)
                yi = max(ni / n_gas_total, 1e-300)
                s_i = (
                    (deltaH_list[gi] - deltaG_list[gi]) / T0
                    - R * np.log(P)
                    - R * np.log(yi)
                    + int_cp_vals[gi]
                )
                grad[gi] = -s_i
                c = comp_list[gi]
                cp_over_T = R * (c.get('a', 0) / T_g + c.get('b', 0)
                                 + c.get('c', 0) * T_g + c.get('d', 0) / T_g ** 3)
                dS_dT += ni * cp_over_T
            grad[nc] = -dS_dT
            return grad

        def constraints_eq(x):
            n = x[:nc]
            T = float(x[nc])
            # Element balance
            elem = self.A.T @ n - b_elem  # shape (total_species,)
            # Enthalpy balance
            H_final_list = enthalpy_T(T, self.data)
            H_final_sum = sum(n[j] * H_final_list[j] for j in range(nc))
            enthalpy = np.array([H_final_sum - H_init_sum])
            return np.append(elem, enthalpy)

        def constraints_jac(x):
            # Analytical Jacobian — avoids nc+2 constraints_eq() calls per iteration.
            # Layout: rows = [elem_balance (total_species), enthalpy_balance (1)]
            #         cols = [n_0..n_{nc-1}, T]
            n = x[:nc]
            T = float(x[nc])
            T_j = max(T, 1.0)
            n_species = self.total_species

            jac = np.zeros((n_species + 1, nc + 1))

            # Element balance: d(A.T @ n - b)/dn = A.T,  d/dT = 0
            jac[:n_species, :nc] = self.A.T

            # Enthalpy balance: d(sum_j n_j*H_j(T) - H_init)/dn_k = H_k(T)
            H_T = enthalpy_T(T_j, self.data)
            jac[n_species, :nc] = H_T

            # d(sum_j n_j*H_j(T))/dT = sum_j n_j * Cp_j(T)
            comp_list = list(self.data.values())
            dH_dT = 0.0
            for j in range(nc):
                c = comp_list[j]
                Cp_j = R * (c.get('a', 0) + c.get('b', 0) * T_j
                            + c.get('c', 0) * T_j ** 2 + c.get('d', 0) / T_j ** 2)
                dH_dT += n[j] * Cp_j
            jac[n_species, nc] = dH_dT

            return jac

        result = cyipopt.minimize_ipopt(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=list(zip(lb, ub)),
            constraints=[{
                'type': 'eq',
                'fun': constraints_eq,
                'jac': constraints_jac,
            }],
            options={
                'max_iter': 2000,
                'tol': 1e-8,
                'print_level': 0,
            }
        )

        if result.success or np.isfinite(result.fun):
            n_result = list(np.maximum(result.x[:nc], 0.0))
            T_result = float(result.x[nc])
            return n_result, T_result
        else:
            raise Exception(f"Entropy: solução não encontrada. {result.message}")
