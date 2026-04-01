import numpy as np
import cyipopt
from app.auxiliar_func.gibbsZero import gibbs_pad
from app.auxiliar_func.eos import fug


class Gibbs:
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

    def solve_gibbs(self, initial, T, P):
        initial = np.where(initial == 0, 1e-5, initial).astype(float)
        bnds = self.bnds_values(initial)

        gases = self.identify_phases('g')
        solids = self.identify_phases('s')

        df_pad = gibbs_pad(T, self.data)
        R = 8.314

        # Pre-compute element balance RHS once
        b_elem = self.A.T @ initial  # shape (total_species,)

        def objective(n):
            n_safe = np.maximum(n, 1e-300)
            n_gas_total = max(sum(n_safe[i] for i in gases), 1e-300)
            phi = fug(T=T, P=P, eq=self.equation, n=n_safe, components=self.data)

            G = 0.0
            for i in gases:
                phi_i = max(phi[i] if not np.isnan(phi[i]) else 1.0, 1e-300)
                y_i = max(n_safe[i] / n_gas_total, 1e-300)
                G += n_safe[i] * (df_pad[i] + R * T * (
                    np.log(phi_i) + np.log(y_i) + np.log(P)
                ))
            for i in solids:
                G += n_safe[i] * df_pad[i]
            return G + 1e-6

        def gradient(n):
            # Analytical gradient for Ideal Gas (avoids nc+1 fug() calls per iteration)
            if self.equation == 'Ideal Gas':
                n_safe = np.maximum(n, 1e-300)
                n_gas_total = max(sum(n_safe[i] for i in gases), 1e-300)
                grad = np.zeros(len(n))
                for i in gases:
                    y_i = max(n_safe[i] / n_gas_total, 1e-300)
                    grad[i] = df_pad[i] + R * T * (np.log(y_i) + np.log(P))
                for i in solids:
                    grad[i] = df_pad[i]
                return grad
            # Finite differences for non-ideal EOS
            eps = np.sqrt(np.finfo(float).eps)
            grad = np.zeros_like(n)
            f0 = objective(n)
            for k in range(len(n)):
                n_p = n.copy()
                n_p[k] += eps
                grad[k] = (objective(n_p) - f0) / eps
            return grad

        def constraints_eq(n):
            return self.A.T @ n - b_elem

        def constraints_jac(n):
            return self.A.T

        lb = [b[0] for b in bnds]
        ub = [b[1] for b in bnds]

        result = cyipopt.minimize_ipopt(
            fun=objective,
            x0=initial,
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
            return list(np.maximum(result.x, 0.0))
        else:
            raise Exception(f"Gibbs: solução não encontrada. {result.message}")
