import warnings
import pandas as pd
import numpy as np
from app.entropy import Entropy


class RunEntropy:
    def __init__(self, data, species, initial, components, Tmin, Tmax, Pmin, Pmax, nT, nP,
                 reference_componente=None, reference_componente_min=None,
                 reference_componente_max=None, n_reference_componente=None,
                 inhibit_component=None, state_equation='Ideal Gas'):
        self.data = data
        self.species = species
        self.initial = np.array(initial, dtype=float)
        self.components = components
        self.inhibit_component = inhibit_component
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.nT = nT
        self.nP = nP
        self.reference_componente = reference_componente
        self.reference_componente_min = reference_componente_min
        self.reference_componente_max = reference_componente_max
        self.n_reference_componente = n_reference_componente
        self.state_equation = state_equation

    def _format_ranges(self):
        T_vals = np.linspace(self.Tmin, self.Tmax, self.nT)
        P_vals = np.linspace(self.Pmin, self.Pmax, self.nP)

        n_vals = None
        ref_idx = None
        if self.reference_componente and self.reference_componente != '---':
            idx = np.where(self.components == self.reference_componente)[0]
            if idx.size > 0:
                ref_idx = int(idx[0])
                n_vals = np.linspace(
                    self.reference_componente_min,
                    self.reference_componente_max,
                    self.n_reference_componente,
                )
            else:
                warnings.warn(f"Componente de referência '{self.reference_componente}' não encontrado.")

        return T_vals, P_vals, n_vals, ref_idx

    def run_entropy(self, progress_callback=None):
        solver = Entropy(
            self.data, self.species, self.components,
            self.inhibit_component, self.state_equation,
        )
        T_vals, P_vals, n_vals, ref_idx = self._format_ranges()
        nan_row = [float('nan')] * len(self.components)
        result_list = []

        n_iter = len(n_vals) if n_vals is not None else 1
        total = len(T_vals) * len(P_vals) * n_iter
        count = 0

        for T in T_vals:
            for P in P_vals:
                if ref_idx is not None and n_vals is not None:
                    for n in n_vals:
                        initial_copy = self.initial.copy()
                        initial_copy[ref_idx] = n
                        try:
                            result, Teq = solver.solve_entropy(initial_copy, T, P)
                        except Exception as e:
                            warnings.warn(f"Entropy falhou em T={T:.1f}, P={P:.3f}, n={n:.4f}: {e}")
                            result, Teq = nan_row, float('nan')
                        row = {comp: round(val, 3) for comp, val in zip(self.components, result)}
                        row[self.components[ref_idx] + ' Initial'] = n
                        row['Equilibrium Temperature (K)'] = Teq
                        row.update({'Initial Temperature': T, 'Pressure': P})
                        result_list.append(row)
                        count += 1
                        if progress_callback:
                            progress_callback(count, total)
                else:
                    try:
                        result, Teq = solver.solve_entropy(self.initial.copy(), T, P)
                    except Exception as e:
                        warnings.warn(f"Entropy falhou em T={T:.1f}, P={P:.3f}: {e}")
                        result, Teq = nan_row, float('nan')
                    row = {comp: round(val, 3) for comp, val in zip(self.components, result)}
                    row.update({'Initial Temperature': T, 'Pressure': P})
                    result_list.append(row)
                    count += 1
                    if progress_callback:
                        progress_callback(count, total)

        return pd.DataFrame(result_list).round(3)
