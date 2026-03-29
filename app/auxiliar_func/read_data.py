import pandas as pd
import os


class ReadData:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        self.path = path
        self.ext = os.path.splitext(path)[1].lower()
        self.dataframe = None
        self.data, self.species, self.initial, self.components = self._load()

    def _load(self):
        try:
            if self.ext in ('.xls', '.xlsx'):
                df = pd.read_excel(self.path, sheet_name='Informations')
            elif self.ext == '.txt':
                # Auto-detect separator (tab, space, semicolon, comma)
                df = pd.read_csv(self.path, sep=None, engine='python')
            else:
                df = pd.read_csv(self.path)
        except Exception as e:
            raise ValueError(f"Não foi possível ler o arquivo: {e}")

        # Coerce all numeric-looking columns to float for safety
        for col in df.columns:
            if col not in ('Component', 'Phase'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        self.dataframe = df

        for col in ('Component', 'initial', 'C'):
            if col not in df.columns:
                raise KeyError(f"Coluna obrigatória '{col}' não encontrada.")

        initial = df['initial'].values.astype(float)
        components = df['Component'].values
        species = df.columns[df.columns.get_loc('C'):]
        data_dict = {row['Component']: row.to_dict() for _, row in df.iterrows()}

        return data_dict, species, initial, components
