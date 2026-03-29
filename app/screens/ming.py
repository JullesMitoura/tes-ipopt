from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFrame, QPushButton, QLabel,
    QHBoxLayout, QTableWidget, QHeaderView, QFileDialog, QTableWidgetItem,
    QScrollArea, QGridLayout, QLineEdit, QComboBox, QMessageBox, QProgressBar
)
from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtCore import Qt
from app.auxiliar_func.read_data import ReadData
from app.auxiliar_func.run_gibbs import RunGibbs
from app.screens.ming_aux.section03 import Section3
from app.screens.ming_aux.section04 import Section4
from app.screens.sim_worker import SimWorker
from app.find_path import resource_path

_FILE_FILTER = "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"

_LINE_EDIT_STYLE = """
    QLineEdit {
        color: black; background-color: white;
        border: 1px solid #555; border-radius: 5px; padding: 2px;
    }
"""

_COMBO_STYLE = """
    QComboBox {
        border: 1px solid black; border-radius: 5px; color: black;
        background-color: white; padding-left: 5px; font-size: 12px;
    }
    QComboBox QAbstractItemView {
        color: black; background-color: white;
        selection-background-color: lightgray; selection-color: black;
    }
    QComboBox::drop-down { border-radius: 5px; }
    QComboBox::item { color: black; }
"""

_MSG_STYLE = """
    QLabel { color: black; }
    QPushButton {
        color: black; background-color: #E1E1E1;
        border: 1px solid #ADADAD; padding: 5px 15px; border-radius: 3px;
    }
    QPushButton:hover { background-color: #F0F0F0; }
    QPushButton:pressed { background-color: #C0C0C0; }
"""

_PROGRESS_BAR_STYLE = """
    QProgressBar {
        border: 1px solid #bbb; border-radius: 5px;
        background: #f0f0f0; text-align: center;
        color: black; font-size: 10px;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #42a5f5, stop:1 #1565c0);
        border-radius: 4px;
    }
"""

_PROGRESS_BAR_INACTIVE = """
    QProgressBar { border: none; background: transparent; }
    QProgressBar::chunk { background: transparent; }
"""


class MinG(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.table = None
        layout.addWidget(self._section1())
        layout.addWidget(self._separator())
        layout.addWidget(self._section2())
        layout.addWidget(self._separator())

        self.section3 = None
        self.section3_container = QFrame()
        self.section3_container.setFixedHeight(190)
        inner3 = QVBoxLayout()
        inner3.setContentsMargins(0, 0, 0, 0)
        self.section3_container.setLayout(inner3)
        layout.addWidget(self.section3_container)
        layout.addWidget(self._separator())

        self.section4 = None
        self.section4_container = QFrame()
        self.section4_container.setFixedHeight(200)
        inner4 = QVBoxLayout()
        inner4.setContentsMargins(0, 0, 0, 0)
        self.section4_container.setLayout(inner4)
        layout.addWidget(self.section4_container)
        layout.addStretch()

        self.setLayout(layout)

        # Data state
        self.file_path = ""
        self.document = None
        self.dataframe = None
        self.data = None
        self.species = None
        self.initial = None
        self.components = []
        self.results = None
        self._worker = None

        # Simulation parameters
        self.tmin = self.tmax = None
        self.pmin = self.pmax = None
        self.reference_componente = None
        self.reference_componente_min = self.reference_componente_max = None
        self.n_temperature = self.n_pressure = self.n_component_values = 0
        self.state_equation = None
        self.inhibit_component = None

    # ------------------------------------------------------------------ helpers
    def _separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    def _parse_float(self, text, label, errors):
        try:
            return float(text.replace(',', '.'))
        except (ValueError, AttributeError):
            errors.append(f"{label} (valor inválido: '{text}')")
            return None

    def _parse_int(self, text, label, errors):
        try:
            return int(text)
        except (ValueError, AttributeError):
            errors.append(f"{label} (valor inválido: '{text}')")
            return None

    # ------------------------------------------------------------------ validation
    def collect_input_values(self):
        if self.data is None:
            QMessageBox.warning(self, "Sem dados", "Abra um arquivo antes de executar a simulação.")
            return False

        errors = []

        def require(text, label):
            if not text.strip():
                errors.append(label)
                return None
            return text.strip()

        tmax_raw = require(self.max_temp_input.text(), "Max. Temperature")
        tmin_raw = require(self.min_temp_input.text(), "Min. Temperature")
        pmax_raw = require(self.max_pressure_input.text(), "Max. Pressure")
        pmin_raw = require(self.min_pressure_input.text(), "Min. Pressure")
        nt_raw   = require(self.n_values_t_input.text(), "N. Values T")
        np_raw   = require(self.n_values_p_input.text(), "N. Values P")

        if errors:
            QMessageBox.warning(self, "Campos obrigatórios",
                                "Preencha os seguintes campos:\n" +
                                "\n".join(f" • {f}" for f in errors))
            return False

        self.tmax = self._parse_float(tmax_raw, "Max. Temperature", errors)
        self.tmin = self._parse_float(tmin_raw, "Min. Temperature", errors)
        self.pmax = self._parse_float(pmax_raw, "Max. Pressure", errors)
        self.pmin = self._parse_float(pmin_raw, "Min. Pressure", errors)
        self.n_temperature = self._parse_int(nt_raw, "N. Values T", errors)
        self.n_pressure    = self._parse_int(np_raw, "N. Values P", errors)

        self.reference_componente = self.component_combobox.currentText()
        self.inhibit_component    = self.inhibit_component_combox.currentText()
        self.state_equation       = self.state_equation_combobox.currentText()

        if self.reference_componente and self.reference_componente != '---':
            nmin_raw = require(self.min_value_input.text(), "Min. Value (componente referência)")
            nmax_raw = require(self.max_value_input.text(), "Max. Value (componente referência)")
            nn_raw   = require(self.n_values_n_input.text(), "N. Values n (componente referência)")
            if not errors:
                self.reference_componente_min = self._parse_float(nmin_raw, "Min. Value", errors)
                self.reference_componente_max = self._parse_float(nmax_raw, "Max. Value", errors)
                self.n_component_values       = self._parse_int(nn_raw, "N. Values n", errors)
        else:
            self.reference_componente_min = 0.0
            self.reference_componente_max = 0.0
            self.n_component_values       = 1

        if errors:
            QMessageBox.warning(self, "Erro de entrada",
                                "\n".join(f" • {e}" for e in errors))
            return False
        return True

    # ------------------------------------------------------------------ run
    def run_gibbs(self):
        if not self.collect_input_values():
            return

        runner = RunGibbs(
            data=self.data, species=self.species, initial=self.initial,
            components=self.components,
            Tmin=self.tmin, Tmax=self.tmax,
            Pmin=self.pmin, Pmax=self.pmax,
            nT=self.n_temperature, nP=self.n_pressure,
            reference_componente=self.reference_componente,
            reference_componente_min=self.reference_componente_min,
            reference_componente_max=self.reference_componente_max,
            n_reference_componente=self.n_component_values,
            inhibit_component=self.inhibit_component,
            state_equation=self.state_equation,
        )

        total = self.n_temperature * self.n_pressure * self.n_component_values
        self._start_simulation(total)

        self._worker = SimWorker(runner.run_gibbs)
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_gibbs_done)
        self._worker.error_occurred.connect(self._on_sim_error)
        self._worker.start()

    def _start_simulation(self, total):
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        self.btn_run.setEnabled(False)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(_PROGRESS_BAR_STYLE)

    def _on_progress(self, current, total):
        self.progress_bar.setValue(current)

    def _cleanup(self):
        QApplication.restoreOverrideCursor()
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(_PROGRESS_BAR_INACTIVE)

    def _on_gibbs_done(self, results):
        self._cleanup()
        self.results = results

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Simulação concluída")
        msg.setText("A simulação de Gibbs foi concluída com sucesso!")
        msg.setStyleSheet(_MSG_STYLE)
        msg.exec()

        self._show_section3(self.results, self.components, self.reference_componente)
        self._refresh_section4()

    def _on_sim_error(self, error_msg):
        self._cleanup()
        QMessageBox.critical(self, "Erro na simulação", error_msg)

    def _show_section3(self, results, components, ref):
        if self.section3:
            self.section3_container.layout().removeWidget(self.section3)
            self.section3.deleteLater()
        self.section3 = Section3(results, components, ref)
        self.section3_container.layout().addWidget(self.section3)
        self.section3.setVisible(True)

    def _refresh_section4(self):
        if self.section4 is not None:
            for i in reversed(range(self.section4_container.layout().count())):
                w = self.section4_container.layout().itemAt(i).widget()
                if w:
                    w.deleteLater()
        self.section4 = Section4(self.results, self.components)
        self.section4_container.layout().addWidget(self.section4)
        self.section4.setVisible(True)

    # ------------------------------------------------------------------ file
    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir arquivo", "", _FILE_FILTER)
        if not file_name:
            return
        try:
            self.document   = ReadData(file_name)
            self.dataframe  = self.document.dataframe
            self.data       = self.document.data
            self.species    = self.document.species
            self.initial    = self.document.initial
            self.components = self.document.components
            self.file_path  = file_name
            self._populate_table()
            self._populate_combos()
            self.btn_run.setEnabled(True)
        except (FileNotFoundError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Erro ao abrir arquivo", str(e))

    def _populate_table(self):
        self.table.setRowCount(len(self.dataframe))
        for row in range(len(self.dataframe)):
            self.table.setItem(row, 0, QTableWidgetItem(str(self.dataframe['Component'].iloc[row])))
            self.table.setItem(row, 1, QTableWidgetItem(str(self.dataframe['initial'].iloc[row])))

    def _populate_combos(self):
        for cb in (self.component_combobox, self.inhibit_component_combox):
            cb.clear()
            cb.addItem("---")
            cb.addItems(self.components)
            cb.setEnabled(True)
        self.state_equation_combobox.setEnabled(True)

    # ------------------------------------------------------------------ UI build
    def _section1(self):
        section = QFrame()
        section.setFixedHeight(120)
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)

        col1 = QVBoxLayout()
        col1.setSpacing(6)

        btn_open = QPushButton("Open File")
        btn_open.setFixedSize(130, 30)
        btn_open.setStyleSheet("""
            QPushButton { border-radius:10px; background-color:white; color:black;
                          padding:5px; border:1px solid black; }
            QPushButton:hover { background-color:#e0e0e0; }
            QPushButton:pressed { background-color:#0056b3; color:white; }
        """)
        btn_open.clicked.connect(self.open_file_dialog)

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setFixedSize(130, 30)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("""
            QPushButton { border-radius:10px; background-color:#28a745; color:white;
                          padding:5px; border:1px solid #218838; }
            QPushButton:hover { background-color:#218838; }
            QPushButton:pressed { background-color:#1a6e2e; color:white; }
            QPushButton:disabled { background-color:#a5d6a7; border:1px solid #a5d6a7; color:#e8f5e9; }
        """)
        self.btn_run.clicked.connect(self.run_gibbs)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(130)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(_PROGRESS_BAR_INACTIVE)

        col1.addWidget(btn_open,          alignment=Qt.AlignmentFlag.AlignLeft)
        col1.addWidget(self.btn_run,      alignment=Qt.AlignmentFlag.AlignLeft)
        col1.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignLeft)

        col2 = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Component", "Initial (mols)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setFixedHeight(20)
        self.table.setRowCount(0)
        self.table.setStyleSheet("""
            QTableWidget { border:1px solid black; background-color:#f5f5f5;
                           gridline-color:black; font-size:10px; }
            QTableWidget::item { border:1px solid black; padding:5px; color:#333; }
            QTableWidget::item:selected { background-color:#d4e157; color:black; }
            QHeaderView::section { background-color:#3f51b5; color:white;
                                   border:1px solid black; padding:5px;
                                   font-weight:bold; font-size:10px; }
            QTableCornerButton::section { background-color:#3f51b5; }
        """)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedSize(400, 110)
        scroll.setWidget(self.table)
        col2.addWidget(scroll, alignment=Qt.AlignmentFlag.AlignCenter)

        col3 = QVBoxLayout()
        lbl = QLabel()
        px = QPixmap(resource_path("app/imgs/minG.png"))
        lbl.setPixmap(px.scaled(150, 130, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation))
        col3.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(col1)
        layout.addLayout(col2)
        layout.addLayout(col3)
        section.setLayout(layout)
        return section

    def _section2(self):
        section = QFrame()
        section.setFixedHeight(210)
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 4, 8, 4)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(8)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(3, 2)

        def lbl(text):
            w = QLabel(text)
            w.setStyleSheet("color: black;")
            w.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return w

        def inp():
            w = QLineEdit()
            w.setFixedHeight(26)
            w.setStyleSheet(_LINE_EDIT_STYLE)
            return w

        grid.addWidget(lbl("Max. Temperature:"), 0, 0)
        self.max_temp_input = inp(); grid.addWidget(self.max_temp_input, 0, 1)

        grid.addWidget(lbl("Min. Temperature:"), 1, 0)
        self.min_temp_input = inp(); grid.addWidget(self.min_temp_input, 1, 1)

        grid.addWidget(lbl("Max. Pressure:"), 2, 0)
        self.max_pressure_input = inp(); grid.addWidget(self.max_pressure_input, 2, 1)

        grid.addWidget(lbl("Min. Pressure:"), 3, 0)
        self.min_pressure_input = inp(); grid.addWidget(self.min_pressure_input, 3, 1)

        grid.addWidget(lbl("Select a Component:"), 4, 0)
        self.component_combobox = QComboBox()
        self.component_combobox.setEnabled(False)
        self.component_combobox.setStyleSheet(_COMBO_STYLE)
        self.component_combobox.currentTextChanged.connect(self._on_component_changed)
        grid.addWidget(self.component_combobox, 4, 1)

        range_widget = QWidget()
        range_layout = QHBoxLayout(range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(6)
        lbl_min = QLabel("Min:")
        lbl_min.setStyleSheet("color: black;")
        self.min_value_input = inp()
        self.min_value_input.setEnabled(False)
        lbl_max = QLabel("Max:")
        lbl_max.setStyleSheet("color: black;")
        self.max_value_input = inp()
        self.max_value_input.setEnabled(False)
        range_layout.addWidget(lbl_min)
        range_layout.addWidget(self.min_value_input)
        range_layout.addWidget(lbl_max)
        range_layout.addWidget(self.max_value_input)
        grid.addWidget(range_widget, 5, 1)

        grid.addWidget(lbl("N. Values T:"), 0, 2)
        self.n_values_t_input = inp(); grid.addWidget(self.n_values_t_input, 0, 3)

        grid.addWidget(lbl("N. Values P:"), 1, 2)
        self.n_values_p_input = inp(); grid.addWidget(self.n_values_p_input, 1, 3)

        grid.addWidget(lbl("N. Values n:"), 2, 2)
        self.n_values_n_input = inp()
        self.n_values_n_input.setEnabled(False)
        grid.addWidget(self.n_values_n_input, 2, 3)

        grid.addWidget(lbl("State Equation:"), 3, 2)
        self.state_equation_combobox = QComboBox()
        self.state_equation_combobox.addItems(
            ['Ideal Gas', 'Peng-Robinson', 'Soave-Redlich-Kwong', 'Redlich-Kwong', 'Virial'])
        self.state_equation_combobox.setStyleSheet(_COMBO_STYLE)
        grid.addWidget(self.state_equation_combobox, 3, 3)

        grid.addWidget(lbl("Inhibit Component:"), 4, 2)
        self.inhibit_component_combox = QComboBox()
        self.inhibit_component_combox.setEnabled(False)
        self.inhibit_component_combox.setStyleSheet(_COMBO_STYLE)
        grid.addWidget(self.inhibit_component_combox, 4, 3)

        outer.addLayout(grid)
        section.setLayout(outer)
        return section

    def _on_component_changed(self, text):
        enabled = text != '---' and text != ''
        self.max_value_input.setEnabled(enabled)
        self.min_value_input.setEnabled(enabled)
        self.n_values_n_input.setEnabled(enabled)
