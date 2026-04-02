from PyQt6.QtCore import QThread, pyqtSignal


class SimWorker(QThread):
    """
    Runs a simulation runner function in a background thread so the UI
    remains responsive while the solver is working.

    Signals
    -------
    progress(current, total) : emitted after each solved point
    result_ready(DataFrame)  : emitted with the full results DataFrame on success
    error_occurred(str)      : emitted with the error message on failure
    """

    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, runner_fn, parent=None):
        """
        Parameters
        ----------
        runner_fn : callable
            Either RunGibbs.run_gibbs or RunEntropy.run_entropy.
            Must accept a keyword argument ``progress_callback``.
        """
        super().__init__(parent)
        self._runner_fn = runner_fn

    def run(self):
        try:
            df = self._runner_fn(progress_callback=self._emit_progress)
            self.result_ready.emit(df)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _emit_progress(self, current, total):
        self.progress.emit(current, total)
