"""Desktop app entry point: `n0face`.

The installed `n0face.exe` runs without a console, so any startup error is shown in a dialog
instead of disappearing silently.
"""

import sys
import traceback

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox


def make_icon() -> QIcon:
    from n0face.app.style import mono

    pm = QPixmap(64, 64)
    pm.fill(QColor("#07090b"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    accent = QColor("#28e6ff")
    p.setPen(QPen(accent, 2))
    r = QRectF(5, 5, 54, 54)
    arm = 14
    for (x, y), (dx, dy) in (
        ((r.left(), r.top()), (1, 1)), ((r.right(), r.top()), (-1, 1)),
        ((r.left(), r.bottom()), (1, -1)), ((r.right(), r.bottom()), (-1, -1)),
    ):
        p.drawLine(QPointF(x, y), QPointF(x + dx * arm, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + dy * arm))
    p.setFont(mono(22, bold=True))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "0")
    p.end()
    return QIcon(pm)


def _show_error(title: str, details: str) -> None:
    box = QMessageBox(QMessageBox.Icon.Critical, "n0face", title)
    box.setDetailedText(details)
    box.exec()


def _excepthook(exc_type, exc, tb) -> None:
    details = "".join(traceback.format_exception(exc_type, exc, tb))
    sys.__stderr__ and sys.__stderr__.write(details)
    if QApplication.instance() is not None:
        _show_error(f"Unexpected error: {exc}", details)


def build_window():
    """Create the detector, engine and main window. Raises on failure."""
    from n0face.app.window import MainWindow
    from n0face.detector import FaceDetector
    from n0face.netguard import lock_network
    from n0face.pipeline import Engine

    detector = FaceDetector()
    # The model is loaded and nothing else needs the network: turn it off for this process.
    lock_network()
    return MainWindow(Engine(detector)), detector


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("n0face")
    sys.excepthook = _excepthook
    try:
        app.setWindowIcon(make_icon())
        window, detector = build_window()
    except FileNotFoundError as e:
        _show_error(str(e), traceback.format_exc())
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - last resort so the GUI never fails silently
        _show_error(f"n0face failed to start: {e}", traceback.format_exc())
        sys.exit(1)

    window.show()
    code = app.exec()
    detector.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
