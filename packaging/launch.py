"""Entry point for the packaged n0face.exe."""

import importlib.util
import sys
import types

# matplotlib is excluded from the bundle (see n0face.spec), but `import mediapipe` pulls in
# mediapipe/tasks/python/vision/drawing_utils.py, which does `import matplotlib.pyplot` at module
# level. It is only used by mediapipe's plot_landmarks() debug helper, which n0face never calls,
# so an empty stand-in module is enough to let mediapipe import.
if importlib.util.find_spec("matplotlib") is None:
    _mpl = types.ModuleType("matplotlib")
    _mpl.pyplot = types.ModuleType("matplotlib.pyplot")
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _mpl.pyplot

from n0face.app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
