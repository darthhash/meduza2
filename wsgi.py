# wsgi.py
from typing import Any
from dotenv import load_dotenv; load_dotenv()

application: Any = None

# 1) пробуем фабрику из пакета app/
try:
    from app import create_app
    application = create_app()
except Exception as e:
    print("[warn] create_app() not usable:", e)
    application = None

def _has_root(a) -> bool:
    try:
        return any(r.rule == "/" for r in a.url_map.iter_rules())
    except Exception:
        return False

# 2) если фабрика не дала корень — НАСИЛЬНО грузим КОРНЕВОЙ файл app.py
if application is None or not _has_root(application):
    try:
        import importlib.util, pathlib, types
        proj = pathlib.Path(__file__).resolve().parent
        app_py = proj / "app.py"
        spec = importlib.util.spec_from_file_location("legacy_app_module", str(app_py))
        mod = importlib.util.module_from_spec(spec)  # type: types.ModuleType
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        legacy = getattr(mod, "app", None)
        if legacy is None:
            raise RuntimeError("app.py does not define global variable 'app'")
        print("[info] using legacy app.py:app")
        application = legacy
    except Exception as e:
        print("[warn] legacy app.py load failed:", e)

if application is None:
    raise RuntimeError("WSGI application not found (neither app.create_app() nor app.py:app)")
