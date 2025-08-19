"""Hook PyTorch's import to make `_add_docstr` idempotent."""
import builtins

_orig_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _orig_import(name, globals, locals, fromlist, level)
    if name == "torch._C" or (name.startswith("torch") and "_C" in fromlist):
        orig = getattr(module, "_add_docstr", None)
        if orig is not None and not getattr(orig, "_patched", False):
            def _safe_add_docstr(obj, doc):
                if getattr(obj, "__doc__", None):
                    return obj
                return orig(obj, doc)
            _safe_add_docstr._patched = True
            module._add_docstr = _safe_add_docstr
    return module

builtins.__import__ = _patched_import
