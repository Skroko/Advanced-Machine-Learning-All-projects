import os
import sys
import importlib
import pkgutil
import AMLsrc as src


def recursive_find_python_class(name, folder=None, current_module="AMLsrc.models", exit_if_not_found=True):
    # Set default search path to root modules
    if folder is None:
        folder = [os.path.join(src.__path__[0], *current_module.split(".")[1:])]

    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, name):
                tr = getattr(m, name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    name,
                    folder=[os.path.join(folder[0], modname)],
                    current_module=next_current_module,
                    exit_if_not_found=exit_if_not_found,
                )

            if tr is not None:
                break

    if tr is None and exit_if_not_found:
        sys.exit(f"Could not find module {name}")

    return tr