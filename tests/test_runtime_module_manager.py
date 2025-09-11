import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.capability import register_module
from backend.capability.runtime_loader import RuntimeModuleManager


def test_runtime_module_manager_load_and_unload():
    # Register a dummy module
    name = "dummy_runtime_test"
    register_module(name, lambda: {"name": name})

    mgr = RuntimeModuleManager()
    mod = mgr.load(name)
    assert mod == {"name": name}
    assert name in mgr.loaded_modules()

    mgr.unload(name)
    assert name not in mgr.loaded_modules()


def test_runtime_module_manager_update(tmp_path):
    name = "dummy_runtime_test2"
    register_module(name, lambda: {"name": name})
    mgr = RuntimeModuleManager()

    mgr.update([name])
    assert name in mgr.loaded_modules()

    mgr.update([])
    assert name not in mgr.loaded_modules()
