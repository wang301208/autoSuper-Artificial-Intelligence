import importlib.util
import sys
import types
from pathlib import Path


# Create a package-like structure for ``backend`` so modules can be imported
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = []
sys.modules["backend"] = backend_pkg

# Load the world model module directly from its file path
wm_path = Path(__file__).resolve().parents[1] / "__init__.py"
spec = importlib.util.spec_from_file_location("backend.world_model", wm_path)
module = importlib.util.module_from_spec(spec)
sys.modules["backend.world_model"] = module
spec.loader.exec_module(module)  # type: ignore[arg-type]

WorldModel = module.WorldModel


def test_store_and_retrieve_visual_data():
    wm = WorldModel()
    image = [[0, 1], [1, 0]]
    features = [0.1, 0.2, 0.3]

    wm.add_visual_observation("agent1", image=image, features=features)

    retrieved = wm.get_visual_observation("agent1")
    assert retrieved["image"] == image
    assert retrieved["features"] == features

    state = wm.get_state()
    assert "agent1" in state["vision"]
    assert state["vision"]["agent1"]["features"] == features
