from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from algorithm_library import AlgorithmLibrary
import pytest


def test_add_and_get_algorithm():
    library = AlgorithmLibrary()

    def sample(x):
        return x + 1

    library.add("increment", sample)
    retrieved = library.get("increment")
    assert retrieved(1) == 2


def test_list_algorithms():
    library = AlgorithmLibrary()
    library.add("a", lambda x: x)
    library.add("b", lambda x: x * 2)
    assert set(library.list()) == {"a", "b"}


def test_remove_algorithm():
    library = AlgorithmLibrary()
    library.add("a", lambda x: x)
    library.remove("a")
    assert library.list() == []


def test_add_duplicate_algorithm_raises():
    library = AlgorithmLibrary()
    library.add("a", lambda x: x)
    with pytest.raises(ValueError):
        library.add("a", lambda x: x)
