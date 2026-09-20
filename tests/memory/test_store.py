"""JsonListStore: atomic JSON-list persistence behind both memory stores."""

import json
import os

from memory.store import JsonListStore


def test_read_returns_default_when_missing(tmp_path):
    store = JsonListStore(str(tmp_path / "missing.json"))
    assert store.read(default=["x"]) == ["x"]
    assert store.read() == []


def test_write_then_read_roundtrip(tmp_path):
    store = JsonListStore(str(tmp_path / "facts.json"))
    store.write(["one", {"two": 3}])
    assert store.read() == ["one", {"two": 3}]


def test_corrupt_json_returns_default(tmp_path):
    target = tmp_path / "facts.json"
    target.write_text("{ not json", encoding="utf-8")
    store = JsonListStore(str(target))
    assert store.read(default=[]) == []


def test_non_list_json_returns_default(tmp_path):
    target = tmp_path / "facts.json"
    target.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    store = JsonListStore(str(target))
    assert store.read(default=["fallback"]) == ["fallback"]


def test_write_is_atomic_on_disk(tmp_path):
    """Temporary file is rotated into place: no .tmp- litter, file is real."""
    target = tmp_path / "facts.json"
    store = JsonListStore(str(target))
    store.write(["a", "b"])
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith(".tmp-")]
    assert leftovers == []
    assert json.loads(target.read_text(encoding="utf-8")) == ["a", "b"]