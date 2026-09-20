"""Short-term memory store: ordered append, recency pruning, and the cap."""

import json
import os
import time

import pytest

import memory.short_term as short_term
from memory.short_term import PRUNE_AFTER_SECONDS, ShortTermMemory


@pytest.fixture
def memory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        short_term, "_data_dir",
        lambda owner: os.path.join(str(tmp_path), owner),
    )
    return ShortTermMemory("alex")


def test_all_facts_most_recent_first(memory):
    memory.add_fact("first")
    memory.add_fact("second")
    assert [f["text"] for f in memory.all_facts()] == ["second", "first"]


def test_max_facts_drops_oldest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        short_term, "_data_dir", lambda owner: os.path.join(str(tmp_path), owner)
    )
    monkeypatch.setattr(short_term, "MAX_FACTS", 3)
    mem = ShortTermMemory("alex")
    for i in range(5):
        mem.add_fact(f"fact #{i}")
    facts = mem.all_facts()
    assert len(facts) == 3
    texts = {f["text"] for f in facts}
    assert "fact #0" not in texts
    assert "fact #4" in texts


def test_added_facts_persist_to_disk(memory, tmp_path):
    memory.add_fact("kept across reads")
    target = tmp_path / "alex" / "short_term.json"
    assert target.exists()
    entries = json.loads(target.read_text(encoding="utf-8"))
    assert entries[-1]["text"] == "kept across reads"


def test_entries_older_than_prune_window_are_dropped(memory, tmp_path):
    target = tmp_path / "alex" / "short_term.json"
    old_ts = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - PRUNE_AFTER_SECONDS - 60)
    )
    fresh_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    target.write_text(
        json.dumps(
            [
                {"text": "ancient", "recorded_at": old_ts},
                {"text": "current", "recorded_at": fresh_ts},
            ]
        ),
        encoding="utf-8",
    )
    assert [f["text"] for f in memory.all_facts()] == ["current"]


def test_entries_without_timestamp_are_dropped(memory, tmp_path):
    target = tmp_path / "alex" / "short_term.json"
    target.write_text(
        json.dumps([{"text": "no timestamp"}]),
        encoding="utf-8",
    )
    assert memory.all_facts() == []