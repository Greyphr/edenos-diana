"""Long-term memory: keyword recall (the original regression), hard caps,
and forget()."""

import os

import pytest

import memory.long_term as long_term
import memory.short_term as short_term
from memory.long_term import LongTermMemory
from memory.tools import make_memory_handlers


@pytest.fixture
def memory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        long_term, "_data_dir",
        lambda owner: os.path.join(str(tmp_path), owner),
    )
    return LongTermMemory("alex")


@pytest.fixture
def both_memories(monkeypatch, tmp_path):
    monkeypatch.setattr(
        long_term, "_data_dir",
        lambda owner: os.path.join(str(tmp_path), owner),
    )
    monkeypatch.setattr(
        short_term, "_data_dir",
        lambda owner: os.path.join(str(tmp_path), owner),
    )


# --- Keyword-overlap recall (the original search regression) ---------------

def test_differently_worded_query_still_finds_fact(memory):
    memory.add_fact("prefers dark roast coffee in the morning")
    hits = memory.search("what kind of coffee does she like")
    assert len(hits) == 1
    assert hits[0]["text"] == "prefers dark roast coffee in the morning"


def test_shared_keyword_without_verbatim_match_finds_fact(memory):
    memory.add_fact("loves jazz music and late records")
    hits = memory.search("jazz")
    assert len(hits) == 1
    assert hits[0]["text"] == "loves jazz music and late records"


def test_stopword_only_query_matches_nothing(memory):
    memory.add_fact("loves jazz music")
    assert memory.search("in the morning is it") == []


def test_no_keyword_overlap_returns_nothing(memory):
    memory.add_fact("loves jazz music")
    assert memory.search("gardening") == []


def test_search_orders_by_keyword_match_count(memory):
    memory.add_fact("likes coffee")
    memory.add_fact("prefers coffee and dark roast")
    memory.add_fact("nothing relevant here")
    hits = memory.search("coffee dark roast")
    assert hits[0]["text"] == "prefers coffee and dark roast"
    assert hits[1]["text"] == "likes coffee"


# --- Caps -------------------------------------------------------------------

def test_max_facts_drops_oldest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        long_term, "_data_dir", lambda owner: os.path.join(str(tmp_path), owner)
    )
    monkeypatch.setattr(long_term, "MAX_FACTS", 3)
    mem = LongTermMemory("alex")
    for i in range(5):
        mem.add_fact(f"fact #{i}")
    facts = mem.all_facts()
    assert len(facts) == 3
    texts = {f["text"] for f in facts}
    assert "fact #0" not in texts  # oldest dropped
    assert "fact #4" in texts      # newest kept


async def test_over_long_fact_is_rejected_at_the_tools_layer(both_memories):
    remember_spec = make_memory_handlers("alex")[0]
    result = await remember_spec.handler(fact="x" * (long_term.MAX_FACT_LENGTH + 1))
    assert result["status"] == "error"
    assert "too long" in result["reason"]
    assert LongTermMemory("alex").all_facts() == []


# --- forget ------------------------------------------------------------------

def test_forget_removes_matching_facts_and_reports_count(memory):
    memory.add_fact("loves coffee in the morning")
    memory.add_fact("hates cilantro")
    memory.add_fact("wants coffee after dinner")
    removed = memory.remove_facts_containing("coffee")
    assert removed == 2
    remaining = {f["text"] for f in memory.all_facts()}
    assert remaining == {"hates cilantro"}


def test_forget_with_no_match_is_a_noop(memory):
    memory.add_fact("loves coffee")
    assert memory.remove_facts_containing("jazz") == 0
    assert len(memory.all_facts()) == 1


def test_all_facts_most_recent_first(memory):
    memory.add_fact("older")
    memory.add_fact("newer")
    assert [f["text"] for f in memory.all_facts()] == ["newer", "older"]


# --- invalid owner names ------------------------------------------------------

def test_invalid_owner_name_raises():
    with pytest.raises(ValueError, match="Invalid owner name"):
        long_term._data_dir("../escape")