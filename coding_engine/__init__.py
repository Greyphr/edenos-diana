"""coding_engine package.

Deliberately scoped: this folder currently holds the isolated-execution
foundation only — execution_policy.py (per-task loop ceilings and the
escalate path) and sandbox_manager.py (default-deny Docker sandboxes).

The model-driven loop, workspace/git management, and the tool wiring are
follow-up work; nothing here is wired into Eden's conversation yet.
"""