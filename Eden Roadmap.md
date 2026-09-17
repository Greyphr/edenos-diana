# Eden Rebuild — Roadmap & Architecture Scaffold

Fresh codebase. One block at a time, each working before the next starts.
Nothing here is cut from the original vision — it's sequenced.

## Block status

| # | Block | Status |
|---|---|---|
| 1 | Conversation engine (voice in/out, Gemini Live) | ✅ Done |
| 2 | Speaker identity (enrollment + recognition) | ⬅ Next |
| 3 | Memory (persistent, session-start load + mid-session tools) | Planned |
| 4 | Tool system + policy engine | Planned |
| 5 | Homelab control (Proxmox, TrueNAS, Docker) | Planned |
| 6 | Smart home (Home Assistant) | Planned |
| 7 | Personal integrations (calendar, email, Spotify, messaging, Obsidian, NotebookLM) | Planned |
| 8 | Coding engine (autonomous + pair-programmer modes) | Planned |
| 9 | Owner console (React/Three.js dashboard) | Planned |
| — | Local model migration (cloud → local realtime + local LLM) | Ongoing, cross-cutting |

Roles/permission tiers, secrets vault, and multi-terminal deployment aren't
their own numbered blocks — they get built into blocks 2, 4, and the console
respectively, since they don't stand alone.

---

## Why this order

Blocks 2–3 (identity, memory) come immediately after the conversation engine
because almost nothing else makes sense without knowing *who's* talking and
*what Eden already knows* — a tool system with no identity to check
permissions against, or a coding engine with no memory of what you asked for
five minutes ago, isn't buildable yet.

Block 4 (tool system + policy engine) comes before any actual
integration (homelab, smart home, Spotify, coding) because every one of
those is "Eden calls a tool that does something in the real world" — and per
the non-negotiable from the original design, the model proposes, it never
executes directly. Building the policy/confirmation/audit layer once, before
the first real tool exists, means blocks 5–8 are just "write a tool," not
"write a tool and also reinvent authorization."

Blocks 5–8 are ordered by blast radius and dependency, not difficulty:
homelab and smart home are foundational (Eden's own home), personal
integrations are lower-risk and high-value, coding engine last because it
needs its own sandboxed VM/container and is the most complex tool by far.

The console (9) comes after there's real data to show — building it against
mock data first per your earlier call, but it doesn't get wired to live
panels until the things it displays (Guardian/roles, Nodes/homelab,
Security, Coding Engine status) actually exist.

Local model migration isn't a block with a start and end — it's something
that becomes possible incrementally as your homelab GPU box comes online,
and gets slotted in wherever the cloud/local swap is easiest first (likely
the conversation engine, since block 1 was already built behind a thin
provider interface for exactly this reason).

---

## Architecture scaffold

Target end-state layout. Folders marked `(future)` don't exist yet —
don't scaffold them early; each gets created when its block starts.

```text
eden/
├── main.py                      # entry point
├── .env                         # secrets — local only, never committed
├── requirements.txt
│
├── conversation/                # BLOCK 1 — done
│   ├── realtime_session.py      # Gemini Live wrapper (connect/send/receive/interrupt)
│   └── audio_io.py              # mic capture + speaker playback
│
├── identity/                    # BLOCK 2 — next
│   ├── enrollment.py            # capture + store voiceprint
│   ├── recognition.py           # embedding extraction + match against enrolled profiles
│   └── voiceprints/             # persisted profile data (gitignored)
│
├── memory/                      # BLOCK 3
│   ├── store.py                 # atomic read/write, per-owner scoping
│   ├── short_term.py
│   ├── long_term.py
│   └── data/                    # persisted memory (gitignored)
│
├── tools/                       # BLOCK 4
│   ├── policy_engine.py         # propose → check → confirm-if-required → execute → verify
│   ├── registry.py              # tool contracts Eden can call
│   ├── vault.py                 # secrets, redacted before anything reaches cloud models
│   └── roles.py                 # Discord-style custom roles, default-deny
│
├── integrations/                # BLOCKS 5–7, one subpackage per integration
│   ├── homelab/       (future)  # Proxmox, TrueNAS, Docker/Jellyfin
│   ├── home_assistant/ (future)
│   ├── calendar/       (future)
│   ├── email/          (future)
│   ├── spotify/        (future)
│   ├── messaging/      (future) # SMS / WhatsApp
│   ├── obsidian/       (future)
│   └── notebooklm/     (future)
│
├── coding_engine/      (future) # BLOCK 8 — runs in its own VM/container
│   ├── autonomous.py            # execute-and-report mode
│   └── pair_programmer.py       # approval-gated mode
│
├── console/             (future)# BLOCK 9 — separate app, not this package
│   ├── client/                  # React + Three.js
│   └── server/                  # Node.js/Express
│
└── terminals/           (future)# lightweight capture clients for other rooms/devices
```

### How the pieces connect (once blocks 1–4 exist)

```text
🎤 Mic ──┬── conversation/ (Gemini Live session, streaming voice)
         │
         └── identity/ (VAD → embedding → match, resolves ~1-2s after
             conversation starts responding)
                    │
                    ↓
             memory/ (session-start context load,
             mid-session remember/recall via
             function-call tools)
                    │
                    ↓
             tools/policy_engine (every real-world action
             routes through here — propose, check role,
             confirm if required, execute, verify)
                    │
                    ↓
             integrations/* (each one is just a tool the
             policy engine can call, once it exists)
```

Identity and memory feed context *into* the conversation session (system
instructions, tool results); the policy engine sits *between* Eden deciding
to do something and it actually happening. Nothing in `integrations/`
should ever be called directly by `conversation/` — always through
`tools/`.

---

## Cross-cutting decisions already made

- Fresh codebase — `eden-os-main` is reference only, not a base to extend
- Realtime speech-to-speech (Gemini Live), not turn-based STT→LLM→TTS
- opencode as the coding assistant, one block-prompt at a time
- Provider-swappable by design (block 1's `realtime_session.py` interface
  exists specifically so the eventual cloud→local swap doesn't mean a
  rewrite)
- No commercial ambitions — single-owner, personal use