# Pharma Voice Assistant — Real-Time Multimodal Pharmaceutical Digital Twin

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Demo Tests](https://img.shields.io/badge/demo%20tests-15%2F15%20passing-brightgreen.svg)]()

A voice-enabled GxP knowledge assistant — Gourav Pandey's "Digital Twin" — that accepts a
typed or spoken question, retrieves grounded answers from a curated pharmaceutical SOP
corpus using BM25 lexical search, and replies with synthesized speech, optionally
lip-synced through a streaming HeyGen / LiveAvatar video.

---

## Demo edition — what is bundled

This release ships with a focused, hand-written knowledge base of **five fundamental GxP
SOPs** (~50 pages of professional content) so the system can be demonstrated end-to-end
with no external corpus or paid LLM calls beyond TTS.

| ID | SOP | What it covers |
|---|---|---|
| SOP-QA-001 | Good Documentation Practice (GDP) | ALCOA+ principles, paper corrections, electronic records, 21 CFR Part 11, audit trails |
| SOP-QA-002 | Batch Record Review | Two-stage review, CPP/IPC checks, yield calculation and reconciliation, deviation triggers, release prerequisites |
| SOP-QA-003 | Batch Release / Disposition | QP authority, decision inputs, conditional release per Annex 16, rejection, regulatory hold |
| SOP-QA-004 | Change Control | Class 1 / 2 / 3 classification, impact assessment, regulatory filing triggers, effectiveness checks, CAPA linkage |
| SOP-QA-005 | Deviation and CAPA Management | Level 1 / 2 / 3 classification, root cause analysis, CAPA hierarchy of effectiveness, trending, escalation |

Each SOP is written against current EU GMP, ICH Q9 (R1) / Q10, 21 CFR 211 / Part 11,
PIC/S PI 041, and MHRA guidance. The same architecture scales unchanged to a 200K-document
production brain.

---

## Architecture

```
+================================================================+
|              Real-Time Multimodal Pipeline                       |
+================================================================+
|                                                                  |
|  Browser  (frontend/index.html or avatar_frontend.html)          |
|   - text input + Web Speech API mic                              |
|   - HeyGen LiveAvatar SDK (streaming video)                      |
|   - HTML5 audio for ElevenLabs TTS                               |
|                                                                  |
|             | /ask, /speak, /avatar/*                            |
|             v                                                    |
|                                                                  |
|  Flask API  (src/api.py / start_server_with_avatar.py)           |
|   - BM25 retrieval                                               |
|   - ElevenLabs TTS  (audio/mpeg streamed back)                   |
|   - HeyGen LiveAvatar token mint + video fallback                |
|                                                                  |
|             |                                                    |
|             v                                                    |
|                                                                  |
|  BM25 search index  (brain_testing_v1.pkl)                       |
|   - 148 chunks, 1663 unique terms, 154 KB                        |
|   - k1=1.5, b=0.75                                               |
|                                                                  |
|             |  built from                                        |
|             v                                                    |
|                                                                  |
|  sops_for_testing/  (5 markdown SOPs, ~50 pages)                 |
|                                                                  |
+==================================================================+
```

See `ARCHITECTURE.md` for stage-by-stage latency budgets and fallback paths.

---

## Test prompts

Fifteen prompts cover every SOP and every major topic. They render as clickable buttons
in the UI and are also exposed at `GET /test_prompts` for programmatic access.

| # | Question | Expected SOP |
|---|---|---|
| 1 | What are the ALCOA+ principles for data integrity? | GDP |
| 2 | How should I correct an error in a paper batch record? | GDP |
| 3 | What are the requirements for electronic records and audit trails per 21 CFR Part 11? | GDP |
| 4 | How should I review a batch record before release? | Batch Record Review |
| 5 | What are the critical process parameters for tablet manufacturing? | Batch Record Review |
| 6 | How do I calculate batch yield and what are the acceptance limits? | Batch Record Review |
| 7 | Who has the authority to make a batch disposition decision and what does the QP review? | Batch Release |
| 8 | When can a batch be conditionally released? | Batch Release |
| 9 | What triggers a batch rejection and how is regulatory hold managed? | Batch Release |
| 10 | What is the change control process for a critical change? | Change Control |
| 11 | How do I classify a change as Class 1, 2, or 3? | Change Control |
| 12 | What is an effectiveness check for a change control? | Change Control |
| 13 | What triggers a Level 1 critical deviation? | Deviation / CAPA |
| 14 | When is a CAPA required and how should it be designed? | Deviation / CAPA |
| 15 | How do I perform root cause analysis using 5 Whys or fishbone? | Deviation / CAPA |

---

## Test results (latest run)

The bundled `run_tests.py` sends each prompt to `/ask`, verifies the answer is non-empty
and references the expected SOP, and exercises `/speak` once.

| Metric | Value |
|---|---|
| Prompts tested | 15 |
| Passed | **15** |
| Failed | 0 |
| Pass rate | **100%** |
| /ask median latency | ~2.07 s |
| /speak round-trip | ~2.6 s for 44.8 KB MP3 |

Full per-prompt output (answer + sources + response time) is in `test_results.json`.

---

## Use cases

1. **Pharmaceutical training and onboarding** — new hires self-serve answers about GDP,
   batch review, change control, and CAPA without blocking a senior SME.
2. **GMP compliance Q&A assistant** — shop-floor operators and QC analysts get instant,
   sourced answers about the right corrective action, the correct SOP step, or the
   classification of a change.
3. **Audit preparation support** — the SME team rehearses against a voice agent that
   quotes back actual SOP language an auditor will see.
4. **Real-time shop floor guidance** — voice-first interface keeps operators hands-free;
   the avatar puts a face on a process that would otherwise be a paper SOP binder.
5. **Knowledge preservation (employee digital twin)** — codify a senior expert's answers
   into the corpus so institutional knowledge survives turnover and retirement.
6. **Regulatory inspection readiness** — front-line staff rehearse against questions
   inspectors typically ask ("what triggers a Level 1 deviation?", "what does ALCOA+
   require for paper records?").
7. **Cross-functional GxP education** — manufacturing, QC, regulatory, and IT teams
   share a single source of truth phrased in plain language with citations.
8. **Remote expert consultation** — LiveAvatar mode lets a remote SME's voice and face
   answer questions in real time, lowering the barrier to async support.

---

## Setup

### Prerequisites
- Python 3.10+
- A modern Chrome / Edge browser (the Web Speech API powers the mic).
- API keys (paid services — only needed for the spoken / video paths):
  - **ElevenLabs** — text-to-speech.
  - **HeyGen LiveAvatar** — real-time streaming avatar.
  - **HeyGen video** — pre-rendered fallback.

### Install

```bash
git clone https://github.com/gauravpandey36/pharma-voice-assistant.git
cd pharma-voice-assistant
pip install -r requirements.txt
```

### Configure

Create `config.json` in the project root:

```json
{
  "elevenlabs_api_key": "sk_...",
  "elevenlabs_voice_id": "your_voice_id",
  "liveavatar_api_key": "...",
  "liveavatar_avatar_id": "...",
  "heygen_api_key": "...",
  "heygen_avatar_id": "...",
  "heygen_voice_id": "...",
  "tts_engine": "elevenlabs"
}
```

### Build the demo brain

```bash
python build_testing_brain.py
```

Produces `brain_testing_v1.pkl` (~150 KB, 148 chunks, 1663 unique terms) from the five
SOPs in `sops_for_testing/`.

### Run the server

```bash
python start_server_with_avatar.py
```

The server defaults to `brain_testing_v1.pkl`. To switch to the full production brain:

```bash
BRAIN_FILE=search_index.pkl python start_server_with_avatar.py
```

Open <http://localhost:5000> in Chrome. The avatar auto-starts on page load (or first
user gesture if the browser blocks autoplay) and the test prompts render as buttons.

### Run the tests

```bash
python run_tests.py
```

Writes `test_results.json` and prints a pass/fail summary.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | Serves the avatar frontend |
| GET  | `/health` | Liveness + brain mode + integration status |
| GET  | `/brain_info` | Loaded brain file, document count, SOP list |
| GET  | `/test_prompts` | The 15 demo prompts (rendered as buttons) |
| POST | `/ask` | `{ "query": "..." }` -> `{ "answer", "sources" }` |
| POST | `/speak` | `{ "text": "..." }` -> `audio/mpeg` |
| POST | `/avatar/token` | Mints a LiveAvatar streaming session token |
| POST | `/avatar/video` | Generates and polls a pre-rendered HeyGen video |

---

## Test evidence

- `test_prompts.json` — the curated 15-question test set.
- `test_results.json` — full server responses for the most recent run (15/15 passing).
- `brain_testing_v1_stats.json` — index size and per-SOP chunk counts.

---

## Repository layout

```
pharma-voice-assistant/
+-- README.md                              # this file
+-- ARCHITECTURE.md                        # stage-by-stage architecture
+-- Makefile                               # build / test convenience targets
+-- requirements.txt
+-- src/
|   +-- api.py                             # core API (modular)
|   +-- asr.py
|   +-- avatar.py
|   +-- graceful_degradation.py
|   +-- llm_engine.py
|   +-- pipeline.py
|   +-- tts.py
|   +-- websocket_server.py
+-- frontend/
|   +-- index.html                         # browser UI
+-- start_server_with_avatar.py            # demo Flask server (BM25 + TTS + avatar)
+-- avatar_frontend.html                   # demo UI (chat + mic + avatar)
+-- build_testing_brain.py                 # rebuilds brain_testing_v1.pkl
+-- brain_testing_v1.pkl                   # demo BM25 index (5 SOPs)
+-- brain_testing_v1_stats.json
+-- sops_for_testing/                      # demo SOP corpus
|   +-- SOP-001-Good-Documentation-Practice.md
|   +-- SOP-002-Batch-Record-Review.md
|   +-- SOP-003-Batch-Release-Disposition.md
|   +-- SOP-004-Change-Control.md
|   +-- SOP-005-Deviation-CAPA-Management.md
+-- test_prompts.json
+-- run_tests.py
+-- test_results.json
```

---

## License & disclaimer

MIT license. The included SOPs are illustrative and reference current guidance, but
**must not be used as company SOPs** without organisation-specific adaptation, review,
and formal approval. Always rely on your own quality system and qualified personnel
for regulated activities.
