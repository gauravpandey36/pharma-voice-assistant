# Real-Time Multimodal Streaming: Pharmaceutical Voice Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: In Progress](https://img.shields.io/badge/status-in%20progress-orange.svg)]()

---

## Overview

The Real-Time Multimodal Streaming project is a pharmaceutical expert voice assistant -- Gourav Pandey's "Digital Twin" -- that accepts voice input, processes queries through the pharmaceutical knowledge engine, and delivers responses via synthesized speech and optional avatar video. It combines Automatic Speech Recognition (ASR), large language model reasoning, text-to-speech synthesis, and avatar generation into a low-latency streaming pipeline designed for real-time conversational interaction with a pharmaceutical domain expert.

---

## Key Features

| Feature | Description |
|---|---|
| **Text-to-Speech Pipeline** | High-quality speech synthesis via ElevenLabs, Google TTS, and HeyGen |
| **Avatar Video Generation** | Realistic talking-head video via HeyGen integration |
| **REST API** | HTTP endpoints for programmatic access to all modalities |
| **Demo Frontend** | Browser-based interface for interactive demonstrations |
| **Multi-Provider TTS** | Fallback chain across ElevenLabs, Google TTS for reliability |
| **Pharmaceutical Domain** | Responses grounded in pharmaceutical compliance knowledge |
| **Low-Latency Design** | Streaming architecture with latency budgets for each pipeline stage |

---

## Architecture

```
+================================================================+
|              Real-Time Multimodal Pipeline                       |
|================================================================|
|                                                                  |
|  +------------------+                                            |
|  | Voice Input      |  (Microphone / Audio File)                 |
|  +--------+---------+                                            |
|           |                                                      |
|           v                                                      |
|  +--------+---------+                                            |
|  | ASR (Whisper)    |  Speech-to-Text                            |
|  | [Planned]        |  Target: < 500ms                           |
|  +--------+---------+                                            |
|           |                                                      |
|           v                                                      |
|  +--------+---------+     +-------------------+                  |
|  | Query Engine     | --> | Knowledge Base    |                  |
|  | (LLM + RAG)     |     | (Pharma Corpus)   |                  |
|  +--------+---------+     +-------------------+                  |
|           |                                                      |
|           |  Target TTFT: < 800ms                                |
|           v                                                      |
|  +--------+---------+                                            |
|  | TTS Engine       |  Text-to-Speech                            |
|  | - ElevenLabs     |  Target TTFB: < 300ms                     |
|  | - Google TTS     |                                            |
|  +--------+---------+                                            |
|           |                                                      |
|           +---> Audio Stream --> Speaker / Browser                |
|           |                                                      |
|           v                                                      |
|  +--------+---------+                                            |
|  | Avatar Engine    |  (Optional)                                |
|  | - HeyGen API     |  Video generation                         |
|  +--------+---------+                                            |
|           |                                                      |
|           +---> Video Stream --> Browser                          |
|                                                                  |
+================================================================+
```

---

## Tech Stack

- **Language:** Python 3.10+
- **ASR:** OpenAI Whisper (planned)
- **LLM:** OpenAI GPT-4 / Local Ollama models
- **TTS:** ElevenLabs API, Google Cloud TTS
- **Avatar:** HeyGen API
- **Streaming:** WebSocket (planned), REST API (current)
- **Frontend:** HTML/JavaScript demo interface
- **API Framework:** FastAPI

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- API keys: ElevenLabs and/or Google Cloud TTS
- Optional: HeyGen API key (for avatar video)

### Installation

```bash
# Clone the repository
git clone https://github.com/gauravpandey36/pharma-voice-assistant.git
cd pharma-voice-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
export ELEVENLABS_API_KEY="your-key"
export GOOGLE_TTS_API_KEY="your-key"
export HEYGEN_API_KEY="your-key"  # Optional
```

### Launch the Assistant

```bash
# Start the API server
python server.py --port 8000

# Open the demo frontend
open http://localhost:8000/demo
```

---

## Usage Examples

### Text-to-Speech via API

```bash
# Generate speech from text
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "The maximum hold time for Buffer A is 24 hours at 2 to 8 degrees Celsius.", "provider": "elevenlabs"}' \
  --output response.mp3
```

### Avatar Video Generation

```bash
# Generate a talking-head video
curl -X POST http://localhost:8000/avatar \
  -H "Content-Type: application/json" \
  -d '{"text": "Welcome to the pharmaceutical compliance briefing.", "avatar_id": "default"}' \
  --output briefing.mp4
```

### Full Pipeline (Planned)

```bash
# Stream a voice query end-to-end
# 1. User speaks into microphone
# 2. ASR transcribes speech to text
# 3. Query engine retrieves answer from knowledge base
# 4. TTS synthesizes response audio
# 5. Audio streams back to user in real-time
```

### Python SDK

```python
from voice_assistant import PharmaVoiceAssistant

assistant = PharmaVoiceAssistant(
    tts_provider="elevenlabs",
    knowledge_base="pharma_corpus"
)

# Text query with audio response
audio = assistant.ask("What are the stability requirements for API storage?")
audio.play()

# With avatar video
video = assistant.ask(
    "Explain the difference between OQ and PQ.",
    avatar=True
)
video.save("explanation.mp4")
```

---

## Latency Budget

End-to-end voice interaction must feel conversational. The target latency budget:

| Stage | Component | Target Latency | Status |
|---|---|---|---|
| 1 | ASR (Speech to Text) | < 500 ms | Planned |
| 2 | LLM Time-to-First-Token | < 800 ms | In Progress |
| 3 | TTS Time-to-First-Byte | < 300 ms | Done |
| 4 | Network + Buffering | < 200 ms | In Progress |
| **Total** | **End-to-End** | **< 1,800 ms** | **Target** |

### Graceful Degradation Strategy

When latency budgets are exceeded, the system degrades gracefully:

| Condition | Fallback Behavior |
|---|---|
| ElevenLabs API slow/down | Switch to Google TTS |
| Google TTS slow/down | Switch to local pyttsx3 (lower quality) |
| HeyGen API slow/down | Audio-only response (skip avatar) |
| LLM timeout | Return cached response for common queries |
| ASR failure | Prompt user for text input |

---

## Current Status

### Completed

- [x] ElevenLabs TTS integration
- [x] Google Cloud TTS integration
- [x] HeyGen avatar video generation
- [x] REST API server
- [x] Demo frontend interface
- [x] Multi-provider TTS fallback chain
- [x] Audio file generation and playback

### In Progress

- [ ] WebSocket streaming for real-time audio delivery
- [ ] Latency budgeting and monitoring per pipeline stage
- [ ] LLM integration with pharmaceutical knowledge base

### Planned

- [ ] ASR input via OpenAI Whisper
- [ ] Full voice-to-voice pipeline
- [ ] Graceful degradation logic
- [ ] Conversation context management
- [ ] WebRTC for browser-based voice input
- [ ] Streaming TTS (chunk-by-chunk audio delivery)

---

## Project Structure

```
pharma-voice-assistant/
├── server.py                  # FastAPI server with all endpoints
├── tts/
│   ├── elevenlabs_client.py   # ElevenLabs TTS integration
│   ├── google_tts_client.py   # Google Cloud TTS integration
│   ├── local_tts.py           # Local pyttsx3 fallback
│   └── router.py              # TTS provider selection and fallback
├── asr/
│   └── whisper_client.py      # Whisper ASR integration (planned)
├── avatar/
│   └── heygen_client.py       # HeyGen avatar video generation
├── knowledge/
│   └── query_engine.py        # Pharmaceutical knowledge retrieval
├── streaming/
│   └── websocket_handler.py   # WebSocket streaming (planned)
├── frontend/
│   ├── index.html             # Demo web interface
│   ├── app.js                 # Frontend application logic
│   └── styles.css             # Interface styling
├── tests/
├── requirements.txt
└── README.md
```

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure all existing tests pass
5. Submit a pull request with a clear description of changes

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Gourav Pandey**
GitHub: [@gauravpandey36](https://github.com/gauravpandey36)

---

*Part of the [AI for Regulated Life Sciences](../MASTER_README.md) project portfolio.*
