# Real-Time Multimodal Streaming: System Architecture

## Table of Contents

1. [System Design Overview](#system-design-overview)
2. [Component Descriptions](#component-descriptions)
3. [Data Flow](#data-flow)
4. [Key Design Decisions](#key-design-decisions)
5. [Integration Points](#integration-points)

---

## System Design Overview

The Multimodal Streaming system is designed as a **staged pipeline with parallel output paths**. Each stage (ASR, LLM, TTS, Avatar) operates independently with defined latency budgets and fallback mechanisms. The pipeline is optimized for streaming: each stage begins emitting output as soon as it has enough input, rather than waiting for the previous stage to complete entirely.

```
+=================================================================+
|                Multimodal Streaming Pipeline                      |
|=================================================================|
|                                                                   |
|  INPUT PATH                                                       |
|  +----------+     +----------+     +------------------+           |
|  | Audio    | --> | ASR      | --> | Text Query       |           |
|  | Stream   |     | (Whisper)|     | (Transcribed)    |           |
|  +----------+     +----------+     +--------+---------+           |
|                                             |                     |
|  PROCESSING                                 v                     |
|                                    +--------+---------+           |
|                                    | LLM + RAG        |           |
|                                    | (Knowledge        |           |
|                                    |  Retrieval +      |           |
|                                    |  Generation)      |           |
|                                    +--------+---------+           |
|                                             |                     |
|                                    Token stream begins            |
|                                             |                     |
|  OUTPUT PATHS (Parallel)                    |                     |
|                              +--------------+--------------+      |
|                              |                             |      |
|                              v                             v      |
|                    +---------+--------+          +---------+--+   |
|                    | TTS Engine       |          | Avatar     |   |
|                    | (Streaming       |          | Engine     |   |
|                    |  Synthesis)      |          | (HeyGen)   |   |
|                    +---------+--------+          +---------+--+   |
|                              |                             |      |
|                              v                             v      |
|                    +---------+--------+          +---------+--+   |
|                    | Audio Stream     |          | Video      |   |
|                    | --> Speaker      |          | Stream     |   |
|                    +------------------+          | --> Browser|   |
|                                                  +------------+   |
|                                                                   |
+=================================================================+
```

The key architectural insight is that the LLM generates tokens sequentially, and the TTS engine can begin synthesizing audio from the first sentence while the LLM is still generating subsequent sentences. This **pipelining** dramatically reduces perceived latency.

---

## Component Descriptions

### ASR Module (Planned)

**Technology:** OpenAI Whisper (local or API)

The Automatic Speech Recognition module converts user speech into text. It must operate with minimal latency to avoid disrupting the conversational flow.

| Aspect | Specification |
|---|---|
| Model | Whisper Medium / Large-v3 |
| Latency Target | < 500ms for 5-second utterance |
| Language | English (primary), extensible |
| Input Format | 16kHz PCM audio stream |
| Output | Transcribed text with confidence score |
| Fallback | Text input prompt if ASR fails |

**Voice Activity Detection (VAD):** The ASR module includes endpoint detection to determine when the user has finished speaking, triggering the processing pipeline without manual signaling.

### LLM Query Engine

**Technology:** OpenAI GPT-4 (cloud) / Ollama (local)

The query engine retrieves relevant pharmaceutical knowledge and generates a response. It operates in streaming mode, emitting tokens as they are generated.

| Aspect | Specification |
|---|---|
| Streaming | Token-by-token via SSE or WebSocket |
| TTFT Target | < 800ms (time to first token) |
| Context | Retrieved pharmaceutical knowledge chunks |
| Max Output | 500 tokens (optimized for spoken delivery) |

**Response Optimization for Speech:** Responses are prompted to be concise, use natural spoken cadence, and avoid abbreviations or formatting that does not translate well to audio.

### TTS Engine

**Technology:** ElevenLabs (primary), Google Cloud TTS (secondary), pyttsx3 (local fallback)

The text-to-speech engine converts generated text into audio. It supports streaming synthesis: audio begins playing while text is still being generated.

| Provider | Quality | Latency | Cost | Offline |
|---|---|---|---|---|
| ElevenLabs | Excellent | ~200ms TTFB | Per-character | No |
| Google Cloud TTS | Good | ~250ms TTFB | Per-character | No |
| pyttsx3 | Basic | ~50ms TTFB | Free | Yes |

**Sentence Buffering:** The TTS engine buffers incoming tokens until a sentence boundary is detected (period, question mark, exclamation mark), then synthesizes that sentence while the next sentence accumulates. This provides natural speech pacing without waiting for the full response.

### Avatar Engine

**Technology:** HeyGen API

The avatar engine generates talking-head video synchronized with the audio response. This is an optional enhancement for demo and presentation contexts.

| Aspect | Specification |
|---|---|
| Video Quality | 720p / 1080p |
| Generation Time | 10-30 seconds (asynchronous) |
| Avatar Options | Pre-configured digital twin appearance |
| Audio Sync | Lip-synced to TTS audio output |

Because avatar generation is inherently slower than real-time, it operates asynchronously: the audio response plays immediately, and the avatar video is delivered when ready (or skipped if latency is critical).

### WebSocket Streaming Handler (Planned)

**Technology:** FastAPI WebSocket

The streaming handler manages bidirectional WebSocket connections for real-time voice interaction:

```
Client                                Server
  |                                      |
  |-- Audio chunks (binary) ------------>|
  |                                      |-- ASR processing
  |                                      |-- LLM streaming
  |<------------ Audio chunks (binary) --|
  |<------------ Metadata (JSON) --------|
  |                                      |
```

---

## Data Flow

### Full Voice-to-Voice Pipeline

```
User speaks into microphone
       |
       v
[Audio Capture] ---> 16kHz PCM stream
       |
       v
[Voice Activity Detection] ---> Detect utterance boundaries
       |
       v
[Whisper ASR] ---> Transcribe to text (< 500ms)
       |
       v
[Query Engine] ---> Retrieve pharma knowledge + generate response
       |
       |  (Token stream begins)
       v
[Sentence Buffer] ---> Accumulate tokens until sentence boundary
       |
       |  (First sentence ready)
       +------+---------------------------+
       |      |                           |
       v      v                           v
[TTS]  [Continue accumulating]     [Avatar Engine]
       |      |                    (Async, optional)
       v      v                           |
[Audio]  [Next sentence -> TTS]           v
       |      |                    [Video ready later]
       v      v
[Speaker / Browser]
```

### Latency Timeline (Target)

```
Time (ms)  0    500    1000    1300    1500    1800
           |-----|------|-------|-------|-------|
           |     |      |      |       |       |
           | ASR |      | LLM  | TTS   |       |
           |     |      | TTFT | TTFB  | Audio |
           |     |      |      |       | plays |
           |<-500ms->|  |<800ms>|<300ms>|       |
                       ^
                 Query sent to LLM
```

### Graceful Degradation Flow

```
Normal Operation:
  ASR -> LLM -> ElevenLabs TTS -> Audio + HeyGen Avatar

ElevenLabs Down:
  ASR -> LLM -> Google TTS -> Audio + HeyGen Avatar

Cloud TTS Down:
  ASR -> LLM -> pyttsx3 (local) -> Audio only (no avatar)

ASR Failure:
  Text Input -> LLM -> TTS -> Audio

LLM Timeout:
  ASR -> Cached Response -> TTS -> Audio
```

---

## Key Design Decisions

### 1. Streaming Pipeline Over Batch Processing

**Decision:** Each pipeline stage streams output to the next stage incrementally rather than waiting for full completion.

**Rationale:** In a batch model, the user waits for ASR + LLM + TTS to fully complete before hearing anything. With a 3-second utterance and 2-second LLM generation, this means 5+ seconds of silence. Streaming reduces perceived latency to approximately 1.8 seconds by overlapping stages: TTS begins on the first sentence while the LLM generates the second.

### 2. Multi-Provider TTS with Automatic Fallback

**Decision:** Support three TTS providers in a priority chain rather than depending on a single provider.

**Rationale:** ElevenLabs provides the highest quality but is a cloud dependency. In demo environments, network issues or API quotas could cause total failure. The three-tier fallback (ElevenLabs -> Google TTS -> pyttsx3) ensures the system always produces audio, degrading quality gracefully rather than failing silently.

### 3. Avatar as Asynchronous Optional Enhancement

**Decision:** Avatar video generation runs asynchronously and is treated as optional enrichment rather than a required pipeline stage.

**Rationale:** HeyGen avatar generation takes 10-30 seconds, far exceeding real-time conversational latency. Making it a blocking requirement would destroy the user experience. Instead, audio plays immediately and the avatar video is delivered as a follow-up enhancement for recordings and presentations.

### 4. Sentence-Level TTS Buffering

**Decision:** Buffer LLM tokens until sentence boundaries before sending to TTS, rather than word-by-word or full-response.

**Rationale:** Word-by-word TTS produces choppy, unnatural speech. Full-response TTS eliminates the latency benefit of streaming. Sentence-level buffering provides the optimal balance: natural speech cadence with minimal additional latency (typically 1-3 seconds per sentence of LLM generation before TTS begins on that sentence).

### 5. Latency Budget as Architecture Constraint

**Decision:** Define explicit latency targets for each pipeline stage and treat them as hard architecture constraints.

**Rationale:** Without explicit budgets, latency creeps upward as features are added. By allocating a budget to each stage (ASR: 500ms, LLM TTFT: 800ms, TTS TTFB: 300ms, overhead: 200ms = 1,800ms total), every component choice and optimization decision has a clear numerical target. Components that cannot meet their budget trigger architectural changes rather than acceptance of degraded performance.

---

## Integration Points

### Current Integrations

| Service | Protocol | Direction | Purpose |
|---|---|---|---|
| ElevenLabs | REST API | Outbound | Text-to-speech synthesis |
| Google Cloud TTS | REST API | Outbound | TTS fallback provider |
| HeyGen | REST API | Outbound | Avatar video generation |
| Frontend | HTTP / Static | Inbound | Demo web interface |

### Planned Integrations

| Service | Protocol | Direction | Purpose |
|---|---|---|---|
| OpenAI Whisper | Python / API | Internal | Speech-to-text |
| GxP-Struct RAG | Python import | Internal | Pharmaceutical knowledge retrieval |
| Local AI Assistant | HTTP (localhost) | Internal | Offline query processing |
| WebSocket | WS | Bidirectional | Real-time audio streaming |
| WebRTC | STUN/TURN | Bidirectional | Browser voice input |
| AI Monitoring Layer | JSONL | Outbound | Latency and quality metrics |

---

*For usage instructions, see [README.md](README.md).*
