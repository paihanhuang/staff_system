# The Synapse Council

An autonomous Architecture Review Board (ARB) powered by multiple AI agents that drafts, critiques, and finalizes system designs with human-in-the-loop oversight.

## Features

- **Multi-Agent Collaboration**: Uses OpenAI o3 (Architect), Claude 4.5 Sonnet (Engineer), and Gemini 3 Pro (Auditor)
- **Blind Ideation**: Agents propose solutions independently to prevent groupthink
- **Cross-Critique**: Agents challenge each other's proposals before final audit
- **Human-in-the-Loop**: Automatic pause for clarification when agents need more information
- **Structured Output**: Consistent JSON schemas for all proposals and decisions
- **Real-time Streaming**: WebSocket-based communication for live updates

## Architecture

```
User Question → Supervisor → Blind Ideation → Cross-Critique → Audit → Convergence → Final ADR
                    ↑                                              ↓
                    └──────────── Clarification Loop ──────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd staff_system
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the Backend

```bash
source .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000
```

### 4. Run the Frontend

```bash
source .venv/bin/activate && streamlit run frontend/app.py
```

## Usage

1. Open the Streamlit app in your browser
2. (Optional) Configure system context (tech stack, constraints, SLAs)
3. Enter your system design question
4. Watch the AI agents collaborate in real-time
5. Respond to any clarification requests
6. Review the final Architecture Decision Record (ADR)

## Project Structure

```
staff_system/
├── src/
│   ├── models/          # Pydantic schemas
│   ├── adapters/        # AI model adapters
│   ├── graph/           # LangGraph state machine
│   ├── prompts/         # System prompts
│   ├── api/             # FastAPI backend
│   └── utils/           # Utilities
├── frontend/            # Streamlit app
└── tests/               # Test suite
```

## License

MIT
