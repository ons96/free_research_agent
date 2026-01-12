# Free Research Agent

A cost-optimized, privacy-focused AI assistant built for speed and $0 ongoing maintenance calls.

## Features

- **Zero Cost**: Uses free model tiers (via g4f) and lite search APIs.
- **Fast & Lightweight**: Server-rendered HTML (Jinja2) + minimal Vanilla JS.
- **Modes**:
    - **Chat**: Fast conversational AI.
    - **Search**: Web-augmented answers with citations.
    - **Research**: Deep dive with full-page content extraction and analysis.
- **Deal Finder**: Automatically extracts prices and normalizes units for "best value" analysis.
- **Deployment**: Design for "Always Free" VM tiers (Oracle, AWS, GCP).

## 1. Local Run (Python)

Run the agent on your own machine.

### Prerequisites
- Python 3.10+
- Git

### Steps

1. Clone the repo:
   ```bash
   git clone https://github.com/your-repo/free_research_agent.git
   cd free_research_agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open `http://localhost:8000` in your browser.

## 2. API Server (For Coding Tools)

This agent exposes an OpenAI-compatible endpoint at `/v1/chat/completions`. You can use it with tools like Cursor, Aider, or Roo Code.

### Configuration
Create `config/providers.yaml` (copy from `.example`) to define your providers:
```yaml
providers:
  - name: "puter-bridge"
    type: "openai"
    base_url: "http://localhost:4000/v1"
    
  - name: "g4f-auto"
    type: "g4f"
    models: ["gpt-4", "gpt-3.5-turbo"]
```

### Usage with Aider
```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=sk-dummy
aider
```

## 3. Free Hosting (VM-First)

Ideal for Oracle Cloud Always Free (ARM Ampere) or Google Cloud E2-Micro.

### Option A: Docker Compose (Recommended)

1. **Install Docker** on your VM.
2. **Copy Files**: Transfer this repo to your VM.
3. **Run**:
   ```bash
   cd deployment
   docker compose up -d --build
   ```
4. Access via `http://<vm-ip>:8000`.

### Option B: Systemd (Raw Python)

1. SSH into your VM.
2. Clone repo and install Python/pip.
3. Install dependencies (see Local Run).
4. Create a service file `/etc/systemd/system/agent.service`:
   ```ini
   [Unit]
   Description=Free Research Agent
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/free_research_agent
   ExecStart=/home/ubuntu/free_research_agent/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 80
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
5. Enable and start: `sudo systemctl enable --now agent`.

## Configuration

See `.env.example`. Most credentials are optional/automatic (keyless search), but you can add specific keys for better stability.

## Architecture

- **Backend**: FastAPI (Async)
- **LLM Engine**: g4f (GPT4Free) with auto-fallback routing.
- **Search**: DuckDuckGo Lite (Keyless).
- **Frontend**: Jinja2 + SSE (Server-Sent Events) for streaming.
