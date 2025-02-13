# Shoperator Agent API

This is the API for the Shoperator Agent. This project builds upon the foundation of the browser-use, which is designed to make websites accessible for AI agents.

Expanded LLM Support: We've integrated support for various Large Language Models (LLMs), including: Google, OpenAI, Azure OpenAI, Anthropic, DeepSeek, Ollama etc. And we plan to add support for even more models in the future.

Custom Browser Support: You can use your own browser with our tool, eliminating the need to re-login to sites or deal with other authentication challenges. This feature also supports high-definition screen recording.

Persistent Browser Sessions: You can choose to keep the browser window open between AI tasks, allowing you to see the complete history and state of AI interactions.

## Installation Guide

### Prerequisites

- Python 3.11 or higher
- Git (for cloning the repository)

### Option 1: Local Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/abhinavsanyal/shoperator-back.git
cd shoperator-back
```

#### Step 2: Set Up Python Environment

We recommend using [uv](https://docs.astral.sh/uv/) for managing the Python environment.

Using uv (recommended):

```bash
uv venv --python 3.11
```

Activate the virtual environment:

- Windows (Command Prompt):

```cmd
.venv\Scripts\activate
```

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

#### Step 3: Install Dependencies

Install Python packages:

```bash
uv pip install -r requirements.txt
```

Install Playwright:

```bash
playwright install
```

#### Step 4: Configure Environment

1. Create a copy of the example environment file:

- Windows (Command Prompt):

```bash
copy .env.example .env
```

- macOS/Linux/Windows (PowerShell):

```bash
cp .env.example .env
```

2. Open `.env` in your preferred text editor and add your API keys and other settings

#### Step 5: Run the Server

Start the FastAPI server with hot reload enabled:

```bash
uvicorn server:app --reload
```

The server will be available at:

- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative API Documentation: http://localhost:8000/redoc

> **Note**: The `--reload` flag enables hot reloading, which automatically restarts the server when you make code changes. Remove this flag in production.

### Option 2: Docker Installation

#### Prerequisites

- Docker and Docker Compose installed
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/) (For Windows/macOS)
  - [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) (For Linux)

#### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/abhinavsanyal/shoperator-back.git
cd shoperator-back
```

2. Create and configure environment file:

- Windows (Command Prompt):

```bash
copy .env.example .env
```

- macOS/Linux/Windows (PowerShell):

```bash
cp .env.example .env
```

Edit `.env` with your preferred text editor and add your API keys

3. Run with Docker:

```bash
# Build and start the container with default settings (browser closes after AI tasks)
docker compose up --build
```

```bash
# Or run with persistent browser (browser stays open between AI tasks)
CHROME_PERSISTENT_SESSION=true docker compose up --build
```

## API Documentation

    http://localhost:3030/docs
