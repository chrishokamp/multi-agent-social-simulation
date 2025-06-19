# Multi-Agent Social Simulation Framework

This directory contains the core implementation of the multi-agent social simulation framework.

## Structure

```
src/
├── backend/          # Python backend services
│   ├── api/         # REST API endpoints
│   ├── db/          # MongoDB models
│   ├── engine/      # Autogen simulation engine
│   ├── orchestrator/# Simulation orchestrator
│   ├── config/      # Agent configurations
│   └── main.py      # Entry point
│
└── frontend/        # React frontend application
    ├── src/
    │   ├── pages/   # UI pages
    │   ├── services/# API integration
    │   └── App.jsx  # Main component
    └── public/      # Static assets
```

## Quick Start

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Set up environment:**
   ```bash
   make setup-env
   # Edit src/backend/.env with your API keys
   ```

3. **Start MongoDB:**
   ```bash
   make db-start
   ```

4. **Run development servers:**
   ```bash
   # Terminal 1
   make dev-backend
   
   # Terminal 2
   make dev-frontend
   ```

5. **Access the application:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:5000

## Available Commands

Run `make help` to see all available commands.

## Requirements

- Python 3.10+
- Node.js 16+
- MongoDB 6.0+
- OpenAI API key (or Azure OpenAI/Ollama configuration)