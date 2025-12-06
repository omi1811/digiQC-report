# Civil Quality AI Assistant

This module replaces the Gemini agent with a locally-hosted OpenLLM solution + Web Search.

## 🚀 Setup Instructions

### 1. Install Dependencies
You need `duckduckgo-search` for the web search capability.
```bash
pip install duckduckgo-search
```
(If not installed, web search will be disabled).

### 2. Run OpenLLM
You must have OpenLLM running for the AI to answer.

**Install OpenLLM:**
```bash
pip install openllm
```

**Start the Model (Gemma 3B):**
```bash
openllm serve gemma3:3b
```
*If your hardware struggles, use the fallback:*
```bash
openllm serve gemma2:2b
```

The server should start at `http://localhost:3000`.

### 3. Run the App
Restart your Flask app:
```bash
python app.py
```

## 🧠 How It Works
1. **Slot Detection**: The AI analyzes your question for project specifics (Grade, Member Type, Exposure, Cement).
2. **Follow-up**: If details are missing, it asks you to clarify.
3. **Web Search**: Once details are clear, it searches the web for standards and codes.
4. **Reasoning**: It uses OpenLLM to synthesize the search results into a verified answer.

## 📂 Files
- `civil_quality/engine.py`: Main orchestrator.
- `civil_quality/session.py`: Slot filling logic.
- `civil_quality/search.py`: Web search wrapper.
- `civil_quality/llm.py`: OpenLLM client.
