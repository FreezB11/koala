# Code Analysis - Version 2 (Consolidation)

## Changes from v1

### Removed Files
- `agent0.py` - Duplicate entry point with different memory format
- `agents/tools.py` - Legacy tools (duplicate of root `tools.py`)
- `agents/mstrl.py` - Unused Mistral client
- `agents/groq.py` - Unused Groq client
- `agents/__init__.py` - Empty

### Consolidated
- **Single entry point**: `orch.py` → `orchestrator.py` → `main()`
- **Single tool source**: `tools.py` (root) with `ToolResult`, `FUNCTION_MAP`, `OPENAI_TOOLS`, `GEMINI_TOOLS`
- **Single config**: `config.py` with clean separation (ModelConfig, MemoryConfig, AgentConfig, ToolConfig)
- **Single memory format**: OpenAI-style messages throughout

### Improved
- **Security**: `run_command` uses `shlex.split()` + allowlist, no `shell=True`
- **Path validation**: Centralized `_validate_path()` in tools
- **File size limits**: Configurable max file size
- **Error handling**: Better logging, structured ToolResult
- **Config consistency**: All modules use `config.models.*` and `config.agent.*`

## Remaining Issues for v3

### 1. **No Retry Logic**
- API calls can fail transiently (network, rate limits)
- No exponential backoff for NVIDIA/Google APIs

### 2. **Worker Failure Handling**
- If one Gemini worker fails, the whole turn fails
- No fallback or graceful degradation

### 3. **Memory Management**
- No memory summarization (just concatenation)
- No expiration/TTL for old memories
- No memory importance scoring

### 4. **Conversation Persistence**
- Only memory persists, not full conversation history
- Can't resume a session

### 5. **No Health Checks**
- No startup validation of API connectivity
- No runtime health monitoring

### 6. **Limited Observability**
- Basic logging only
- No metrics (token usage, latency, tool calls)
- No structured logging (JSON)

### 7. **Blocking Workers**
- Workers run sequentially in `execute_tool`
- Could run in parallel for independent tasks

### 8. **No Input Validation**
- Tool arguments not validated against schemas
- No sanitization of user input

### 9. **Hardcoded Values**
- Colors, prompts, thresholds scattered
- Some still in orchestrator.py

### 10. **Testing**
- No unit tests
- No integration tests
- No mock fixtures

## Plan for v3 (Robustness & Reliability)

1. Add retry logic with exponential backoff for all API clients
2. Implement graceful worker failure handling
3. Add memory summarization with LLM
4. Add conversation persistence (JSONL format)
5. Add health checks at startup
6. Add structured logging + metrics
7. Implement parallel worker execution
8. Add input validation for tools
9. Centralize all constants
10. Add basic test structure