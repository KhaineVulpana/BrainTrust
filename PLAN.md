# Dual-Machine Local Coding Assistant
## Implementation Guide

### Architecture Overview

```
┌─────────────────────────────────────┐
│  LAPTOP (4060 + 64GB RAM)           │
│  Intelligent Preprocessor            │
├─────────────────────────────────────┤
│ • Intent Analysis                   │
│ • Structured Extraction             │
│ • RAG Context Retrieval             │
│ • Prompt Optimization               │
│ • Real-time Validation              │
└──────────────┬──────────────────────┘
               │ Structured JSON
               ▼
┌─────────────────────────────────────┐
│  DESKTOP (4080 + 32GB RAM)          │
│  Code Generation Engine              │
├─────────────────────────────────────┤
│ • Code Generation (13B model)       │
│ • Multi-pass Refinement             │
│ • Quality Validation                │
└─────────────────────────────────────┘
```

---

## Hardware Allocation

### Laptop (4060 + 64GB RAM)
**VRAM Usage:**
- 3B intent/extraction model: ~4GB
- Embedding model (all-MiniLM-L6-v2): ~500MB
- **Total VRAM: ~4.5GB / 8GB**

**RAM Usage:**
- ChromaDB vector store: ~5-10GB
- Document cache: ~10GB
- Model context buffer: ~5GB
- System + tools: ~5GB
- **Total RAM: ~25-30GB / 64GB**

### Desktop (4080 + 32GB RAM)
**VRAM Usage:**
- 13B code generation model (4-bit quantized): ~14GB
- **Total VRAM: ~14GB / 16GB**

**RAM Usage:**
- 7B planning model (for large contexts): ~8GB
- Context management: ~10GB
- System: ~5GB
- **Total RAM: ~23GB / 32GB**

---

## System Components

### Laptop Components

#### 1. Intent Analyzer
**Model:** 3B parameter model (Phi-3-mini or similar)
**Purpose:** Classify user request and extract structure
**Input:** Raw user text
**Output:** Task classification and structured data

```python
class IntentAnalyzer:
    def analyze(self, user_input: str) -> dict:
        return {
            "task_type": "feature_implementation",
            "complexity": "medium",
            "domain": "web_backend",
            "confidence": 0.92
        }
```

#### 2. Structured Extractor
**Model:** Same 3B model
**Purpose:** Extract technical details from user request
**Output:** Frameworks, libraries, requirements, constraints

```python
class StructuredExtractor:
    def extract(self, user_input: str, intent: dict) -> dict:
        return {
            "frameworks": ["express", "postgresql"],
            "requirements": ["authentication", "password_reset"],
            "constraints": ["use_bcrypt", "jwt_tokens"],
            "mentioned_files": ["models/User.js"]
        }
```

#### 3. RAG System
**Vector DB:** ChromaDB
**Embedding Model:** all-MiniLM-L6-v2
**Purpose:** Retrieve relevant context from codebase and docs

```python
class RAGSystem:
    def __init__(self):
        self.db = chromadb.Client()
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(self, query: str, n_results: int = 5) -> list:
        query_embedding = self.embeddings.encode(query)
        results = self.db.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
```

#### 4. Prompt Optimizer
**Purpose:** Combine all extracted info into optimal structured prompt
**Output:** Comprehensive JSON payload for desktop

```python
class PromptOptimizer:
    def optimize(self, intent, extracted, context, user_style) -> dict:
        return {
            "task": self._create_task_spec(intent, extracted),
            "context": self._compress_context(context),
            "style": user_style,
            "quality_criteria": self._generate_criteria(intent)
        }
```

#### 5. Style Learner
**Purpose:** Learn and maintain user's coding preferences
**Storage:** SQLite database or JSON file

```python
class StyleLearner:
    def learn_from_code(self, code: str):
        """Analyze user's approved code to learn patterns"""
        patterns = {
            "async_style": self._detect_async_pattern(code),
            "error_handling": self._detect_error_style(code),
            "naming_convention": self._detect_naming(code),
            "preferred_libraries": self._detect_libraries(code)
        }
        self.update_profile(patterns)
```

### Desktop Components

#### 1. Code Generator
**Model:** 13B+ parameter model (CodeLlama-13B-Instruct or DeepSeek-Coder-13B)
**Purpose:** Generate code from structured input

```python
class CodeGenerator:
    def __init__(self):
        self.model = load_model("deepseek-coder-13b-instruct")
    
    def generate(self, structured_input: dict) -> str:
        prompt = self._construct_prompt(structured_input)
        code = self.model.generate(prompt, max_tokens=2000)
        return code
```

#### 2. Multi-pass Refiner
**Purpose:** Iteratively improve code quality

```python
class CodeRefiner:
    def refine(self, code: str, issues: list, max_iterations: int = 3) -> str:
        for i in range(max_iterations):
            if not issues:
                break
            
            refinement_prompt = self._create_refinement_prompt(code, issues)
            code = self.generator.generate(refinement_prompt)
            issues = self.validator.validate(code)
        
        return code
```

#### 3. Quality Validator
**Purpose:** Run validation tools and checks

```python
class QualityValidator:
    def validate(self, code: str, language: str) -> list:
        issues = []
        
        # Syntax check
        issues.extend(self._check_syntax(code, language))
        
        # Linting
        issues.extend(self._run_linter(code, language))
        
        # Security scan
        issues.extend(self._security_scan(code))
        
        # Type checking (if applicable)
        if language in ["typescript", "python"]:
            issues.extend(self._type_check(code, language))
        
        return issues
```

---

## Data Flow

### Request Processing Flow

```
1. USER TYPES
   └─> Laptop: Stream analysis begins at ~15 characters

2. LAPTOP PROCESSING (parallel, 5-8s)
   ├─> Intent Analyzer: Classify request
   ├─> Structured Extractor: Extract technical details
   ├─> RAG System: Search for relevant context
   ├─> Style Learner: Load user preferences
   └─> Prompt Optimizer: Combine into structured payload

3. USER SENDS
   └─> Laptop: Transmit structured JSON to Desktop

4. DESKTOP GENERATION (10-15s)
   └─> Code Generator: Generate code from structured input

5. DESKTOP VALIDATION (3-5s)
   └─> Quality Validator: Check for issues

6. REFINEMENT LOOP (if issues found, 8-12s per iteration)
   ├─> Desktop: Generate fixes
   └─> Desktop: Re-validate
   └─> Repeat until pass or max iterations

7. RETURN TO USER
   └─> Stream code back to Laptop for display
```

### Message Protocol

**Laptop → Desktop Request:**
```json
{
  "request_id": "uuid",
  "timestamp": "2025-10-01T10:30:00Z",
  "task": {
    "type": "feature_implementation",
    "description": "Add JWT authentication to Express API",
    "complexity": "medium",
    "components": [
      "user_model",
      "login_endpoint",
      "password_reset"
    ]
  },
  "context": {
    "existing_code": {
      "models/User.js": "...",
      "routes/api.js": "..."
    },
    "relevant_docs": [...],
    "examples": [...]
  },
  "requirements": {
    "security": ["bcrypt", "rate_limiting"],
    "testing": true,
    "documentation": true
  },
  "style": {
    "async_pattern": "async_await",
    "error_handling": "try_catch_custom_errors",
    "testing_framework": "jest"
  },
  "quality_criteria": {
    "min_test_coverage": 80,
    "max_complexity": 10,
    "security_scan": true
  }
}
```

**Desktop → Laptop Response:**
```json
{
  "request_id": "uuid",
  "status": "success",
  "code": "...",
  "validation": {
    "passed": true,
    "issues": [],
    "iterations": 2
  },
  "metadata": {
    "generation_time": 12.3,
    "model_used": "deepseek-coder-13b",
    "tokens_generated": 1842
  }
}
```

---

## Technology Stack

### Laptop Stack
```yaml
Models:
  - Ollama (model server)
  - phi-3-mini-4k-instruct (3B intent/extraction)
  - all-MiniLM-L6-v2 (embeddings)

RAG:
  - ChromaDB (vector database)
  - sentence-transformers (embeddings)

Framework:
  - Python 3.11+
  - FastAPI (web server for UI)
  - WebSockets (real-time streaming)

Tools:
  - tree-sitter (code parsing)
  - SQLite (user preferences)
```

### Desktop Stack
```yaml
Models:
  - Ollama or vLLM (model server)
  - deepseek-coder-13b-instruct (code generation)
  - codellama-7b-instruct (planning for large contexts)

Validation:
  - ESLint/Pylint (linting)
  - Bandit/Semgrep (security)
  - Pytest/Jest (testing)

Framework:
  - Python 3.11+
  - FastAPI (API server)
```

### Network
```yaml
Protocol: HTTP/2 + WebSockets
Format: JSON
Security: mTLS (if exposed beyond LAN)
```

---

## Implementation Phases

### Phase 1: Basic Pipeline (Days 1-3)

**Day 1: Setup Infrastructure**
```bash
# Laptop
ollama pull phi-3-mini-4k-instruct
pip install chromadb sentence-transformers fastapi

# Desktop
ollama pull deepseek-coder:13b-instruct
pip install fastapi uvicorn
```

**Day 2: Implement Core Flow**
- Laptop: Basic intent analysis + RAG search
- Desktop: Simple code generation endpoint
- Test: Send structured request, get code back

**Day 3: Add Network Layer**
- FastAPI server on Desktop
- FastAPI client on Laptop
- WebSocket streaming for results

### Phase 2: Quality Enhancement (Days 4-7)

**Day 4-5: Multi-pass Refinement**
- Implement validation tools integration
- Add refinement loop on Desktop
- Test: Code generation with quality checks

**Day 6-7: RAG Enhancement**
- Index your codebase
- Add documentation sources
- Test: Context retrieval quality

### Phase 3: Intelligence Layer (Days 8-14)

**Day 8-10: Style Learning**
- Implement pattern detection
- Build user preference database
- Test: Style consistency

**Day 11-12: Advanced Extraction**
- Improve structured extraction
- Add dependency detection
- Test: Complex request handling

**Day 13-14: Optimization**
- Parallel processing on Laptop
- Caching frequently used contexts
- Performance tuning

---

## Quick Start Guide

### Installation

**1. Clone Repository (you'll create this)**
```bash
git clone https://github.com/yourusername/dual-machine-coder
cd dual-machine-coder
```

**2. Setup Laptop**
```bash
cd laptop
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi-3-mini-4k-instruct

# Initialize ChromaDB
python init_rag.py
```

**3. Setup Desktop**
```bash
cd desktop
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-coder:13b-instruct

# Configure API
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

**4. Configure Network**
```bash
# In laptop/config.yaml
desktop_url: "http://192.168.1.100:8000"  # Your desktop IP

# In desktop/config.yaml
listen_address: "0.0.0.0"
port: 8000
```

**5. Start Services**
```bash
# On Desktop (Terminal 1)
cd desktop
python main.py

# On Laptop (Terminal 2)
cd laptop
python main.py

# On Laptop (Terminal 3 - Web UI)
cd laptop
python ui.py
# Navigate to http://localhost:3000
```

---

## Configuration Examples

### Laptop config.yaml
```yaml
models:
  intent: "phi-3-mini-4k-instruct"
  embedding: "all-MiniLM-L6-v2"

rag:
  db_path: "./chroma_db"
  max_results: 5
  similarity_threshold: 0.7

desktop:
  url: "http://192.168.1.100:8000"
  timeout: 120
  retry_attempts: 3

preprocessing:
  min_confidence: 0.8
  parallel_extraction: true
  cache_ttl: 3600
```

### Desktop config.yaml
```yaml
models:
  code_generation: "deepseek-coder:13b-instruct"
  planning: "codellama:7b-instruct"

generation:
  max_tokens: 2000
  temperature: 0.2
  top_p: 0.95

validation:
  max_iterations: 3
  enable_security_scan: true
  enable_type_checking: true

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
```

---

## Example Usage

### Simple Request
```python
# User input
"Add a login endpoint to my Express app"

# Laptop processes → structured output
{
  "task_type": "api_endpoint",
  "framework": "express",
  "component": "authentication",
  "requirements": ["login", "password_validation", "jwt_token"]
}

# Desktop generates
# ... code for login endpoint with JWT ...
```

### Complex Request
```python
# User input
"Refactor the user authentication system to use OAuth2 
instead of JWT, maintaining backward compatibility"

# Laptop processes
{
  "task_type": "refactoring",
  "complexity": "high",
  "existing_code": {...},  # Retrieved from RAG
  "requirements": [
    "oauth2_implementation",
    "backward_compatibility",
    "migration_strategy"
  ],
  "constraints": ["no_breaking_changes"]
}

# Desktop generates migration plan + code
```

---

## Performance Targets

### Latency Goals
- **Simple requests (<10k context):** 10-15s total
- **Medium requests (10-30k context):** 15-25s total  
- **Complex requests (>30k context):** 25-40s total

### Quality Goals
- **First-pass success rate:** >70%
- **After refinement success rate:** >90%
- **Style consistency:** >85%
- **Context relevance:** >80%

### Resource Utilization
- **Laptop idle time:** <20%
- **Desktop idle time:** <30%
- **Network bandwidth:** <5MB per request
- **Memory growth:** <100MB per hour

---

## Troubleshooting

### Common Issues

**1. Models not loading**
```bash
# Check Ollama is running
ollama list

# Verify model is pulled
ollama pull deepseek-coder:13b-instruct

# Check VRAM availability
nvidia-smi
```

**2. Network connection fails**
```bash
# Test connectivity
curl http://192.168.1.100:8000/health

# Check firewall
sudo ufw allow 8000

# Verify config
cat laptop/config.yaml | grep desktop_url
```

**3. RAG returns poor results**
```bash
# Re-index codebase
python laptop/reindex.py

# Check embedding quality
python laptop/test_embeddings.py

# Adjust similarity threshold in config
```

**4. Code quality issues**
```bash
# Enable stricter validation
# In desktop/config.yaml:
validation:
  max_iterations: 5  # Increase from 3
  
# Add custom quality rules
# In desktop/quality_rules.yaml
```

---

## Future Enhancements

### Short-term (1-2 months)
- [ ] Web-based UI (replace CLI)
- [ ] Conversation history
- [ ] Multiple programming languages
- [ ] Custom validation rules
- [ ] Performance metrics dashboard

### Medium-term (3-6 months)
- [ ] Agentic workflows (planning → coding → testing)
- [ ] Integration with VS Code
- [ ] Fine-tuning on user's codebase
- [ ] Collaborative features (team style learning)
- [ ] Advanced debugging capabilities

### Long-term (6+ months)
- [ ] Multi-project context management
- [ ] Automated test generation
- [ ] Performance optimization suggestions
- [ ] Architecture refactoring agent
- [ ] Code review agent

---

## Resources

### Recommended Models
- **Intent/Extraction:** Phi-3-mini, TinyLlama-1.1B
- **Code Generation:** DeepSeek-Coder-13B, CodeLlama-13B, StarCoder2-15B
- **Embeddings:** all-MiniLM-L6-v2, bge-small-en-v1.5

### Documentation
- [Ollama](https://ollama.ai/docs)
- [ChromaDB](https://docs.trychroma.com/)
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
- [FastAPI](https://fastapi.tiangolo.com/)

### Community
- [LocalLLaMA Reddit](https://reddit.com/r/LocalLLaMA)
- [Ollama Discord](https://discord.gg/ollama)
- [AI Code Generation Forum](https://discuss.huggingface.co/)

---

## License

MIT License - feel free to modify and extend

## Contributing

Issues and PRs welcome! Please follow the contribution guidelines in CONTRIBUTING.md

---

**Built with ❤️ for maximizing local hardware potential**