# BSSC_QA Framework

**BSSC_QA** is an intelligent, agentic framework for automated Question-Answer pair generation from documents. Built with LangChain v1.0, it leverages multiple LLM providers and ChromaDB for efficient, scalable, and high-quality QA dataset creation.

---

## 🎯 Overview

BSSC_QA automates the creation of question-answer datasets through a multi-agent system that generates, synthesizes, and evaluates QA pairs from your documents. It's designed to be modular, config-driven, and LLM-agnostic.

**Key Features:**
- 🤖 Multi-agent architecture (Generator, Synthesizer, Evaluator)
- 🔄 Support for 4 LLM providers (Gemini, DeepSeek, Mistral, HuggingFace)
- 💾 Local vector storage with ChromaDB
- 📝 Multi-format document support (TXT, PDF, HTML, DOCX, URLs)
- ⚙️ Fully config-driven - no code changes needed
- 📊 Quality metrics and validation
- 🎓 Optional Bloom's taxonomy support

---

## 📦 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BSSC_QA.git
cd BSSC_QA

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create or update `config.json` with your API keys:

```json
{
  "llm": {
    "default_provider": "gemini",
    "providers": {
      "gemini": {
        "api_key": "YOUR_GEMINI_API_KEY",
        "model": "gemini-2.0-flash-exp",
        "temperature": 0.7
      }
    }
  }
}
```

### Running the Pipeline

```bash
# Full pipeline
python run_pipeline.py

# Or use the notebook
jupyter notebook notebooks/01_BSSC_QA_Complete.ipynb
```

---

## 🏗️ Architecture

![Architecture Diagram](docs/architecture_diagram_placeholder.png)

The framework consists of three main agents working in sequence:

1. **Generator Agent** - Creates diverse questions from document chunks
2. **Synthesis Agent** - Generates comprehensive, evidence-based answers
3. **Evaluator Agent** - Assesses quality and validates QA pairs

Each agent uses specialized tools and can be powered by different LLM providers.

---

## 📂 Project Structure

```
BSSC_QA/
├── bssc_qa/
│   └── src/
│       ├── core/              # Config, LLM factory, vector store
│       ├── agents/            # Generator, Synthesis, Evaluator
│       ├── tools/             # Retrieval, validation, chunking tools
│       ├── pipeline/          # Ingestion and orchestration
│       └── utils/             # Text processing utilities
├── data/
│   ├── chroma_db/            # Vector database
│   └── output/               # Generated datasets
├── notebooks/                 # Jupyter notebooks
├── config.json               # Configuration file
├── run_pipeline.py          # Main runner script
└── requirements.txt         # Dependencies
```

---

## 💡 Usage Example

### Basic Usage

```python
from bssc_qa import QAPipelineOrchestrator
from core.config import load_config

# Load configuration
cfg = load_config('config.json')

# Run pipeline
orchestrator = QAPipelineOrchestrator(cfg)
results = orchestrator.generate_qa_from_chunks(
    num_chunks=10,
    questions_per_chunk=3
)
```

### Document Ingestion

```python
from pipeline.ingestion import IngestionPipeline

# Initialize pipeline
pipeline = IngestionPipeline(vs_manager, chunk_size=512)

# Ingest documents
results = pipeline.ingest_directory(
    directory='data/documents',
    pattern='*.txt',
    max_files=50
)
```

---

## ⚙️ Configuration

The framework is fully controlled via `config.json`:

**Key Settings:**
- **LLM Providers** - Configure multiple LLMs (Gemini, DeepSeek, Mistral, HuggingFace)
- **Chunking** - Adjust chunk size and overlap
- **Quality Threshold** - Set minimum acceptable QA quality scores
- **Bloom Levels** - Enable cognitive complexity targeting (optional)
- **Human Review** - Enable manual verification for low-confidence pairs (optional)

See [framework_presentation.md](framework_presentation.md) for detailed configuration options.

---

## 📊 Output Format

Generated QA pairs are exported in JSON format:

```json
{
  "metadata": {
    "total_qa_pairs": 30,
    "passed_qa_pairs": 28,
    "timestamp": "2025-11-01T12:34:56"
  },
  "qa_pairs": [
    {
      "qa_id": "qa_001",
      "question": "What are the key features of BSSC_QA?",
      "answer": "BSSC_QA features include...",
      "evidence_spans": ["..."],
      "scores": {
        "relevance": 0.95,
        "clarity": 0.92,
        "completeness": 0.88,
        "factuality": 0.94,
        "diversity": 0.90
      },
      "overall_score": 0.92
    }
  ]
}
```

---

## 🎓 Advanced Features

### Bloom's Taxonomy Support

Enable cognitive complexity targeting:

```json
{
  "bloom_level": {
    "enabled": true,
    "levels": ["remember", "understand", "apply", "analyze", "evaluate", "create"]
  }
}
```

### Multi-LLM Configuration

Use different LLMs for different agents:

```json
{
  "agents": {
    "generator": {"provider": "gemini"},
    "synthesis": {"provider": "deepseek"},
    "evaluator": {"provider": "mistral"}
  }
}
```

---

## 🔧 Development

### Running Tests

```bash
# Unit tests
python -m pytest tests/

# Specific test
python -m pytest tests/test_agents.py
```

### Code Structure

Each module is designed to be:
- **Modular** - Easy to swap components
- **Testable** - Clean interfaces and mock-friendly
- **Extensible** - Add new agents or tools easily

---

## 📚 Documentation

For comprehensive documentation, see:
- [**Framework Presentation**](framework_presentation.md) - Detailed technical documentation
- [**Blueprint**](BSSC_QA_blueprint.txt) - Original design document
- [**Notebooks**](notebooks/) - Interactive tutorials and examples

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - Agent framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy QA Generation!** 🚀
