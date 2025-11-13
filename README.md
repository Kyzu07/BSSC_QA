# BSSC_QA Framework

**BSSC_QA** is an intelligent, multi-agent framework for automated Question-Answer pair generation from documents. Built with LangChain 1.0, it orchestrates multiple LLM providers and ChromaDB to create high-quality QA datasets efficiently and scalably.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Overview

BSSC_QA automates QA dataset creation through a sophisticated three-agent system (Generator → Synthesizer → Evaluator) that works together to produce diverse, accurate, and validated question-answer pairs from your documents.

### Key Features

- 🤖 **Multi-Agent Architecture** - Generator, Synthesizer, and Evaluator agents working in concert
- 🔄 **Multiple LLM Providers** - Gemini, DeepSeek, Mistral, and HuggingFace support
- 💾 **Local Vector Storage** - ChromaDB with efficient similarity search
- 📝 **Multi-Format Support** - Process TXT, PDF, HTML, DOCX files and URLs
- ⚙️ **Config-Driven** - Zero code changes needed, fully customizable via JSON
- 📊 **Quality Metrics** - Comprehensive validation and scoring system
- 🎓 **Bloom's Taxonomy** - Optional cognitive complexity targeting
- 🌏 **Bengali Support** - Full support for Bengali text processing (Make sure the LLM are capable of handling Bengali)

---

## 📦 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for at least one LLM provider (Gemini, DeepSeek, Mistral, or HuggingFace)

### Installation

```bash
# Clone repository
git clone https://github.com/Kyzu07/BSSC_QA.git
cd BSSC_QA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example config and add your API keys:

```bash
cp config.json config.local.json
```

2. Edit `config.local.json` (or `config.json`) with your API credentials:

```json
{
  "llm": {
    "default_provider": "gemini",
    "providers": {
      "gemini": {
        "api_key": "YOUR_GEMINI_API_KEY",      # Not necessary if you export it in the working environment.
        "model": "gemini-2.5-flash",
        "temperature": 0.5
      }
    }
  }
}
```

**Security Note**: Use environment variables for API keys:
```bash
export GEMINI_API_KEY="your-api-key-here"     # Or you can hardcode it in the config.json file
```
*Grab a free tier API from here for testing purpose 
  1. Gemini: (https://aistudio.google.com/api-keys)
  2. Mistral: (https://admin.mistral.ai/organization/api-keys)
  3. Deepseek: (https://platform.deepseek.com/api_keys)
  4. Huggingface: (https://huggingface.co/settings/tokens)
  5. Openrouter: (https://openrouter.ai/settings/keys) # Not tested in this framework
### Quick Demo

```bash
# Run the complete pipeline on demo data
python run_pipeline.py
```

The pipeline will:
1. Load documents from `data/demo/`
2. Process them into chunks
3. Generate QA pairs
4. Output results to `data/output/qa_dataset_[timestamp].json`

---

## 🏗️ Architecture

The framework uses a three-stage agent pipeline:

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  Generator  │  →    │ Synthesizer │  →    │  Evaluator  │
│    Agent    │       │    Agent    │       │    Agent    │
└─────────────┘       └─────────────┘       └─────────────┘
     ↓                      ↓                      ↓
 Questions            Answers + Evidence      Quality Scores
```

1. **Generator Agent** - Creates diverse, high-quality questions from document chunks
2. **Synthesis Agent** - Generates comprehensive, evidence-based answers using vector retrieval
3. **Evaluator Agent** - Assesses quality across relevance, clarity, completeness, and factuality

Each agent can be powered by a different LLM provider for optimal performance.

---

## 📂 Project Structure

```
BSSC_QA/
├── bssc_qa/src/           # Core source code
│   ├── core/              # Config, LLM factory, vector store
│   ├── agents/            # Generator, Synthesis, Evaluator
│   ├── tools/             # Retrieval, validation, chunking
│   ├── pipeline/          # Document loading & orchestration
│   └── utils/             # Text processing utilities
├── data/
│   ├── demo/              # Sample documents
│   ├── chroma_db/         # Vector database (auto-created)
│   └── output/            # Generated QA datasets
├── prompts/               # Customizable agent prompts
├── notebooks/             # Interactive tutorials
│   ├── 01_BSSC_QA_Complete.ipynb
│   └── test_BSSC_QA.ipynb
├── config.json            # Main configuration
├── requirements.txt       # Python dependencies
└── run_pipeline.py        # Main execution script
```

---

## 💡 Usage

### Basic Pipeline

```python
from bssc_qa.src.core.config import load_config
from bssc_qa.src.pipeline.orchestrator import QAPipelineOrchestrator

# Load configuration
cfg = load_config('config.json')

# Initialize orchestrator (see run_pipeline.py for full setup)
orchestrator = QAPipelineOrchestrator(generator, synthesizer, evaluator, vs_manager, cfg)

# Generate QA pairs from 20 chunks, 3 questions each
results = orchestrator.generate_qa_from_chunks(
    num_chunks=20,
    questions_per_chunk=3
)

# Export results
orchestrator.export_results(results, 'output/my_dataset.json')

print(f"Generated {results['passed_qa_pairs']} high-quality QA pairs")
print(f"Average score: {results['statistics']['average_score']:.2f}")
```

### Document Ingestion

```python
from bssc_qa.src.pipeline.ingestion import IngestionPipeline
from bssc_qa.src.core.vector_store import VectorStoreManager

# Initialize vector store
vs_manager = VectorStoreManager(
    persist_directory="./data/chroma_db",
    collection_name="my_documents",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Create ingestion pipeline
pipeline = IngestionPipeline(vs_manager, chunk_size=512, chunk_overlap=50)

# Ingest a directory of documents
results = pipeline.ingest_directory(
    directory='data/my_documents',
    pattern='*.txt',
    max_files=100
)

print(f"Processed: {results['processed']} files")
print(f"Total chunks: {results['total_chunks']}")
```

### Interactive Notebooks

For step-by-step tutorials and examples:

1. **Complete Pipeline**: `notebooks/01_BSSC_QA_Complete.ipynb`
2. **Testing & Evaluation**: `notebooks/test_BSSC_QA.ipynb`

```bash
jupyter notebook notebooks/
```

---

## ⚙️ Configuration Guide

### LLM Provider Configuration

The framework supports mixing different LLMs for different agents:

```json
{
  "agents": {
    "generator": {
      "provider": "gemini",
      "max_retries": 3
    },
    "synthesis": {
      "provider": "deepseek",
      "max_evidence_spans": 3
    },
    "evaluator": {
      "provider": "mistral",
      "quality_threshold": 0.75
    }
  }
}
```

**Supported Providers**:
- **Gemini** (`gemini-2.5-flash`) - Fast, efficient for generation
- **DeepSeek** (`deepseek-chat`) - Excellent reasoning for synthesis
- **Mistral** (`mistral-large-latest`) - Strong evaluation capabilities
- **HuggingFace** (`meta-llama/Llama-3.1-8B-Instruct`) - Open-source option

### Chunking Configuration

```json
{
  "chunking": {
    "chunk_size": 512,        // Tokens per chunk
    "chunk_overlap": 50,      // Overlapping tokens for context
    "auto_adjust": true       // Smart boundary detection
  }
}
```

**Recommendations**:
- Smaller chunks (256-512): Better for factual QA
- Larger chunks (512-1024): Better for conceptual QA
- Overlap: 10-20% of chunk_size for context preservation

### Quality Threshold

```json
{
  "agents": {
    "evaluator": {
      "quality_threshold": 0.75,
      "metrics": ["relevance", "clarity", "completeness", "factuality"]
    }
  }
}
```

Scores are averaged across all metrics. Only QA pairs with `overall_score >= threshold` are included in the final dataset.

---

## 📊 Output Format

Generated QA datasets follow this structure:

```json
{
  "metadata": {
    "total_qa_pairs": 28,
    "passed_qa_pairs": 26,
    "timestamp": "2025-11-08T10:30:00",
    "statistics": {
      "pass_rate": 0.928,
      "average_score": 0.87,
      "question_type_distribution": {
        "factual": 12,
        "conceptual": 10,
        "analytical": 4
      }
    }
  },
  "qa_pairs": [
    {
      "qa_id": "uuid-string",
      "question": "What are the key architectural features?",
      "answer": "The structure features...",
      "evidence_spans": ["Evidence text 1", "Evidence text 2"],
      "scores": {
        "relevance": 0.95,
        "clarity": 0.92,
        "completeness": 0.88,
        "factuality": 0.94
      },
      "overall_score": 0.92,
      "question_type": "conceptual"
    }
  ]
}
```

---

## 🎓 Advanced Features

### Bloom's Taxonomy Support

Target specific cognitive complexity levels for educational datasets:

```json
{
  "bloom_level": {
    "enabled": true,
    "levels": ["remember", "understand", "apply", "analyze"]
  }
}
```

**Cognitive Levels** (from simple to complex):
1. **Remember** - Recall facts and basic concepts
2. **Understand** - Explain ideas or concepts  
3. **Apply** - Use information in new situations
4. **Analyze** - Draw connections among ideas
5. **Evaluate** - Justify decisions or actions
6. **Create** - Produce new or original work

The generator will create questions distributed across specified levels.

### Custom Prompts

Customize agent behavior by editing prompt templates in `prompts/`:

- `prompts/default_prompt.json` - Prompts for full detailed answers
- `prompts/short_prompt.json` - Prompts for shorter answers

To use custom prompts:
**Example Prompt:** *(short_prompt.json)*
```json
{
  "generator": {
    "system": "You author crisp recall questions that elicit one- or two-word replies while still covering key facts. Keep prompts lean and avoid fluff.",
    "user_template": "From the text chunk below craft {count} short-answer questions. Each answer should be a single word or a tight two-word phrase. Avoid yes/no questions.\n\nTEXT CHUNK:\n{chunk_text}\n\nQuestions:"
  },
  "synthesis": {
    "system": "You respond with the briefest accurate answer possible—prefer a single word, never exceed two words—strictly grounded in the supplied evidence.",
    "user_template": "Question: {question}\nQuestion Type: {question_type}\n\nEvidence:\n{evidence}\n\nAnswer in one or two words, nothing more."
  },
  "evaluator": {
    "system": "You evaluate ultra-short QA pairs. Ensure the one- or two-word answer is relevant, clear, complete for the question scope, and factually supported.",
    "user_template": "Review this short QA pair:\n\nQuestion: {question}\nAnswer: {answer}\n\nEvidence:\n{evidence}\n\nScore 0.0-1.0 for relevance, clarity, completeness, factuality (in that order) assuming concise answers are expected. Format as\nrelevance: X.X\nclarity: X.X\ncompleteness: X.X\nfactuality: X.X"
  }
}

```

---

## 🔧 Development

### To/Do

```bash
1. Improve Evaluator Agent
2. Implement Bloom's Taxonomy properly
3. Add Deepseek OCR for consistant pdf support
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Collection name is empty or invalid"**
- Ensure `collection_name` in config has 3+ characters
- Use only alphanumeric, dots, underscores, or hyphens

**2. "API key not found"**
- Set environment variables for API keys
- Or add keys directly to `config.json` (not recommended for production)

**3. "ChromaDB persistence error"**
- Check write permissions for `persist_directory`
- Ensure directory exists or can be created

**4. Low-quality QA pairs**
- Increase `max_evidence_spans` for better context
- Use higher-quality LLM providers

---

## 📚 Documentation

- **[Complete Technical Report](bssc_qa_presentation.md)** - Detailed architecture and implementation
- **[Notebooks](notebooks/)** - Interactive tutorials with examples
- **[Config Reference](config.json)** - Full configuration options

---

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Keep changes focused and atomic

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with powerful open-source tools:

- **[LangChain](https://github.com/langchain-ai/langchain)** - Agent orchestration framework
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for embeddings
- **[Sentence Transformers](https://www.sbert.net/)** - State-of-the-art text embeddings

Special thanks to the open-source AI community.

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Kyzu07/BSSC_QA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Kyzu07/BSSC_QA/discussions)
- **Email**: shah.imran.1599@gmail.com

---

## 🚀 Roadmap

- [ ] Web UI for easier interaction
- [ ] Support for more document formats (PPT, Excel)
- [ ] Real-time streaming QA generation
- [ ] Multi-language support expansion
- [ ] Automated hyperparameter tuning
- [ ] Integration with annotation tools
- [ ] Pre-built datasets and benchmarks

---
