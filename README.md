
# DataSleuth 🔍

**A local-first AI-powered data analysis platform built with Python and Qwen3.**

DataSleuth is an intelligent data analysis project designed to help users explore datasets, discover patterns, identify anomalies, and ask analytical questions using a locally running Large Language Model.

The project combines deterministic Python-based analysis with local AI interpretation to produce explainable, evidence-backed results.

> 🚧 DataSleuth is actively under development.

## ✨ Implemented Features

### Dataset Analysis
- CSV dataset loading
- Automated dataset profiling
- Missing-value detection
- Duplicate detection
- Numerical and categorical analysis
- Outlier detection
- Data quality checks

### Automated Intelligence
- Automated insight generation
- Skewness detection
- Correlation analysis
- Categorical imbalance detection
- Constant-column detection

### Local AI Integration
- Local LLM integration using Ollama
- Qwen3 8B model support
- Structured AI responses
- Evidence-based claim verification
- AI-generated result explanations

### Natural Language Analytics
- Natural-language dataset queries
- Structured query interpretation
- Safe aggregation execution
- Natural-language filtering
- Multiple filter conditions
- AND/OR logical operations
- Filtered aggregations

## 🧠 Architecture

DataSleuth follows a hybrid architecture:

1. **Data layer:** Loads and prepares datasets using pandas.
2. **Analysis layer:** Performs profiling, quality checks and insight detection.
3. **AI layer:** Uses locally hosted Qwen3 to interpret natural-language questions.
4. **Execution layer:** Validates structured plans and executes calculations using Python.
5. **Verification layer:** Checks AI-generated numerical claims against analytical evidence.

The LLM interprets questions; Python performs the actual calculations.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Ollama
- Qwen3 8B
- Pytest

## ⚙️ Requirements

- Python 3.10 or newer
- Ollama installed locally
- Qwen3 8B model
- Git

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sreeragsajitha-bit/DataSleuth.git
cd DataSleuth
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 🤖 Local AI Setup

Install Ollama from:

https://ollama.com/

Download Qwen3 8B:

```bash
ollama pull qwen3:8b
```

Start the model:

```bash
ollama run qwen3:8b
```

DataSleuth uses the local Ollama service to interpret analytical questions and explain calculated results.

No paid AI API is required.

## ▶️ Running DataSleuth

Run the analysis pipeline:

```bash
python src/analysis/data_pipeline.py
```

Run natural-language dataset queries:

```bash
python src/analysis/query_runner.py
```

Run natural-language filtering:

```bash
python src/analysis/filter_runner.py
```

Run filtered aggregation:

```bash
python src/analysis/filtered_aggregation.py
```

## 🧪 Testing

Run the complete test suite:

```bash
pytest src/analysis -v
```

Latest verified result:

**161 tests passed.**

## 🗺️ Development Roadmap

- [x] Dataset loading
- [x] Automated dataset profiling
- [x] Data quality engine
- [x] Automated insight engine
- [x] Unified analysis pipeline
- [x] Local AI integration and claim verification
- [x] Natural-language dataset querying
- [x] Intelligent filtering and filtered aggregation
- [ ] Unified natural-language analytics
- [ ] Advanced statistical analysis
- [ ] Machine learning intelligence
- [ ] Multi-dataset analysis
- [ ] Automated report generation
- [ ] Interactive dashboard
- [ ] Benchmarking and public release

## 🔒 Privacy and Design Philosophy

DataSleuth is designed around local-first processing.

- No paid AI APIs.
- No requirement for cloud-based LLM inference.
- Dataset calculations are performed locally.
- AI-generated numerical claims are checked against analytical evidence.

## 👨‍💻 Project Status

Actively developed as a learning and engineering project.

More features, tests and documentation are being added incrementally.
