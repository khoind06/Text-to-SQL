# High-Performance Single-Agent Text-to-SQL Engine

[![Spider Benchmark](https://img.shields.io/badge/Spider%20Dev-78.72%25%20EX-brightgreen)](https://yale-lily.github.io/spider)
[![LLM](https://img.shields.io/badge/LLM-Qwen2.5--Coder--7B-blue)](https://ollama.com/library/qwen2.5-coder:7b)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, high-accuracy **Text-to-SQL** system powered by `qwen2.5-coder:7b` and LangGraph. Designed to achieve state-of-the-art zero-shot performance across complex cross-domain relational databases, validated on the renowned **Yale Spider Benchmark**.

---

## 🌟 Key Highlights & Benchmark Results

Evaluated on the full **Yale Spider Validation Set (1,034 queries across 20+ real-world relational databases)** without any manual prompt engineering or dataset fine-tuning:

| Metric | Result | Notes |
| :--- | :---: | :--- |
| **Execution Accuracy (EX)** | **78.72%** | **814 / 1,034 samples** exact result match |
| **SQL Generation Success Rate** | **98.26%** | Syntactically valid SQLite queries |
| **Average Query Latency** | **1.55s** | End-to-end (LLM inference + schema extraction + execution) |
| **AST Security Interception** | **100%** | Zero destructive/injection queries executed |

### Error Taxonomy Breakdown (220 Non-Matching Samples)
- **Column / Aggregation Granularity Discrepancy (45.5%)**: Model queried valid results with slight projection variations (e.g., querying `T1.name, T2.score` instead of `T1.name` only, or returning unique records without `DISTINCT`).
- **Missing / Incorrect JOIN Keys (27.3%)**: Complex 3+ table transitive joins where foreign keys were implicitly named.
- **Complex Subquery / Set Operations (18.2%)**: Nested `EXCEPT` / `INTERSECT` vs. `NOT IN` variations.
- **Syntactic / Execution Error (9.1%)**: SQLite column not found or syntax mismatch.

---

## 🏛️ System Architecture

```
                                      [User Natural Language Question]
                                                     │
                                                     ▼
                                       ┌───────────────────────────┐
                                       │    Relational Database    │
                                       │     Schema Extractor      │
                                       └─────────────┬─────────────┘
                                                     │ DDL, PK/FKs, Sample Rows
                                                     ▼
┌───────────────────────────┐          ┌───────────────────────────┐
│     qwen2.5-coder:7b      │ ◄─────── │        Agent Node         │
│      (Ollama Local)       │ ───────► │  (Prompt & Chain Context) │
└───────────────────────────┘          └─────────────┬─────────────┘
                                                     │ Candidate SQL
                                                     ▼
                                       ┌───────────────────────────┐
                                       │     Risk Guardrails       │
                                       │   (AST sqlglot Parser)    │
                                       └─────────────┬─────────────┘
                                                     │ Approved Read-Only SQL
                                                     ▼
                                       ┌───────────────────────────┐
                                       │  SQLite Execution Engine  │
                                       │   (Read-Only URI Mount)   │
                                       └─────────────┬─────────────┘
                                                     │
                                                     ▼
                                           [Structured Data Result]
```

### 1. Dynamic Relational Schema Introspection
- Reads table structures directly via SQLite `sqlite_master`.
- Resolves Primary Keys, Foreign Keys (`PRAGMA foreign_key_list`), and extracts representative sample rows (3 rows per table) to provide deep semantic context for foreign join resolution.

### 2. AST Risk Guardrails (`src/security/risk_guardrails.py`)
- Employs **`sqlglot`** abstract syntax tree parsing to guarantee query safety.
- Strictly blocks non-`SELECT` statements (`DROP`, `DELETE`, `UPDATE`, `ALTER`, `INSERT`).
- Rejects multi-statement injections and SQL comments used for syntax evasion.

### 3. Isolated Read-Only Execution Engine (`src/engine/execution_engine.py`)
- Mounts SQLite databases using `file:<db_path>?mode=ro&immutable=1`.
- Enforces statement execution timeouts and automatic transaction rollbacks.

---

## 📁 Repository Structure

```
├── data/
│   ├── spider/
│   │   └── dev_gold.json            # 1,034 Gold queries for validation
├── docs/
│   └── spider_evaluation_results.json # Full benchmark scorecard & execution logs
├── scripts/
│   ├── download_spider.py           # Automated Spider dataset downloader
│   └── evaluate_spider.py           # Full validation benchmark runner
├── src/
│   ├── agents/
│   │   ├── orchestrator.py          # LangGraph single-agent graph workflow
│   │   └── sql_generator.py         # LLM Prompt generation & schema grounding
│   ├── engine/
│   │   └── execution_engine.py      # SQLite read-only query executor
│   ├── security/
│   │   └── risk_guardrails.py       # AST-based SQL query sanitizer
│   └── utils/
│       └── logger.py                # Logging configuration
├── tests/
│   ├── test_spider_agent.py         # End-to-end workflow verification
│   └── test_sql_generator.py        # Schema & prompt unit tests
├── requirements.txt                 # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.10 or higher
- **Ollama**: Running locally with `qwen2.5-coder:7b` installed:
  ```bash
  ollama run qwen2.5-coder:7b
  ```

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/khoind06/Text-to-SQL.git
cd Text-to-SQL
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download Spider Dataset
Download and extract the Spider dataset databases into `data/spider/database/`:
```bash
python scripts/download_spider.py
```

### 4. Run Unit Tests
Verify the pipeline, schema parser, and guardrails:
```bash
pytest tests/ -v
```

---

## 📊 Reproducing the Benchmark

To evaluate the system against all 1,034 questions from the Spider evaluation dataset:

```bash
python scripts/evaluate_spider.py --all
```

*Key evaluation arguments:*
- `--all`: Run on all 1,034 validation samples.
- `--samples N`: Run on a subset of N samples for quick checks (e.g. `--samples 50`).
- `--checkpoint`: Progressively checkpoint evaluation data every 10 samples to `docs/spider_evaluation_results.json`.

---

## 🛡️ Security & Guardrails

The engine implements defense-in-depth:
1. **Abstract Syntax Tree (AST) Validation**: Any destructive statement (`DROP`, `ALTER`, `TRUNCATE`, `UPDATE`, `DELETE`) is immediately flagged and aborted before reaching the database.
2. **Read-Only SQLite Engine**: Databases are opened with `mode=ro` connection flags, preventing disk writes even in case of prompt injection attacks.
3. **Deterministic Clean Output**: Markdown code blocks and hallucinated explanatory tokens are automatically stripped, ensuring only pure SQL is emitted.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
