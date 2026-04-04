# DealAgent 🤖

An autonomous AI deal hunting system that scans the internet for discounted products, estimates their true market value using a fine-tuned LLM, and sends real-time push notifications when a significant bargain is found.

---

## What It Does

DealAgent runs continuously in the background. Every few minutes it:

1. Scrapes deal RSS feeds from the internet and filters the 5 most promising listings
2. Estimates the true market value of each deal using an ensemble of two AI models
3. Compares the estimated value against the listed price
4. Sends a push notification to your phone when a deal is significantly underpriced
5. Displays all discovered deals in a live web dashboard

---

## System Architecture

```
Internet RSS Feeds
        ↓
ScannerAgent        — scrapes and filters deals using GPT
        ↓
EnsembleAgent       — combines two price estimators
    ├── SpecialistAgent   — fine-tuned Llama 3.2 deployed on Modal.com
    └── FrontierAgent     — RAG pipeline with ChromaDB + frontier LLM
        ↓
AutonomousPlanningAgent  — orchestrates the full pipeline with tool calling
        ↓
MessengerAgent      — sends push notification to your phone via Pushover
        ↓
React.js Web Dashboard   — live deal tracking interface
```

### Agents

| Agent | Role |
|---|---|
| ScannerAgent | Scrapes RSS feeds and uses GPT to select the 5 most promising deals with clear prices |
| SpecialistAgent | Calls the fine-tuned Llama 3.2 3B model deployed on Modal.com for price estimation |
| FrontierAgent | Uses ChromaDB to find similar products then passes them to a frontier LLM for context-aware estimation |
| EnsembleAgent | Averages estimates from SpecialistAgent and FrontierAgent for higher accuracy |
| AutonomousPlanningAgent | GPT-5.1 powered brain that autonomously decides when to scan, estimate, and notify |
| MessengerAgent | Delivers push notifications to your phone via Pushover API |

---

## How the Price Prediction Works

The price estimation model was built across multiple stages:

- **Baselines** — Linear Regression, Random Forest, XGBoost trained on 800,000+ Amazon products
- **Neural Network** — Custom 8-layer PyTorch network as a deep learning baseline
- **Zero-shot LLMs** — GPT-4, Claude, Gemini evaluated with no training
- **Fine-tuned Llama 3.2 3B** — QLoRA fine-tuning on 800,000 product descriptions, outperforming all baselines and zero-shot frontier models including GPT-5.1

The fine-tuned model is deployed as a live REST API on Modal.com and combined with a RAG pipeline backed by ChromaDB for ensemble estimation.

---

## Tech Stack

| Category | Tools |
|---|---|
| ML & Training | PyTorch, Scikit-learn, HuggingFace, QLoRA, PEFT |
| LLMs | OpenAI GPT, Anthropic Claude, Meta Llama 3.2 |
| Vector Database | ChromaDB |
| Deployment | Modal.com |
| Data | HuggingFace Datasets, Groq Batch API |
| Frontend | React.js |
| Notifications | Pushover API |
| Backend | Python, LiteLLM |

---

## Setup

### Prerequisites

- Python 3.12+
- Modal.com account
- OpenAI API key
- HuggingFace account and token
- Pushover account (for push notifications)

### Installation

```bash
# Clone the repo
git clone https://github.com/your_username/DealAgent
cd DealAgent

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
PUSHOVER_USER=your_pushover_user_key
PUSHOVER_TOKEN=your_pushover_app_token
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

### Modal Setup

```bash
# Authenticate with Modal
modal token set --token-id your_token_id --token-secret your_token_secret

# Add HuggingFace secret to Modal
modal secret create huggingface-secret HF_TOKEN=your_hf_token

# Deploy the fine-tuned Llama model
modal deploy -m pricer_service2
```

### Running the App

```bash
python price_is_right.py
```

The web dashboard will open automatically in your browser.

---

## Data

The price prediction model was trained on 800,000+ Amazon product descriptions across 8 categories processed and hosted on HuggingFace:

- Raw dataset: `ed-donner/items_raw_full`
- Preprocessed dataset: `ed-donner/items_full`
- Prompt dataset for fine-tuning: `ed-donner/items_prompts_full`

---

## Model Training

Fine-tuning was done on Google Colab using a T4 GPU (lite mode) or A100 (full mode). See the training notebook in `fine_tuning/` for full details. The fine-tuned model weights are hosted privately on HuggingFace.

---

## Project Structure

```
DealAgent/
├── agents/
│   ├── specialist_agent.py      # Fine-tuned Llama inference via Modal
│   ├── frontier_agent.py        # RAG pipeline with ChromaDB
│   ├── ensemble_agent.py        # Combines both estimators
│   ├── scanner_agent.py         # RSS feed scraping and filtering
│   ├── messaging_agent.py       # Push notifications via Pushover
│   └── planning_agent.py        # Autonomous orchestration agent
├── baseline/                    # Traditional ML and neural network models
├── fine_tuning/                 # QLoRA training notebooks
├── frontend/                    # React.js web dashboard
├── products_vectorstore/        # ChromaDB vector database
├── src/                         # Core utilities and helpers
├── hello.py                     # Modal setup verification
├── llama.py                     # Base Llama on Modal
├── pricer_service2.py           # Deployed fine-tuned pricer
├── deal_agent_framework.py      # Main agent orchestration framework
├── price_is_right.py            # App entry point
├── memory.json                  # Persistent deal memory
├── .env                         # Environment variables (not committed)
└── requirements.txt
```

---

## Acknowledgements

Built following the LLM Engineering course by Ed Donner. The dataset is sourced from the McAuley-Lab Amazon Reviews 2023 collection on HuggingFace.
