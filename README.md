# MLOps Text Summarizer Using Hugging Face Transformers

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-orange.svg)](https://huggingface.co/transformers/)

An end-to-end MLOps pipeline for text summarization using the Google Pegasus model fine-tuned on the SAMSum dataset. This project demonstrates best practices in ML engineering, including modular code structure, configuration management, automated pipelines, and API deployment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Model Information](#model-information)
- [Development Workflow](#development-workflow)
- [Contributing](#contributing)

## Overview

The **MLOps Text Summarizer** is a production-ready machine learning system that automatically generates concise summaries from longer texts. Built with modern MLOps principles, this project showcases:

- **End-to-end ML Pipeline**: From data ingestion to model deployment
- **Modular Architecture**: Reusable components following SOLID principles
- **Configuration Management**: YAML-based configuration for easy experimentation
- **API-First Design**: RESTful API using FastAPI for seamless integration
- **Reproducibility**: Structured logging and artifact management

The system uses the **Google Pegasus-CNN/DailyMail** model, fine-tuned on the **SAMSum** conversational dataset for dialogue summarization tasks.

## Features

- **Automated ML Pipeline**: Four-stage pipeline (ingestion, transformation, training, evaluation)
- **Pre-trained Model Fine-tuning**: Leverages Hugging Face Pegasus model
- **RESTful API**: FastAPI endpoints for training and inference
- **Comprehensive Logging**: Structured logging throughout the pipeline
- **Configurable Training**: Easy parameter tuning via YAML files
- **Model Evaluation**: ROUGE score metrics for performance assessment
- **Dockerized Deployment**: Container-ready application
- **Modular Components**: Easy to extend and customize

## Project Structure

```
mlopstextsummarizer/
├── .github/
│   └── workflows/            # CI/CD pipeline configurations
├── artifacts/                # Generated artifacts (models, data, metrics)
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
├── config/
│   └── config.yaml          # Pipeline configuration
├── research/                 # Jupyter notebooks for experimentation
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_transformation.ipynb
│   ├── 3_model_trainer.ipynb
│   ├── 4_model_evaluation.ipynb
│   └── mlopstextsummarizer.ipynb
├── src/
│   └── mlopstextsummarizer/
│       ├── components/       # Core pipeline components
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── config/           # Configuration management
│       │   └── configuration.py
│       ├── constants/        # Project constants
│       ├── entity/           # Data classes and entities
│       ├── pipeline/         # Pipeline orchestration
│       │   ├── stage_1_data_ingestion_pipeline.py
│       │   ├── stage_2_data_transformation_pipeline.py
│       │   ├── stage_3_model_trainer_pipeline.py
│       │   ├── stage_4_model_evaluation.py
│       │   └── predicition_pipeline.py
│       ├── logging/          # Logging configuration
│       └── utils/            # Utility functions
├── app.py                   # FastAPI application
├── main.py                  # Pipeline execution script
├── params.yaml              # Training hyperparameters
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── Dockerfile              # Docker configuration
└── README.md               # Project documentation
```

## Architecture

The project follows a layered architecture:

```
┌─────────────────────────────────────┐
│         FastAPI Application         │
│         (app.py)                    │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│       Pipeline Orchestration        │
│  (stage_1, stage_2, stage_3, stage_4) │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│         Core Components             │
│  (DataIngestion, DataTransformation,│
│   ModelTrainer, ModelEvaluation)    │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│    Configuration Management         │
│    (config.yaml, params.yaml)       │
└─────────────────────────────────────┘
```

## Pipeline Stages

### 1. Data Ingestion Stage

**Purpose**: Downloads and extracts the SAMSum dataset for training.

**Key Features**:
- Automatic dataset download from remote sources
- ZIP file extraction and validation
- Artifact storage in structured directories
- Comprehensive error handling and logging

**Implementation**: [data_ingestion.py](src/mlopstextsummarizer/components/data_ingestion.py)

**Configuration**:
```yaml
data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: <dataset_url>
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion
```

### 2. Data Transformation Stage

**Purpose**: Tokenizes raw text data into model-compatible format.

**Key Features**:
- Hugging Face tokenizer integration (Pegasus)
- Batch processing for efficient transformation
- Train/validation/test split handling
- Tokenized dataset persistence

**Implementation**: [data_transformation.py](src/mlopstextsummarizer/components/data_transformation.py)

**Configuration**:
```yaml
data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/samsum_dataset
  tokenizer_name: google/pegasus-cnn_dailymail
```

### 3. Model Training Stage

**Purpose**: Fine-tunes the Pegasus model on the SAMSum dataset.

**Key Features**:
- Pre-trained model loading from Hugging Face
- Customizable training arguments (epochs, batch size, learning rate)
- Gradient accumulation for memory efficiency
- Model and tokenizer checkpoint saving
- Training metrics logging

**Implementation**: [model_trainer.py](src/mlopstextsummarizer/components/model_trainer.py)

**Training Parameters** (from [params.yaml](params.yaml)):
```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1000000
  gradient_accumulation_steps: 16
```

### 4. Model Evaluation Stage

**Purpose**: Assesses model performance using ROUGE metrics.

**Key Features**:
- ROUGE-1, ROUGE-2, ROUGE-L score calculation
- Batch prediction for efficiency
- CSV export of evaluation metrics
- Comparison with reference summaries

**Implementation**: [model_evaluation.py](src/mlopstextsummarizer/components/model_evaluation.py)

**Output**: Evaluation metrics saved to `artifacts/model_evaluation/metrics.csv`

## Installation

### Prerequisites

- Python 3.12.7 or higher
- Conda (recommended) or virtualenv
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/alassanepaulyaro/mlopstextsummarizer.git
cd mlopstextsummarizer
```

### Step 2: Create Virtual Environment

**Using Conda (Recommended)**:
```bash
conda create -p venv python==3.12.7 -y
conda activate venv/
```

**Using venv**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Package

```bash
pip install -e .
```

## Usage

### Running the Complete Training Pipeline

Execute all four pipeline stages sequentially:

```bash
python main.py
```

This will:
1. Download and extract the dataset
2. Transform and tokenize the data
3. Train the model
4. Evaluate model performance

### Starting the FastAPI Application

```bash
# Method 1: Using the app directly
python app.py

# Method 2: Using uvicorn
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Access the application at: `http://localhost:8080`

### Using the Prediction Pipeline

```python
from src.mlopstextsummarizer.pipeline.predicition_pipeline import PredictionPipeline

# Initialize pipeline
predictor = PredictionPipeline()

# Generate summary
text = "Your long text to summarize goes here..."
summary = predictor.predict(text)
print(summary)
```

## API Documentation

Once the application is running, access the interactive API documentation:

- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

### API Endpoints

#### 1. Root Endpoint

```http
GET /
```

**Description**: Redirects to the API documentation (Swagger UI).

**Response**: Redirects to `/docs`

#### 2. Training Endpoint

```http
GET /train
```

**Description**: Triggers the complete training pipeline.

**Response**:
```json
{
  "message": "Training successful !!"
}
```

**Note**: This endpoint executes the entire pipeline (data ingestion, transformation, training, evaluation) and may take considerable time depending on your hardware.

#### 3. Prediction Endpoint

```http
POST /predict
```

**Description**: Generates a summary for the provided text.

**Request Body**:
```json
{
  "text": "Your long text to summarize..."
}
```

**Response**:
```json
{
  "summary": "Generated summary text..."
}
```

**Example using cURL**:
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"Your text here"}'
```

**Example using Python**:
```python
import requests

url = "http://localhost:8080/predict"
payload = {"text": "Your long text to summarize..."}
response = requests.post(url, json=payload)
print(response.json())
```

## Configuration

### Main Configuration ([config/config.yaml](config/config.yaml))

Defines paths and settings for each pipeline stage:

- **artifacts_root**: Base directory for all generated artifacts
- **data_ingestion**: Dataset source URL and storage paths
- **data_transformation**: Tokenizer and data paths
- **model_trainer**: Model checkpoint and training data paths
- **model_evaluation**: Evaluation data and output paths

### Training Parameters ([params.yaml](params.yaml))

Controls model training behavior:

- **num_train_epochs**: Number of training iterations
- **per_device_train_batch_size**: Batch size per GPU/CPU
- **gradient_accumulation_steps**: Steps before updating weights
- **warmup_steps**: Learning rate warmup steps
- **weight_decay**: L2 regularization factor
- **logging_steps**: Frequency of logging metrics
- **evaluation_strategy**: When to evaluate (steps/epoch)

## Model Information

### Base Model

**Model**: [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)

**Architecture**: Pegasus (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence)

**Training Data**: CNN/DailyMail news articles (original pre-training)

### Fine-tuning Dataset

**Dataset**: SAMSum Corpus

**Description**: A corpus of conversational dialogues with human-written summaries, ideal for dialogue summarization tasks.

**Size**: 16,000+ messenger-like conversations

### Performance Metrics

Model performance is evaluated using ROUGE scores:

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

Results are saved in `artifacts/model_evaluation/metrics.csv`.

## Development Workflow

The project follows a structured development approach:

1. **Research Phase**: Experimentation in Jupyter notebooks ([research/](research/))
   - Prototype pipeline stages
   - Test different models and hyperparameters
   - Validate approach before implementation

2. **Implementation Phase**: Convert notebooks to production code
   - Create modular components in [src/mlopstextsummarizer/components/](src/mlopstextsummarizer/components/)
   - Define configuration entities
   - Implement configuration manager

3. **Pipeline Integration**: Orchestrate components into pipelines
   - Create stage-specific pipelines in [src/mlopstextsummarizer/pipeline/](src/mlopstextsummarizer/pipeline/)
   - Implement main training script ([main.py](main.py))

4. **API Development**: Build FastAPI application
   - Create training and prediction endpoints ([app.py](app.py))
   - Add API documentation

5. **Deployment**: Containerize and deploy
   - Docker configuration ([Dockerfile](Dockerfile))
   - CI/CD pipelines ([.github/workflows/](.github/workflows/))

### Development Commands

```bash
# Run specific pipeline stage
python -c "from src.mlopstextsummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline; DataIngestionTrainingPipeline().initiate_data_ingestion()"

# Run tests (if implemented)
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Run pre-commit hooks
pre-commit install
```

## Author

**Yaro Alassane Paul**
- Email: alassane-paul.yaro@outlook.fr
- GitHub: [@alassanepaulyaro](https://github.com/alassanepaulyaro)

## License

This project is open-source and available for educational and commercial use.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library and model hosting
- [SAMSum Dataset](https://huggingface.co/datasets/samsum) creators
- Google Research for the Pegasus model
- FastAPI team for the excellent web framework

---

**Note**: This project is designed for educational purposes and demonstrates MLOps best practices. For production deployment, consider additional aspects such as model monitoring, A/B testing, and advanced CI/CD pipelines.  