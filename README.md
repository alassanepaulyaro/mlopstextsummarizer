# MLOPS Text Summarizer Using Huggingface

The **MLOPS Text Summarizer** is an end-to-end machine learning project that implements a complete pipeline for text summarization. It includes several stages such as data ingestion, data transformation, model training, and model evaluation. The project is also equipped with a FastAPI application to provide a web interface for training and prediction.

## Pipeline Stages

The project pipeline is divided into the following stages:

1. **Data Ingestion Stage**  
   - **Description:** Downloads and extracts the source data.  
   - **Implementation:**  
     - Uses the `DataIngestion` class to download a zip file if it does not exist locally.
     - Extracts the downloaded zip file to a specified directory.
   - **Logging:**  
     - Logs when the data ingestion stage is initiated and completed.
     
2. **Data Transformation Stage**  
   - **Description:** Processes the ingested data and converts it into features suitable for model training.  
   - **Implementation:**  
     - The `DataTransformation` class uses a tokenizer (from Hugging Face) to convert text data into token IDs.
     - Maps the raw dataset into a tokenized dataset and saves it for later stages.
   - **Logging:**  
     - Logs the initiation and completion of the transformation stage.
     
3. **Model Trainer Stage**  
   - **Description:** Trains a text summarization model using a pre-trained checkpoint.  
   - **Implementation:**  
     - The `ModelTrainer` class leverages the Hugging Face `Trainer` API to fine-tune a model (e.g., Pegasus) on the transformed data.
     - Training configurations such as number of epochs, batch sizes, and logging steps are specified.
     - After training, the model and tokenizer are saved for inference.
   - **Logging:**  
     - Logs the start and successful completion of the model training stage.
     
4. **Model Evaluation Stage**  
   - **Description:** Evaluates the performance of the trained model on a test dataset using metrics like ROUGE.  
   - **Implementation:**  
     - The `ModelEvaluation` class generates predictions in batches and computes ROUGE scores by comparing predictions with reference summaries.
     - The evaluation results are saved in CSV format.
   - **Logging:**  
     - Logs the start and completion of the model evaluation stage.

## Application Endpoints

The project also provides a FastAPI application with the following endpoints:

- **GET /**  
  - **Description:** Redirects to the API documentation (Swagger UI).
  
- **GET /train**  
  - **Description:** Triggers the training pipeline by executing the main training script.
  
- **POST /predict**  
  - **Description:** Accepts a text payload and returns the summarized text using the prediction pipeline.


## Project Workflows 

1. Config.yaml
2. Params.yaml
3. Config entity
4. Configuration Manager
5. Update the components- Data Ingestion,Data Transformation, Model Trainer
6. Create our Pipeline-- Training Pipeline,PRediction Pipeline
7. Front end-- Api's, Training APi's, Batch Prtediction API's

## Installation and Setup

1. Create and Activate a Virtual Environment:
    > conda create -p venv python==3.12.7 -y
    > conda activate venv/

2. Install Dependencies
    > pip install -r requirements.txt

3. Running the Application
    > python app.py
    > uvicorn app:app --host 127.0.0.1 --port 8080  