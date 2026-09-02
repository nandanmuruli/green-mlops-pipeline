import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import mlflow
from src.telemetry.energy_tracker import GreenTracker

def train_baseline():
    print("1. Loading local IMDb dataset...")
    # Load the data we saved with DVC
    dataset = load_from_disk("./data/raw/imdb")
    
    # SHRINK THE DATASET for a fast local test run
    small_train = dataset["train"].shuffle(seed=42).select(range(500))
    small_test = dataset["test"].shuffle(seed=42).select(range(100))

    print("2. Tokenizing the text (translating words to numbers)...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # This cuts long reviews off at 128 words so the computer reads faster
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = small_train.map(tokenize_function, batched=True)
    tokenized_test = small_test.map(tokenize_function, batched=True)

    print("3. Setting up the AI Model and MLflow Scoreboard...")
    # Load a blank DistilBERT model with 2 labels (Positive or Negative)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tell MLflow where to save the scoreboard data
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    mlflow.set_experiment("Green_MLOps_Baseline")

    # Set the rules for the classroom (Batch size, epochs, etc.)
    training_args = TrainingArguments(
        output_dir="./models/baseline",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        report_to="mlflow", # Sends accuracy/loss grades to MLflow
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    print("4. Starting Green Training Loop! 🚀")
    # We wrap the training process inside your electricity meter!
    with GreenTracker(project_name="imdb_baseline_run"):
        trainer.train()

    print("\nTraining Complete! Model saved to ./models/baseline and energy logged.")

if __name__ == "__main__":
    train_baseline()