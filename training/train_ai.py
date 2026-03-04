import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from library.models import AdvancedNeuralClassifier
from library.utils import FileAnalyzer, TextExtractor
import csv


class DatasetTrainer:
    """Load training data from folder structure"""
    
    def __init__(self):
        # Resolve datasets relative to the project root so training works from any CWD.
        training_datasets = PROJECT_ROOT / "training" / "datasets"
        local_datasets = PROJECT_ROOT / "datasets"
        
        if training_datasets.exists():
            self.training_data_dir = training_datasets
        elif local_datasets.exists():
            self.training_data_dir = local_datasets
        else:
            self.training_data_dir = local_datasets
        
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
    def train_from_folder_structure(self):
        """Load from organized folder structure"""
        print("Loading training data from folder structure...")
        
        texts = []
        labels = []
        
        for category_dir in self.training_data_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            
            category = category_dir.name
            file_count = 0
            
            csv_file = category_dir / f"{category}.csv"
            if csv_file.exists():
                try:
                    with csv_file.open("r", encoding="utf-8", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            text = (row.get("text") or "").strip()
                            if text:
                                texts.append(text)
                                labels.append(category)
                                file_count += 1
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")

            for data_file in category_dir.iterdir():
                if not data_file.is_file():
                    continue
                if data_file.suffix.lower() == ".csv":
                    continue
                if not FileAnalyzer.is_supported(data_file):
                    continue
                try:
                    text = TextExtractor.extract(str(data_file)).strip()
                    if text:
                        texts.append(text)
                        labels.append(category)
                        file_count += 1
                except Exception as e:
                    print(f"Error reading {data_file}: {e}")
            
            if file_count > 0:
                print(f"  {category}: {file_count} samples")
        
        if texts:
            return texts, labels
        else:
            print("No training data found in training/datasets/")
            return None, None


def main():
    """Main training function"""
    
    try:
        parser = argparse.ArgumentParser(description="Train AI SmartDoc classifier")
        parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
        parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
        args = parser.parse_args()

        print("Training AI Model\n")
        
        trainer = DatasetTrainer()
        texts, labels = trainer.train_from_folder_structure()
        
        if not texts:
            print("No training data found!")
            sys.exit(1)
        
        print(f"\nLoaded {len(texts)} training samples\n")
        
        print("Training neural network model...")
        model = AdvancedNeuralClassifier()

        # Sync categories from dataset folders for flexible training
        dataset_categories = sorted(set(labels))
        model.CATEGORIES = dataset_categories

        # Keep keywords only for categories present in dataset
        model.KEYWORDS = {
            category: model.KEYWORDS.get(category, [category.lower()])
            for category in dataset_categories
        }
        model.save_categories()

        result = model.train(texts, labels, epochs=args.epochs, batch_size=args.batch_size)
        
        print("\nModel trained successfully!")
        if isinstance(result, dict) and result.get("status") == "success":
            print(
                "Metrics (held-out test set): "
                f"Accuracy={result.get('test_accuracy', 0.0):.2%}, "
                f"F1={result.get('weighted_f1', 0.0):.2%}"
            )
        print("Model saved to: library/trained_models/\n")
        
        print("Categories recognized:")
        categories = list(set(labels))
        for i, cat in enumerate(sorted(categories), 1):
            count = labels.count(cat)
            print(f"  {i}. {cat} ({count} samples)")
        
        print("\nNext: python app.py")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
