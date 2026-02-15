"src/classifier.py - Document classification engine with batch processing"

from pathlib import Path
from library.models import Classifier
from library.utils import FileAnalyzer, FileOrganizer, TextExtractor
from typing import List


class DocClassifier:
    "Main classification engine"
    
    def __init__(self):
        self.organizer = None  # Set when processing starts
        self.extractor = TextExtractor()
        self.unorganized_category = "Unorganized"
        # Route low-confidence predictions to Unorganized (tunable tradeoff: automation vs precision).
        self.unorganized_threshold = 0.3
        # Use the same threshold for model acceptance (Others vs category) so that routing behavior is consistent:
        # if confidence >= unorganized_threshold, a concrete category may be returned and organized.
        self.classifier = Classifier(threshold=self.unorganized_threshold)
        
        # Load trained models
        if not self.classifier.load_models():
            print("[WARN] No trained model found. Run: python training/train_ai.py")
        else:
            print("[OK] ML models loaded successfully!")
    
    def classify_file(self, file_path, mode="copy", rename_mode=False, output_dir=None):
        "Classify single file"
        file_path = Path(file_path)

        # Ensure organizer is initialized for single-file classification
        if self.organizer is None:
            self.organizer = FileOrganizer(output_dir or ".")
        
        if not file_path.exists():
            return {"status": "error", "file": file_path.name, "message": "File not found"}
        
        # Check if empty
        if FileAnalyzer.is_empty(file_path):
            dest, success = self.organizer.organize(file_path, self.unorganized_category, mode, rename_mode)
            return {
                "status": "success" if success else "error",
                "file": file_path.name,
                "category": self.unorganized_category,
                "confidence": None,
                "destination": dest,
                "reason": "Empty file",
            }
        
        # Check if supported format
        if not FileAnalyzer.is_supported(file_path):
            dest, success = self.organizer.organize(file_path, self.unorganized_category, mode, rename_mode)
            return {
                "status": "success" if success else "error",
                "file": file_path.name,
                "category": self.unorganized_category,
                "confidence": None,
                "destination": dest,
                "reason": "Unsupported format",
            }
        
        # Extract text
        text = self.extractor.extract(str(file_path))
        if not text:
            dest, success = self.organizer.organize(file_path, self.unorganized_category, mode, rename_mode)
            return {
                "status": "success" if success else "error",
                "file": file_path.name,
                "category": self.unorganized_category,
                "confidence": None,
                "destination": dest,
                "reason": "No text extracted",
            }
        
        # Classify
        category, confidence = self.classifier.classify(text)
        reason = None
        if (
            category not in self.classifier.CATEGORIES
            or category == "Others"
            or confidence < self.unorganized_threshold
        ):
            if category not in self.classifier.CATEGORIES:
                reason = "Unknown category"
            elif category == "Others":
                reason = "Model uncertain (below acceptance threshold)"
            else:
                reason = "Low confidence"
            category = self.unorganized_category
        
        # Organize
        dest, success = self.organizer.organize(file_path, category, mode, rename_mode)
        
        result = {
            "status": "success" if success else "error",
            "file": file_path.name,
            "category": category,
            "confidence": confidence,
            "destination": dest
        }
        if reason:
            result["reason"] = reason
        return result
    
    def classify_batch(self, file_paths: List[str], mode="copy", output_dir=None, rename_mode=False, callback=None, cancel_check=None):
        "Classify multiple files"
        if output_dir:
            self.organizer = FileOrganizer(output_dir)
        elif not self.organizer:
            self.organizer = FileOrganizer(".")
        
        results = {
            "total": len(file_paths),
            "successful": 0,
            "failed": 0,
            "files": [],
            "summary": {},
            "cancelled": False,
            "processed": 0,
            "rename_mode": rename_mode
        }
        
        for idx, file_path in enumerate(file_paths):
            if cancel_check and cancel_check():
                results["cancelled"] = True
                break
            try:
                result = self.classify_file(file_path, mode, rename_mode)
                
                if result["status"] == "success":
                    results["successful"] += 1
                    category = result.get("category", "Unknown")
                    results["summary"][category] = results["summary"].get(category, 0) + 1
                else:
                    results["failed"] += 1
                
                results["files"].append(result)
                results["processed"] += 1
                
                if callback:
                    callback(idx + 1, len(file_paths))
            except Exception as e:
                results["failed"] += 1
                results["files"].append({"status": "error", "message": str(e)})
                results["processed"] += 1
        
        return results
    
    def classify_folder(self, folder_path, mode="copy", output_dir=None, callback=None, recursive=True):
        "Classify all files in folder"
        folder = Path(folder_path)
        if not folder.is_dir():
            return {"status": "error", "message": "Folder not found"}
        
        # Find all files
        if recursive:
            files = list(folder.rglob("*"))
        else:
            files = list(folder.glob("*"))
        
        files = [f for f in files if f.is_file()]
        
        return self.classify_batch(
            [str(f) for f in files],
            mode=mode,
            output_dir=output_dir,
            callback=callback,
        )
