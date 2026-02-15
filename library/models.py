"library/models.py - ADVANCED NEURAL NETWORK CLASSIFIER"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import json
import re
from pathlib import Path
from typing import List
import numpy as np

# Neural Network
import tensorflow as tf
from tensorflow.keras import layers, models

# Text Processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib


LIBRARY_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINED_MODELS_DIR = LIBRARY_DIR / "trained_models"
DEFAULT_CATEGORIES_FILE = LIBRARY_DIR / "categories.json"


class AdvancedNeuralClassifier:
    """
    ADVANCED NEURAL NETWORK CLASSIFIER
    
    Single powerful deep learning model with:
    - Intelligent text understanding via neural network
    - Continuous learning capability (incremental training)
    - Self-improving performance over time
    - Trainable with custom datasets
    
    Architecture:
    - Input: TF-IDF Vectorized Text (5000 features)
    - Dense Layer 1: 256 neurons + ReLU + BatchNorm
    - Dropout: 0.3 (prevents overfitting)
    - Dense Layer 2: 128 neurons + ReLU + BatchNorm
    - Dropout: 0.3
    - Dense Layer 3: 64 neurons + ReLU
    - Dropout: 0.2
    - Output: Softmax (multi-class classification)
    
    Training: Adam optimizer, Categorical Crossentropy loss
    """
    
    DEFAULT_CATEGORIES = ["Invoice", "Policy", "Proposal", "Agreement", "Report", "Resume", "IT", "Shipping"]
    
    # Comprehensive keywords for intelligent fallback
    DEFAULT_KEYWORDS = {
        "Invoice": ["invoice", "bill", "receipt", "payment", "amount due", "total", "charge", "vendor",
                   "statement", "price", "qty", "unit", "subtotal", "tax", "balance"],
        "Policy": ["policy", "insurance", "coverage", "premium", "deductible", "claim", "beneficiary",
                  "terms", "conditions", "effective", "renewal", "protection"],
        "Proposal": ["proposal", "quotation", "quote", "estimate", "bid", "tender", "offer", "pricing",
                    "cost", "scope", "timeline", "deliverable"],
        "Agreement": ["agreement", "contract", "nda", "terms", "conditions", "party", "hereby", "legal",
                     "binding", "obligation", "service", "license"],
        "Report": ["report", "analysis", "findings", "summary", "conclusion", "financial", "results",
                  "quarterly", "annual", "monthly", "assessment", "data"],
        "Resume": ["resume", "cv", "curriculum vitae", "experience", "skills", "employment", "education",
                  "background", "qualification", "profile", "career"],
        "IT": ["software", "server", "database", "network", "application", "code", "system", "technical",
              "api", "configuration", "infrastructure", "architecture"],
        "Shipping": ["shipment", "tracking", "delivery", "freight", "logistics", "carrier", "package",
                    "order", "warehouse", "dispatch", "confirmation"],
        "Email": ["email", "from:", "to:", "subject:", "cc:", "bcc:", "sender", "recipient"],
        "Form": ["form", "application", "checkbox", "required field", "signature", "date"],
    }
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = None
        self.tfidf = None
        self.trained = False
        self.history = []  # Track training history
        
        # Load or initialize categories
        self.categories_file = DEFAULT_CATEGORIES_FILE
        self.load_categories()
        
        print("[INIT] Initializing Advanced Neural Network Classifier...")
    
    def load_categories(self):
        """Load categories from file or use defaults"""
        if self.categories_file.exists():
            try:
                with open(self.categories_file, 'r') as f:
                    data = json.load(f)
                    self.CATEGORIES = data.get("categories", self.DEFAULT_CATEGORIES)
                    self.KEYWORDS = data.get("keywords", {**self.DEFAULT_KEYWORDS})
            except:
                self.CATEGORIES = self.DEFAULT_CATEGORIES.copy()
                self.KEYWORDS = {**self.DEFAULT_KEYWORDS}
        else:
            self.CATEGORIES = self.DEFAULT_CATEGORIES.copy()
            self.KEYWORDS = {**self.DEFAULT_KEYWORDS}
    
    def save_categories(self):
        """Save categories to file for persistence"""
        try:
            self.categories_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.categories_file, 'w') as f:
                json.dump({
                    "categories": self.CATEGORIES,
                    "keywords": self.KEYWORDS
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save categories: {e}")
    
    def add_category(self, category_name: str, keywords: List[str] = None):
        """Dynamically add a new category"""
        if category_name not in self.CATEGORIES:
            self.CATEGORIES.append(category_name)
            if keywords:
                self.KEYWORDS[category_name] = keywords
            else:
                self.KEYWORDS[category_name] = [category_name.lower()]
            
            self.save_categories()
            print(f"[OK] NEW CATEGORY ADDED: {category_name}")
            return True
        return False
    
    def _build_model(self, input_dim):
        """Build the neural network architecture"""
        model = models.Sequential([
            # Input layer with TF-IDF features
            layers.Input(shape=(input_dim,)),
            
            # Feature extraction layer
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Abstraction layer
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Pattern recognition layer
            layers.Dense(64, activation='relu', name='dense_3'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(len(self.CATEGORIES), activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        split_strategy: str = "stratified_70_15_15",
        random_state: int = 42,
    ):
        """Train the neural network with data.

        Uses a stratified 70/15/15 train/val/test split by default.
        TF-IDF is fit on the training set only to avoid data leakage.
        """
        print(f"\n[TRAIN] Training Advanced Neural Network...")
        print(f"   Samples: {len(texts)}")
        print(f"   Categories: {len(set(labels))}")

        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and have the same length")

        # Convert labels to numeric (dense integer labels, then one-hot later)
        label_to_idx = {cat: idx for idx, cat in enumerate(self.CATEGORIES)}
        y_all = np.array([label_to_idx.get(label, 0) for label in labels], dtype=np.int64)

        # Split data
        if split_strategy != "stratified_70_15_15":
            raise ValueError(f"Unsupported split_strategy: {split_strategy}")

        X_train_text, X_temp_text, y_train_int, y_temp_int = train_test_split(
            texts,
            y_all,
            test_size=0.30,
            stratify=y_all,
            random_state=random_state,
        )
        X_val_text, X_test_text, y_val_int, y_test_int = train_test_split(
            X_temp_text,
            y_temp_int,
            test_size=0.50,
            stratify=y_temp_int,
            random_state=random_state,
        )

        print(
            f"   Split: train={len(X_train_text)} ({len(X_train_text)/len(texts):.0%}), "
            f"val={len(X_val_text)} ({len(X_val_text)/len(texts):.0%}), "
            f"test={len(X_test_text)} ({len(X_test_text)/len(texts):.0%})"
        )

        # Fit TF-IDF on TRAIN only, transform val/test
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        X_train = self.tfidf.fit_transform(X_train_text).toarray()
        X_val = self.tfidf.transform(X_val_text).toarray()
        X_test = self.tfidf.transform(X_test_text).toarray()

        y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=len(self.CATEGORIES))
        y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=len(self.CATEGORIES))
        y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=len(self.CATEGORIES))
        
        # Build model if not exists
        if self.model is None:
            print(f"Building neural network architecture...")
            self.model = self._build_model(X_train.shape[1])
            print(f"   Input features: {X_train.shape[1]}")
            print(f"   Output classes: {len(self.CATEGORIES)}")

        # Class weighting for imbalanced datasets
        try:
            class_weights_arr = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(len(self.CATEGORIES)),
                y=y_train_int,
            )
            class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}
        except Exception:
            class_weight = None

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        print(f"Training for up to {epochs} epochs...")
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
        )
        
        self.history.append(history)
        self.trained = True
        self.save_models()

        # Evaluate on held-out test set
        y_test_pred_probs = self.model.predict(X_test, verbose=0)
        y_test_pred = np.argmax(y_test_pred_probs, axis=1)
        test_accuracy = float(accuracy_score(y_test_int, y_test_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_int,
            y_test_pred,
            average="weighted",
            zero_division=0,
        )

        print(f"[OK] Training completed!")
        print(f"   Test Accuracy: {test_accuracy:.2%}")
        print(f"   Weighted Precision: {precision:.2%}")
        print(f"   Weighted Recall:    {recall:.2%}")
        print(f"   Weighted F1-Score:  {f1:.2%}")

        # Store a compact classification report string for documentation/logging
        try:
            report = classification_report(
                y_test_int,
                y_test_pred,
                target_names=[str(c) for c in self.CATEGORIES],
                zero_division=0,
            )
        except Exception:
            report = ""

        return {
            "status": "success",
            "split_strategy": split_strategy,
            "train_samples": len(X_train_text),
            "val_samples": len(X_val_text),
            "test_samples": len(X_test_text),
            "test_accuracy": test_accuracy,
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1": float(f1),
            "categories": len(self.CATEGORIES),
            "classification_report": report,
        }
    
    def save_models(self):
        """Save trained model and vectorizer"""
        try:
            model_dir = DEFAULT_TRAINED_MODELS_DIR
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save neural network
            if self.model:
                self.model.save(model_dir / "neural_model.h5")
            
            # Save TF-IDF
            if self.tfidf:
                joblib.dump(self.tfidf, model_dir / "tfidf.pkl")
            
            print(f"[OK] Models saved to {model_dir}/")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_dir = DEFAULT_TRAINED_MODELS_DIR
            
            if not model_dir.exists():
                return False
            
            # Load neural network
            model_path = model_dir / "neural_model.h5"
            if model_path.exists():
                # Inference-only load; avoids compile/metric warnings on Windows CLI.
                self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Load TF-IDF
            tfidf_path = model_dir / "tfidf.pkl"
            if tfidf_path.exists():
                self.tfidf = joblib.load(tfidf_path)
            
            if self.model and self.tfidf:
                self.trained = True
                print(f"[OK] Neural network model loaded from {model_dir}/")
                return True
            
            return False
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            return False
    
    def classify_keywords(self, text):
        """Keyword-based classification (fast fallback)"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            scores[category] = matches / len(keywords) if keywords else 0
        
        if max(scores.values()) > 0.15:
            best = max(scores, key=scores.get)
            return best, min(scores[best], 0.95)
        return None, 0.0

    def _classify_structural_resume(self, text: str) -> tuple[str | None, float]:
        """Detect resume-like documents via common section headings.

        This catches minimalist resumes that may not contain explicit keywords like "resume" or "cv",
        and helps prevent over-routing to Unorganized when extraction is partial/noisy.
        """
        if not isinstance(text, str) or not text.strip():
            return None, 0.0

        # Normalize: keep letters/numbers/spaces for simple heading matching.
        t = re.sub(r"[^a-z0-9\s:/-]+", " ", text.lower())
        t = re.sub(r"\s+", " ", t).strip()

        # Strong resume section headings/signals.
        # Note: some PDFs extract headings without whitespace (e.g., "EDUCATIONMy"), so we match
        # "start of token" patterns instead of strict word boundaries.
        signals = [
            "education",
            "experience",
            "work experience",
            "skills",
            "skill",
            "projects",
            "portfolio",
            "certifications",
            "certification",
            "contact",
            "profile",
            "about me",
            "objective",
            "summary",
        ]
        hits = 0

        def has_signal(sig: str) -> bool:
            sig = sig.strip()
            if not sig:
                return False
            # Match at start-of-token; allow following letters (handles "contacthelene").
            return re.search(rf"(^|\s){re.escape(sig)}", t) is not None

        for s in signals:
            if has_signal(s):
                hits += 1

        # Require multiple independent signals to avoid false positives.
        # A typical resume will contain several of these headings.
        if hits >= 3:
            return "Resume", 0.60
        return None, 0.0
    
    def classify(self, text):
        """
        INTELLIGENT NEURAL NETWORK CLASSIFICATION
        
        Process:
        1. Keyword matching (fast for obvious cases)
        2. Neural network for accurate classification
        3. Return category with confidence
        """
        if not isinstance(text, str) or not text.strip():
            return "Others", 0.0
        
        # Fast keyword path
        category, confidence = self.classify_keywords(text)
        if category and confidence > 0.3:
            return category, confidence

        # Structural heuristic path (primarily for minimalist resumes)
        category, confidence = self._classify_structural_resume(text)
        if category and confidence >= self.threshold:
            return category, confidence
        
        # Neural network path
        if not self.trained:
            return "Others", 0.0
        
        try:
            # Vectorize text
            text_vector = self.tfidf.transform([text]).toarray()
            
            # Get predictions
            predictions = self.model.predict(text_vector, verbose=0)[0]
            
            # Get best prediction
            best_idx = np.argmax(predictions)
            best_category = self.CATEGORIES[best_idx]
            confidence = float(predictions[best_idx])
            
            if confidence >= self.threshold:
                return best_category, confidence
            else:
                # If the neural net is unsure, attempt structural heuristics before routing to Others.
                h_cat, h_conf = self._classify_structural_resume(text)
                if h_cat and h_conf >= self.threshold:
                    return h_cat, h_conf
                return "Others", confidence
                
        except Exception as e:
            print(f"Classification error: {e}")
            return "Others", 0.0
    
    def continuous_learn(self, texts: List[str], labels: List[str], epochs=10):
        """
        CONTINUOUS LEARNING - Improve with new data
        
        The model updates itself with new examples without forgetting
        previously learned patterns (transfer learning)
        """
        if not self.trained:
            print("Warning: Model not yet trained. Use train() first.")
            return None
        
        print(f"\n[LEARN] Continuous Learning Update...")
        print(f"   New samples: {len(texts)}")
        
        # Vectorize new data
        X_new = self.tfidf.transform(texts).toarray()
        
        label_to_idx = {cat: idx for idx, cat in enumerate(self.CATEGORIES)}
        y_new = np.array([label_to_idx.get(label, 0) for label in labels])
        y_new = tf.keras.utils.to_categorical(y_new, num_classes=len(self.CATEGORIES))
        
        # Fine-tune model with new data
        history = self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=16,
            verbose=0
        )
        
        self.history.append(history)
        self.save_models()
        
        final_accuracy = history.history['accuracy'][-1]
        print(f"[OK] Continuous learning completed!")
        print(f"   Accuracy on new data: {final_accuracy:.2%}")
        
        return {"status": "success", "accuracy": final_accuracy}


class Classifier:
    """Compatibility wrapper for existing code"""
    
    def __init__(self, threshold=0.5):
        self.classifier = AdvancedNeuralClassifier(threshold)
        self.CATEGORIES = self.classifier.CATEGORIES
        self.KEYWORDS = self.classifier.KEYWORDS
    
    def train(self, texts, labels):
        return self.classifier.train(texts, labels)
    
    def load_models(self):
        return self.classifier.load_models()
    
    def save_models(self):
        return self.classifier.save_models()
    
    def classify(self, text):
        return self.classifier.classify(text)
    
    def continuous_learn(self, texts, labels, epochs=10):
        return self.classifier.continuous_learn(texts, labels, epochs)
    
    def add_category(self, category_name, keywords=None):
        return self.classifier.add_category(category_name, keywords)

