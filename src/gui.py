"Desktop GUI for AI SmartDoc Classifier System"
import sys
import os
import json
import re
from pathlib import Path
from itertools import zip_longest
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTabWidget, QFileDialog, QProgressBar,
                             QTextEdit, QRadioButton, QButtonGroup, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QHeaderView)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
from src.classifier import DocClassifier

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
    SKLEARN_METRICS_AVAILABLE = True
except Exception:
    SKLEARN_METRICS_AVAILABLE = False


UNORGANIZED_LABEL = "Unorganized"


def _format_confidence(value):
    "Format classifier confidence as a user-friendly percentage"
    try:
        if value is None:
            return "N/A"
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def _average_confidence(file_results):
    "Compute average confidence across successful classified files"
    confidences = []
    for r in file_results or []:
        if isinstance(r, dict) and r.get("status") == "success" and "confidence" in r:
            try:
                confidences.append(float(r["confidence"]))
            except Exception:
                pass
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _safe_float(value):
    "Convert numeric input to float or return None."
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _canonical_eval_label(label, valid_labels):
    "Normalize a label into one of the valid evaluation labels."
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None

    alias_map = {
        "others": UNORGANIZED_LABEL,
        "unknown": UNORGANIZED_LABEL,
        "unorganized": UNORGANIZED_LABEL,
    }
    lookup = {str(item).lower(): str(item) for item in (valid_labels or [])}
    lowered = text.lower()

    if lowered in alias_map:
        lowered = alias_map[lowered].lower()

    return lookup.get(lowered)


def _infer_true_label_from_path(file_path, valid_labels):
    """Infer ground-truth from source path folder names.

    Rule: first matching parent folder name is used. Unknown/Others map to Unorganized.
    """
    if not file_path:
        return None

    try:
        path_obj = Path(file_path)
        for depth, parent in enumerate(path_obj.parents):
            if depth > 4:
                break
            matched = _canonical_eval_label(parent.name, valid_labels)
            if matched:
                return matched
    except Exception:
        return None
    return None


def _infer_true_label_from_filename(file_ref, valid_labels, keyword_map=None):
    """Infer a weak ground-truth label from filename text.

    Priority:
    1) Direct category name in filename
    2) Keyword score from category keyword list
    """
    if not file_ref:
        return None

    try:
        stem = Path(str(file_ref)).stem.lower()
    except Exception:
        stem = str(file_ref).lower()

    normalized = re.sub(r"[^a-z0-9]+", " ", stem).strip()
    if not normalized:
        return None

    # Skip Unorganized for filename-derived truth labels.
    candidate_labels = [lbl for lbl in (valid_labels or []) if lbl != UNORGANIZED_LABEL]

    # Domain aliases for filename-only labeling (helps templates/abbreviations/typos).
    alias_map = {
        "Agreement": ["agreement", "contract", "nda", "terms", "sla"],
        "Invoice": ["invoice", "bill", "receipt", "purchase order", "po", "gpo"],
        "Policy": ["policy", "procedure", "checklist", "guideline", "compliance"],
        "Proposal": ["proposal", "project definition", "project-definition", "sow", "scope"],
        "Report": ["report", "analysis", "summary", "assessment"],
        "Resume": ["resume", "resome", "cv", "curriculum vitae"],
        "Shipping": ["shipping", "shipment", "bill of lading", "tracking", "delivery"],
        "IT": ["software", "technical", "infrastructure", "system", "api"],
    }

    # 1) Direct category name hit.
    for label in candidate_labels:
        key = str(label).lower()
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            return label

    # 1b) Alias hit.
    for label in candidate_labels:
        for alias in alias_map.get(str(label), []):
            a = str(alias).strip().lower()
            if not a:
                continue
            if re.search(rf"\b{re.escape(a)}\b", normalized):
                return label

    # 2) Keyword score hit.
    scores = {}
    for label in candidate_labels:
        keywords = []
        if isinstance(keyword_map, dict):
            keywords = keyword_map.get(label) or keyword_map.get(str(label)) or []
        score = 0
        for kw in keywords:
            k = str(kw).strip().lower()
            if not k:
                continue
            if k in normalized:
                score += 1
        if score > 0:
            scores[label] = score

    if not scores:
        return None

    best_label, best_score = max(scores.items(), key=lambda x: x[1])
    ties = [lbl for lbl, sc in scores.items() if sc == best_score]
    if len(ties) > 1:
        return None
    return best_label


def _prob_vector_from_prediction(predicted_label, confidence, labels):
    "Build a probability-like vector from top-1 label and confidence."
    if not labels:
        return []

    labels = list(labels)
    predicted = _canonical_eval_label(predicted_label, labels) or (
        UNORGANIZED_LABEL if UNORGANIZED_LABEL in labels else labels[0]
    )

    conf = _safe_float(confidence)
    if conf is None:
        conf = 1.0
    conf = max(0.0, min(1.0, conf))

    if len(labels) == 1:
        return [1.0]

    base = (1.0 - conf) / (len(labels) - 1)
    probs = [base for _ in labels]
    probs[labels.index(predicted)] = conf
    return probs

class ProcessorThread(QThread):
    "Background processing thread"
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, classifier, files, mode, output_dir, rename_mode=False):
        super().__init__()
        self.classifier = classifier
        self.files = files
        self.mode = mode
        self.output_dir = output_dir
        self.rename_mode = rename_mode
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            results = self.classifier.classify_batch(
                self.files,
                mode=self.mode,
                output_dir=self.output_dir,
                rename_mode=self.rename_mode,
                callback=lambda c, t: self.progress.emit(c, t),
                cancel_check=lambda: self._cancelled
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class SmartDocGUI(QMainWindow):
    "Main GUI Window"
    
    def __init__(self):
        super().__init__()
        self.classifier = DocClassifier()
        self.selected_files = []
        self.processing_history = []  # Track processed files and destinations
        
        # Auto-detect OS and set appropriate default destination (portable)
        if os.name == 'nt':  # Windows
            self.output_dir = str(Path.home() / "Documents" / "AI_SmartDoc_Output")
        elif sys.platform == 'darwin':  # macOS
            self.output_dir = str(Path.home() / "Documents" / "AI_SmartDoc_Output")
        else:  # Linux and others
            self.output_dir = str(Path.home() / "AI_SmartDoc_Output")
        
        self.mode = "copy"  # Default mode
        self.rename_mode = False  # Default: no auto-rename
        # Remember last used paths for file browser - use safer defaults
        try:
            docs_path = str(Path.home() / "Documents")
            if not Path(docs_path).exists():
                docs_path = str(Path.home())
        except:
            docs_path = str(Path.home())
        
        self.last_file_path = docs_path
        self.last_folder_path = docs_path
        
        # Set up persistent history file
        self.history_file = Path.home() / ".ai_smartdoc_history.json"
        
        self.init_ui()
        # Load saved history after UI is initialized
        self.load_history_from_file()
    
    def init_ui(self):
        "Initialize UI"
        # Get current category count from dynamic system
        self.setWindowTitle(f"AI SmartDoc Classifier System")
        self.setGeometry(100, 100, 1100, 750)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("AI SmartDoc - Intelligent Document Organization")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Create 2-tab interface
        tabs = QTabWidget()
        tabs.addTab(self.create_setup_tab(), "Setup & Select")
        tabs.addTab(self.create_advanced_tab(), "Results")
        layout.addWidget(tabs)
        
        main_widget.setLayout(layout)
    
    def create_setup_tab(self):
        "Setup & File Selection Tab"
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Add spacing
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # SECTION 1: Destination Folder
        layout.addWidget(self.create_section_header("Destination Folder"))
        dest_layout = QHBoxLayout()
        self.dest_label = QLabel(self.output_dir + "\\")
        self.dest_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; border: 1px solid #ddd;")
        dest_layout.addWidget(self.dest_label, 1)
        btn_dest = QPushButton("Browse...")
        btn_dest.setMaximumWidth(120)
        btn_dest.setStyleSheet("padding: 8px 12px; border-radius: 5px;")
        btn_dest.clicked.connect(self.select_destination)
        dest_layout.addWidget(btn_dest)
        layout.addLayout(dest_layout)
        
        # SECTION 2: File Operation Mode
        layout.addWidget(self.create_section_header("File Operation Mode"))
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(8)
        self.mode_group = QButtonGroup()
        self.radio_copy = QRadioButton("Copy (Keep original)")
        self.radio_copy.setChecked(True)
        self.radio_copy.toggled.connect(lambda checked: self.set_mode("copy") if checked else None)
        self.mode_group.addButton(self.radio_copy, 0)
        mode_layout.addWidget(self.radio_copy)
        self.radio_move = QRadioButton("Move (Save space)")
        self.radio_move.toggled.connect(lambda checked: self.set_mode("move") if checked else None)
        self.mode_group.addButton(self.radio_move, 1)
        mode_layout.addWidget(self.radio_move)
        layout.addLayout(mode_layout)
        
        # SECTION 2B: Auto-Rename Option
        layout.addWidget(self.create_section_header("Auto-Rename Option"))
        from PyQt5.QtWidgets import QCheckBox
        self.checkbox_rename = QCheckBox("Auto-rename files")
        self.checkbox_rename.setChecked(False)
        self.checkbox_rename.stateChanged.connect(lambda state: self.set_rename_mode(state == 2))  # 2 = checked
        layout.addWidget(self.checkbox_rename)
        
        # SECTION 3: File Selection with Toggle & Action Buttons
        layout.addWidget(self.create_section_header("Select Files"))
        
        # Left side: Toggle buttons
        select_toggle_layout = QHBoxLayout()
        select_toggle_layout.setSpacing(10)
        
        # Left side: Radio buttons for selection mode
        self.select_mode_group = QButtonGroup()
        self.radio_select_files = QRadioButton("Select Individual Files")
        self.radio_select_files.setChecked(True)
        self.select_mode_group.addButton(self.radio_select_files, 0)
        
        self.radio_select_folder = QRadioButton("Select Entire Folder")
        self.select_mode_group.addButton(self.radio_select_folder, 1)
        
        toggle_group = QWidget()
        toggle_layout = QVBoxLayout()
        toggle_layout.setSpacing(8)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.addWidget(self.radio_select_files)
        toggle_layout.addWidget(self.radio_select_folder)
        toggle_group.setLayout(toggle_layout)
        
        select_toggle_layout.addWidget(toggle_group, 1)
        
        # Right side: Action buttons (Compact Design)
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(5)
        
        btn_add = QPushButton("+ Add")
        btn_add.setMaximumWidth(80)
        btn_add.setMaximumHeight(32)
        btn_add.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; border-radius: 4px; padding: 5px;")
        btn_add.clicked.connect(self.add_files)
        action_buttons_layout.addWidget(btn_add)
        
        btn_clear = QPushButton("✕ Clear")
        btn_clear.setMaximumWidth(80)
        btn_clear.setMaximumHeight(32)
        btn_clear.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; border-radius: 4px; padding: 5px;")
        btn_clear.clicked.connect(self.clear_files)
        action_buttons_layout.addWidget(btn_clear)
        
        action_buttons_layout.addStretch()
        
        select_toggle_layout.addLayout(action_buttons_layout, 0)
        layout.addLayout(select_toggle_layout)
        
        # SECTION 4: Selected Files List
        layout.addWidget(self.create_section_header("Selected Files"))
        self.file_list = QTextEdit()
        self.file_list.setReadOnly(True)
        self.file_list.setMinimumHeight(120)
        self.file_list.setStyleSheet("border: 1px solid #ddd; border-radius: 5px; padding: 8px;")
        layout.addWidget(self.file_list)
        
        # SECTION 5: START/Cancel Buttons
        action_layout = QHBoxLayout()
        self.start_btn_tab1 = QPushButton("START PROCESSING")
        self.start_btn_tab1.setMinimumHeight(45)
        self.start_btn_tab1.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;")
        self.start_btn_tab1.clicked.connect(self.start_processing)
        action_layout.addWidget(self.start_btn_tab1)
        
        self.cancel_btn_tab1 = QPushButton("CANCEL")
        self.cancel_btn_tab1.setMinimumHeight(45)
        self.cancel_btn_tab1.setMaximumWidth(140)
        self.cancel_btn_tab1.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;")
        self.cancel_btn_tab1.clicked.connect(self.cancel_processing)
        self.cancel_btn_tab1.setEnabled(False)
        action_layout.addWidget(self.cancel_btn_tab1)
        
        layout.addLayout(action_layout)
        
        # SECTION 6: Processing Progress
        layout.addWidget(self.create_section_header("Processing Progress"))
        self.progress_bar_tab1 = QProgressBar()
        self.progress_bar_tab1.setMinimumHeight(20)
        self.progress_bar_tab1.setStyleSheet("border: 1px solid #ddd; border-radius: 5px;")
        layout.addWidget(self.progress_bar_tab1)
        
        self.progress_label_tab1 = QLabel("Ready to process")
        self.progress_label_tab1.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.progress_label_tab1)
        
        # SECTION 7: Open Organized Folder Button
        self.open_folder_btn_tab1 = QPushButton("Open Organized Folder")
        self.open_folder_btn_tab1.setMinimumHeight(40)
        self.open_folder_btn_tab1.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 5px;")
        self.open_folder_btn_tab1.clicked.connect(self.open_results_folder)
        self.open_folder_btn_tab1.setEnabled(False)
        layout.addWidget(self.open_folder_btn_tab1)
        
        layout.addStretch()


        widget.setLayout(layout)
        return widget
    
    def create_advanced_tab(self):
        "Results Tab - Shows processing results and history"
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Add spacing
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # SECTION 1: Results Summary
        layout.addWidget(self.create_section_header("Results Summary"))
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(150)
        self.summary_text.setStyleSheet("border: 1px solid #ddd; border-radius: 5px; padding: 8px; background-color: #f9f9f9;")
        self.summary_text.setText("No processing yet. Use 'Setup & Select' tab to process documents.")
        layout.addWidget(self.summary_text)
        
        # SECTION 2: Processing History (Expandable Tree)
        # Header with buttons
        history_header_layout = QHBoxLayout()
        history_header_label = self.create_section_header("Scan History (Files → Destination)")
        history_header_layout.addWidget(history_header_label)
        history_header_layout.addStretch()
        
        btn_remove = QPushButton("Remove Selected")
        btn_remove.setMaximumWidth(130)
        btn_remove.setMaximumHeight(32)
        btn_remove.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold; border-radius: 4px; padding: 5px;")
        btn_remove.clicked.connect(self.remove_history_item)
        history_header_layout.addWidget(btn_remove)
        
        btn_clear = QPushButton("Clear All")
        btn_clear.setMaximumWidth(100)
        btn_clear.setMaximumHeight(32)
        btn_clear.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; border-radius: 4px; padding: 5px;")
        btn_clear.clicked.connect(self.clear_all_history)
        history_header_layout.addWidget(btn_clear)
        
        layout.addLayout(history_header_layout)
        
        # Create tree widget for history
        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabels(["Scan Date/Time", "Destination", "Success", "Failed", "Files", "Avg Confidence (%)", "Actions"])
        self.history_tree.setMinimumHeight(250)
        self.history_tree.setStyleSheet(
            "border: 1px solid #ddd; border-radius: 5px; padding: 0px; background-color: #f9f9f9;"
            "QTreeWidget::item { padding: 5px; height: 28px; }"
        )
        self.history_tree.setColumnCount(7)
        self.history_tree.setUniformRowHeights(False)
        self.history_tree.setSelectionMode(self.history_tree.SingleSelection)
        
        # Make columns resizable
        header = self.history_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        header.setSectionResizeMode(6, QHeaderView.Interactive)
        header.setStretchLastSection(False)
        
        # Initial empty message
        empty_item = QTreeWidgetItem(self.history_tree)
        empty_item.setText(0, "No scan history yet")
        
        layout.addWidget(self.history_tree)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_section_header(self, title):
        "Create a styled section header"
        label = QLabel(title)
        label.setFont(QFont("Arial", 11, QFont.Bold))
        label.setStyleSheet("color: #333; padding-top: 10px;")
        return label
    
    def add_files(self):
        "Add files to existing selection"
        if self.radio_select_files.isChecked():
            # Add individual files
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select Files to Add",
                self.last_file_path,
                "All Files (*);;PDF (*.pdf);;DOCX (*.docx);;XLSX (*.xlsx);;CSV (*.csv)"
            )
            if files:
                # Remember last used path
                self.last_file_path = str(Path(files[0]).parent)
                self.selected_files.extend(files)
                self.update_file_list()
        else:
            # Add folder
            folder = QFileDialog.getExistingDirectory(self, "Select Folder to Add", self.last_folder_path)
            if folder:
                # Remember last used path
                self.last_folder_path = folder
                files = list(Path(folder).rglob("*"))
                new_files = [str(f) for f in files if f.is_file()]
                self.selected_files.extend(new_files)
                self.update_file_list()
    
    def clear_files(self):
        "Clear all selected files"
        self.selected_files = []
        self.file_list.setText("Total files: 0\n\nNo files selected")
    
    
    def update_file_list(self):
        "Update file list display"
        text = f"Total files: {len(self.selected_files)}\n\n"
        # Show all files (scroll if too many)
        for f in self.selected_files:
            text += f"{Path(f).name}\n"
        self.file_list.setText(text)
    
    def set_mode(self, mode):
        "Set file operation mode (copy or move)"
        self.mode = mode
    
    def set_rename_mode(self, enabled):
        "Set auto-rename mode"
        self.rename_mode = enabled
    
    def start_processing(self):
        "Start processing"
        if not self.selected_files:
            # Show error in appropriate progress label
            if hasattr(self, 'progress_label_tab1'):
                self.progress_label_tab1.setText("[ERROR] No files selected!")
            if hasattr(self, 'progress_label'):
                self.progress_label.setText("[ERROR] No files selected!")
            return
        
        # Create output directory only when processing starts
        try:
            from pathlib import Path
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if hasattr(self, 'progress_label_tab1'):
                self.progress_label_tab1.setText(f"[ERROR] Error creating folder: {e}")
            return
        
        mode = self.mode if hasattr(self, 'mode') else "copy"
        
        # Disable both start buttons
        if hasattr(self, 'start_btn'):
            self.start_btn.setEnabled(False)
        if hasattr(self, 'start_btn_tab1'):
            self.start_btn_tab1.setEnabled(False)
        if hasattr(self, 'cancel_btn_tab1'):
            self.cancel_btn_tab1.setEnabled(True)
        
        # Update Tab 1 progress label
        if hasattr(self, 'progress_label_tab1'):
            self.progress_label_tab1.setText(f"[PROCESSING] Started...")
        
        self.thread = ProcessorThread(self.classifier, self.selected_files, mode, self.output_dir, self.rename_mode)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def cancel_processing(self):
        "Cancel processing"
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            if hasattr(self, 'progress_label_tab1'):
                self.progress_label_tab1.setText("[CANCEL] Cancelling...")
    
    def update_progress(self, current, total):
        "Update progress"
        # Avoid division by zero
        if total == 0:
            return
        
        percentage = int(current * 100 / total)
        
        # Also update Tab 1 progress
        if hasattr(self, 'progress_bar_tab1'):
            self.progress_bar_tab1.setValue(percentage)
        
        if hasattr(self, 'progress_label_tab1'):
            self.progress_label_tab1.setText(f"Processing: {current}/{total} files")

    def compute_scan_evaluation(self, results, file_paths):
        """Compute scan-time metrics when ground-truth labels are inferable.

        Ground truth is inferred from parent folder names of source files,
        with filename-keyword fallback when folder labels are absent.
        """
        total_files = len(file_paths or []) or len(results.get("files", []))
        evaluation = {
            "label_source": "source_path_parent_folder_or_filename_heuristic",
            "label_rule": (
                "Use source parent folder as label; if missing, infer from filename/category keywords. "
                "Unknown/Others map to Unorganized."
            ),
            "label_quality": "none",
            "quality_note": "No inferable labels.",
            "strict_ground_truth_files": 0,
            "proxy_label_files": 0,
            "default_label_files": 0,
            "metrics_available": False,
            "total_files": int(total_files),
            "evaluated_files": 0,
            "skipped_files": int(total_files),
            "coverage": 0.0,
            "accuracy": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
            "roc_auc_ovr_weighted": None,
            "confusion_labels": [],
            "confusion_matrix": [],
            "per_file": [],
        }

        if not SKLEARN_METRICS_AVAILABLE:
            evaluation["reason"] = "scikit-learn metrics unavailable in runtime environment"
            return evaluation

        known_categories = list(getattr(self.classifier.classifier, "CATEGORIES", []))
        eval_labels = []
        for label in known_categories + [UNORGANIZED_LABEL]:
            text = str(label).strip()
            if text and text not in eval_labels:
                eval_labels.append(text)

        evaluation["confusion_labels"] = eval_labels

        file_results = list(results.get("files", []))
        y_true = []
        y_pred = []
        y_scores = []
        keyword_map = getattr(self.classifier.classifier, "KEYWORDS", {}) or {}
        path_label_count = 0
        filename_label_count = 0
        default_label_count = 0

        for source_path, file_result in zip_longest(file_paths or [], file_results, fillvalue=None):
            if not isinstance(file_result, dict):
                continue

            predicted_label = (
                _canonical_eval_label(file_result.get("category"), eval_labels)
                or UNORGANIZED_LABEL
            )
            true_label = _infer_true_label_from_path(source_path, eval_labels)
            label_method = "path_parent_folder" if true_label is not None else None
            if true_label is None:
                true_label = _infer_true_label_from_filename(
                    source_path or file_result.get("file"),
                    eval_labels,
                    keyword_map=keyword_map,
                )
                if true_label is not None:
                    label_method = "filename_heuristic"
            if true_label is None:
                # Full-coverage fallback: treat unlabelable files as Unorganized for evaluation.
                true_label = UNORGANIZED_LABEL if UNORGANIZED_LABEL in eval_labels else predicted_label
                label_method = "default_unorganized"
            confidence = _safe_float(file_result.get("confidence"))

            row = {
                "source_path": str(source_path) if source_path else None,
                "file": file_result.get("file"),
                "predicted_label": predicted_label,
                "true_label": true_label,
                "label_method": label_method,
                "confidence": confidence,
                "evaluated": true_label is not None,
                "correct": None,
            }

            if true_label is not None:
                if label_method == "path_parent_folder":
                    path_label_count += 1
                elif label_method == "filename_heuristic":
                    filename_label_count += 1
                elif label_method == "default_unorganized":
                    default_label_count += 1
                row["correct"] = bool(true_label == predicted_label)
                y_true.append(true_label)
                y_pred.append(predicted_label)
                y_scores.append(_prob_vector_from_prediction(predicted_label, confidence, eval_labels))

            evaluation["per_file"].append(row)

        evaluated_files = len(y_true)
        evaluation["evaluated_files"] = evaluated_files
        evaluation["skipped_files"] = max(0, total_files - evaluated_files)
        evaluation["coverage"] = (evaluated_files / total_files) if total_files else 0.0
        evaluation["strict_ground_truth_files"] = path_label_count
        evaluation["proxy_label_files"] = filename_label_count
        evaluation["default_label_files"] = default_label_count

        if evaluated_files == 0:
            return evaluation

        if path_label_count > 0 and filename_label_count == 0 and default_label_count == 0:
            evaluation["label_quality"] = "strict"
            evaluation["quality_note"] = "All evaluated labels came from source folder ground truth."
        elif path_label_count > 0 and (filename_label_count > 0 or default_label_count > 0):
            evaluation["label_quality"] = "mixed"
            evaluation["quality_note"] = (
                "Metrics mix strict folder labels with proxy/default labels; interpret with caution."
            )
        elif filename_label_count > 0 and default_label_count == 0:
            evaluation["label_quality"] = "proxy_filename_only"
            evaluation["quality_note"] = (
                "Metrics are based only on filename-derived proxy labels and can be optimistic."
            )
        elif filename_label_count == 0 and default_label_count > 0:
            evaluation["label_quality"] = "default_unorganized_only"
            evaluation["quality_note"] = (
                "Unlabelable files were assigned Unorganized as fallback; metrics are weak proxies."
            )
        else:
            evaluation["label_quality"] = "proxy_plus_default"
            evaluation["quality_note"] = (
                "Metrics combine filename-derived labels and Unorganized defaults; use as rough proxy only."
            )

        evaluation["metrics_available"] = True
        evaluation["accuracy"] = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=eval_labels,
            average="weighted",
            zero_division=0,
        )
        evaluation["weighted_precision"] = float(precision)
        evaluation["weighted_recall"] = float(recall)
        evaluation["weighted_f1"] = float(f1)
        cm_labels = [
            lbl for lbl in eval_labels
            if (lbl in set(y_true)) or (lbl in set(y_pred))
        ]
        evaluation["confusion_labels"] = cm_labels
        evaluation["confusion_matrix"] = confusion_matrix(
            y_true,
            y_pred,
            labels=cm_labels,
        ).tolist()

        # Scan-time ROC-AUC is intentionally disabled because scan pipeline does not preserve
        # full calibrated class probability vectors for all classes.
        evaluation["roc_auc_ovr_weighted"] = None

        return evaluation

    def add_evaluation_tree_items(self, parent_item, evaluation):
        "Render evaluation metrics under a scan history entry."
        if not isinstance(evaluation, dict):
            return

        eval_parent = QTreeWidgetItem(parent_item)
        eval_parent.setText(0, "Evaluation Metrics")
        eval_parent.setText(
            4,
            f"{evaluation.get('evaluated_files', 0)}/{evaluation.get('total_files', 0)}"
        )

        if not evaluation.get("metrics_available"):
            note = QTreeWidgetItem(eval_parent)
            note.setText(0, "Not enough labeled files (folder labels or recognizable filenames) to compute metrics")
            return

        metric_rows = [
            ("Accuracy", evaluation.get("accuracy")),
            ("Precision (weighted)", evaluation.get("weighted_precision")),
            ("Recall (weighted)", evaluation.get("weighted_recall")),
            ("F1 (weighted)", evaluation.get("weighted_f1")),
        ]
        for metric_name, metric_value in metric_rows:
            row = QTreeWidgetItem(eval_parent)
            row.setText(0, metric_name)
            if metric_value is None:
                row.setText(5, "N/A")
            else:
                row.setText(5, f"{float(metric_value) * 100:.2f}%")

        labels = evaluation.get("confusion_labels", [])
        matrix = evaluation.get("confusion_matrix", [])
        if labels and matrix:
            cm_parent = QTreeWidgetItem(eval_parent)
            cm_parent.setText(0, "Confusion Matrix")
            cm_parent.setText(1, ", ".join(labels))
            for label, row_values in zip(labels, matrix):
                row = QTreeWidgetItem(cm_parent)
                row.setText(0, f"True {label}")
                row.setText(1, str(row_values))
    
    def processing_finished(self, results):
        "Processing finished"
        # Enable start button in Tab 1
        if hasattr(self, 'start_btn_tab1'):
            self.start_btn_tab1.setEnabled(True)
        if hasattr(self, 'cancel_btn_tab1'):
            self.cancel_btn_tab1.setEnabled(False)

        # Compute and attach evaluation metrics for this scan session.
        evaluation = self.compute_scan_evaluation(results, self.selected_files)
        results["evaluation"] = evaluation

        # Update summary for Results tab
        processed = results.get('processed', results['successful'] + results['failed'])
        status_label = "[CANCELLED]" if results.get('cancelled') else "[COMPLETE] 100%"
        summary = (
            f"{status_label} Processing Complete!\n\n"
            f"Total Files: {results['total']}\n"
            f"Processed: {processed}\n"
            f"[OK] Successful: {results['successful']}\n"
            f"[FAILED] Failed: {results['failed']}\n\n"
            "Category Breakdown:\n"
        )
        for cat, count in results['summary'].items():
            summary += f"- {cat}: {count}\n"

        if evaluation.get("metrics_available"):
            summary += (
                "\nEvaluation Metrics:\n"
                f"- Evaluated Files: {evaluation.get('evaluated_files', 0)}/{evaluation.get('total_files', 0)}\n"
                f"- Accuracy: {float(evaluation.get('accuracy', 0.0)) * 100:.2f}%\n"
                f"- Precision (weighted): {float(evaluation.get('weighted_precision', 0.0)) * 100:.2f}%\n"
                f"- Recall (weighted): {float(evaluation.get('weighted_recall', 0.0)) * 100:.2f}%\n"
                f"- F1 (weighted): {float(evaluation.get('weighted_f1', 0.0)) * 100:.2f}%\n"
            )
        else:
            summary += (
                "\nEvaluation Metrics: N/A\n"
                "Reason: source files are not in labeled folders and filenames did not match known categories/keywords.\n"
            )

        summary += "\nFINISHED"
        
        if hasattr(self, 'summary_text'):
            self.summary_text.setText(summary)
        
        # Update history with processing results (store timestamp and destination)
        self.update_history_tree(results)
        
        # Reset file list after processing
        self.clear_files()
        
        # Enable Open Folder button in Tab 1
        if hasattr(self, 'open_folder_btn_tab1'):
            self.open_folder_btn_tab1.setEnabled(True)
        
        # Show completion popup
        if results.get('cancelled'):
            QMessageBox.information(self, "Processing Cancelled", 
                                    f"Processing cancelled.\n\n"
                                    f"Processed: {processed}/{results['total']}\n"
                                    f"Successful: {results['successful']}\n"
                                    f"Failed: {results['failed']}")
        else:
            QMessageBox.information(self, "Processing Completed", 
                                    f"All {results['total']} files processed successfully!\n\n"
                                    f"Successful: {results['successful']}\n"
                                    f"Failed: {results['failed']}")
    
    def select_destination(self):
        "Select destination folder for organized files"
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select Destination Folder",
            self.output_dir if hasattr(self, 'output_dir') else str(Path.home() / "Documents")
        )
        if folder:
            # Always create AI_SmartDoc_Output inside the selected folder
            self.output_dir = str(Path(folder) / "AI_SmartDoc_Output")
            # Remember last destination path
            self.last_folder_path = folder
            if hasattr(self, 'dest_label'):
                self.dest_label.setText(self.output_dir + "\\")
    
    def processing_error(self, error):
        "Processing error"
        # Re-enable start button in Tab 1
        if hasattr(self, 'start_btn_tab1'):
            self.start_btn_tab1.setEnabled(True)
        if hasattr(self, 'cancel_btn_tab1'):
            self.cancel_btn_tab1.setEnabled(False)
        
        # Update Tab 1 progress label
        if hasattr(self, 'progress_label_tab1'):
            self.progress_label_tab1.setText(f"[ERROR] {error}")
    
    def update_history_tree(self, results):
        "Update processing history tree view with expandable entries"
        from datetime import datetime
        
        # Remove empty placeholder if this is the first entry
        if self.history_tree.topLevelItemCount() == 1:
            item = self.history_tree.topLevelItem(0)
            if item.text(0) == "No scan history yet":
                self.history_tree.takeTopLevelItem(0)
        
        # Create parent item for this scan session
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        parent = QTreeWidgetItem()
        parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
        parent.setCheckState(0, Qt.Unchecked)
        parent.setText(0, timestamp)
        parent.setText(1, self.output_dir)
        parent.setText(2, str(results['successful']))
        parent.setText(3, str(results['failed']))
        parent.setText(4, str(results['total']))
        avg_conf = _average_confidence(results.get("files"))
        parent.setText(5, _format_confidence(avg_conf))
        parent.setData(6, Qt.UserRole, self.output_dir)  # Store folder path
        
        # Create open folder button for this row
        btn_open = QPushButton("Open")
        btn_open.setMaximumWidth(70)
        btn_open.setMaximumHeight(24)
        btn_open.setStyleSheet("padding: 2px; font-size: 10px; border-radius: 3px;")
        folder_path = self.output_dir
        btn_open.clicked.connect(lambda: self.open_history_folder(folder_path))
        
        # Insert at the beginning (index 0) to show latest first
        self.history_tree.insertTopLevelItem(0, parent)
        self.history_tree.setItemWidget(parent, 6, btn_open)
        
        # Add category breakdown as child items
        category_parent = QTreeWidgetItem(parent)
        category_parent.setText(0, "Category Breakdown")
        
        for cat, count in sorted(results['summary'].items()):
            cat_item = QTreeWidgetItem(category_parent)
            cat_item.setText(0, f"{cat}")
            cat_item.setText(4, str(count))
        
        # Add scanned files as child items
        files_parent = QTreeWidgetItem(parent)
        files_parent.setText(0, "Scanned Files")

        file_paths = list(self.selected_files)
        file_results = list(results.get("files", []))
        rename_mode = results.get("rename_mode", False)
        for i, (filepath, file_result) in enumerate(zip(file_paths, file_results), 1):
            # Show new filename if auto-rename enabled and destination available
            if rename_mode and isinstance(file_result, dict) and file_result.get("destination"):
                filepath_display = file_result.get("destination")
                filename = Path(filepath_display).name
            else:
                filename = Path(filepath).name
            
            try:
                filesize = Path(filepath).stat().st_size
                size_str = self.format_filesize(filesize)
            except:
                size_str = "N/A"
            
            file_item = QTreeWidgetItem(files_parent)
            file_item.setText(0, f"{i}. {filename}")
            file_item.setText(1, size_str)
            conf_text = _format_confidence(file_result.get("confidence") if isinstance(file_result, dict) else None)
            reason = file_result.get("reason") if isinstance(file_result, dict) else None
            if conf_text == "N/A" and reason:
                file_item.setText(5, "N/A*")
                file_item.setToolTip(5, reason)
            else:
                file_item.setText(5, conf_text)

        # If for any reason we have more paths than results, still list them
        if len(file_paths) > len(file_results):
            for j, filepath in enumerate(file_paths[len(file_results):], len(file_results) + 1):
                filename = Path(filepath).name
                try:
                    filesize = Path(filepath).stat().st_size
                    size_str = self.format_filesize(filesize)
                except:
                    size_str = "N/A"
                file_item = QTreeWidgetItem(files_parent)
                file_item.setText(0, f"{j}. {filename}")
                file_item.setText(1, size_str)
                file_item.setText(5, "N/A")

        self.add_evaluation_tree_items(parent, results.get("evaluation"))

        # Expand only the parent item, NOT the category and files sections
        self.history_tree.expandItem(parent)
        
        # Store history entry
        self.processing_history.append({
            'timestamp': timestamp,
            'folder': self.output_dir,
            'results': results,
            'files': self.selected_files.copy()
        })
        
        # Save history to file for persistence
        self.save_history_to_file()
    
    def open_history_folder(self, folder_path):
        "Open a specific scan folder from history"
        folder = Path(folder_path)
        if folder.exists():
            os.startfile(str(folder))
        else:
            QMessageBox.warning(self, "Folder Not Found", f"The folder '{folder_path}' no longer exists.")
    
    def save_history_to_file(self):
        "Save history to persistent JSON file"
        try:
            history_data = []
            for entry in self.processing_history:
                history_data.append({
                    'timestamp': entry['timestamp'],
                    'folder': entry['folder'],
                    'results': entry['results'],
                    'files': entry['files']
                })
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def load_history_from_file(self):
        "Load history from persistent JSON file"
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                updated = False
                
                # Clear the empty placeholder
                if self.history_tree.topLevelItemCount() == 1:
                    item = self.history_tree.topLevelItem(0)
                    if item.text(0) == "No scan history yet":
                        self.history_tree.takeTopLevelItem(0)
                
                # Rebuild history from file
                for entry in history_data:
                    results = entry.get("results", {})
                    if isinstance(results, dict):
                        existing_eval = results.get("evaluation")
                        needs_rebuild = (
                            not isinstance(existing_eval, dict)
                            or "label_quality" not in existing_eval
                            or "strict_ground_truth_files" not in existing_eval
                            or "default_label_files" not in existing_eval
                        )
                        if needs_rebuild:
                            results["evaluation"] = self.compute_scan_evaluation(
                                results,
                                entry.get("files", []),
                            )
                            entry["results"] = results
                            updated = True
                    self.processing_history.append(entry)
                    self.display_history_item(entry)
                if updated:
                    self.save_history_to_file()
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def display_history_item(self, entry):
        "Display a single history item in the tree"
        # Create parent item for this scan session
        parent = QTreeWidgetItem()
        parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
        parent.setCheckState(0, Qt.Unchecked)
        parent.setText(0, entry['timestamp'])
        parent.setText(1, entry['folder'])
        parent.setText(2, str(entry['results'].get('successful', 0)))
        parent.setText(3, str(entry['results'].get('failed', 0)))
        parent.setText(4, str(entry['results'].get('total', 0)))
        avg_conf = _average_confidence(entry['results'].get('files'))
        parent.setText(5, _format_confidence(avg_conf))
        parent.setData(6, Qt.UserRole, entry['folder'])
        parent.setData(0, Qt.UserRole, entry['results'].get('rename_mode', False))
        
        # Create open folder button for this row
        btn_open = QPushButton("Open")
        btn_open.setMaximumWidth(70)
        btn_open.setMaximumHeight(24)
        btn_open.setStyleSheet("padding: 2px; font-size: 10px; border-radius: 3px;")
        folder_path = entry['folder']
        btn_open.clicked.connect(lambda: self.open_history_folder(folder_path))
        
        # Insert at the beginning (index 0) to show latest first
        self.history_tree.insertTopLevelItem(0, parent)
        self.history_tree.setItemWidget(parent, 6, btn_open)
        
        # Add category breakdown as child items
        category_parent = QTreeWidgetItem(parent)
        category_parent.setText(0, "Category Breakdown")
        
        for cat, count in sorted(entry['results'].get('summary', {}).items()):
            cat_item = QTreeWidgetItem(category_parent)
            cat_item.setText(0, f"{cat}")
            cat_item.setText(4, str(count))
        
        # Add scanned files as child items
        files_parent = QTreeWidgetItem(parent)
        files_parent.setText(0, "Scanned Files")

        file_paths = list(entry.get('files', []))
        file_results = list(entry.get('results', {}).get('files', []))
        rename_mode = entry.get('results', {}).get('rename_mode', False)
        for i, (filepath, file_result) in enumerate(zip(file_paths, file_results), 1):
            # Show new filename if auto-rename enabled and destination available
            if rename_mode and isinstance(file_result, dict) and file_result.get("destination"):
                filepath_display = file_result.get("destination")
                filename = Path(filepath_display).name
            else:
                filename = Path(filepath).name
            
            try:
                filesize = Path(filepath).stat().st_size
                size_str = self.format_filesize(filesize)
            except:
                size_str = "N/A"
            
            file_item = QTreeWidgetItem(files_parent)
            file_item.setText(0, f"{i}. {filename}")
            file_item.setText(1, size_str)
            conf_text = _format_confidence(file_result.get("confidence") if isinstance(file_result, dict) else None)
            reason = file_result.get("reason") if isinstance(file_result, dict) else None
            if conf_text == "N/A" and reason:
                file_item.setText(5, "N/A*")
                file_item.setToolTip(5, reason)
            else:
                file_item.setText(5, conf_text)

        if len(file_paths) > len(file_results):
            for j, filepath in enumerate(file_paths[len(file_results):], len(file_results) + 1):
                filename = Path(filepath).name
                try:
                    filesize = Path(filepath).stat().st_size
                    size_str = self.format_filesize(filesize)
                except:
                    size_str = "N/A"
                file_item = QTreeWidgetItem(files_parent)
                file_item.setText(0, f"{j}. {filename}")
                file_item.setText(1, size_str)
                file_item.setText(5, "N/A")

        self.add_evaluation_tree_items(parent, entry.get("results", {}).get("evaluation"))

        # Expand only the parent item, NOT the category and files sections
        self.history_tree.expandItem(parent)
    
    def remove_history_item(self):
        "Remove selected history items (with checkboxes)"
        # Find all checked items
        checked_timestamps = []
        for i in range(self.history_tree.topLevelItemCount()):
            item = self.history_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                checked_timestamps.append(item.text(0))
        
        if not checked_timestamps:
            QMessageBox.warning(self, "No Selection", "Please check the history items you want to remove.")
            return
        
        # Confirm deletion
        count = len(checked_timestamps)
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove {count} selected scan(s)? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Remove from history list
        self.processing_history = [h for h in self.processing_history if h['timestamp'] not in checked_timestamps]
        
        # Remove from tree (iterate backwards to avoid index issues)
        for i in range(self.history_tree.topLevelItemCount() - 1, -1, -1):
            item = self.history_tree.topLevelItem(i)
            if item.text(0) in checked_timestamps:
                self.history_tree.takeTopLevelItem(i)
        
        # Save updated history
        self.save_history_to_file()
        
        # Show empty message if no history left
        if self.history_tree.topLevelItemCount() == 0:
            empty_item = QTreeWidgetItem(self.history_tree)
            empty_item.setText(0, "No scan history yet")
        
        QMessageBox.information(self, "Success", f"Removed {count} scan(s) from history.")
    
    def clear_all_history(self):
        "Clear all history"
        reply = QMessageBox.question(
            self, 
            "Clear All History", 
            "Are you sure you want to delete all scan history? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.processing_history = []
            self.history_tree.clear()
            
            # Add empty message
            empty_item = QTreeWidgetItem(self.history_tree)
            empty_item.setText(0, "No scan history yet")
            
            # Save (delete) history file
            self.save_history_to_file()
    
    def format_filesize(self, bytes):
        "Format file size in human readable format"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.1f}{unit}"
            bytes /= 1024
        return f"{bytes:.1f}TB"
    
    def open_results_folder(self):
        "Open results folder"
        folder = Path(self.output_dir)
        if folder.exists():
            os.startfile(str(folder))


def run_gui():
    "Run GUI application"
    app = QApplication(sys.argv)
    window = SmartDocGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
