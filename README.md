Name: CHEN JUN YUAN
Student ID: i25033940
Github: https://github.com/jy3940/AI-SmartDoc-Classifier-System/

# AI SmartDoc Classifier System
AI SmartDoc is a desktop document-organization system that extracts text from files, classifies each file into a related category folder, and copies or moves files into destination.

## 1) What this project includes
- Trained AI model artifacts (`library/trained_models/`) for direct inference.
- GUI application for non-technical users (`python app.py`).
- Optional CLI mode for batch runs.
- Retraining pipeline from CSV-based datasets (`training/train_ai.py`).

## 2) Features
- Multi-format input support: `PDF`, `DOCX`, `XLSX`, `CSV`, `JSON`, `XML`, `TXT`, `PNG`, `JPG`, `JPEG`.
- Hybrid classification logic:
  - Keyword rules (fast path)
  - TF-IDF + neural network (main model)
  - Confidence-based fallback to `Unorganized`
- Auto organization by category folder:
  - `Agreement`, `Invoice`, `IT`, `Policy`, `Proposal`, `Report`, `Resume`, `Shipping`, `Unorganized`.
- GUI history log saved to `~/.ai_smartdoc_history.json`.

## 3) Project structure

```text
AI_smartdoc/
|-- .vscode/
|   |-- settings.json
|-- library/
|   |-- models.py
|   |-- utils.py
|   |-- categories.json
|   `-- trained_models/
|       |-- tfidf.pkl
|       `-- neural_model.h5
|-- src/
|   |-- gui.py
|   `-- classifier.py
|-- training/
|   |-- train_ai.py
|   `-- datasets/
|       `-- <Category>/<Category>.csv
|-- app.py
|-- README.md
|-- requirements.txt
|-- Sample Unsorted Files (Sample unsorted files inside this folder can be used to test with the AI SmartDoc Classifier System and verify the results)
```

## 4) System requirements

- OS: Windows 10/11, Linux/macOS also possible.
- Python: `3.12.6` recommended.
- RAM: minimum 8 GB and above.

OCR requirements:
- Tesseract OCR installed and available in `PATH`.
- Poppler installed for PDF-to-image OCR (`pdf2image`).

## 5) Installation

This section fixes the common issue where one PC has multiple Python versions and `pip` installs into the wrong interpreter.

### A) For computer havent the following requirement (run once):

```powershell
winget install --id Python.Python.3.12 --version 3.12.6 --exact --accept-package-agreements --accept-source-agreements
winget install --id Python.Launcher --exact --accept-package-agreements --accept-source-agreements
winget install --id Microsoft.VCRedist.2015+.x64 --exact --accept-package-agreements --accept-source-agreements
winget install --id tesseract-ocr.tesseract --exact --accept-package-agreements --accept-source-agreements
winget install --id oschwartz10612.Poppler --exact --accept-package-agreements --accept-source-agreements
```
### B) Project environment setup (always use venv Python path)

```powershell
cd C:\<your-path-location>\<Project_Name>

Example:
cd C:\python\i25033940_Chen Jun Yuan_AI-SmartDoc-Classifier-System
```

Create venv with explicit Python 3.12 path (recommended, works even if `py` is missing):

```powershell
$PY312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (!(Test-Path $PY312)) { $PY312 = "C:\Program Files\Python312\python.exe" }
& $PY312 -m venv venv
```

Alternative (only if `py` launcher works):

```powershell
py -3.12 -m venv venv
```

Install dependencies into this venv:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
```

Optional: activate venv (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Deactivate when done:

```powershell
deactivate
```

### C) Verify environment before running (If version no recognize refer to section 10: Troubleshooting)

```
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -c "import numpy,pandas,sklearn,tensorflow,PyQt5,PyPDF2,docx,pytesseract,pdf2image; print('Environment OK')"
tesseract --version
pdftoppm -v
```

## 6) Run the system

Always run with venv Python:

```powershell
.\venv\Scripts\python.exe app.py
```
## 7) Train or retrain the model

If model files are missing or you want retrain with own dataset (CSV):

```powershell
.\venv\Scripts\python.exe training\train_ai.py
```

### Required dataset format

Folder-per-category structure:

```
training/datasets/
|-- Invoice/Invoice.csv
|-- Resume/Resume.csv
|-- Agreement/Agreement.csv
`-- ...
```

Each CSV must contain at least a `text` column:

```csv
filename,text
sample_001.txt,"invoice number INV-1001 total amount due RM 3500 ..."
sample_002.txt,"payment terms net 30 ..."
```

Notes:
- `filename` is optional.
- Empty `text` rows are ignored.
- Category name is taken from folder name.

Training outputs:
- `library/trained_models/neural_model.h5`
- `library/trained_models/tfidf.pkl`
- `library/categories.json`

## 8) Sample data and public dataset download links (CSV)

You can use the included dataset in `training/datasets/` directly, or download public datasets:

1. The RVL-CDIP Dataset test (tif format, after download can convert tif to CSV format using AI like chatgpt)  
   https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test
2. Resume Classification Dataset (CSV format, Hugging Face)  
   https://huggingface.co/datasets/C0ldSmi1e/resume-dataset
3. Financial Documents Dataset (CSV format, Hugging Face)  
   https://huggingface.co/datasets/Adityaaaa468/Financial_Documents_dataset


### How to adapt external CSVs to this AI SmartDoc system

1. Keep/create a `text` column containing document text.
2. Create or use existing category folders under `training/datasets/`.
3. Save one CSV per category with name `<Category>.csv` inside that category folder.
4. Run `.\venv\Scripts\python.exe training\train_ai.py` to train.

## 9) Reproducibility checklist

1. Install Python `3.12.6` (Python Launcher optional).
2. Install dependencies from `requirements.txt` inside venv.
3. Install OCR tools (Tesseract and Poppler).
4. Run app with `.\venv\Scripts\python.exe app.py`.
5. Optional: run retraining with `.\venv\Scripts\python.exe training\train_ai.py`.
6. Verify outputs under `AI_SmartDoc_Output` and model artifacts under `library/trained_models/`.

## 10) Troubleshooting

- `py` not recognized:
  - Install launcher: `winget install --id Python.Launcher --exact`.
  - Or use explicit Python 3.12.
- `pip install -r requirements.txt` tries to compile NumPy:
  - Recreate venv using Section 5B and install using `.\venv\Scripts\python.exe -m pip ...`.
- `Failed to load the native TensorFlow runtime` or DLL errors:
  - Install VC++ runtime: `winget install --id Microsoft.VCRedist.2015+.x64 --exact`.
  - Reinstall in fresh venv from Section 5.
- `No trained model found`:
  - Run `.\venv\Scripts\python.exe training\train_ai.py`.
- OCR returns empty text:
  - Verify `tesseract --version` and `pdftoppm -v`.
  - If `tesseract` is not recognized:
    ```powershell
    $TESS = "C:\Program Files\Tesseract-OCR"
    if (!(Test-Path "$TESS\tesseract.exe")) { $TESS = "C:\Program Files (x86)\Tesseract-OCR" }
    & "$TESS\tesseract.exe" --version
    $env:Path += ";$TESS"
    ```
