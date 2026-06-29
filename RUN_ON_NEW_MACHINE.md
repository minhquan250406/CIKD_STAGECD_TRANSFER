# RUN ON NEW MACHINE - CIKD Stage C/D Training Package Setup

This guide explains how to restore and run Stage C baselines and Stage D CIKD model training on a new machine equipped with an NVIDIA RTX 4070 Ti Super 16GB VRAM.

## Step-by-Step Setup

### 1. Restore Project Folder Structure
* Copy the entire package folder to the target machine.
* **Recommended Location**: To avoid path mismatch issues, rename/copy the folder to:
  `D:\CIKD`
  If you keep it in another folder (e.g. `D:\CIKD_STAGECD_TRANSFER`), update all absolute paths in scripts if any script assumes `D:\CIKD`.

### 2. Create the Python Virtual Environment
Navigate to the root of the project folder (e.g. `D:\CIKD`) and run:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Package Dependencies
Install the frozen requirements:
```bash
pip install -r requirements_stage_cd.txt
```
If the frozen requirements fail due to system-specific mismatches, install the dependencies manually:
```bash
pip install numpy pandas scikit-learn matplotlib tqdm pillow transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 4. Verify the Package Integrity
Verify the package files and caches are fully complete and intact:
```powershell
powershell -ExecutionPolicy Bypass -File VERIFY_PACKAGE.ps1
```
This will run the PowerShell audit and then run the Python shape validation script.

### 5. Optional Feature Cache Smoke Audit
Verify feature cached arrays have proper distributions and labels align:
```bash
.venv\Scripts\python.exe src\stage_b_smoke_audit.py
```

### 6. Run Stage C (Baselines)
Train the Stage C baseline models:
```bash
.venv\Scripts\python.exe src\run_stage_cd.py --stage C
```

### 7. Run Stage D (CIKD Model)
After Stage C completes successfully, train the Stage D CIKD models:
```bash
.venv\Scripts\python.exe src\run_stage_cd.py --stage D --model cikd_light --lambda_tvcs 0.3 --seed 42
```
