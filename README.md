# Image Resizer (CLI/GUI)

Simple, dependency-light image resizer with a CLI and an optional Tkinter GUI. Built with Python and Pillow.

## Features

- Select input via file paths and/or a folder (recursive by default)
- Supported formats: JPG/JPEG, PNG, TIFF
- Resize: percent-only (e.g., 50 means 50% of original size)
- Output: saved next to the original with prefix `resized_` (always overwrite)
- Preserves metadata and EXIF orientation when available
- Preview step before processing; optional `--dry-run`

## Requirements

- Python 3.9+
- Windows/macOS/Linux

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage (CLI)

Preview and process a folder recursively (50% default):

```powershell
python app.py --folder "C:\\images" -y
```

GUI mode (Tkinter):

```powershell
python app.py --gui
```

See all options:

```powershell
python app.py -h
```

## Notes

- For TIFF, JPEG-compressed TIFF is used when saving.
- Metadata is preserved when available; orientation is respected.
- Best-quality resampling (Lanczos) is used under the hood.

## Packaging (optional)

You can later bundle as an .exe using PyInstaller:

```powershell
pip install pyinstaller
pyinstaller --onefile app.py
```
