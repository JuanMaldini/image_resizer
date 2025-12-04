# Image Resizer (CLI/GUI)

Simple, dependency-light image resizer with a CLI and an optional Tkinter GUI. Built with Python and Pillow.

## Features

- Select input via file paths and/or a folder (recursive by default)
- Supported formats: JPG/JPEG, PNG, TIFF
- Resize modes: scale dimensions by percent **or** compress to a target weight (MB)
- Output: saved next to the original with prefix `resized_` (always overwrite)
- Preserves metadata and EXIF orientation when available
- Preview step before processing; optional `--dry-run`
- GUI table now shows both resolution and file weight for every entry

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

Preview and process a folder recursively (50% default, percent mode):

```powershell
python app.py --folder "C:\\images" -y
```

Compress everything in a folder to ~1.5 MB per file:

```powershell
python app.py --folder "C:\images" --resize-mode target-size --target-size-mb 1.5 -y
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

- Percent mode always keeps the image format and only scales width/height (1-100%).
- Target-size mode keeps dimensions intact and iteratively adjusts compression until the requested MB size (default 10 MB) is reached. If Pillow cannot achieve that weight without changing dimensions/format, processing for that file is skipped so you never get an output above the requested size.
- For TIFF, JPEG-compressed TIFF is used when saving.
- Metadata is preserved when available; orientation is respected.
- Best-quality resampling (Lanczos) is used under the hood.

## Packaging (optional)

You can later bundle as an .exe using PyInstaller:

```powershell
pip install pyinstaller
pyinstaller --onefile app.py
```
