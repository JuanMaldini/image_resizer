import argparse
import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

from PIL import Image, ImageOps
from PIL import __version__ as PILLOW_VERSION


APP_TITLE = "Image Resizer (CLI/GUI)"
OUTPUT_PREFIX = "resized_"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RESIZE_MODE_PERCENT = "percent"
RESIZE_MODE_TARGET = "target-size"
RESIZE_MODES = (RESIZE_MODE_PERCENT, RESIZE_MODE_TARGET)
BYTES_IN_MB = 1024 * 1024

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 10
except AttributeError:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 1))


@dataclass
class ResizeOptions:
    mode: str = RESIZE_MODE_PERCENT
    percent: int = 50
    target_size_mb: Optional[float] = 10.0


@dataclass
class ProcessResult:
    success: bool
    message: str
    output_path: Optional[str] = None
    output_bytes: Optional[int] = None
    skipped: bool = False


log = logging.getLogger("image_resizer")


def configure_logging(log_file: Optional[str], verbose: bool) -> Optional[Path]:
    level = logging.DEBUG if verbose else logging.INFO
    log.setLevel(level)
    # Clear existing handlers if reconfigured
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    # File
    # Write logs to temp dir by default to avoid polluting the repo folder
    file_path = Path(log_file) if log_file else (Path(tempfile.gettempdir()) / "image_resizer.log")
    file_handler_added = False
    try:
        fh = logging.FileHandler(str(file_path), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        log.addHandler(fh)
        file_handler_added = True
    except OSError as e:
        # Fall back to console-only if file not writable
        log.warning("Could not open log file '%s': %s", file_path, e)
    return file_path if file_handler_added else None


def is_image_file(path: Path, include_outputs: bool = False) -> bool:
    # Exclude outputs created by this app unless explicitly allowed
    if not include_outputs and path.name.lower().startswith(OUTPUT_PREFIX.lower()):
        return False
    return path.suffix.lower() in SUPPORTED_EXTS


def human_size(num_bytes: int) -> str:
    mb = num_bytes / BYTES_IN_MB
    return f"{mb:.2f} MB"


def resolve_format_and_extension(img: Image.Image, src_path: Path) -> Tuple[str, str]:
    fmt = (img.format or src_path.suffix.replace(".", "")).upper()
    if fmt in {"JPG", "JPEG"}:
        return "JPEG", ".jpg"
    if fmt == "PNG":
        return "PNG", ".png"
    if fmt in {"TIF", "TIFF"}:
        return "TIFF", ".tif"
    # Fallback to original suffix or JPG if missing
    suffix = src_path.suffix.lower() or ".jpg"
    return suffix.replace(".", "").upper(), suffix


def build_output_path(src_path: Path, prefix: Optional[str], ext: str) -> Path:
    dst_dir = src_path.parent
    use_prefix = prefix if prefix is not None else OUTPUT_PREFIX
    return dst_dir / f"{use_prefix}{src_path.stem}{ext}"


def extract_exif_bytes(img: Image.Image) -> Optional[bytes]:
    try:
        exif = img.getexif()
        if exif:
            return exif.tobytes()
    except (OSError, ValueError):
        return None
    return None


def ensure_mode_for_format(img: Image.Image, fmt: str) -> Image.Image:
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        return img.convert("RGB")
    return img


def build_base_save_params(fmt: str) -> Dict[str, object]:
    if fmt == "JPEG":
        return {"quality": 95, "optimize": True, "subsampling": 0}
    if fmt == "PNG":
        return {"optimize": True, "compress_level": 3}
    if fmt == "TIFF":
        return {"compression": "jpeg", "quality": 95}
    return {}


def encode_image(img: Image.Image, fmt: str, save_params: Dict[str, object], exif_bytes: Optional[bytes]) -> bytes:
    params = dict(save_params)
    if exif_bytes and fmt in {"JPEG", "TIFF"}:
        params["exif"] = exif_bytes
    buf = io.BytesIO()
    img.save(buf, format=fmt, **params)
    return buf.getvalue()


def encode_with_params(
    img: Image.Image,
    fmt: str,
    exif_bytes: Optional[bytes],
    **overrides,
) -> bytes:
    params = build_base_save_params(fmt)
    params.update(overrides)
    return encode_image(img, fmt, params, exif_bytes)


def _write_bytes(dst_path: Path, data: bytes) -> None:
    with open(dst_path, "wb") as fh:
        fh.write(data)


def _compress_with_quality(
    img: Image.Image,
    fmt: str,
    dst_path: Path,
    exif_bytes: Optional[bytes],
    target_bytes: int,
    *,
    min_quality: int = 5,
    max_quality: int = 95,
) -> Tuple[Optional[str], Optional[int]]:
    low, high = min_quality, max_quality
    best_data: Optional[bytes] = None
    best_size: Optional[int] = None
    best_quality: Optional[int] = None
    while low <= high:
        mid = (low + high) // 2
        data = encode_with_params(img, fmt, exif_bytes, quality=mid)
        size = len(data)
        if size <= target_bytes:
            best_data = data
            best_size = size
            best_quality = mid
            low = mid + 1  # try higher quality while staying under target
        else:
            high = mid - 1  # need more compression
    if best_data is None or best_size is None:
        return None, None
    _write_bytes(dst_path, best_data)
    log.debug("%s compressed with quality %s => %s bytes", fmt, best_quality, best_size)
    return str(dst_path), best_size


def _compress_png(
    img: Image.Image,
    dst_path: Path,
    exif_bytes: Optional[bytes],
    target_bytes: int,
) -> Tuple[Optional[str], Optional[int]]:
    # Try original image first (no quantization), then a palette-based version for stronger reduction.
    candidates: List[Tuple[Image.Image, bool]] = [(img, False)]
    try:
        palette_img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        candidates.append((palette_img, True))
    except ValueError:
        pass
    for candidate, quantized in candidates:
        for level in range(0, 10):  # Pillow compress level 0-9
            data = encode_with_params(candidate, "PNG", exif_bytes, compress_level=level)
            size = len(data)
            if size <= target_bytes:
                _write_bytes(dst_path, data)
                log.debug(
                    "PNG compressed%s at level %d => %s bytes",
                    " (quantized)" if quantized else "",
                    level,
                    size,
                )
                return str(dst_path), size
    return None, None



def scan_folder(folder: Path, recursive: bool = True, include_outputs: bool = False) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and is_image_file(p, include_outputs=include_outputs):
                files.append(p)
    else:
        for p in folder.glob("*"):
            if p.is_file() and is_image_file(p, include_outputs=include_outputs):
                files.append(p)
    return files


def compute_new_size(img: Image.Image, opts: ResizeOptions) -> Tuple[int, int]:
    w, h = img.size
    factor = max(1, int(opts.percent)) / 100.0
    return max(1, int(w * factor)), max(1, int(h * factor))


def save_resized_image(
    img: Image.Image,
    src_path: Path,
    new_size: Tuple[int, int],
    prefix: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[int]]:
    try:
        oriented = ImageOps.exif_transpose(img)
        out_img = oriented.resize(new_size, RESAMPLE_LANCZOS)
        fmt, ext = resolve_format_and_extension(img, src_path)
        out_img = ensure_mode_for_format(out_img, fmt)
        save_params = build_base_save_params(fmt)
        exif_bytes = extract_exif_bytes(img)
        data = encode_image(out_img, fmt, save_params, exif_bytes)
        dst_path = build_output_path(src_path, prefix, ext)
        with open(dst_path, "wb") as fh:
            fh.write(data)
        return True, str(dst_path), len(data)
    except (OSError, ValueError) as e:
        log.error("Save failed for %s: %s", src_path, e)
        return False, str(e), None


def compress_image_to_target(
    img: Image.Image,
    src_path: Path,
    target_bytes: int,
    prefix: Optional[str] = None,
) -> Tuple[Optional[str], Optional[int]]:
    try:
        oriented = ImageOps.exif_transpose(img)
        fmt, ext = resolve_format_and_extension(img, src_path)
        working = ensure_mode_for_format(oriented, fmt)
        exif_bytes = extract_exif_bytes(img)
        dst_path = build_output_path(src_path, prefix, ext)
        fmt_upper = fmt.upper()
        if fmt_upper in {"JPEG", "TIFF"}:
            return _compress_with_quality(working, fmt_upper, dst_path, exif_bytes, target_bytes)
        if fmt_upper == "PNG":
            return _compress_png(working, dst_path, exif_bytes, target_bytes)
        # Fallback: treat as JPEG-style quality adjustments
        return _compress_with_quality(working, fmt_upper, dst_path, exif_bytes, target_bytes)
    except (OSError, ValueError) as e:
        log.error("Compression failed for %s: %s", src_path, e)
        return None, None


def _process_percent_mode(
    img: Image.Image,
    src_path: Path,
    opts: ResizeOptions,
    prefix: str,
) -> ProcessResult:
    new_size = compute_new_size(img, opts)
    success, detail, out_bytes = save_resized_image(img, src_path, new_size, prefix)
    if success:
        message = (
            f"{detail}: {img.width}x{img.height} -> {new_size[0]}x{new_size[1]}"
            f" ({human_size(out_bytes or 0)})"
        )
        return ProcessResult(True, message, output_path=detail, output_bytes=out_bytes)
    return ProcessResult(False, detail or "Save failed")


def _process_target_mode(
    img: Image.Image,
    src_path: Path,
    opts: ResizeOptions,
    prefix: str,
) -> ProcessResult:
    if opts.target_size_mb is None or opts.target_size_mb <= 0:
        return ProcessResult(False, "Target size (MB) must be greater than zero")
    target_bytes = int(opts.target_size_mb * BYTES_IN_MB)
    original_size = src_path.stat().st_size
    if target_bytes >= original_size:
        return ProcessResult(
            False,
            (
                f"Target {human_size(target_bytes)} is greater than or equal to current size "
                f"{human_size(original_size)}; skipping."
            ),
            skipped=True,
        )
    saved_path, out_bytes = compress_image_to_target(
        img,
        src_path,
        target_bytes,
        prefix,
    )
    if saved_path and out_bytes is not None:
        message = (
            f"{saved_path}: {human_size(original_size)} -> {human_size(out_bytes)}"
            f" (target {human_size(target_bytes)})"
        )
        return ProcessResult(True, message, output_path=saved_path, output_bytes=out_bytes)
    error_msg = (
        f"Could not reach {human_size(target_bytes)} for {src_path.name}. "
        "Increase the target size a bit and try again."
    )
    return ProcessResult(False, error_msg)


def process_image_file(path: Path, opts: ResizeOptions, prefix: str) -> ProcessResult:
    try:
        with Image.open(path) as im:
            if opts.mode == RESIZE_MODE_TARGET:
                return _process_target_mode(im, path, opts, prefix)
            return _process_percent_mode(im, path, opts, prefix)
    except (OSError, ValueError) as exc:
        return ProcessResult(False, f"ERROR opening {path.name}: {exc}")


def gather_inputs(files: List[str], folder: Optional[str], recursive: bool) -> List[Path]:
    paths: List[Path] = []
    if files:
        for f in files:
            p = Path(f)
            if p.is_dir():
                paths.extend(scan_folder(p, recursive=True))
            elif p.is_file() and is_image_file(p):
                paths.append(p)
    if folder:
        paths.extend(scan_folder(Path(folder), recursive=recursive))
    # de-duplicate and keep order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    return unique


def preview(files: List[Path], opts: ResizeOptions):
    if opts.mode == RESIZE_MODE_TARGET:
        target_bytes = int((opts.target_size_mb or 0) * BYTES_IN_MB)
        log.info("Preview (target size):")
        for f in files:
            try:
                size = f.stat().st_size
                log.info(
                    "- %s: %s -> target %s",
                    f.name,
                    human_size(size),
                    human_size(target_bytes),
                )
            except OSError as e:
                log.error("- %s: ERROR reading size: %s", f.name, e)
    else:
        log.info("Preview (percent scale):")
        for f in files:
            try:
                with Image.open(f) as im:
                    nw, nh = compute_new_size(im, opts)
                    log.info("- %s: %dx%d -> %dx%d", f.name, im.width, im.height, nw, nh)
            except (OSError, ValueError) as e:
                log.error("- %s: ERROR reading image: %s", f.name, e)


def run(files: List[Path], opts: ResizeOptions, prefix: str):
    if not files:
        log.warning("No input images found.")
        return 1
    log.info("Processing %d file(s)...", len(files))
    ok = 0
    for i, f in enumerate(files, 1):
        result = process_image_file(f, opts, prefix)
        if result.success:
            ok += 1
            log.info("[%d/%d] OK: %s", i, len(files), result.message)
        else:
            level = log.warning if result.skipped else log.error
            level("[%d/%d] %s", i, len(files), result.message)
    log.info("Done. %d/%d succeeded.", ok, len(files))
    return 0 if ok == len(files) else 2


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=APP_TITLE)
    src = p.add_argument_group("Input sources")
    src.add_argument("--files", nargs="*", default=[], help="Image files and/or folders to include")
    src.add_argument("--folder", help="Folder to scan for images", default=None)
    src.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders")
    resize = p.add_argument_group("Resize")
    resize.add_argument(
        "--resize-mode",
        choices=RESIZE_MODES,
        default=RESIZE_MODE_PERCENT,
        help="Select percent (dimension) or target-size (weight) mode",
    )
    resize.add_argument("--percent", type=int, default=50, help="Percent size (1-100, percent mode only)")
    resize.add_argument(
        "--target-size-mb",
        type=float,
        default=10.0,
        help="Desired file size in MB (target-size mode only)",
    )

    # Fixed prefix and overwrite/metadata policy; no CLI controls now

    flow = p.add_argument_group("Flow")
    flow.add_argument("--yes", "-y", action="store_true", help="Do not prompt for confirmation")
    flow.add_argument("--dry-run", action="store_true", help="Only preview, do not write files")
    flow.add_argument("--verbose", "-v", action="store_true", help="Verbose logging (DEBUG)")
    flow.add_argument("--log-file", default=None, help="Log file path (default: image_resizer.log)")
    flow.add_argument("--pause-on-exit", action="store_true", help="Wait for Enter before exiting")
    flow.add_argument("--open-log", action="store_true", help="Open the log file on exit")
    flow.add_argument("--gui", action="store_true", help="Launch the GUI instead of CLI")
    return p.parse_args()


def make_resize_options(args: argparse.Namespace) -> ResizeOptions:
    if args.resize_mode == RESIZE_MODE_PERCENT:
        if args.percent < 1 or args.percent > 100:
            raise ValueError("Percent mode only accepts values between 1 and 100.")
        return ResizeOptions(
            mode=RESIZE_MODE_PERCENT,
            percent=args.percent,
        )
    if args.target_size_mb is None or args.target_size_mb <= 0:
        raise ValueError("Target-size mode requires --target-size-mb > 0.")
    return ResizeOptions(
        mode=RESIZE_MODE_TARGET,
        target_size_mb=args.target_size_mb,
    )


# --- GUI (Tkinter) integration ---
def launch_gui() -> None:
    """Start the Tkinter GUI. Imported lazily to keep CLI lightweight."""
    import tkinter as tk
    from tkinter import ttk, filedialog

    class _Tooltip:
        """Minimal tooltip for a widget. Shows only when widget is disabled."""
        def __init__(self, widget, text: str):
            self.widget = widget
            self.text = text
            self.tipwindow = None
            widget.bind("<Enter>", self._enter)
            widget.bind("<Leave>", self._leave)

        def _enter(self, event=None):
            try:
                state = str(self.widget["state"]).lower()
            except Exception:
                state = "normal"
            if state != "disabled":
                return
            if self.tipwindow is not None:
                return
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
            self.tipwindow = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                             background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                             font=("Segoe UI", 9))
            label.pack(ipadx=6, ipady=3)

        def _leave(self, event=None):
            self.hidetip()

        def hidetip(self):
            tw = self.tipwindow
            self.tipwindow = None
            if tw is not None:
                try:
                    tw.destroy()
                except Exception:
                    pass

    class ImageResizerGUI(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Image Resizer")
            self.geometry("1080x640")
            self.minsize(920, 540)
            self.columnconfigure(0, weight=1)
            self.rowconfigure(1, weight=1)

            # State
            self.files: List[Path] = []
            self.row_to_path: dict[str, Path] = {}
            self.status_var = tk.StringVar(value="Select files or a folder to begin.")
            self._progress_total = 0

            # Top controls
            top = ttk.Frame(self)
            top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
            top.columnconfigure(4, weight=1)

            self.recursive_var = tk.BooleanVar(value=True)

            ttk.Button(top, text="Select Files", command=self.on_select_files).grid(row=0, column=0, padx=(0, 6))
            ttk.Button(top, text="Select Folder", command=self.on_select_folder).grid(row=0, column=1, padx=(0, 6))
            ttk.Checkbutton(top, text="Recursive", variable=self.recursive_var).grid(row=0, column=2, padx=(12, 0))
            ttk.Button(top, text="Clear List", command=self.on_clear).grid(row=0, column=3, padx=(12, 0))

            # Middle: responsive table + options via paned window
            middle = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
            middle.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)

            # Table
            table_frame = ttk.Frame(middle)
            table_frame.columnconfigure(0, weight=1)
            table_frame.rowconfigure(0, weight=1)

            cols = ("name", "folder", "dimensions", "weight")
            self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
            self.tree.heading("name", text="File")
            self.tree.heading("folder", text="Folder")
            self.tree.heading("dimensions", text="W x H")
            self.tree.heading("weight", text="Size (MB)")
            self.tree.column("name", minwidth=150, width=260, stretch=True)
            self.tree.column("folder", minwidth=200, width=420, stretch=True)
            self.tree.column("dimensions", width=120, anchor=tk.CENTER, stretch=False)
            self.tree.column("weight", width=110, anchor=tk.CENTER, stretch=False)

            vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
            hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
            self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
            self.tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")

            def _adjust_columns(event=None):
                try:
                    total = self.tree.winfo_width()
                    fixed = 240
                    remaining = max(total - fixed, 200)
                    name_w = int(remaining * 0.55)
                    folder_w = remaining - name_w
                    self.tree.column("name", width=max(name_w, 160))
                    self.tree.column("folder", width=max(folder_w, 240))
                except Exception:
                    pass

            self.tree.bind("<Configure>", _adjust_columns)

            middle.add(table_frame, weight=5)

            # Options panel
            opts = ttk.Frame(middle, padding=(14, 12))
            opts.columnconfigure(0, weight=1)

            self.mode_var = tk.StringVar(value=RESIZE_MODE_PERCENT)
            self.percent_var = tk.IntVar(value=50)
            self.target_mb_var = tk.DoubleVar(value=10.0)

            mode_group = ttk.LabelFrame(opts, text="Mode")
            mode_group.grid(row=0, column=0, sticky="ew")
            ttk.Radiobutton(
                mode_group,
                text="Percent (resize)",
                value=RESIZE_MODE_PERCENT,
                variable=self.mode_var,
                command=self._update_option_states,
            ).grid(row=0, column=0, sticky="w", padx=4, pady=2)
            ttk.Radiobutton(
                mode_group,
                text="Target weight",
                value=RESIZE_MODE_TARGET,
                variable=self.mode_var,
                command=self._update_option_states,
            ).grid(row=1, column=0, sticky="w", padx=4, pady=(0, 2))

            percent_group = ttk.LabelFrame(opts, text="Percent scale")
            percent_group.grid(row=1, column=0, sticky="ew", pady=(12, 0))
            ttk.Label(percent_group, text="Percent (1-100)").grid(row=0, column=0, sticky="w", padx=4, pady=4)
            self.percent_spin = ttk.Spinbox(percent_group, from_=1, to=100, textvariable=self.percent_var, width=6)
            self.percent_spin.grid(row=0, column=1, sticky="w", padx=4, pady=4)

            weight_group = ttk.LabelFrame(opts, text="Target size (MB)")
            weight_group.grid(row=2, column=0, sticky="ew", pady=(12, 0))
            ttk.Label(weight_group, text="Target MB").grid(row=0, column=0, sticky="w", padx=4, pady=4)
            self.target_spin = ttk.Spinbox(
                weight_group,
                from_=0.1,
                to=500.0,
                increment=0.1,
                textvariable=self.target_mb_var,
                width=6,
            )
            self.target_spin.grid(row=0, column=1, sticky="w", padx=4, pady=4)

            action_group = ttk.Frame(opts)
            action_group.grid(row=3, column=0, sticky="ew", pady=(16, 0))
            self.btn_process = ttk.Button(action_group, text="Process", command=self.on_process, state=tk.DISABLED)
            self.btn_process.pack(side=tk.LEFT)

            middle.add(opts, weight=2)

            # Status bar
            status_bar = ttk.Frame(self, padding=(12, 8))
            status_bar.grid(row=2, column=0, sticky="ew")
            status_bar.columnconfigure(1, weight=1)
            ttk.Label(status_bar, text="Status:").grid(row=0, column=0, sticky="w")
            self.status_label = ttk.Label(status_bar, textvariable=self.status_var)
            self.status_label.grid(row=0, column=1, sticky="w", padx=(6, 0))
            self.progress = ttk.Progressbar(status_bar, mode="determinate", maximum=100, length=220)
            self.progress.grid(row=0, column=2, sticky="e", padx=(12, 0))

            # Tooltip for disabled Process button
            self._process_tooltip = _Tooltip(self.btn_process, "Select one or more files to process")

            # Selection change binding to enable/disable Process button
            self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
            self._update_option_states()

        def refresh_table(self):
            self.tree.delete(*self.tree.get_children())
            self.row_to_path.clear()
            for p in self.files:
                size_str = "-"
                weight_str = "-"
                try:
                    with Image.open(p) as im:
                        size_str = f"{im.width} x {im.height}"
                except (OSError, ValueError):
                    pass
                try:
                    weight_str = human_size(p.stat().st_size)
                except OSError:
                    pass
                iid = self.tree.insert("", tk.END, values=(p.name, str(p.parent), size_str, weight_str))
                self.row_to_path[str(iid)] = p
            self._update_process_state()
            count = len(self.files)
            if count:
                self._set_status(f"{count} file(s) ready.")
            else:
                self._set_status("List is empty.")

        # Event handlers
        def on_select_files(self):
            paths = filedialog.askopenfilenames(
                title="Select image files",
                filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")],
            )
            if not paths:
                return
            added = 0
            for s in paths:
                p = Path(s)
                if p.is_file() and is_image_file(p, include_outputs=True):
                    rp = p.resolve()
                    if rp not in self.files:
                        self.files.append(rp)
                        added += 1
            if added:
                self._set_status(f"Added {added} file(s).")
            else:
                self._set_status("No new files were added.")
            self.refresh_table()

        def on_select_folder(self):
            folder = filedialog.askdirectory(title="Select folder")
            if not folder:
                return
            recursive = self.recursive_var.get()
            to_add = scan_folder(Path(folder), recursive=recursive, include_outputs=True)
            before = len(self.files)
            existing = set(self.files)
            for p in to_add:
                rp = p.resolve()
                if rp not in existing:
                    self.files.append(rp)
            added = len(self.files) - before
            if added:
                self._set_status(f"Added {added} file(s) from folder.")
            else:
                self._set_status("No images met the criteria in that folder.")
            self.refresh_table()

        def on_clear(self):
            self.files.clear()
            self.refresh_table()
            self._set_status("List cleared.")

        def _on_tree_select(self, event=None):
            self._update_process_state()

        def _update_process_state(self):
            has_selection = len(self.tree.selection()) > 0
            self.btn_process.configure(state=(tk.NORMAL if has_selection else tk.DISABLED))
            # Tooltip shows only when disabled; managed inside Tooltip class

        def _update_option_states(self):
            mode = self.mode_var.get()
            percent_state = tk.NORMAL if mode == RESIZE_MODE_PERCENT else tk.DISABLED
            weight_state = tk.NORMAL if mode == RESIZE_MODE_TARGET else tk.DISABLED
            self.percent_spin.configure(state=percent_state)
            self.target_spin.configure(state=weight_state)

        # Utility methods
        def _set_status(self, message: str):
            self.status_var.set(message)
            self.update_idletasks()

        def _set_progress(self, current: int, total: int):
            value = 0.0
            if total > 0:
                value = max(0.0, min(100.0, (current / float(total)) * 100.0))
            self.progress["value"] = value
            self.update_idletasks()

        def _reset_progress(self):
            self.progress["value"] = 0
            self.update_idletasks()

        def _set_window_disabled(self, disabled: bool):
            try:
                self.attributes("-disabled", disabled)
            except tk.TclError:
                pass
            self.configure(cursor="wait" if disabled else "")
            self.update_idletasks()

        def gather_opts(self) -> ResizeOptions:
            mode = self.mode_var.get()
            if mode == RESIZE_MODE_TARGET:
                target = float(self.target_mb_var.get())
                if target <= 0:
                    raise ValueError("Target size must be greater than 0 MB")
                return ResizeOptions(
                    mode=RESIZE_MODE_TARGET,
                    target_size_mb=target,
                )
            percent = int(self.percent_var.get())
            if percent < 1 or percent > 100:
                raise ValueError("Percent must stay between 1 and 100")
            return ResizeOptions(mode=RESIZE_MODE_PERCENT, percent=percent)

        def on_process(self):
            if not self.files:
                self._set_status("Add files before processing.")
                return
            # Determine selected paths
            selected_iids = list(self.tree.selection())
            selected_paths = [self.row_to_path[iid] for iid in selected_iids if iid in self.row_to_path]
            if not selected_paths:
                self._set_status("Select one or more files to process.")
                return
            try:
                opts = self.gather_opts()
            except ValueError as exc:
                self._set_status(str(exc))
                return
            prefix = OUTPUT_PREFIX
            total = len(selected_paths)
            self._set_status(f"Processing {total} file(s)...")
            self._set_progress(0, total)
            self._set_window_disabled(True)
            try:
                ok = 0
                for i, p in enumerate(selected_paths, 1):
                    result = process_image_file(p, opts, prefix)
                    status = "OK" if result.success else ("SKIP" if result.skipped else "ERROR")
                    msg = result.message
                    if result.success:
                        ok += 1
                    self._set_status(f"[{i}/{total}] {status}: {msg}")
                    self._set_progress(i, total)
                self._set_status(f"Done. {ok}/{total} succeeded.")
                self.refresh_table()
            finally:
                self._reset_progress()
                self._set_window_disabled(False)

    app = ImageResizerGUI()
    app.mainloop()


def main() -> int:
    args = build_args()
    log_file_path = configure_logging(args.log_file, args.verbose)
    # Startup banner and environment info
    log.info("=== Image Resizer starting ===")
    log.info("Python: %s (%s)", sys.version.split()[0], sys.executable)
    log.info("Pillow: %s", PILLOW_VERSION)
    log.info("CWD: %s", Path.cwd())
    log.info("Script: %s", Path(__file__).resolve())
    log.info("Args: %s", sys.argv)
    if log_file_path:
        log.info("Logging to: %s", log_file_path)
    # Launch GUI if requested or if no CLI inputs were provided (double-click/default)
    if getattr(args, "gui", False) or (not args.files and not args.folder):
        try:
            launch_gui()
        except Exception as exc:
            # Catch broadly here to surface Tkinter errors reliably on Windows
            log.error("GUI failed: %s", exc)
            return 1
        return 0
    files = gather_inputs(args.files, args.folder, not args.no_recursive)
    log.info("Found %d image(s)", len(files))
    try:
        opts = make_resize_options(args)
    except ValueError as exc:
        log.error("%s", exc)
        if args.pause_on_exit:
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass
        return 2

    # Auto-pause only applies to explicit CLI usage
    auto_pause = args.pause_on_exit
    if not files:
        log.warning("No images found. Provide --files and/or --folder.")
        if auto_pause:
            try:
                input("No images found. Press Enter to exit...")
            except EOFError:
                pass
        return 1

    preview(files, opts)
    if args.dry_run:
        log.info("Dry-run complete. No files written.")
        if auto_pause:
            try:
                input("Dry-run complete. Press Enter to exit...")
            except EOFError:
                pass
        return 0

    if not args.yes:
        try:
            answer = input("Proceed with processing? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("y", "yes"):
            log.info("Aborted.")
            return 0

    code = run(files, opts, OUTPUT_PREFIX)
    if args.open_log and log_file_path and os.name == "nt":
        try:
            os.startfile(str(log_file_path))  # Windows only
        except OSError as exc:
            log.warning("Could not open log file automatically: %s", exc)
    if auto_pause:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        sys.exit(130)
