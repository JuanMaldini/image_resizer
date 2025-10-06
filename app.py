import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile

from PIL import Image, ImageOps
from PIL import __version__ as PILLOW_VERSION


APP_TITLE = "Image Resizer (CLI/GUI)"
OUTPUT_PREFIX = "resized_"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 10
except AttributeError:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 1))


@dataclass
class ResizeOptions:
    percent: int = 50


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


def is_image_file(path: Path) -> bool:
    # Exclude outputs created by this app
    if path.name.lower().startswith(OUTPUT_PREFIX.lower()):
        return False
    return path.suffix.lower() in SUPPORTED_EXTS


def scan_folder(folder: Path, recursive: bool = True) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and is_image_file(p):
                files.append(p)
    else:
        for p in folder.glob("*"):
            if p.is_file() and is_image_file(p):
                files.append(p)
    return files


def compute_new_size(img: Image.Image, opts: ResizeOptions) -> Tuple[int, int]:
    w, h = img.size
    factor = max(1, int(opts.percent)) / 100.0
    return max(1, int(w * factor)), max(1, int(h * factor))


def save_image(
    img: Image.Image,
    src_path: Path,
    new_size: Tuple[int, int],
    prefix: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    try:
        img = ImageOps.exif_transpose(img)  # respect orientation
        out_img = img.resize(new_size, RESAMPLE_LANCZOS)

        dst_dir = src_path.parent
        use_prefix = prefix if prefix is not None else OUTPUT_PREFIX
        # Determine format and extension first
        fmt = img.format or src_path.suffix.replace(".", "").upper()
        if fmt is None:
            fmt = src_path.suffix.replace(".", "").upper()
        if fmt.upper() in ("JPG", "JPEG"):
            fmt = "JPEG"
            ext = ".jpg"
        elif fmt.upper() in ("PNG",):
            fmt = "PNG"
            ext = ".png"
        elif fmt.upper() in ("TIF", "TIFF"):
            fmt = "TIFF"
            ext = ".tif"
        else:
            # Fallback to original suffix if unknown
            ext = src_path.suffix or ".jpg"

        # Build output path (change suffix if forced format)
        stem = src_path.stem
        dst_name = f"{use_prefix}{stem}{ext}"
        dst_path = dst_dir / dst_name

        save_params = {}
        if fmt == "JPEG":
            save_params.update({"quality": 95, "optimize": True, "subsampling": 0})
            if out_img.mode not in ("RGB", "L"):
                out_img = out_img.convert("RGB")
        elif fmt == "PNG":
            save_params.update({"optimize": True})
        elif fmt == "TIFF":
            save_params.update({"compression": "jpeg", "quality": 95})

        # Always try to preserve metadata when available
        exif_bytes = None
        try:
            exif = img.getexif()
            if exif:
                exif_bytes = exif.tobytes()
        except (OSError, ValueError):
            exif_bytes = None
        if exif_bytes:
            save_params["exif"] = exif_bytes

        out_img.save(dst_path, format=fmt, **save_params)
        return True, str(dst_path)
    except (OSError, ValueError) as e:
        log.error("Save failed for %s: %s", src_path, e)
        return False, str(e)


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
        try:
            with Image.open(f) as im:
                new_size = compute_new_size(im, opts)
                success, msg = save_image(im, f, new_size, prefix)
                if success:
                    ok += 1
                    log.info("[%d/%d] OK: %s", i, len(files), msg)
                else:
                    log.error("[%d/%d] ERROR: %s", i, len(files), msg)
        except (OSError, ValueError) as e:
            log.error("[%d/%d] ERROR opening %s: %s", i, len(files), f.name, e)
    log.info("Done. %d/%d succeeded.", ok, len(files))
    return 0 if ok == len(files) else 2


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=APP_TITLE)
    src = p.add_argument_group("Input sources")
    src.add_argument("--files", nargs="*", default=[], help="Image files and/or folders to include")
    src.add_argument("--folder", help="Folder to scan for images", default=None)
    src.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders")
    resize = p.add_argument_group("Resize")
    resize.add_argument("--percent", type=int, default=50, help="Percent size (1-100)")

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


# --- GUI (Tkinter) integration ---
def launch_gui() -> None:
    """Start the Tkinter GUI. Imported lazily to keep CLI lightweight."""
    import threading
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

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
            self.geometry("980x600")
            self.minsize(900, 520)

            # State
            self.files: List[Path] = []
            self.row_to_path: dict[str, Path] = {}

            # Top controls
            top = ttk.Frame(self)
            top.pack(fill=tk.X, padx=8, pady=6)

            self.recursive_var = tk.BooleanVar(value=True)

            ttk.Button(top, text="Select Files", command=self.on_select_files).pack(side=tk.LEFT)
            ttk.Button(top, text="Select Folder", command=self.on_select_folder).pack(side=tk.LEFT, padx=(6, 0))
            ttk.Checkbutton(top, text="Recursive", variable=self.recursive_var).pack(side=tk.LEFT, padx=(12, 0))
            ttk.Button(top, text="Clear List", command=self.on_clear).pack(side=tk.LEFT, padx=(12, 0))

            # Middle: table and options
            middle = ttk.Frame(self)
            middle.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

            # Table
            table_frame = ttk.Frame(middle)
            table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            cols = ("name", "folder", "size")
            self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
            self.tree.heading("name", text="File")
            self.tree.heading("folder", text="Folder")
            self.tree.heading("size", text="W x H")
            # Make columns responsive: name and folder stretch, size stays fixed
            self.tree.column("name", minwidth=150, width=240, stretch=True)
            self.tree.column("folder", minwidth=200, width=420, stretch=True)
            self.tree.column("size", width=100, anchor=tk.CENTER, stretch=False)

            vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
            hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
            self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
            self.tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            table_frame.rowconfigure(0, weight=1)
            table_frame.columnconfigure(0, weight=1)

            # Adjust column widths on resize to fill available space
            def _adjust_columns(event=None):
                try:
                    total = self.tree.winfo_width()
                    size_w = 100  # keep fixed width for size column
                    remaining = max(total - size_w - 2, 100)
                    name_w = int(remaining * 0.5)
                    folder_w = remaining - name_w
                    self.tree.column("name", width=max(name_w, 150))
                    self.tree.column("folder", width=max(folder_w, 200))
                    self.tree.column("size", width=size_w)
                except Exception:
                    pass

            self.tree.bind("<Configure>", _adjust_columns)

            # Options panel
            opts = ttk.LabelFrame(middle, text="Options")
            opts.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

            row = 0
            ttk.Label(opts, text="Percent").grid(row=row, column=0, sticky="w", padx=6, pady=(6, 2))
            self.percent_var = tk.IntVar(value=50)
            ttk.Spinbox(opts, from_=1, to=100, textvariable=self.percent_var, width=6).grid(row=row, column=1, sticky="w", padx=6, pady=2)

            row += 1
            ttk.Separator(opts).grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 4))

            row += 1
            btns = ttk.Frame(opts)
            btns.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
            self.btn_process = ttk.Button(btns, text="Process", command=self.on_process, state=tk.DISABLED)
            self.btn_process.pack(side=tk.LEFT)

            for c in range(2):
                opts.columnconfigure(c, weight=1)

            # Log area
            log_frame = ttk.Frame(self)
            log_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))
            ttk.Label(log_frame, text="Log:").pack(anchor="w")
            self.log = tk.Text(log_frame, height=8, wrap="word")
            self.log.pack(fill=tk.BOTH, expand=True)

            # Tooltip for disabled Process button
            self._process_tooltip = _Tooltip(self.btn_process, "Select one or more files to process")

            # Selection change binding to enable/disable Process button
            self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Utility methods
        def log_print(self, msg: str):
            self.log.insert(tk.END, msg + "\n")
            self.log.see(tk.END)

        def refresh_table(self):
            self.tree.delete(*self.tree.get_children())
            self.row_to_path.clear()
            for p in self.files:
                size_str = "-"
                try:
                    with Image.open(p) as im:
                        size_str = f"{im.width} x {im.height}"
                except (OSError, ValueError):
                    pass
                iid = self.tree.insert("", tk.END, values=(p.name, str(p.parent), size_str))
                self.row_to_path[str(iid)] = p
            self._update_process_state()

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
                if p.is_file() and is_image_file(p):
                    rp = p.resolve()
                    if rp not in self.files:
                        self.files.append(rp)
                        added += 1
            self.log_print(f"Added {added} file(s)")
            self.refresh_table()

        def on_select_folder(self):
            folder = filedialog.askdirectory(title="Select folder")
            if not folder:
                return
            recursive = self.recursive_var.get()
            to_add = scan_folder(Path(folder), recursive=recursive)
            before = len(self.files)
            existing = set(self.files)
            for p in to_add:
                rp = p.resolve()
                if rp not in existing:
                    self.files.append(rp)
            self.log_print(f"Added {len(self.files) - before} file(s) from folder")
            self.refresh_table()

        def on_clear(self):
            self.files.clear()
            self.refresh_table()
            self.log_print("Cleared list")

        def _on_tree_select(self, event=None):
            self._update_process_state()

        def _update_process_state(self):
            has_selection = len(self.tree.selection()) > 0
            self.btn_process.configure(state=(tk.NORMAL if has_selection else tk.DISABLED))
            # Tooltip shows only when disabled; managed inside Tooltip class

        def gather_opts(self) -> ResizeOptions:
            return ResizeOptions(percent=int(self.percent_var.get()))

        def on_process(self):
            if not self.files:
                messagebox.showinfo("Image Resizer", "No files to process.")
                return
            # Determine selected paths
            selected_iids = list(self.tree.selection())
            selected_paths = [self.row_to_path[iid] for iid in selected_iids if iid in self.row_to_path]
            if not selected_paths:
                messagebox.showinfo("Image Resizer", "Select one or more files to process.")
                return
            opts = self.gather_opts()
            prefix = OUTPUT_PREFIX
            self.log_print("About to process these images:")
            for p in selected_paths:
                self.log_print(f"- {p}")

            def worker():
                ok = 0
                for i, p in enumerate(selected_paths, 1):
                    try:
                        with Image.open(p) as im:
                            new_size = compute_new_size(im, opts)
                            success, msg = save_image(im, p, new_size, prefix)
                            if success:
                                ok += 1
                                self.log_print(f"[{i}/{len(selected_paths)}] OK: {msg}")
                            else:
                                self.log_print(f"[{i}/{len(selected_paths)}] ERROR: {msg}")
                    except (OSError, ValueError) as e:
                        self.log_print(f"[{i}/{len(selected_paths)}] ERROR opening {p.name}: {e}")
                self.log_print(f"Done. {ok}/{len(selected_paths)} succeeded.")

            threading.Thread(target=worker, daemon=True).start()

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
    opts = ResizeOptions(percent=args.percent)

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
