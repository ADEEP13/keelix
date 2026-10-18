"""Tkinter front end for navsim.

Four tabs, in the order the project should actually be worked:

**Physics** comes first deliberately.  It answers the questions that decide
what is worth attempting -- the MTF budget, hull-versus-water contrast per
band, the speed below which a wake carries no usable signal, and which ship
classes are provably inseparable at 3 m -- and all of it costs nothing and
needs no imagery.  Run it before spending quota.

**Simulator** composes a scene directly, which is the right tool for
developing the measurement module: you choose how many vessels are underway,
at what speed, under what illumination and view geometry.

**Campaign** runs the honest sampling model instead, where sorties make units
absent and cloud removes runs of days.  Use it to find out what the pipeline
will actually be fed.

**Planet** holds the API key and the archive tools, with the quota cost of
every action shown before you can take it.

**Real imagery** loads a folder of PlanetScope deliveries and runs the same
pipeline over them -- the identical code path, because every stage depends only
on :class:`navsim.scenes.SceneLike`.
"""

from __future__ import annotations

import csv
import math
import queue
import threading
import traceback
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from . import credentials
from .credentials import PROVIDERS
from .aoi import (AOIS, DEFAULT_REGION, MONTHLY_QUOTA_KM2, REGIONS,
                  WITHHOLD_DAYS, quota_plan)

#: Pad the withhold by a day so the newest searchable date always exists.
WITHHOLD_PAD = WITHHOLD_DAYS + 1
from .catalog import CATALOG, Domain, sample_instance
from .config import (BAND_NAMES, CloudConfig, GeometryConfig, GridConfig,
                     NIR, SeaConfig, SensorConfig, SimConfig, rng_for)
from .metrics import class_confusability, detection_metrics, measurement_metrics
from .pipeline.classify import LocationPrior, TwoStageClassifier
from .pipeline.detect import DetectorConfig, ShipDetector
from .pipeline.measure import HullFitMeasurer, MeasurerConfig
from .pipeline.signature import separable_category_count
from .overlay import (build_annotations, draw_annotations, draw_corridor,
                      filter_annotations, write_geojson)
from .run import run_pipeline
from .scenes import as_scene, build_static_masks, load_directory
from .movement import generate_history, solar_position
from .port import make_port
from .radiometry import contrast_report
from .scene import simulate_scene
from .sensor import noise_coefficients, system_mtf
from .state import KNOTS, Pose, ShipState
from .wake import narrow_wake_angle, wake_resolvability

MONO = ("Menlo", 10)


# ==========================================================================
# Small widget helpers
# ==========================================================================
class Slider:
    """A labelled ttk.Scale with a live numeric readout."""

    def __init__(self, parent, label, lo, hi, init, fmt="{:.1f}", row=0,
                 unit="", command=None):
        self.var = tk.DoubleVar(value=init)
        self.fmt = fmt
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(2, 6))
        self.scale = ttk.Scale(parent, from_=lo, to=hi, variable=self.var,
                               orient="horizontal", length=150,
                               command=self._on_move)
        self.scale.grid(row=row, column=1, sticky="ew")
        self.readout = ttk.Label(parent, text=fmt.format(init) + unit, width=9,
                                 anchor="e")
        self.readout.grid(row=row, column=2, sticky="e", padx=(4, 2))
        self.unit = unit
        self.command = command

    def _on_move(self, _=None):
        self.readout.config(text=self.fmt.format(self.var.get()) + self.unit)
        if self.command:
            self.command()

    def get(self) -> float:
        return float(self.var.get())

    def set(self, v: float) -> None:
        self.var.set(v)
        self._on_move()


class IntField:
    def __init__(self, parent, label, init, row=0, width=8, lo=0, hi=10_000):
        self.var = tk.IntVar(value=init)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(2, 6))
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=self.var, width=width
                    ).grid(row=row, column=1, columnspan=2, sticky="w")

    def get(self) -> int:
        return int(self.var.get())


def mono_text(parent, height=24, width=96) -> tk.Text:
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=4, pady=4)
    txt = tk.Text(frame, height=height, width=width, font=MONO, wrap="none")
    ysb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
    xsb = ttk.Scrollbar(frame, orient="horizontal", command=txt.xview)
    txt.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
    txt.grid(row=0, column=0, sticky="nsew")
    ysb.grid(row=0, column=1, sticky="ns")
    xsb.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return txt


class JobCancelled(Exception):
    """Raised inside a worker when the user presses Cancel."""


def fmt_clock(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s_ = divmod(rem, 60)
    return f"{h}:{m:02d}:{s_:02d}" if h else f"{m}:{s_:02d}"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def confirm_dialog(parent, title: str, message: str,
                   ok_text: str = "Go ahead", cancel_text: str = "Cancel"
                   ) -> bool:
    """A modal yes/no dialog that is guaranteed to be visible.

    ``tkinter.messagebox`` on macOS can open *behind* the main window, which
    is indistinguishable from the application having frozen: the app stops
    responding because it is blocked waiting for input on a dialog the user
    cannot see.  This builds the dialog explicitly, centres it on the parent,
    lifts it and grabs focus, so that cannot happen.
    """
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    win.resizable(False, False)

    frame = ttk.Frame(win, padding=16)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text=message, justify="left", font=MONO,
             anchor="w").pack(fill="both", expand=True)

    result = {"ok": False}

    def choose(val: bool) -> None:
        result["ok"] = val
        win.destroy()

    btns = ttk.Frame(frame)
    btns.pack(fill="x", pady=(14, 0))
    ttk.Button(btns, text=cancel_text, command=lambda: choose(False)
               ).pack(side="right", padx=4)
    go = ttk.Button(btns, text=ok_text, command=lambda: choose(True))
    go.pack(side="right")
    win.bind("<Return>", lambda e: choose(True))
    win.bind("<Escape>", lambda e: choose(False))

    win.update_idletasks()
    px, py = parent.winfo_rootx(), parent.winfo_rooty()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    w, h = win.winfo_width(), win.winfo_height()
    win.geometry(f"+{px + max((pw - w) // 2, 0)}+{py + max((ph - h) // 3, 0)}")

    win.lift()
    go.focus_set()
    try:
        # A modal grab that leaks -- because the dialog was destroyed by an
        # exception rather than by a button -- leaves the main window visible
        # but deaf to every click, which is indistinguishable from a freeze.
        # So the grab is always released, whatever happens here.
        win.grab_set()
        parent.wait_window(win)
    finally:
        try:
            if parent.grab_current() is not None:
                parent.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
    return bool(result["ok"])


def set_text(widget: tk.Text, content: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


# ==========================================================================
# Main application
# ==========================================================================
class NavsimApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("navsim -- PlanetScope naval vessel simulator")
        self.geometry("1480x940")
        self.minsize(1180, 760)

        self.scene = None                 # current SceneProduct
        self.history = None               # current MovementHistory
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        # Worker threads must never touch Tk widgets; they push status strings
        # here and the main-thread poll loop renders them.
        self.msgq: "queue.Queue[str]" = queue.Queue()
        # Callables posted by background threads, run on the Tk thread.
        self.uiq: "queue.Queue" = queue.Queue()

        self.status = tk.StringVar(value="ready")
        self._build_progress_panel()
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        self._build_setup(nb)
        self._build_physics(nb)
        self._build_simulator(nb)
        self._build_campaign(nb)
        self._build_planet(nb)
        self._build_real(nb)
        self._build_wide(nb)
        # Land on Setup when nothing is configured, otherwise the simulator.
        nb.select(0 if not credentials.configured("planet") else 2)

        self.after(120, self._poll)
        self._raise_to_front()
        # Escape hatch: if a dialog ever leaves the app unclickable, this
        # releases the grab without needing to kill the process.
        self.bind_all("<Escape>", lambda e: self._unstick())


    # ---------------------------------------------------------------- jobs
    def _run_async(self, fn, done, label: str) -> None:
        """Run ``fn`` off the main thread; call ``done(result)`` back on it."""
        if getattr(self, "_busy", False):
            self.status.set("already working -- wait for the current job")
            return
        import time as _t
        self._busy = True
        self._job_id += 1
        job_id = self._job_id
        cancel_ev = threading.Event()
        self._cancel_flag = cancel_ev
        self._job_t0 = _t.time()
        self._job_frac = None
        self._job_text = label
        self.status.set(label)
        self.job_detail.set("")
        # Indeterminate until the job reports a real fraction: an animated
        # bar is honest about "working, extent unknown"; a bar parked at 0%
        # looks dead.
        self.job_pct.set(0.0)
        self.job_bar.configure(mode="indeterminate")
        self.job_bar.start(15)
        self.btn_cancel.configure(state="normal")
        self._tick()

        def worker():
            self._tls.cancel = cancel_ev
            self._tls.job = job_id
            try:
                self.jobs.put(("ok", done, fn(), job_id))
            except JobCancelled:
                self.jobs.put(("cancelled", done, None, job_id))
            except Exception:
                self.jobs.put(("err", done, traceback.format_exc(), job_id))

        threading.Thread(target=worker, daemon=True).start()

    def say(self, text: str, fraction: float | None = None,
            detail: str = "") -> None:
        """Thread-safe progress update, callable from a worker thread.

        ``fraction`` is overall completion in [0, 1]; pass ``None`` to leave
        the bar where it is and only change the text.  Raises
        :class:`JobCancelled` once the user has pressed Cancel, which is what
        lets every long loop be cancellable without plumbing a flag through
        it: they all already report progress.
        """
        ev = getattr(self._tls, "cancel", None) or self._cancel_flag
        if ev.is_set():
            raise JobCancelled()
        self.msgq.put({"text": text, "fraction": fraction, "detail": detail,
                       "job": getattr(self._tls, "job", self._job_id)})

    def _apply_progress(self, msg) -> None:
        if isinstance(msg, str):
            msg = {"text": msg, "fraction": None, "detail": ""}
        # A cancelled job's worker may still be finishing a download in the
        # background; its late messages must not overwrite the next job.
        if msg.get("job") is not None and msg["job"] != self._job_id:
            return
        text = msg.get("text") or ""
        self.status.set(text)
        self._job_text = text
        frac = msg.get("fraction")
        if frac is not None:
            frac = max(0.0, min(float(frac), 1.0))
            if str(self.job_bar.cget("mode")) != "determinate":
                self.job_bar.stop()
                self.job_bar.configure(mode="determinate", maximum=100.0)
            self._job_frac = frac
            self.job_pct.set(frac * 100.0)
        self.job_detail.set(msg.get("detail") or "")
        self._tick()

    def _poll(self) -> None:
        try:
            while True:
                fn = self.uiq.get_nowait()
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        try:
            while True:
                self._apply_progress(self.msgq.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                kind, done, payload, jid = self.jobs.get_nowait()
                if jid != self._job_id:
                    # Result of a job the user already cancelled: discard.
                    continue
                import time as _t
                el = _t.time() - self._job_t0
                self._busy = False
                self.job_bar.stop()
                self.job_bar.configure(mode="determinate")
                self.btn_cancel.configure(state="disabled")
                if kind == "cancelled":
                    self.job_pct.set(0.0)
                    self.job_label.set(f"cancelled after {fmt_clock(el)}")
                    self.job_detail.set("anything already downloaded is "
                                        "kept and will be reused")
                    self.status.set("cancelled")
                    continue
                if kind == "err":
                    self.status.set("failed")
                    self.job_pct.set(0.0)
                    self.job_label.set(f"failed after {fmt_clock(el)} -- "
                                       f"see the message")
                    self.job_detail.set("")
                    self.lift()
                    messagebox.showerror("navsim", payload[-2000:], parent=self)
                else:
                    done(payload)
                    self.job_pct.set(100.0)
                    self.job_label.set(f"done in {fmt_clock(el)}   "
                                       f"{self.status.get()}")
        except queue.Empty:
            pass
        self._beat = getattr(self, "_beat", 0) + 1
        if self._beat % 8 == 0:
            self._tick()
        # Re-arm the poll, and nothing else: this runs eight times a second,
        # so anything added here runs eight times a second too.  Raising the
        # window from inside this loop made the app impossible to interact
        # with -- it re-lifted and re-took focus faster than a click could
        # land.  Startup-only work belongs in __init__.
        self.after(120, self._poll)

    def _build_progress_panel(self) -> None:
        """One progress panel for the whole app, visible from every tab.

        It used to live on the Get-imagery tab only, so a long job started
        from any other tab -- radar detection is the worst, at 45 downloads
        for the default box -- showed nothing but a line of small text and
        then up to a minute of silence per download.  That reads as "the
        button did nothing".  Now every job gets a bar, a phase line, a
        detail line, a live elapsed clock, an estimate of time remaining
        once one can honestly be made, and a Cancel button.
        """
        panel = ttk.Frame(self, relief="groove", borderwidth=1)
        panel.pack(fill="x", side="bottom")

        top = ttk.Frame(panel)
        top.pack(fill="x", padx=6, pady=(5, 0))
        self.job_pct = tk.DoubleVar(value=0.0)
        self.job_bar = ttk.Progressbar(top, orient="horizontal",
                                       mode="determinate", maximum=100.0,
                                       variable=self.job_pct)
        self.job_bar.pack(side="left", fill="x", expand=True)
        self.btn_cancel = ttk.Button(top, text="Cancel", width=9,
                                     command=self._cancel_job,
                                     state="disabled")
        self.btn_cancel.pack(side="right", padx=(8, 0))

        self.job_label = tk.StringVar(value="idle")
        ttk.Label(panel, textvariable=self.job_label, anchor="w",
                  font=MONO).pack(fill="x", padx=8, pady=(3, 0))
        self.job_detail = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.job_detail, anchor="w",
                  foreground="#246", font=MONO).pack(fill="x", padx=8)
        ttk.Label(panel, textvariable=self.status, anchor="w",
                  foreground="#666").pack(fill="x", padx=8, pady=(0, 4))

        # Kept for older call sites; the panel above is what users see.
        self.progress = self.job_bar
        self._cancel_flag = threading.Event()
        self._job_id = 0
        self._tls = threading.local()
        self._job_t0 = 0.0
        self._job_frac = None
        self._job_text = ""

    def _cancel_job(self) -> None:
        """Cancel immediately, without waiting for a download to finish.

        A radar request can block for a minute inside one HTTP call, and
        nothing can interrupt that call.  So rather than waiting, the job is
        retired here and now: the UI is freed, the job id moves on so any
        late result is discarded, and the worker stops at its next progress
        report.  A download it was in the middle of still lands in the cache,
        which is a bonus, not a leak.
        """
        if not getattr(self, "_busy", False):
            return
        import time as _t
        self._cancel_flag.set()
        el = _t.time() - self._job_t0
        self._job_id += 1
        self._busy = False
        self.job_bar.stop()
        self.job_bar.configure(mode="determinate")
        self.job_pct.set(0.0)
        self.btn_cancel.configure(state="disabled")
        self.job_label.set(f"cancelled after {fmt_clock(el)}")
        self.job_detail.set("anything already downloaded is kept and will be "
                            "reused next time")
        self.status.set("cancelled")

    def _tick(self) -> None:
        """Heartbeat: keep the clock moving even when no message arrives."""
        if not getattr(self, "_busy", False):
            return
        import time as _t
        el = _t.time() - self._job_t0
        frac = self._job_frac
        if frac is not None and frac > 0.03:
            remain = el * (1.0 - frac) / frac
            eta = f"   ~{fmt_clock(remain)} left"
            pct = f"{frac * 100:3.0f}%  "
        else:
            eta = ""
            pct = ""
        self.job_label.set(f"{pct}{self._job_text}   "
                           f"[elapsed {fmt_clock(el)}{eta}]")

    def _raise_to_front(self) -> None:
        """Bring the window forward without seizing the screen.

        The obvious trick -- setting ``-topmost`` and calling ``focus_force``
        -- is wrong on macOS.  It pins the window above every other
        application and, worse, can leave it floating there without keyboard
        focus, so the entry fields cannot be clicked at all.  The cure was
        worse than the disease it was meant to fix.

        ``lift()`` on its own raises the window within the normal stacking
        order, which is all that is wanted.  Any stale grab from a dialog
        that failed to tear down is released here too, because a leaked grab
        has exactly the same symptom: a visible window that ignores clicks.
        """
        try:
            self.lift()
            self.attributes("-topmost", False)
        except Exception:
            pass
        try:
            if self.grab_current() is not None:
                self.grab_release()
        except Exception:
            pass

    def _unstick(self) -> None:
        """Release any modal grab and drop any stacking override."""
        try:
            if self.grab_current() is not None:
                self.grab_current().grab_release()
        except Exception:
            pass
        try:
            self.attributes("-topmost", False)
            self.lift()
        except Exception:
            pass
        self.status.set("released any stuck dialog grab")

    # ==================================================================
    # Tab: Setup -- every credential in one place
    # ==================================================================
    def _build_setup(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="0. Setup")

        ttk.Label(tab, text="Paste your keys here. Everything is optional: "
                            "the pipeline uses whatever is configured and "
                            "tells you what it cannot do without the rest.",
                  foreground="#444").pack(anchor="w", padx=10, pady=(10, 2))
        ttk.Label(tab, text=f"Stored at {credentials.CREDENTIALS_PATH} with "
                            f"owner-only permissions, outside the project so "
                            f"it cannot be committed. Shown only as ****last4.",
                  foreground="#246").pack(anchor="w", padx=10, pady=(0, 8))

        self.cred_vars: dict[tuple[str, str], tk.StringVar] = {}
        self.cred_status: dict[str, tk.StringVar] = {}

        for prov in PROVIDERS:
            box = ttk.LabelFrame(tab, text=prov.label)
            box.pack(fill="x", padx=10, pady=4)
            box.columnconfigure(1, weight=1)
            ttk.Label(box, text=prov.help, foreground="#555",
                      wraplength=820, justify="left").grid(
                row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 2))
            for i, field in enumerate(prov.fields):
                ttk.Label(box, text=field.replace("_", " ")).grid(
                    row=1 + i, column=0, sticky="w", padx=6, pady=2)
                var = tk.StringVar()
                self.cred_vars[(prov.key, field)] = var
                ttk.Entry(box, textvariable=var, show="*", width=52).grid(
                    row=1 + i, column=1, sticky="w")
            r = 1 + len(prov.fields)
            st = tk.StringVar()
            self.cred_status[prov.key] = st
            ttk.Label(box, textvariable=st, foreground="#246").grid(
                row=r, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))
            ttk.Button(box, text="Save",
                       command=lambda k=prov.key: self._save_cred(k)).grid(
                row=r, column=2, padx=3, pady=(2, 6))
            ttk.Button(box, text="Clear",
                       command=lambda k=prov.key: self._clear_cred(k)).grid(
                row=r, column=3, padx=3, pady=(2, 6))
            ttk.Label(box, text=f"sign up: {prov.signup}",
                      foreground="#0645ad").grid(
                row=r + 1, column=0, columnspan=4, sticky="w", padx=6,
                pady=(0, 6))

        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=10, pady=8)
        ttk.Button(bar, text="Test all connections",
                   command=self._test_all).pack(side="left")
        self.txt_setup = mono_text(tab, height=9, width=110)
        self._refresh_creds()

    def _refresh_creds(self) -> None:
        lines = []
        for s_ in credentials.status():
            mark = "ready" if s_["configured"] else "not set"
            self.cred_status[s_["key"]].set(
                f"{mark}   " + "   ".join(f"{k}: {v}"
                                          for k, v in s_["fields"].items()))
            lines.append(f"{s_['label']:48s} {mark}")
        set_text(self.txt_setup,
                 "\n".join(lines) + "\n\n"
                 "What each unlocks:\n"
                 "  Planet       Tier 3/4 - 3 m imagery, hull measurement, "
                 "classification\n"
                 "  Copernicus   Tier 1 - free Sentinel-1 radar wide-area "
                 "search (the big win)\n"
                 "  Earthdata    optional alternative radar source\n"
                 "  AIS          optional - US coastal AIS is free as CSV "
                 "and needs no key\n\n"
                 "Bathymetry for Tier 2 needs no key at all: NOAA ERDDAP is "
                 "fetched on demand.")

    def _save_cred(self, provider: str) -> None:
        prov = next(p for p in PROVIDERS if p.key == provider)
        vals = {f: self.cred_vars[(provider, f)].get()
                for f in prov.fields}
        if not any(v.strip() for v in vals.values()):
            messagebox.showinfo("navsim", "nothing to save", parent=self)
            return
        try:
            # Blank fields must not wipe an already-stored value.
            existing = credentials.get_all(provider)
            merged = {f: (vals[f].strip() or existing.get(f, ""))
                      for f in prov.fields}
            path = credentials.set_credentials(provider, **merged)
        except Exception as exc:
            messagebox.showerror("navsim", str(exc), parent=self)
            return
        for f in prov.fields:
            self.cred_vars[(provider, f)].set("")
        self._refresh_creds()
        self.status.set(f"saved {prov.label} to {path}")

    def _clear_cred(self, provider: str) -> None:
        credentials.clear(provider)
        self._refresh_creds()
        self.status.set(f"cleared {provider}")

    def _test_all(self) -> None:
        def work():
            out = []
            if credentials.configured("planet"):
                try:
                    from .planet import search
                    from .aoi import AOIS
                    aoi = AOIS["norfolk_base_piers"]
                    end = datetime.now(timezone.utc) - timedelta(days=40)
                    n = len(search(aoi, end - timedelta(days=10), end,
                                   max_cloud=1.0, limit=3))
                    out.append(f"Planet       OK  ({n} scenes returned)")
                except Exception as exc:
                    out.append(f"Planet       FAILED  {exc}")
            else:
                out.append("Planet       not configured")
            if credentials.configured("copernicus"):
                try:
                    from .sentinel import get_token
                    get_token(force=True)
                    out.append("Copernicus   OK  (token issued)")
                except Exception as exc:
                    out.append(f"Copernicus   FAILED  {exc}")
            else:
                out.append("Copernicus   not configured -- Tier 1 radar "
                           "search unavailable")
            try:
                from .bathymetry import fetch_erddap
                b = fetch_erddap((-76.5, 36.8, -76.0, 37.1))
                out.append(f"Bathymetry   OK  ({b.shape[0]}x{b.shape[1]} "
                           f"cells, no key needed)")
            except Exception as exc:
                out.append(f"Bathymetry   unreachable  {str(exc)[:70]}")
            return out

        def done(lines):
            set_text(self.txt_setup, "\n".join(lines))
            self.status.set("connection test complete")

        self._run_async(work, done, "testing connections...")

    # ==================================================================
    # Tab: Physics
    # ==================================================================
    def _build_physics(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="1. Physics (free)")

        top = ttk.Frame(tab)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="These answer what is worth attempting, before any "
                            "imagery is ordered.", foreground="#444"
                  ).pack(side="left")

        inner = ttk.Notebook(tab)
        inner.pack(fill="both", expand=True, padx=4, pady=4)

        # --- MTF budget --------------------------------------------------
        f_mtf = ttk.Frame(inner); inner.add(f_mtf, text="MTF budget")
        self.fig_mtf = Figure(figsize=(7, 4.4), dpi=96)
        self.canvas_mtf = FigureCanvasTkAgg(self.fig_mtf, master=f_mtf)
        self.canvas_mtf.get_tk_widget().pack(fill="both", expand=True)
        self.txt_mtf = mono_text(f_mtf, height=7)

        # --- contrast ----------------------------------------------------
        f_con = ttk.Frame(inner); inner.add(f_con, text="Hull / water contrast")
        self.txt_con = mono_text(f_con, height=20)

        # --- wake --------------------------------------------------------
        f_wake = ttk.Frame(inner); inner.add(f_wake, text="Wake resolvability")
        self.txt_wake = mono_text(f_wake, height=24)

        # --- identifiability --------------------------------------------
        f_id = ttk.Frame(inner); inner.add(f_id, text="Class identifiability")
        ctl = ttk.Frame(f_id); ctl.pack(fill="x", padx=4, pady=4)
        self.sig_len = Slider(ctl, "sigma length (m)", 0.5, 10.0, 3.0, "{:.1f}", 0)
        self.sig_beam = Slider(ctl, "sigma beam (m)", 0.5, 8.0, 2.0, "{:.1f}", 1)
        ctl.columnconfigure(1, weight=1)
        ttk.Button(ctl, text="Recompute", command=self._refresh_identifiability
                   ).grid(row=0, column=3, rowspan=2, padx=8)
        self.id_scope = tk.StringVar(value="naval only")
        ttk.Combobox(ctl, textvariable=self.id_scope, state="readonly", width=16,
                     values=("naval only", "all classes")
                     ).grid(row=0, column=4, rowspan=2, padx=4)
        self.txt_id = mono_text(f_id, height=24)

        self.after(200, self._refresh_physics)

    def _refresh_physics(self) -> None:
        sen = SensorConfig()
        gsd = 3.0

        m = system_mtf(sen, gsd)
        ax = self.fig_mtf.add_subplot(111)
        ax.plot(m["freq_over_nyquist"], m["optics"], label="optics + smear")
        ax.plot(m["freq_over_nyquist"], m["detector"], label="detector footprint")
        ax.plot(m["freq_over_nyquist"], m["system"], lw=2.2, label="system")
        ax.axvline(1.0, color="k", ls=":", lw=1)
        ax.set_xlabel("spatial frequency / Nyquist")
        ax.set_ylabel("MTF")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        self.fig_mtf.tight_layout()
        self.canvas_mtf.draw()
        set_text(self.txt_mtf,
                 f"GSD                {gsd:.1f} m\n"
                 f"Nyquist            {m['nyquist_cyc_per_m']:.4f} cyc/m\n"
                 f"PSF sigma          {m['psf_sigma_m']:.3f} m "
                 f"({m['psf_sigma_m'] / gsd:.3f} output px)\n"
                 f"Optics at Nyquist  {m['optics'][-1]:.3f}  (config value)\n"
                 f"Detector at Nyq.   {m['detector'][-1]:.3f}  = sinc(pi/2), "
                 f"applied by the box average to the output grid\n"
                 f"System at Nyquist  {m['system_at_nyquist']:.3f}\n"
                 f"navsim.sensor.slanted_edge_mtf recovers 0.203 against this "
                 f"0.204 on synthetic data: forward model and estimator agree.")

        rep = contrast_report(SeaConfig(), SimConfig().atmosphere)
        a, b = noise_coefficients(sen)
        lines = ["TOA reflectance contrast of hull materials against water.",
                 "Positive = brighter than water.  Grey paint is a HIGH contrast "
                 "target in the NIR;", "the low-contrast target is a coated "
                 "submarine.", "",
                 f"{'material':16s}" + "".join(f"{n[:7]:>9s}" for n in BAND_NAMES)]
        for k, v in rep.items():
            lines.append(f"{k:16s}" + "".join(f"{x:9.4f}" for x in v))
        lines += ["", f"{'per-pixel contrast SNR':16s}" +
                  "".join(f"{n[:7]:>9s}" for n in BAND_NAMES)]
        water = rep["water_toa"]
        for k, v in rep.items():
            if k == "water_toa":
                continue
            snr = []
            for i in range(len(BAND_NAMES)):
                sig = np.sqrt(a * max(water[i], 1e-4) + b)
                snr.append(v[i] / sig)
            lines.append(f"{k:16s}" + "".join(f"{x:9.0f}" for x in snr))
        set_text(self.txt_con, "\n".join(lines))

        wl = ["Transverse Kelvin wavelength lambda = 2 pi v^2 / g, and whether "
              "3 m sampling resolves it.",
              "Below about 4 pixels per wavelength an FFT cannot recover speed "
              "at all.", "",
              f"{'kn':>5s} {'v m/s':>7s} {'lambda m':>9s} {'px @3m':>7s} "
              f"{'resolvable':>11s} {'cusp 20m beam':>14s} {'cusp 6m beam':>13s}"]
        for kn in (4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35):
            v = kn / KNOTS
            r = wake_resolvability(v)
            wl.append(f"{kn:5d} {v:7.2f} {r['transverse_wavelength_m']:9.1f} "
                      f"{r['wavelength_px']:7.1f} {str(r['resolvable']):>11s} "
                      f"{narrow_wake_angle(v, 20.0):14.2f} "
                      f"{narrow_wake_angle(v, 6.0):13.2f}")
        v_min = wake_resolvability(1.0)["min_resolvable_speed_m_s"]
        wl += ["", f"Minimum speed for wake-FFT speed retrieval: "
                   f"{v_min:.2f} m/s = {v_min * KNOTS:.1f} kn.",
               "",
               "The cusp half-angle is 19.47 deg until the WIDTH-based Froude "
               "number v/sqrt(gB) exceeds 1.414.",
               "A 20 m-beam destroyer never reaches that, so a fixed 19.47 deg "
               "is safe for large combatants;",
               "a 6 m-beam fast boat does, and a fixed-angle heading estimator "
               "is badly biased there."]
        set_text(self.txt_wake, "\n".join(wl))

        self._refresh_identifiability()

    def _refresh_identifiability(self) -> None:
        classes = ([c for c in CATALOG.values() if c.domain is Domain.NAVAL]
                   if self.id_scope.get() == "naval only" else None)
        c = class_confusability(
            classes,
            sigma={"length_m": self.sig_len.get(), "beam_m": self.sig_beam.get()},
        )
        head = ("Analytic separability under measurement error -- no imagery "
                "needed.\nRun this before committing quota: it tells you which "
                "claims are available to you.\n\n")
        set_text(self.txt_id, head + c.report(max_pairs=16))

    # ==================================================================
    # Tab: Simulator
    # ==================================================================
    def _build_simulator(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="2. Simulator")

        left = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=6, pady=6)
        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=6)

        def box(title):
            f = ttk.LabelFrame(left, text=title)
            f.pack(fill="x", pady=3)
            f.columnconfigure(1, weight=1)
            return f

        g = box("Scene")
        self.sc_w = IntField(g, "width (px)", 400, 0, lo=64, hi=2000)
        self.sc_h = IntField(g, "height (px)", 400, 1, lo=64, hi=2000)
        self.sc_ss = IntField(g, "supersample", 4, 2, lo=1, hi=8)
        self.sc_seed = IntField(g, "seed", 11, 3, lo=0, hi=999999)
        self.port_style = tk.StringVar(value="naval_base")
        ttk.Label(g, text="port style").grid(row=4, column=0, sticky="w", padx=(2, 6))
        ttk.Combobox(g, textvariable=self.port_style, state="readonly", width=13,
                     values=("naval_base", "commercial", "open_water")
                     ).grid(row=4, column=1, columnspan=2, sticky="w")

        il = box("Illumination and view")
        self.sun_az = Slider(il, "sun azimuth", 0, 360, 150, "{:.0f}", 0, " deg")
        self.sun_el = Slider(il, "sun elevation", 5, 85, 45, "{:.0f}", 1, " deg")
        self.view_az = Slider(il, "view azimuth", 0, 360, 280, "{:.0f}", 2, " deg")
        self.view_zen = Slider(il, "off-nadir", 0, 25, 8, "{:.1f}", 3, " deg")
        self.lbl_glint = ttk.Label(il, text="", foreground="#8a4b00")
        self.lbl_glint.grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 2))
        for s in (self.sun_az, self.sun_el, self.view_az, self.view_zen):
            s.command = self._update_glint_label
        self._update_glint_label()

        se = box("Sea and atmosphere")
        self.wind = Slider(se, "wind speed", 0.5, 18, 6, "{:.1f}", 0, " m/s")
        self.wind_dir = Slider(se, "wind direction", 0, 360, 200, "{:.0f}", 1, " deg")
        self.turb = Slider(se, "turbidity", 0, 1, 0.25, "{:.2f}", 2)
        self.chl = Slider(se, "chlorophyll", 0.05, 5, 0.4, "{:.2f}", 3)
        self.cloud = Slider(se, "cloud fraction", 0, 0.9, 0.0, "{:.2f}", 4)

        tr = box("Traffic")
        self.n_naval_moored = IntField(tr, "naval, moored", 3, 0, lo=0, hi=40)
        self.n_civ_moored = IntField(tr, "civil, moored", 2, 1, lo=0, hi=40)
        self.n_underway = IntField(tr, "underway", 2, 2, lo=0, hi=40)
        self.n_other = IntField(tr, "'other' domain", 1, 3, lo=0, hi=20)
        self.uw_speed = Slider(tr, "underway speed", 0, 32, 18, "{:.0f}", 4, " kn")
        self.force_class = tk.StringVar(value="(mixed)")
        ttk.Label(tr, text="force class").grid(row=5, column=0, sticky="w", padx=(2, 6))
        ttk.Combobox(tr, textvariable=self.force_class, state="readonly", width=22,
                     values=("(mixed)",) + tuple(sorted(CATALOG))
                     ).grid(row=5, column=1, columnspan=2, sticky="w")

        rn = box("Forward model")
        self.r_wakes = tk.BooleanVar(value=True)
        self.r_shadows = tk.BooleanVar(value=True)
        self.r_layover = tk.BooleanVar(value=True)
        self.r_clutter = tk.BooleanVar(value=True)
        for i, (lbl, var) in enumerate((("wakes", self.r_wakes),
                                        ("shadows", self.r_shadows),
                                        ("off-nadir layover", self.r_layover),
                                        ("deck clutter", self.r_clutter))):
            ttk.Checkbutton(rn, text=lbl, variable=var).grid(
                row=i, column=0, columnspan=3, sticky="w")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Generate scene", command=self._generate
                   ).pack(fill="x", pady=2)
        ttk.Button(btns, text="Random seed + generate", command=self._reseed
                   ).pack(fill="x", pady=2)
        ttk.Button(btns, text="Run full pipeline on this scene",
                   command=self._run_detector).pack(fill="x", pady=2)
        ttk.Button(btns, text="Benchmark over many scenes",
                   command=self._run_benchmark).pack(fill="x", pady=2)
        exp = ttk.Frame(left); exp.pack(fill="x")
        ttk.Button(exp, text="Save PNG", command=self._save_png
                   ).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(exp, text="GeoTIFF", command=self._save_tif
                   ).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(exp, text="Truth CSV", command=self._save_csv
                   ).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(exp, text="GeoJSON", command=self._save_geojson
                   ).pack(side="left", expand=True, fill="x", padx=1)

        # ---- display ----------------------------------------------------
        disp = ttk.Frame(right); disp.pack(fill="x")
        ttk.Label(disp, text="display").pack(side="left", padx=(0, 4))
        self.disp_mode = tk.StringVar(value="RGB")
        cb = ttk.Combobox(disp, textvariable=self.disp_mode, state="readonly",
                          width=14,
                          values=("RGB",) + BAND_NAMES + ("UDM2",))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._draw_scene())
        ttk.Label(disp, text="  max").pack(side="left")
        self.vmax = tk.DoubleVar(value=0.30)
        sc = ttk.Scale(disp, from_=0.02, to=0.8, variable=self.vmax,
                       orient="horizontal", length=140,
                       command=lambda e: self._draw_scene())
        sc.pack(side="left")
        self.show_truth = tk.BooleanVar(value=True)
        self.show_dets = tk.BooleanVar(value=True)
        self.show_outline = tk.BooleanVar(value=True)
        self.show_box = tk.BooleanVar(value=False)
        self.show_labels = tk.BooleanVar(value=True)
        self.hide_rejected = tk.BooleanVar(value=True)
        for lbl, var in (("truth", self.show_truth),
                         ("detections", self.show_dets),
                         ("outline", self.show_outline),
                         ("box", self.show_box),
                         ("labels", self.show_labels),
                         ("hide rejected", self.hide_rejected)):
            ttk.Checkbutton(disp, text=lbl, variable=var,
                            command=self._draw_scene).pack(side="left", padx=3)
        ttk.Label(disp, text=" min L").pack(side="left")
        self.min_len = tk.DoubleVar(value=0.0)
        ttk.Spinbox(disp, from_=0, to=400, increment=10, width=5,
                    textvariable=self.min_len,
                    command=self._draw_scene).pack(side="left")

        self.fig_scene = Figure(figsize=(7.6, 6.2), dpi=96)
        self.canvas_scene = FigureCanvasTkAgg(self.fig_scene, master=right)
        self.canvas_scene.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_scene, right).update()

        cols = ("id", "class", "domain", "L true", "L apparent", "B true",
                "px", "state", "berth", "kn")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (62, 160, 70, 66, 82, 62, 56, 62, 128, 48)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x", pady=(4, 0))
        self.lbl_scene = ttk.Label(right, text="", foreground="#444")
        self.lbl_scene.pack(fill="x")

        ttk.Label(right, text="pipeline output (measured, not truth)",
                  foreground="#246").pack(fill="x", pady=(6, 0))
        rcols = ("det", "domain", "P(naval)", "category", "L m", "B m",
                 "hdg", "R2", "rej")
        self.rtree = ttk.Treeview(right, columns=rcols, show="headings",
                                  height=7)
        for c, w in zip(rcols, (62, 70, 70, 150, 58, 54, 46, 46, 40)):
            self.rtree.heading(c, text=c)
            self.rtree.column(c, width=w, anchor="center")
        self.rtree.pack(fill="x", pady=(2, 0))

        self.detections: list = []
        self.annotations: list = []
        self.corridor = None
        self.results: list = []

    def _update_glint_label(self) -> None:
        g = GeometryConfig(self.sun_az.get(), self.sun_el.get(),
                           self.view_az.get(), self.view_zen.get())
        sep = g.glint_separation_deg()
        warn = "  <-- near the glitter point, expect heavy glint" if sep < 12 else ""
        self.lbl_glint.config(
            text=f"glint separation {sep:.1f} deg | shadow {g.shadow_scale:.2f}x "
                 f"height | layover {g.layover_scale:.3f}x height{warn}")

    def _build_config(self) -> SimConfig:
        return SimConfig(
            grid=GridConfig(self.sc_w.get(), self.sc_h.get(), 3.0, self.sc_ss.get()),
            geometry=GeometryConfig(self.sun_az.get(), self.sun_el.get(),
                                    self.view_az.get(), self.view_zen.get()),
            sea=SeaConfig(self.wind.get(), self.wind_dir.get(), self.chl.get(),
                          self.turb.get()),
            cloud=CloudConfig(cloud_fraction=self.cloud.get()),
            seed=self.sc_seed.get(),
            render_wakes=self.r_wakes.get(),
            render_shadows=self.r_shadows.get(),
            render_layover=self.r_layover.get(),
            deck_clutter=self.r_clutter.get(),
        )

    def _compose(self):
        """Build a scene from the traffic controls (not the campaign model)."""
        cfg = self._build_config()
        port = make_port(cfg.grid, cfg.seed + 7, "gui", self.port_style.get())
        rng = rng_for(cfg.seed, "gui_traffic")
        W, H = cfg.grid.extent_m

        forced = self.force_class.get()
        def pick(domain):
            if forced != "(mixed)":
                return CATALOG[forced]
            pool = [c for c in CATALOG.values() if c.domain is domain]
            w = np.array([c.abundance for c in pool], float)
            return pool[int(rng.choice(len(pool), p=w / w.sum()))]

        ships, insts = [], {}
        n = 0
        plan = ([(Domain.NAVAL, True)] * self.n_naval_moored.get()
                + [(Domain.MERCHANT, True)] * self.n_civ_moored.get()
                + [(Domain.OTHER, False)] * self.n_other.get()
                + [(None, False)] * self.n_underway.get())
        for domain, moored in plan:
            cls = pick(domain if domain is not None else
                       (Domain.NAVAL if rng.random() < 0.5 else Domain.MERCHANT))
            inst = sample_instance(cls, rng, ship_id=f"G{n:02d}")
            insts[inst.ship_id] = inst
            n += 1
            if moored and port.berths:
                b = (port.berth_for(inst.length_m, rng,
                                    naval=(cls.domain is Domain.NAVAL))
                     or port.berth_for(inst.length_m, rng) or port.berths[0])
                x, y, hd = b.mooring_pose(inst.beam_m)
                pose, bid, mo = Pose(x, y, hd, 0.0), b.berth_id, True
            else:
                pose = Pose(float(rng.uniform(0.1 * W, 0.9 * W)),
                            float(rng.uniform(0.08 * H, 0.60 * H)),
                            float(rng.uniform(0, 360)),
                            self.uw_speed.get() / KNOTS)
                bid, mo = None, False
            ships.append(ShipState(
                inst.ship_id, cls.key, pose, mo, bid, inst.length_m,
                inst.beam_m, cls.domain.value, cls.category.value,
                inst.broadcasts_ais, inst.mmsi))
        return cfg, port, ships, insts

    def _generate(self) -> None:
        try:
            cfg, port, ships, insts = self._compose()
        except Exception as exc:
            messagebox.showerror("navsim", str(exc))
            return
        px = cfg.grid.width * cfg.grid.height * cfg.grid.supersample ** 2
        self._run_async(
            lambda: simulate_scene(cfg, port, ships, insts, scene_id="GUI"),
            self._on_scene,
            f"rendering {cfg.grid.width}x{cfg.grid.height} px "
            f"({cfg.grid.area_km2:.2f} km2, {px / 1e6:.1f} Mpx canvas), "
            f"{len(ships)} ships...")

    def _reseed(self) -> None:
        self.sc_seed.var.set(int(np.random.default_rng().integers(1, 999_999)))
        self._generate()

    def _on_scene(self, scene) -> None:
        self.scene = scene
        self.detections = []
        self.annotations = []
        self.corridor = None
        self.status.set(f"scene ready: {scene.cfg.grid.area_km2:.2f} km2, "
                        f"{len(scene.truth)} ships")
        self._draw_scene()
        self._fill_tree()
        m = scene.meta
        self.lbl_scene.config(
            text=f"sun {m['sun_azimuth']:.0f}/{m['sun_elevation']:.0f} deg | "
                 f"view {m['view_azimuth']:.0f}/{m['view_zenith']:.1f} deg | "
                 f"glint sep {m['glint_separation_deg']:.1f} deg | "
                 f"wind {m['wind_speed']:.1f} m/s | "
                 f"Hs {4 * m['sea_sigma_eta_m']:.2f} m | "
                 f"cloud {m['cloud_fraction']:.2f} | "
                 f"mean glint {m['mean_glint_reflectance']:.5f}")

    def _draw_scene(self) -> None:
        if self.scene is None:
            return
        sp = self.scene
        self.fig_scene.clear()
        ax = self.fig_scene.add_subplot(111)
        mode = self.disp_mode.get()
        if mode == "RGB":
            ax.imshow(sp.rgb(stretch=(0.0, float(self.vmax.get()))))
        elif mode == "UDM2":
            ax.imshow(sp.udm2, cmap="tab10", vmin=0, vmax=9,
                      interpolation="nearest")
        else:
            ax.imshow(sp.band(mode), cmap="gray", vmin=0,
                      vmax=float(self.vmax.get()))
        ax.set_title(f"{mode}  |  3 m GSD  |  {sp.cfg.grid.area_km2:.2f} km2",
                     fontsize=9)
        ax.axis("off")

        if self.show_truth.get():
            gsd = sp.cfg.grid.gsd
            for t in sp.truth:
                c = t["x_m"] / gsd
                r = sp.cfg.grid.height - t["y_m"] / gsd
                col = {"naval": "#ff2d2d", "merchant": "#30d5ff",
                       "other": "#ffd400"}.get(t["domain"], "w")
                ax.plot(c, r, marker="o", mfc="none", mec=col, ms=13, mew=1.5)
                ax.annotate(t["ship_id"], (c, r), xytext=(8, 6),
                            textcoords="offset points", color=col, fontsize=7)
        if self.show_dets.get() and self.annotations:
            shown = filter_annotations(
                self.annotations, hide_rejected=self.hide_rejected.get(),
                min_length_m=float(self.min_len.get()))
            draw_annotations(ax, shown,
                             show_outline=self.show_outline.get(),
                             show_box=self.show_box.get(),
                             show_labels=self.show_labels.get())
        elif self.show_dets.get() and self.detections:
            for d in self.detections:
                ax.plot(d.col, d.row, marker="x", color="#00ff66", ms=9, mew=1.6)
        if self.corridor is not None:
            draw_corridor(ax, self.corridor)
        self.fig_scene.tight_layout()
        self.canvas_scene.draw()

    def _fill_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if self.scene is None:
            return
        for t in self.scene.truth:
            state = "moored" if t["moored"] else (
                "slow" if t["speed_kn"] < 3 else "underway")
            self.tree.insert("", "end", values=(
                t["ship_id"], t["class_key"], t["domain"],
                f"{t['length_m']:.1f}", f"{t['apparent_length_m']:.1f}",
                f"{t['beam_m']:.1f}", f"{t['pixels_on_target']:.0f}",
                state, t.get("berth_id") or "-", f"{t['speed_kn']:.0f}"))

    def _run_detector(self) -> None:
        """Run the real detection + measurement + classification stages."""
        if self.scene is None:
            messagebox.showinfo("navsim", "generate a scene first")
            return
        sp = self.scene
        sc = as_scene(sp)
        aoi_kind = ("base" if self.port_style.get() == "naval_base"
                    else "commercial")

        def work():
            det = ShipDetector()
            res = det.run(sc)
            meas = HullFitMeasurer().measure(sc, res.detections)
            clf = TwoStageClassifier(
                stage_a=__import__(
                    "navsim.pipeline.classify", fromlist=["x"]
                ).NavalVsCivilClassifier(
                    prior=LocationPrior.for_context({"aoi_kind": aoi_kind})))
            a, b = clf.classify(meas, {"aoi_kind": aoi_kind})
            dm = detection_metrics(sp.truth, res.detections,
                                   sp.cfg.grid.area_km2)
            mm = measurement_metrics(sp.truth, meas, res.detections)
            return res, meas, a, b, dm, mm

        def done(out):
            res, meas, a, b, dm, mm = out
            self.detections = res.detections
            self.annotations = build_annotations(
                res.detections, meas, a, b, gsd=sp.cfg.grid.gsd)
            self.results = meas
            self._fill_results()
            self._draw_scene()
            self.status.set(f"pipeline: {len(res.detections)} detections, "
                            f"Pd={dm.pd:.2f}")
            by_m = {m.det_id: m for m in meas}
            lines = [res.report(), "", dm.report(), "", mm.report(), "",
                     "CLASSIFICATION (posterior and prior-free log LR)", "",
                     f"  {'det':10s} {'P(naval)':>9s} {'logLR':>8s} "
                     f"{'L m':>7s} {'B m':>6s} {'rej':>4s} {'category':>26s}"]
            for p in a:
                m = by_m.get(p.det_id)
                cat = b.get(p.det_id)
                label = "-"
                if cat is not None:
                    k, v = cat.top()
                    label = f"{k} {v:.2f}"
                lines.append(
                    f"  {p.det_id.split(':')[-1][:10]:10s} "
                    f"{(p.p_naval or 0):9.3f} "
                    f"{(p.log_likelihood_ratio or 0):8.2f} "
                    f"{(m.length_m if m else float('nan')):7.1f} "
                    f"{(m.beam_m if m else float('nan')):6.1f} "
                    f"{'Y' if p.rejected else '':>4s} {label[:26]:>26s}")
            lines += ["",
                      "Uncertainties here are DEFAULTS, not calibrated: there "
                      "is no AIS in a synthetic",
                      "scene. On real imagery, supply AIS so navsim.calibrate "
                      "can measure the error",
                      "model before any of these posteriors is treated as "
                      "quantitative."]
            set_text(self.txt_pipe, "\n".join(lines))

        self._run_async(work, done, "running detection, measurement and "
                                    "classification...")

    # ---- export ------------------------------------------------------
    def _save_png(self) -> None:
        if self.scene is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.fig_scene.savefig(p, dpi=150, bbox_inches="tight")
            self.status.set(f"wrote {p}")

    def _save_tif(self) -> None:
        if self.scene is None:
            return
        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            messagebox.showerror("navsim", "rasterio is not installed")
            return
        p = filedialog.asksaveasfilename(defaultextension=".tif",
                                         filetypes=[("GeoTIFF", "*.tif")])
        if not p:
            return
        sp = self.scene
        aoi = AOIS[self.aoi_key.get()] if hasattr(self, "aoi_key") else None
        lon, lat = aoi.centre if aoi else (0.0, 0.0)
        # Local UTM-like placement so the file opens in GIS at the AOI.
        zone = int((lon + 180) / 6) + 1
        crs = f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"
        try:
            from pyproj import Transformer
            tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            cx, cy = tx.transform(lon, lat)
        except Exception:
            cx, cy = 0.0, 0.0
        g = sp.cfg.grid
        tr = from_origin(cx - g.width * g.gsd / 2,
                         cy + g.height * g.gsd / 2, g.gsd, g.gsd)
        with rasterio.open(p, "w", driver="GTiff", height=g.height,
                           width=g.width, count=sp.dn.shape[0], dtype="uint16",
                           crs=crs, transform=tr,
                           compress="deflate") as dst:
            dst.write(sp.dn)
            for i, n in enumerate(BAND_NAMES, start=1):
                dst.set_band_description(i, n)
        self.status.set(f"wrote {p}")

    def _save_geojson(self) -> None:
        """Export detections as vectors that open over the scene in QGIS."""
        if not self.annotations:
            messagebox.showinfo("navsim", "run the pipeline first")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".geojson",
            filetypes=[("GeoJSON", "*.geojson")])
        if not path:
            return
        n = write_geojson(path, self.annotations, as_scene(self.scene))
        self.status.set(f"wrote {n} features to {path}")

    def _fill_results(self) -> None:
        """Populate the measurement/signature table under the image."""
        self.rtree.delete(*self.rtree.get_children())
        for a in self.annotations:
            q = a.quality or {}
            self.rtree.insert("", "end", values=(
                a.det_id.split(":")[-1],
                a.domain,
                f"{a.p_naval:.3f}" if a.p_naval is not None else "-",
                f"{a.category} {a.category_prob:.2f}" if a.category else "-",
                f"{a.length_m:.1f}" if a.length_m else "-",
                f"{a.beam_m:.1f}" if a.beam_m else "-",
                f"{a.heading_deg:.0f}" if a.heading_deg is not None else "-",
                f"{q.get('r_squared', float('nan')):.2f}",
                "yes" if a.rejected else "",
            ))

    def _save_csv(self) -> None:
        if self.scene is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        rows = self.scene.truth
        keys = sorted({k for r in rows for k in r})
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        self.status.set(f"wrote {p} ({len(rows)} rows)")

    def _run_benchmark(self) -> None:
        from .benchmark import run_benchmark
        n = 6

        def work():
            return run_benchmark(
                n_scenes=n, size=min(self.sc_w.get(), 360),
                progress=lambda i, k: self.say(
                    f"benchmark scene {i} of {k}", i / max(k, 1)))

        def done(r):
            set_text(self.txt_pipe, r.report())
            self.status.set(f"benchmark: {len(r.records)} vessel-observations "
                            f"over {r.n_scenes} scenes")

        self._run_async(work, done,
                        f"benchmarking over {n} scenes (this takes a while)...")

    # ==================================================================
    # Tab: Campaign
    # ==================================================================
    def _build_campaign(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="3. Campaign")

        ctl = ttk.LabelFrame(tab, text="Sampling model")
        ctl.pack(fill="x", padx=6, pady=6)
        ctl.columnconfigure(1, weight=1)
        self.cp_days = IntField(ctl, "days", 60, 0, lo=5, hi=365)
        self.cp_naval = IntField(ctl, "naval units homeported", 6, 1, lo=1, hi=30)
        self.cp_seed = IntField(ctl, "seed", 3, 2, lo=0, hi=999999)
        self.cp_lat = Slider(ctl, "latitude", -60, 60, 36.95, "{:.2f}", 3, " deg")
        self.cp_hour = Slider(ctl, "local pass hour", 8, 14, 10.5, "{:.2f}", 4)
        ttk.Button(ctl, text="Generate campaign", command=self._gen_campaign
                   ).grid(row=0, column=3, rowspan=2, padx=8, sticky="ns")
        ttk.Button(ctl, text="Render first usable scene",
                   command=self._campaign_scene
                   ).grid(row=2, column=3, rowspan=3, padx=8, sticky="ns")
        self.txt_camp = mono_text(tab, height=30)
        set_text(self.txt_camp,
                 "Generate a campaign to see what the pipeline will actually be "
                 "fed.\n\nThe sampling model is honest about three things that "
                 "decide feasibility:\n"
                 "  - sorties make units ABSENT for days, not slow;\n"
                 "  - cloud removes RUNS of days, via a two-state Markov chain;\n"
                 "  - sun elevation swings by tens of degrees across the year, "
                 "and shadow\n    height accuracy goes as 1/tan(elevation).")

    def _gen_campaign(self) -> None:
        cfg = self._build_config()
        port = make_port(cfg.grid, cfg.seed + 7, "campaign", "naval_base")
        days, nn = self.cp_days.get(), self.cp_naval.get()
        seed, lat, hour = self.cp_seed.get(), self.cp_lat.get(), self.cp_hour.get()

        def work():
            h = generate_history(port, n_days=days, seed=seed, n_naval=nn,
                                 latitude=lat, local_pass_hour=hour)
            rng = np.random.default_rng(seed)
            ns, nais = [], []
            mix = {"moored": 0, "slow": 0, "underway": 0}
            for p in h.passes:
                if not p.usable:
                    continue
                st = h.states_at(p.timestamp)
                ns.append(len(st))
                nais.append(len(h.ais_at(p.timestamp, rng)))
                for s in st:
                    k = ("moored" if s.moored else
                         "slow" if s.pose.speed < 3 else "underway")
                    mix[k] += 1
            return h, h.visibility_summary(), ns, nais, mix

        def done(res):
            h, vs, ns, nais, mix = res
            self.history = h
            els = [p.geometry.sun_elevation for p in h.passes]
            lines = [
                f"Passes scheduled      {vs['n_passes']}",
                f"Usable (cloud < 0.45) {vs['n_usable_passes']}",
                f"Lost to cloud         {vs['cloud_loss_fraction']:.1%}",
                f"Sun elevation range   {min(els):.1f} - {max(els):.1f} deg "
                f"-> shadow scale {1 / np.tan(np.radians(max(els))):.2f}x to "
                f"{1 / np.tan(np.radians(min(els))):.2f}x height",
                f"Ships per usable scene  mean {np.mean(ns):.1f}  "
                f"range {min(ns)}-{max(ns)}" if ns else "no usable scenes",
                f"AIS records per scene   mean {np.mean(nais):.1f}  "
                f"<- this is your calibration set" if nais else "",
                f"State mix               {mix}",
                "",
                "Naval unit visibility across usable passes:",
                f"  {'unit':8s} {'class':24s} {'alongside':>10s} "
                f"{'underway in AOI':>16s} {'not visible':>12s}",
            ]
            tot_uw = 0
            for k, v in vs["units"].items():
                tot_uw += v["underway_in_aoi"]
                lines.append(f"  {k:8s} {v['class']:24s} {v['alongside']:10d} "
                             f"{v['underway_in_aoi']:16d} {v['not_visible']:12d}")
            denom = max(vs["n_usable_passes"] * len(vs["units"]), 1)
            lines += [
                "",
                f"Naval unit-passes caught underway inside the AOI: "
                f"{tot_uw} of {denom} ({tot_uw / denom:.1%}).",
                "",
                "That number is the reason Section 7 has to be rebuilt around "
                "berth occupancy.",
                "A homeported unit is either alongside or gone; the transit "
                "window is an hour or two",
                "out of days, so one look per day almost never catches it "
                "moving. Combined with a",
                "24 h reachable area of 5.6 million km2 at 30 kn, inter-day "
                "at-sea association is not",
                "recoverable. Berth-occupancy state, gate counts and group "
                "sailings are where the",
                "information actually is.",
            ]
            set_text(self.txt_camp, "\n".join(x for x in lines if x != ""))
            self.status.set(f"campaign: {vs['n_usable_passes']} usable passes")

        self._run_async(work, done, f"generating {days}-day campaign...")

    def _campaign_scene(self) -> None:
        if self.history is None:
            messagebox.showinfo("navsim", "generate a campaign first")
            return
        from .movement import iter_scenes
        base = self._build_config()
        hist = self.history

        def work():
            for p, sp in iter_scenes(hist, base):
                return p, sp
            raise RuntimeError("no usable passes in this campaign")

        def done(res):
            p, sp = res
            self._on_scene(sp)
            self.status.set(f"campaign scene {p.scene_id} at "
                            f"{p.timestamp:%Y-%m-%d %H:%M}Z")

        self._run_async(work, done, "rendering first usable campaign scene...")

    # ==================================================================
    # Tab: Planet
    # ==================================================================
    def _build_planet(self, nb) -> None:
        """Guided flow: key, place, dates, one button for everything."""
        tab = ttk.Frame(nb)
        nb.add(tab, text="4. Get imagery")

        kf = ttk.LabelFrame(tab, text="Step 1  -  Planet API key")
        kf.pack(fill="x", padx=8, pady=(8, 4))
        self.key_status = tk.StringVar()
        ttk.Label(kf, textvariable=self.key_status, foreground="#246").pack(
            anchor="w", padx=8, pady=(8, 2))
        ttk.Button(kf, text="Open Setup tab to paste keys",
                   command=lambda: self.nb.select(0)).pack(anchor="w", padx=8,
                                                           pady=(0, 8))
        self._refresh_key_status()

        # ---- step 2: place ----------------------------------------------
        pf = ttk.LabelFrame(tab, text="Step 2  -  Where")
        pf.pack(fill="x", padx=8, pady=4)
        ttk.Label(pf, text="place").grid(row=0, column=0, sticky="w", padx=6,
                                         pady=6)
        self.aoi_key = tk.StringVar(value="norfolk_base_piers")
        ac = ttk.Combobox(pf, textvariable=self.aoi_key, state="readonly",
                          width=26, values=tuple(AOIS))
        ac.grid(row=0, column=1, sticky="w")
        ac.bind("<<ComboboxSelected>>", lambda e: self._refresh_quota())
        self.aoi_info = tk.StringVar()
        ttk.Label(pf, textvariable=self.aoi_info, foreground="#246").grid(
            row=0, column=2, sticky="w", padx=10)
        self.region = tk.StringVar(value=DEFAULT_REGION)

        # ---- step 3: when -----------------------------------------------
        df = ttk.LabelFrame(tab, text="Step 3  -  When")
        df.pack(fill="x", padx=8, pady=4)
        end_default = datetime.now(timezone.utc) - timedelta(days=WITHHOLD_PAD)
        ttk.Label(df, text="last").grid(row=0, column=0, sticky="w", padx=6,
                                        pady=6)
        self.days_back = tk.IntVar(value=60)
        ttk.Spinbox(df, from_=7, to=365, textvariable=self.days_back, width=6
                    ).grid(row=0, column=1, sticky="w")
        ttk.Label(df, text="days, ending").grid(row=0, column=2, sticky="w",
                                                padx=4)
        self.d_end = tk.StringVar(value=end_default.strftime("%Y-%m-%d"))
        ttk.Entry(df, textvariable=self.d_end, width=12).grid(row=0, column=3,
                                                              sticky="w")
        ttk.Label(df, text="max cloud").grid(row=0, column=4, sticky="w",
                                             padx=(14, 4))
        self.max_cloud = tk.DoubleVar(value=0.25)
        ttk.Spinbox(df, from_=0.0, to=1.0, increment=0.05, width=6,
                    textvariable=self.max_cloud).grid(row=0, column=5,
                                                      sticky="w")
        ttk.Label(df, text=f"30-day withhold: nothing newer than "
                           f"{end_default:%Y-%m-%d} exists",
                  foreground="#8a4b00").grid(row=1, column=0, columnspan=6,
                                             sticky="w", padx=6, pady=(0, 6))
        self.d_start = tk.StringVar()

        # ---- step 4: go --------------------------------------------------
        gf = ttk.LabelFrame(tab, text="Step 4  -  Get it")
        gf.pack(fill="x", padx=8, pady=4)
        ttk.Label(gf, text="save to").grid(row=0, column=0, sticky="w", padx=6,
                                           pady=6)
        self.dl_dir = tk.StringVar(
            value=str(Path.home() / "navsim_data" / "norfolk_base_piers"))
        ttk.Entry(gf, textvariable=self.dl_dir, width=52).grid(row=0, column=1,
                                                               sticky="w")
        ttk.Button(gf, text="Browse", command=lambda: self.dl_dir.set(
            filedialog.askdirectory() or self.dl_dir.get())
        ).grid(row=0, column=2, padx=4)
        ttk.Button(gf, text="Check what's available",
                   command=self._search).grid(row=1, column=0, columnspan=2,
                                              sticky="w", padx=6, pady=6)
        self.btn_go = ttk.Button(
            gf, text="Order, download and analyse",
            command=self._order_download_analyse)
        self.btn_go.grid(row=1, column=1, sticky="e", padx=6)
        self.max_scenes = tk.IntVar(value=30)
        ttk.Label(gf, text="max scenes").grid(row=1, column=2, sticky="e")
        ttk.Spinbox(gf, from_=1, to=200, textvariable=self.max_scenes,
                    width=6).grid(row=1, column=3, sticky="w", padx=4)

        self.txt_quota = mono_text(tab, height=10, width=112)
        self.txt_search = mono_text(tab, height=14, width=112)
        self._search_results: list = []
        self._refresh_quota()
        set_text(self.txt_search,
                 "Press 'Check what's available' first. Nothing is ordered and "
                 "no quota is spent by\nsearching -- it only reads metadata.\n\n"
                 "Then 'Order, download and analyse' does the whole thing: "
                 "submits a clipped order,\nwaits for Planet to process it, "
                 "downloads the imagery with its UDM2 masks and\nmetadata, and "
                 "runs the pipeline. Results land on the Real imagery tab.")

    def _order_download_analyse(self) -> None:
        """Search -> clipped order -> download -> pipeline, in one go.

        If no search has been run for the currently selected place, this runs
        it first rather than refusing: a button that silently does nothing is
        indistinguishable from a frozen application.
        """
        aoi = AOIS[self.aoi_key.get()]
        if not self._search_results or getattr(self, "_search_aoi", None) != aoi.key:
            self.status.set("checking what is available first...")
            self._search(then=self._order_download_analyse)
            return
        self._do_order()

    def _do_order(self) -> None:
        from .planet import describe_downloaded, order_download_and_wait
        aoi = AOIS[self.aoi_key.get()]
        picks = self._search_results[:max(1, int(self.max_scenes.get()))]
        ids = [s.item_id for s in picks]
        from .planet import estimate_order_km2
        est = estimate_order_km2(aoi, len(ids))
        dest = Path(self.dl_dir.get().strip() or
                    (Path.home() / "navsim_data" / aoi.key))

        ok = confirm_dialog(
            self, "Confirm  -  this spends download quota",
            f"Place:    {aoi.name}\n"
            f"Scenes:   {len(ids)}\n"
            f"Area:     {aoi.area_km2():.2f} km2 each\n\n"
            f"DOWNLOAD: about {est:.1f} km2\n"
            f"That is {est / MONTHLY_QUOTA_KM2 * 100:.1f}% of your "
            f"{MONTHLY_QUOTA_KM2:.0f} km2 monthly quota.\n\n"
            f"Saving to {dest}\n\n"
            f"Planet processing usually takes a few minutes.",
            ok_text="Order it")
        if not ok:
            self.status.set("cancelled")
            return

        # Overall progress is split into weighted phases.  Planet's own
        # processing has no completion signal -- the order sits in "running"
        # until it does not -- so that band advances on an asymptotic curve
        # against elapsed time and is labelled with a clock, rather than
        # pretending to know a percentage it cannot know.
        P_ORDER, P_PROC, P_DL, P_END = 0.06, 0.55, 0.90, 1.00

        def work():
            import math as _m
            self.say(f"submitting order for {len(ids)} scenes...", 0.01)

            def on_state(state, elapsed):
                frac = P_ORDER + (P_PROC - P_ORDER) * (
                    1.0 - _m.exp(-elapsed / 240.0))
                mins = elapsed / 60.0
                self.say(f"Planet is processing the order ({state})",
                         frac,
                         f"{mins:.1f} min elapsed  -  a few minutes is normal; "
                         f"this is their side, not yours")

            seen = {"bytes": 0}

            def on_file(i, n, name, done_b=0, total_b=0):
                within = (done_b / total_b) if total_b else 0.0
                frac = P_PROC + (P_DL - P_PROC) * ((i - 1 + within) / max(n, 1))
                size = (f"{human_bytes(done_b)} of {human_bytes(total_b)}"
                        if total_b else human_bytes(done_b))
                self.say(f"downloading file {i} of {n}", frac,
                         f"{name[:52]}   {size}")
                seen["bytes"] = done_b

            oid, files = order_download_and_wait(
                aoi, ids, dest, on_state=on_state, on_file=on_file)

            self.say("reading the downloaded scenes...", P_DL)
            scenes = load_directory(dest)
            if not scenes:
                self.say("no analytic scenes found", 1.0)
                return oid, files, None, None, dest

            self.say(f"building static masks from {len(scenes)} dates...",
                     P_DL + 0.01)
            def step(i, n):
                self.say(f"running the pipeline: scene {i} of {n}",
                         P_DL + (P_END - P_DL) * 0.15 + (P_END - P_DL) * 0.75
                         * (i / max(n, 1)),
                         "detect, measure, calibrate, classify")

            res = run_pipeline(scenes, aoi_kind=aoi.kind, progress=step)
            self.say("drawing outlines...", 0.98)
            anns = {}
            for sc, out in zip(scenes, res.scenes):
                anns[sc.scene_id] = build_annotations(
                    out.detections, out.measurements, out.stage_a,
                    out.stage_b, gsd=sc.gsd)
            self.say(f"done: {res.n_detections} detections over "
                     f"{len(scenes)} scenes", 1.0)
            return oid, files, (scenes, res, anns), describe_downloaded(files), dest

        def done(payload):
            oid, files, loaded, summary, folder = payload
            lines = [f"Order {oid} complete.", "", summary or "", ""]
            if loaded is None:
                lines.append(
                    "No analytic scenes were found after download. Check the "
                    "folder contents --\nthe order may have delivered a "
                    "bundle without the 8-band analytic asset.")
                set_text(self.txt_search, "\n".join(lines))
                return
            scenes, res, anns = loaded
            self.real_scenes = scenes
            self.real_result = res
            self.real_annotations = anns
            self.real_folder.set(str(folder))
            ids_ = [sc.scene_id for sc in scenes]
            self.real_combo.configure(values=ids_)
            self.real_pick.set(ids_[0])
            set_text(self.txt_real, res.report())
            _real_show(self)
            lines += [f"Loaded {len(scenes)} scenes and ran the pipeline: "
                      f"{res.n_detections} detections.",
                      "",
                      "Open the 'Real imagery' tab to see the outlines on the "
                      "pictures."]
            if res.warnings:
                lines += ["", "Warnings:"] + [f"  - {w}" for w in res.warnings]
            set_text(self.txt_search, "\n".join(lines))
            self.status.set(f"done: {len(scenes)} scenes, "
                            f"{res.n_detections} detections")
            self.job_detail.set("results are on the Real imagery tab")

        self._run_async(work, done, "ordering...")

    def _refresh_key_status(self) -> None:
        ok = credentials.configured("planet")
        self.key_status.set(
            f"Planet key: {credentials.masked('planet')}   "
            f"({'ready' if ok else 'NOT SET - ordering will fail'})")

    def _test_key(self) -> None:
        self.nb.select(0)
        self._test_all()

    def _refresh_quota(self) -> None:
        """Keep the place summary, download folder and quota panel in step."""
        aoi = AOIS[self.aoi_key.get()]
        ew, ns = aoi.extent_km()
        self.aoi_info.set(f"{aoi.area_km2():.2f} km2  ({ew:.1f} x {ns:.1f} km)"
                          f"   kind={aoi.kind}   AIS: {aoi.ais_source}")
        base = (Path(self.dl_dir.get()).parent if self.dl_dir.get()
                else Path.home() / "navsim_data")
        self.dl_dir.set(str(base / aoi.key))
        # A new place invalidates the previous search: ordering must never
        # use item ids from whichever place was selected before.
        self._search_results = []
        self._search_aoi = None

        days = int(self.days_back.get())
        single = quota_plan([aoi.key], days=days, looks_per_day=1)
        region = None
        for name, r in REGIONS.items():
            if aoi.key in r["aois"]:
                region = (name, r)
                break
        txt = [
            f"{aoi.name}",
            f"  {aoi.notes}" if aoi.notes else "",
            "",
            f"Daily coverage for {days} days would be "
            f"{single.total_km2:.1f} km2, i.e. "
            f"{single.fraction * 100:.1f}% of the {MONTHLY_QUOTA_KM2:.0f} km2 "
            f"monthly quota.",
            "",
        ]
        if region:
            name, r = region
            txt += [f"Recommended set for {r['name']}:", r["why"], "",
                    quota_plan(r["aois"], days=days,
                               looks_per_day=1).report(), ""]
        txt += [
            "On this account Planet bills at least 100 km2 per scene, however "
            "small the clip.",
            "So cost is set by the NUMBER of scenes, not their size: any area "
            "up to 10 x 10 km costs",
            "the same, and a 3,000 km2 quota buys about 30 scenes. Draw areas "
            "generously (they're",
            "free up to 100 km2) and order few dates.",
        ]
        set_text(self.txt_quota, "\n".join(x for x in txt if x != ""))

    def _search(self, then=None) -> None:
        from .planet import coverage_stats, search
        aoi = AOIS[self.aoi_key.get()]
        try:
            end = datetime.strptime(self.d_end.get(), "%Y-%m-%d") \
                .replace(tzinfo=timezone.utc)
        except ValueError:
            messagebox.showerror("navsim", "end date must be YYYY-MM-DD")
            return
        start = end - timedelta(days=int(self.days_back.get()))
        self.d_start.set(start.strftime("%Y-%m-%d"))
        mc = float(self.max_cloud.get())

        def work():
            sc = search(aoi, start, end, max_cloud=mc)
            return sc, coverage_stats(aoi, sc)

        def done(res):
            sc, cs = res
            self._search_results = sc
            lines = [cs.report(), "",
                     "Verify these before committing: PlanetScope coverage is "
                     "reliable near coasts but",
                     "selective over open water, and the off-nadir "
                     "distribution drives layover bias.", "",
                     f"  {'item_id':34s} {'acquired':20s} {'cloud':>6s} "
                     f"{'view':>6s} {'sun el':>7s} {'sat':>8s}"]
            for s in sc[:60]:
                lines.append(
                    f"  {s.item_id:34s} {s.acquired:%Y-%m-%d %H:%M}Z    "
                    f"{s.cloud_cover:6.2f} "
                    f"{(s.view_angle if s.view_angle is not None else float('nan')):6.1f} "
                    f"{(s.sun_elevation if s.sun_elevation is not None else float('nan')):7.1f} "
                    f"{str(s.satellite_id or '-'):>8s}")
            if len(sc) > 60:
                lines.append(f"  ... {len(sc) - 60} more")
            n_order = min(len(sc), int(self.max_scenes.get()))
            cost = aoi.area_km2() * n_order
            lines.insert(1, "")
            lines.insert(2, f"  Ordering the first {n_order} would download "
                            f"{cost:.1f} km2 "
                            f"({cost / MONTHLY_QUOTA_KM2 * 100:.1f}% of the "
                            f"monthly quota).")
            set_text(self.txt_search, "\n".join(lines))
            self.status.set(f"{len(sc)} scenes found for {aoi.key}")
            self._search_aoi = aoi.key
            if then is not None and sc:
                self.after(50, then)
            elif then is not None:
                self.status.set("nothing available for that place and window")

        self._run_async(work, done, f"searching archive for {aoi.key}...")

    def _order(self) -> None:
        from .planet import create_clip_order, estimate_order_km2
        if not self._search_results:
            messagebox.showinfo("navsim", "run a search first")
            return
        aoi = AOIS[self.aoi_key.get()]
        ids = [s.item_id for s in self._search_results]
        est = estimate_order_km2(aoi, len(ids))
        ok = messagebox.askyesno(
            "Confirm order -- this spends download quota",
            f"AOI:     {aoi.name}\n"
            f"Scenes:  {len(ids)}\n"
            f"Area:    {aoi.area_km2():.2f} km2 per scene\n\n"
            f"ESTIMATED DOWNLOAD: {est:.1f} km2\n"
            f"That is {est / MONTHLY_QUOTA_KM2 * 100:.1f}% of your "
            f"{MONTHLY_QUOTA_KM2:.0f} km2 monthly quota.\n\n"
            f"Submit this order?")
        if not ok:
            self.status.set("order cancelled")
            return
        name = f"navsim-{aoi.key}-{datetime.now():%Y%m%d-%H%M}"

        def work():
            return create_clip_order(aoi, ids, name, confirm=True,
                                     budget_km2=MONTHLY_QUOTA_KM2)

        def done(res):
            oid = res.get("id", "?")
            self.status.set(f"order submitted: {oid}")
            messagebox.showinfo("navsim", f"Order {oid} submitted.\n"
                                          f"Estimated {est:.1f} km2.")

        self._run_async(work, done, "submitting order...")


def main() -> None:
    NavsimApp().mainloop()


if __name__ == "__main__":
    main()


# ==========================================================================
# Real-imagery tab, appended as mixin methods
# ==========================================================================
def _build_real(self, nb) -> None:
    tab = ttk.Frame(nb)
    nb.add(tab, text="5. Real imagery")

    left = ttk.Frame(tab)
    left.pack(side="left", fill="y", padx=6, pady=6)
    right = ttk.Frame(tab)
    right.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=6)

    src = ttk.LabelFrame(left, text="PlanetScope deliveries")
    src.pack(fill="x")
    ttk.Label(src, text="scene folder").grid(row=0, column=0, sticky="w", padx=3)
    self.real_folder = tk.StringVar()
    ttk.Entry(src, textvariable=self.real_folder, width=30).grid(
        row=1, column=0, columnspan=2, sticky="ew", padx=3)
    ttk.Button(src, text="Browse...", command=lambda: self.real_folder.set(
        filedialog.askdirectory() or self.real_folder.get())
    ).grid(row=2, column=0, sticky="ew", padx=3, pady=2)

    ttk.Label(src, text="AIS CSV (for calibration)").grid(
        row=3, column=0, sticky="w", padx=3, pady=(8, 0))
    self.real_ais = tk.StringVar()
    ttk.Entry(src, textvariable=self.real_ais, width=30).grid(
        row=4, column=0, columnspan=2, sticky="ew", padx=3)
    ttk.Button(src, text="Browse...", command=lambda: self.real_ais.set(
        filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        or self.real_ais.get())).grid(row=5, column=0, sticky="ew", padx=3,
                                      pady=2)

    ttk.Label(src, text="AOI kind").grid(row=6, column=0, sticky="w", padx=3,
                                         pady=(8, 0))
    self.real_kind = tk.StringVar(value="base")
    ttk.Combobox(src, textvariable=self.real_kind, state="readonly", width=14,
                 values=("base", "commercial", "gate", "anchorage")
                 ).grid(row=7, column=0, sticky="w", padx=3)
    ttk.Label(src, text="sets the location prior;\nprior-free log LR is\n"
                        "reported separately", foreground="#666",
              justify="left").grid(row=8, column=0, sticky="w", padx=3,
                                   pady=(2, 4))

    act = ttk.Frame(left)
    act.pack(fill="x", pady=8)
    ttk.Button(act, text="1. Load scenes", command=lambda: _real_load(self)
               ).pack(fill="x", pady=2)
    ttk.Button(act, text="2. Run pipeline", command=lambda: _real_run(self)
               ).pack(fill="x", pady=2)
    ttk.Button(act, text="Export GeoJSON", command=lambda: _real_export(self)
               ).pack(fill="x", pady=2)

    view = ttk.LabelFrame(left, text="View")
    view.pack(fill="x", pady=4)
    ttk.Label(view, text="scene").pack(anchor="w", padx=3)
    self.real_pick = tk.StringVar()
    self.real_combo = ttk.Combobox(view, textvariable=self.real_pick,
                                   state="readonly", width=28, values=())
    self.real_combo.pack(fill="x", padx=3)
    self.real_combo.bind("<<ComboboxSelected>>", lambda e: _real_show(self))
    ttk.Label(view, text="display").pack(anchor="w", padx=3, pady=(6, 0))
    self.real_mode = tk.StringVar(value="RGB")
    cb = ttk.Combobox(view, textvariable=self.real_mode, state="readonly",
                      width=28,
                      values=("RGB",) + BAND_NAMES + ("hull index", "NDWI",
                                                      "water mask"))
    cb.pack(fill="x", padx=3)
    cb.bind("<<ComboboxSelected>>", lambda e: _real_show(self))
    self.real_vmax = tk.DoubleVar(value=0.30)
    ttk.Scale(view, from_=0.02, to=0.8, variable=self.real_vmax,
              orient="horizontal", command=lambda e: _real_show(self)
              ).pack(fill="x", padx=3, pady=2)
    self.real_outline = tk.BooleanVar(value=True)
    self.real_labels = tk.BooleanVar(value=True)
    self.real_hide_rej = tk.BooleanVar(value=True)
    for lbl, var in (("hull outlines", self.real_outline),
                     ("labels", self.real_labels),
                     ("hide rejected", self.real_hide_rej)):
        ttk.Checkbutton(view, text=lbl, variable=var,
                        command=lambda: _real_show(self)).pack(anchor="w",
                                                               padx=3)
    ttk.Label(view, text="min length (m)").pack(anchor="w", padx=3,
                                                pady=(4, 0))
    self.real_min_len = tk.DoubleVar(value=40.0)
    ttk.Spinbox(view, from_=0, to=400, increment=10, width=8,
                textvariable=self.real_min_len,
                command=lambda: _real_show(self)).pack(anchor="w", padx=3)

    self.fig_real = Figure(figsize=(7.4, 5.6), dpi=96)
    self.canvas_real = FigureCanvasTkAgg(self.fig_real, master=right)
    self.canvas_real.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(self.canvas_real, right).update()

    self.txt_real = mono_text(right, height=16, width=104)
    set_text(self.txt_real,
             "Point this at a folder of PlanetScope deliveries, then Load and "
             "Run.\n\n"
             "What happens, in order:\n"
             "  1. Each ortho_analytic_8b GeoTIFF is read and DN converted to "
             "TOA reflectance using\n"
             "     the per-band reflectanceCoefficient from the metadata XML; "
             "UDM2 is read too.\n"
             "  2. Static water and persistent-clutter masks are built from "
             "ALL dates together.\n"
             "     One date is not enough: a single-date NDWI threshold cuts "
             "bright ships out of\n"
             "     their own water mask, and clutter cannot be identified from "
             "a single look.\n"
             "  3. Detect, measure, match AIS, CALIBRATE, then classify. "
             "Calibration comes before\n"
             "     classification because posteriors inherit whatever "
             "covariance they are given.\n"
             "  4. Berth occupancy, groups, and association tracks with their "
             "entropy.\n\n"
             "Results draw on the image above: fitted hull outlines coloured "
             "by domain (red naval,\n"
             "cyan merchant, yellow other), dotted if the classifier rejected "
             "them, with length,\n"
             "beam, P(naval) and category as labels. Export GeoJSON to open "
             "them in QGIS.\n\n"
             "Product note: ortho_analytic_8b_sr is the wrong asset -- "
             "resampled, altering the PSF\n"
             "the hull fit assumes, with a land-tuned atmospheric correction "
             "that misbehaves over\n"
             "dark water. Use ortho_analytic_8b. The 8-band product is "
             "required for the hull index.")

    self.real_scenes: list = []
    self.real_result = None
    self.real_annotations: dict = {}


def _real_load(self) -> None:
    folder = self.real_folder.get().strip()
    if not folder:
        messagebox.showinfo("navsim", "choose a folder first")
        return

    def work():
        scenes = load_directory(folder)
        masks = build_static_masks(scenes) if len(scenes) >= 2 else None
        return scenes, masks

    def done(out):
        scenes, masks = out
        self.real_scenes = scenes
        self.real_annotations = {}
        self.real_result = None
        if not scenes:
            set_text(self.txt_real,
                     f"No analytic scenes found under {folder}.\n\n"
                     f"Expected files matching *AnalyticMS*.tif with their "
                     f"_udm2.tif and _metadata.xml sidecars alongside.")
            return
        ids = [s.scene_id for s in scenes]
        self.real_combo.configure(values=ids)
        self.real_pick.set(ids[0])
        lines = [f"Loaded {len(scenes)} scenes from {folder}", "",
                 f"  {'scene_id':38s} {'acquired':20s} {'gsd':>5s} "
                 f"{'size':>11s} {'clear':>6s} {'reflectance source'}"]
        for sc in scenes:
            lines.append(
                f"  {sc.scene_id[:38]:38s} {sc.timestamp:%Y-%m-%d %H:%M}Z   "
                f"{sc.gsd:5.2f} {f'{sc.shape[1]}x{sc.shape[0]}':>11s} "
                f"{sc.clear_fraction():6.3f} "
                f"{str(sc.meta.get('reflectance_source'))[:40]}")
        if masks is not None:
            lines += ["", masks.report()]
        else:
            lines += ["", "Only one scene loaded: no static water or clutter "
                          "map is possible. Load several",
                      "dates of the same clipped AOI before trusting any "
                      "detection count."]
        set_text(self.txt_real, "\n".join(lines))
        self.status.set(f"{len(scenes)} real scenes loaded")
        _real_show(self)

    self._run_async(work, done, "loading deliveries...")


def _real_run(self) -> None:
    if not self.real_scenes:
        messagebox.showinfo("navsim", "load scenes first")
        return
    scenes = self.real_scenes
    ais_path = self.real_ais.get().strip()
    kind = self.real_kind.get()

    def work():
        reports = None
        if ais_path:
            from .ais import load_ais_csv, project_reports
            reports = load_ais_csv(ais_path)
            project_reports(reports, scenes[0].crs)
        res = run_pipeline(
            scenes, ais_reports=reports, aoi_kind=kind,
            progress=lambda i, n: self.say(
                f"scene {i} of {n}: detect, measure, classify",
                0.05 + 0.9 * i / max(n, 1)))
        anns = {}
        for sc, out in zip(scenes, res.scenes):
            anns[sc.scene_id] = build_annotations(
                out.detections, out.measurements, out.stage_a, out.stage_b,
                gsd=sc.gsd)
        return res, reports, anns

    def done(out):
        res, reports, anns = out
        self.real_result = res
        self.real_annotations = anns
        extra = ("" if reports is None
                 else f"\n\nAIS reports loaded: {len(reports)}")
        set_text(self.txt_real, res.report() + extra)
        self.status.set(f"pipeline done: {res.n_detections} detections "
                        f"over {len(scenes)} scenes")
        _real_show(self)

    self._run_async(work, done, "running pipeline on real imagery...")


def _real_show(self) -> None:
    if not self.real_scenes:
        return
    pick = self.real_pick.get()
    sc = next((x for x in self.real_scenes if x.scene_id == pick),
              self.real_scenes[0])
    mode = self.real_mode.get()
    vmax = float(self.real_vmax.get())

    self.fig_real.clear()
    ax = self.fig_real.add_subplot(111)
    try:
        if mode == "RGB":
            ax.imshow(sc.rgb(stretch=(0.0, vmax)))
        elif mode == "hull index":
            from .radiometry import hull_index
            ax.imshow(hull_index(sc.bands), cmap="viridis",
                      vmin=-0.02, vmax=max(vmax * 0.5, 0.05))
        elif mode == "NDWI":
            ax.imshow(sc.ndwi(), cmap="BrBG", vmin=-0.6, vmax=0.6)
        elif mode == "water mask":
            ax.imshow(sc.water_mask(), cmap="gray", interpolation="nearest")
        else:
            ax.imshow(sc.band(mode), cmap="gray", vmin=0, vmax=vmax)
    except Exception as exc:
        ax.text(0.5, 0.5, f"cannot display: {exc}", ha="center",
                transform=ax.transAxes)
    anns = self.real_annotations.get(sc.scene_id, [])
    shown = filter_annotations(
        anns, hide_rejected=self.real_hide_rej.get(),
        min_length_m=float(self.real_min_len.get())) if anns else []
    if shown:
        draw_annotations(ax, shown, show_outline=self.real_outline.get(),
                         show_box=False, show_labels=self.real_labels.get())
    n = f"{len(shown)} of {len(anns)}" if anns else "0"
    ax.set_title(f"{sc.scene_id}   {sc.timestamp:%Y-%m-%d %H:%M}Z   "
                 f"{mode}   {n} detections shown", fontsize=9)
    ax.axis("off")
    self.fig_real.tight_layout()
    self.canvas_real.draw()


def _real_export(self) -> None:
    if not self.real_annotations:
        messagebox.showinfo("navsim", "run the pipeline first")
        return
    pick = self.real_pick.get()
    sc = next((x for x in self.real_scenes if x.scene_id == pick),
              self.real_scenes[0])
    anns = self.real_annotations.get(sc.scene_id, [])
    if not anns:
        messagebox.showinfo("navsim", f"no detections in {sc.scene_id}")
        return
    path = filedialog.asksaveasfilename(
        defaultextension=".geojson", initialfile=f"{sc.scene_id}.geojson",
        filetypes=[("GeoJSON", "*.geojson")])
    if not path:
        return
    n = write_geojson(path, anns, sc)
    self.status.set(f"wrote {n} features ({sc.crs}) to {path}")


NavsimApp._build_real = _build_real


# ==========================================================================
# Wide-area tab: Tier 1 radar -> Tier 2 routing -> Tier 3 cueing
# ==========================================================================
def _build_wide(self, nb) -> None:
    tab = ttk.Frame(nb)
    nb.add(tab, text="6. Wide search")

    left = ttk.Frame(tab)
    left.pack(side="left", fill="y", padx=6, pady=6)
    right = ttk.Frame(tab)
    right.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=6)

    area = ttk.LabelFrame(left, text="Area and date")
    area.pack(fill="x")
    ttk.Label(area, text="preset").grid(row=0, column=0, sticky="w", padx=4)
    self.wide_preset = tk.StringVar(value="norfolk_approaches")
    wc = ttk.Combobox(area, textvariable=self.wide_preset, state="readonly",
                      width=22, values=tuple(WIDE_PRESETS))
    wc.grid(row=0, column=1, sticky="w", pady=3)
    wc.bind("<<ComboboxSelected>>", lambda e: _wide_preset(self))
    for i, (lbl, attr, default) in enumerate(
            (("west", "wide_w", "-76.6"), ("south", "wide_s", "36.7"),
             ("east", "wide_e", "-75.8"), ("north", "wide_n", "37.2"))):
        ttk.Label(area, text=lbl).grid(row=1 + i, column=0, sticky="w", padx=4)
        v = tk.StringVar(value=default)
        setattr(self, attr, v)
        ttk.Entry(area, textvariable=v, width=14).grid(row=1 + i, column=1,
                                                       sticky="w")
    ttk.Label(area, text="date").grid(row=5, column=0, sticky="w", padx=4)
    self.wide_date = tk.StringVar(
        value=(datetime.now(timezone.utc) - timedelta(days=45))
        .strftime("%Y-%m-%d"))
    ttk.Entry(area, textvariable=self.wide_date, width=14).grid(
        row=5, column=1, sticky="w", pady=(0, 4))

    t1 = ttk.LabelFrame(left, text="Tier 1  -  Sentinel-1 radar (free)")
    t1.pack(fill="x", pady=4)
    ttk.Button(t1, text="Find available radar passes",
               command=lambda: _wide_search(self)).pack(fill="x", padx=4,
                                                        pady=2)
    ttk.Button(t1, text="Detect ships on radar",
               command=lambda: _wide_detect(self)).pack(fill="x", padx=4,
                                                        pady=2)
    ttk.Label(t1, text="AIS CSV (optional, for dark detection)",
              foreground="#555").pack(anchor="w", padx=4)
    self.wide_ais = tk.StringVar()
    ttk.Entry(t1, textvariable=self.wide_ais, width=28).pack(fill="x", padx=4)
    ttk.Button(t1, text="Browse", command=lambda: self.wide_ais.set(
        filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        or self.wide_ais.get())).pack(fill="x", padx=4, pady=(2, 6))

    t2 = ttk.LabelFrame(left, text="Tier 2  -  where a hull can go")
    t2.pack(fill="x", pady=4)
    ttk.Label(t2, text="draft (m)").grid(row=0, column=0, sticky="w", padx=4)
    self.wide_draft = tk.DoubleVar(value=12.0)
    ttk.Spinbox(t2, from_=1.0, to=25.0, increment=0.5, width=8,
                textvariable=self.wide_draft).grid(row=0, column=1, sticky="w")
    ttk.Label(t2, text="speed (kn)").grid(row=1, column=0, sticky="w", padx=4)
    self.wide_speed = tk.DoubleVar(value=20.0)
    ttk.Spinbox(t2, from_=4.0, to=35.0, increment=1.0, width=8,
                textvariable=self.wide_speed).grid(row=1, column=1, sticky="w")
    ttk.Label(t2, text="hours").grid(row=2, column=0, sticky="w", padx=4)
    self.wide_hours = tk.DoubleVar(value=24.0)
    ttk.Spinbox(t2, from_=1.0, to=168.0, increment=1.0, width=8,
                textvariable=self.wide_hours).grid(row=2, column=1, sticky="w")
    ttk.Label(t2, text="GEBCO file (optional, sharper)",
              foreground="#555").grid(row=3, column=0, columnspan=2,
                                      sticky="w", padx=4)
    self.wide_bathy = tk.StringVar()
    ttk.Entry(t2, textvariable=self.wide_bathy, width=28).grid(
        row=4, column=0, columnspan=2, sticky="ew", padx=4)
    ttk.Button(t2, text="Browse", command=lambda: self.wide_bathy.set(
        filedialog.askopenfilename() or self.wide_bathy.get())).grid(
        row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
    ttk.Button(t2, text="Build navigable map + reachable set",
               command=lambda: _wide_reach(self)).grid(
        row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 2))
    ttk.Label(t2, text="start  lon,lat (blank = centre)",
              foreground="#555").grid(row=7, column=0, columnspan=2,
                                      sticky="w", padx=4)
    self.wide_from = tk.StringVar()
    ttk.Entry(t2, textvariable=self.wide_from, width=28).grid(
        row=8, column=0, columnspan=2, sticky="ew", padx=4)
    ttk.Label(t2, text="destination  lon,lat (optional)",
              foreground="#555").grid(row=9, column=0, columnspan=2,
                                      sticky="w", padx=4)
    self.wide_to = tk.StringVar()
    ttk.Entry(t2, textvariable=self.wide_to, width=28).grid(
        row=10, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 6))

    t3 = ttk.LabelFrame(left, text="Tier 3  -  cue Planet")
    t3.pack(fill="x", pady=4)
    ttk.Label(t3, text="budget (km2)").grid(row=0, column=0, sticky="w",
                                            padx=4)
    self.wide_budget = tk.DoubleVar(value=200.0)
    ttk.Spinbox(t3, from_=10, to=3000, increment=10, width=8,
                textvariable=self.wide_budget).grid(row=0, column=1,
                                                    sticky="w")
    ttk.Button(t3, text="Plan cues (free)",
               command=lambda: _wide_plan(self)).grid(
        row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
    ttk.Button(t3, text="Order cues (spends quota)",
               command=lambda: _wide_order(self)).grid(
        row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 6))

    layers = ttk.Frame(right)
    layers.pack(fill="x", pady=(0, 2))
    ttk.Label(layers, text="map").pack(side="left")
    self.wide_basemap = tk.StringVar(value="satellite")
    bcb = ttk.Combobox(layers, textvariable=self.wide_basemap, width=10,
                       state="readonly",
                       values=("satellite", "streets", "ocean", "none"))
    bcb.pack(side="left", padx=(2, 8))
    bcb.bind("<<ComboboxSelected>>", lambda e: _wide_draw(self))
    self.wide_show = {}
    for key, label, default in (("radar", "radar", True),
                                ("depth", "depth", True),
                                ("reach", "reachable", True),
                                ("contacts", "contacts", True),
                                ("rejected", "rejected", True),
                                ("labels", "labels", True),
                                ("cues", "chips", True),
                                ("findings", "classified", True),
                                ("track", "track", True)):
        v = tk.BooleanVar(value=default)
        self.wide_show[key] = v
        ttk.Checkbutton(layers, text=label, variable=v,
                        command=lambda: _wide_draw(self)).pack(side="left",
                                                               padx=2)
    ttk.Label(layers, text="  radar opacity").pack(side="left")
    self.wide_radar_alpha = tk.DoubleVar(value=0.7)
    ttk.Scale(layers, from_=0.1, to=1.0, variable=self.wide_radar_alpha,
              length=90, command=lambda e: _wide_draw(self)).pack(side="left")
    ttk.Button(layers, text="Show map", command=lambda: _wide_draw(self)
               ).pack(side="right")
    self.wide_select = tk.BooleanVar(value=False)
    self.wide_click = tk.StringVar(value="chip")
    ttk.Combobox(layers, textvariable=self.wide_click, state="readonly",
                 width=6, values=("chip", "track")).pack(side="right")
    ttk.Label(layers, text="click makes").pack(side="right", padx=(8, 2))
    ttk.Checkbutton(layers, text="SELECT ON MAP", variable=self.wide_select,
                    command=lambda: _wide_draw(self)).pack(side="right",
                                                           padx=4)

    self.fig_wide = Figure(figsize=(7.2, 4.9), dpi=96)
    self.canvas_wide = FigureCanvasTkAgg(self.fig_wide, master=right)
    self.canvas_wide.get_tk_widget().pack(fill="both", expand=True)
    self.toolbar_wide = NavigationToolbar2Tk(self.canvas_wide, right)
    self.toolbar_wide.update()
    bottom = ttk.Frame(right)
    bottom.pack(fill="both", expand=False)
    tools = ttk.Notebook(bottom)
    tools.pack(side="left", fill="y", padx=(0, 6))
    chip_page = ttk.Frame(tools)
    track_page = ttk.Frame(tools)
    tools.add(chip_page, text="Planet chips")
    tools.add(track_page, text="Track a ship")
    _build_chip_panel(self, chip_page)
    _build_track_panel(self, track_page)
    txt_holder = ttk.Frame(bottom)
    txt_holder.pack(side="left", fill="both", expand=True)
    self.txt_wide = mono_text(txt_holder, height=14, width=70)
    set_text(self.txt_wide, WIDE_INTRO)

    self.wide_scenes: list = []
    self.wide_dark = None
    self.wide_graph = None
    self.wide_reach = None
    self.wide_plan = None
    self.wide_bathymetry = None
    self.wide_route = None
    self.wide_chips: list = []
    self.track = None
    self.track_pred = None
    self.track_looks: list = []
    self.track_assoc = None
    self._wide_selector = None
    self._bm_cache: dict = {}
    self._bm_loading: set = set()
    self.after(300, lambda: _wide_draw(self))


WIDE_PRESETS = {
    "norfolk_approaches": (-76.6, 36.7, -75.8, 37.2, 12.0),
    "hampton_roads_gate": (-76.40, 36.95, -76.25, 37.05, 12.0),
    "malacca_strait":     (100.0, 1.0, 104.5, 4.0, 15.0),
    "hormuz":             (55.6, 25.6, 57.2, 26.9, 15.0),
    "bab_el_mandeb":      (43.0, 12.3, 43.8, 13.2, 15.0),
    "taiwan_strait":      (118.0, 23.0, 121.5, 25.5, 12.0),
    "san_diego_approach": (-117.35, 32.55, -117.10, 32.78, 12.0),
}

WIDE_INTRO = """Tier 1 -> Tier 3: find ships cheaply over a big area, then spend
quota only where it is worth spending.

  Tier 1  Sentinel-1 radar. Free, sees through cloud and at night, 10 m.
          A 155 m destroyer is 16 pixels: enough to detect and roughly size,
          not to classify. Subtracting AIS leaves the "dark" contacts.

  Tier 2  Where a hull of a given draft can actually go. Bathymetry turns a
          flat reachable disc into a corridor. In a strait that is a ~1400x
          reduction in search area; in open ocean it buys nothing, and the
          report says so rather than pretending otherwise.

  Tier 3  Cue PlanetScope with chips instead of blanket coverage.
          Planet bills at least 100 km2 per scene on this account, so a chip
          costs ~100 km2 however small: 3,000 km2 buys about 30 scenes. Chips
          inside imagery you already have are classified for free.

Three honest limits, enforced in code:
  - 30-day withhold: this is retrospective, never live.
  - Planet barely images open ocean, so every cue is coverage-checked first.
  - A MOVING contact cannot be re-acquired across a multi-hour sensor gap:
    at 20 kn an unknown-speed vessel is +/-91 km after 7 hours. Only
    stationary contacts and fixed gates are orderable; the rest are listed
    as not viable with the reason.

Needs the free Copernicus credentials from the Setup tab. Bathymetry needs
no key at all."""


def _wide_bbox(self):
    return (float(self.wide_w.get()), float(self.wide_s.get()),
            float(self.wide_e.get()), float(self.wide_n.get()))


def _wide_preset(self) -> None:
    w, s_, e, n, draft = WIDE_PRESETS[self.wide_preset.get()]
    for var, val in ((self.wide_w, w), (self.wide_s, s_),
                     (self.wide_e, e), (self.wide_n, n)):
        var.set(f"{val}")
    self.wide_draft.set(draft)
    from .sentinel import bbox_size_m, tile_bbox
    wm, hm = bbox_size_m((w, s_, e, n))
    self.status.set(f"{self.wide_preset.get()}: {wm / 1000:.0f} x "
                    f"{hm / 1000:.0f} km, {len(tile_bbox((w, s_, e, n)))} "
                    f"radar tiles")


def _wide_search(self) -> None:
    from .sentinel import search_s1
    bbox = _wide_bbox(self)
    try:
        day = datetime.strptime(self.wide_date.get(), "%Y-%m-%d").replace(
            tzinfo=timezone.utc)
    except ValueError:
        messagebox.showerror("navsim", "date must be YYYY-MM-DD", parent=self)
        return

    def work():
        return search_s1(bbox, day - timedelta(days=6), day + timedelta(days=6))

    def done(scenes):
        self.wide_scenes = scenes
        lines = [f"{len(scenes)} Sentinel-1 acquisitions within +/-6 days", ""]
        for sc in scenes[:30]:
            lines.append("  " + str(sc))
        if not scenes:
            lines.append("  none. Widen the date range or the box.")
        set_text(self.txt_wide, "\n".join(lines))
        self.status.set(f"{len(scenes)} radar passes")

    self._run_async(work, done, "searching Sentinel-1 archive...")


def _wide_detect(self) -> None:
    from .sentinel import SentinelError, wide_area_search
    bbox = _wide_bbox(self)
    try:
        day = datetime.strptime(self.wide_date.get(), "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=12)
    except ValueError:
        messagebox.showerror("navsim", "date must be YYYY-MM-DD", parent=self)
        return
    ais_path = self.wide_ais.get().strip()

    def work():
        reports = []
        if ais_path:
            from .ais import load_ais_csv
            self.say("reading the AIS file...", 0.01)
            reports = load_ais_csv(ais_path, bbox=bbox)
            self.say(f"loaded {len(reports)} AIS reports", 0.03)
        return wide_area_search(
            bbox, day, ais_reports=reports,
            progress=lambda i, n, msg, detail="": self.say(
                msg, 0.03 + 0.94 * i / max(n, 1), detail))

    def done(res):
        self.wide_dark = res
        lines = [res.report(), "",
                 "Fetched patches are cached, so re-running this pass is "
                 "fast. The first run fetches",
                 "the chosen pass plus 4 reference passes for the clutter "
                 "map (~1 min each).", ""]
        if res.dark:
            lines.append(f"  {'lat,lon':22s} {'length':>8s} {'SNR':>7s}")
            for d in sorted(res.dark,
                            key=lambda x: -x.extra.get("moment_length_m", 0))[:25]:
                e = d.extra
                pos = f"{e.get('lat', 0):+.4f},{e.get('lon', 0):+.4f}"
                lines.append(f"  {pos:22s} "
                             f"{e.get('moment_length_m', 0):7.0f}m "
                             f"{d.peak_snr:7.1f}")
        set_text(self.txt_wide, "\n".join(lines))
        _wide_draw(self)
        self.status.set(f"{len(res.detections)} radar contacts, "
                        f"{len(res.dark)} dark")

    self._run_async(work, done, "fetching radar and detecting...")


def _wide_reach(self) -> None:
    from .bathymetry import build_nav_graph, load_bathymetry
    bbox = _wide_bbox(self)
    draft = float(self.wide_draft.get())
    speed = float(self.wide_speed.get())
    hours = float(self.wide_hours.get())
    local = self.wide_bathy.get().strip() or None

    def parse(txt):
        try:
            a, b_ = (float(v) for v in txt.replace(" ", "").split(","))
            return (a, b_)
        except Exception:
            return None
    start = parse(self.wide_from.get()) or (0.5 * (bbox[0] + bbox[2]),
                                             0.5 * (bbox[1] + bbox[3]))
    dest = parse(self.wide_to.get())

    def work():
        from .bathymetry import route_bottleneck
        self.say("fetching depth data...", 0.15)
        b = load_bathymetry(bbox, local_path=local)
        self.say(f"building the navigable map at {draft:.1f} m draft...", 0.5)
        g = build_nav_graph(b, draft)
        self.say("working out where it can get to...", 0.75)
        r = g.reachable(start[0], start[1], hours, speed)
        rb = None
        if dest is not None:
            self.say("finding the deepest route to the destination...", 0.9)
            rb = route_bottleneck(b, start, dest)
            rb["dest"] = dest
        return b, g, r, rb

    def done(out):
        b, g, r, rb = out
        self.wide_graph, self.wide_reach = g, r
        self.wide_bathymetry = b
        self.wide_route = rb
        from .bathymetry import default_clearance
        keel = default_clearance(draft)
        prune = ("not measurable: the reach fills the box"
                 if r.get("saturates_box")
                 else f"{r['reduction_vs_disc']:,.0f}x")
        lines = [b.report(), "",
                 f"Draft                {draft:.1f} m "
                 f"(+{keel:.1f} m under-keel clearance = needs "
                 f"{draft + keel:.1f} m of water)",
                 f"Navigable cells      {g.n_nodes:,}  at "
                 f"{g.cell_m:,.0f} m per cell",
                 f"Reach in {hours:.0f} h at {speed:.0f} kn   "
                 f"{r['area_km2']:,.0f} km2  (red on the map)",
                 f"A flat disc would be {r['free_disc_km2']:,.0f} km2",
                 f"PRUNING FACTOR       {prune}", ""]
        if r.get("saturates_box"):
            lines.append("The reachable area fills the box, so the box is the "
                         "limit, not the geography. Widen it to measure real "
                         "pruning.")
        elif r["reduction_vs_disc"] < 3:
            lines.append("Geography barely constrains this: it is effectively "
                         "open ocean. Cue a gate, not a ship.")
        else:
            lines.append("Geography constrains this usefully -- corridor "
                         "cueing is worthwhile here.")
        if rb is not None:
            ok = rb["max_draft_m"] >= draft
            lines += ["", f"ROUTE TO {rb['dest'][1]:.4f},{rb['dest'][0]:.4f}: "
                          + ("reachable" if ok else "BLOCKED")
                          + f" for a {draft:.1f} m draft",
                      "  " + rb["verdict"]]
        set_text(self.txt_wide, "\n".join(lines))
        _wide_draw(self)
        self.status.set(f"reachable {r['area_km2']:,.0f} km2 "
                        f"({r['reduction_vs_disc']:,.0f}x prune)")

    self._run_async(work, done, "building navigable water map...")


def _wide_plan(self) -> None:
    from .cue import cues_from_dark_vessels, plan_cues
    if self.wide_dark is None:
        messagebox.showinfo("navsim", "run radar detection first", parent=self)
        return
    budget = float(self.wide_budget.get())
    dark = self.wide_dark.dark

    def work():
        cues = cues_from_dark_vessels(dark)
        self.say(f"{len(cues)} candidate cues; checking Planet coverage...",
                 0.3)
        return plan_cues(cues, budget_km2=budget,
                         progress=lambda i, n, m: self.say(
                             m, 0.3 + 0.65 * i / max(n, 1)))

    def done(plan):
        self.wide_plan = plan
        set_text(self.txt_wide, plan.report())
        _wide_draw(self)
        self.status.set(f"{len(plan.cues)} orderable cues, "
                        f"{plan.total_km2:.1f} km2")

    self._run_async(work, done, "planning cues...")


def _wide_order(self) -> None:
    from .cue import order_cues
    plan = self.wide_plan
    if plan is None or not plan.cues:
        messagebox.showinfo("navsim", "plan cues first", parent=self)
        return
    root = Path.home() / "navsim_data" / "cues"
    ok = confirm_dialog(
        self, "Confirm  -  this spends download quota",
        f"Chips:    {len(plan.cues)}\n"
        f"Download: {plan.total_km2:.1f} km2\n"
        f"That is {plan.fraction * 100:.2f}% of the monthly quota.\n\n"
        f"Saving to {root}", ok_text="Order them")
    if not ok:
        self.status.set("cancelled")
        return

    def work():
        return order_cues(plan, str(root), confirm=True,
                          progress=lambda i, n, m: self.say(
                              m, i / max(n, 1)),
                          on_state=lambda st, el: self.say(
                              f"order {st} ({el / 60:.1f} min)"),
                          on_file=lambda i, n, nm, d=0, t=0: self.say(
                              f"downloading {i}/{n}: {nm[:40]}"))

    def done(results):
        ok_n = sum(1 for r in results if r.get("ordered"))
        lines = [f"Ordered {ok_n} of {len(results)} cues", ""]
        for r in results:
            c = r["cue"]
            lines.append(f"  {c.lat:+.4f},{c.lon:+.4f}  "
                         + ("ordered " + str(r.get("order_id", ""))[:12]
                            if r.get("ordered")
                            else "skipped: " + str(r.get("reason"))[:50]))
        set_text(self.txt_wide, "\n".join(lines))
        self.status.set(f"{ok_n} cue orders complete")

    self._run_async(work, done, "ordering cues...")


def _wide_basemap(self, bbox, provider):
    """Cached basemap; fetched in the background the first time."""
    from .basemap import fetch_basemap
    if provider == "none":
        return None
    key = (provider, tuple(round(float(v), 5) for v in bbox))
    if key in self._bm_cache:
        return self._bm_cache[key]
    if key not in self._bm_loading:
        self._bm_loading.add(key)

        def work():
            try:
                bm = fetch_basemap(bbox, provider, width_px=1400)
            except Exception as exc:
                bm = None
                # Bind the text now: Python deletes ``exc`` when the except
                # block ends, so a lambda that reads it later would raise.
                msg = f"basemap unavailable: {str(exc)[:80]}"
                self.uiq.put(lambda: self.status.set(msg))

            def finish():
                self._bm_loading.discard(key)
                self._bm_cache[key] = bm
                _wide_draw(self)
            self.uiq.put(finish)

        threading.Thread(target=work, daemon=True).start()
        self.status.set(f"loading {provider} map tiles...")
    return None


def _wide_draw(self) -> None:
    """Everything on one real map: basemap, radar, depth, reach, contacts."""
    from .mapview import draw_map, mosaic_radar_tiles

    show = {k: v.get() for k, v in self.wide_show.items()}
    res = self.wide_dark
    bbox = tuple(res.bbox) if (res is not None and getattr(res, "bbox", None)) \
        else _wide_bbox(self)
    if self.track is not None and show.get("track", True):
        bbox = _track_extent(self, bbox)
    bm = _wide_basemap(self, bbox, self.wide_basemap.get())

    radar = None
    if res is not None and show["radar"] and getattr(res, "display_tiles", None):
        cache_key = (id(res), bbox)
        if getattr(self, "_radar_mosaic_key", None) != cache_key:
            self._radar_mosaic = mosaic_radar_tiles(res.display_tiles, bbox)
            self._radar_mosaic_key = cache_key
        radar = self._radar_mosaic

    reach = self.wide_reach
    bathy = self.wide_bathymetry
    title_bits = []
    if res is not None:
        title_bits.append(f"Sentinel-1 {res.acquired:%Y-%m-%d %H:%MZ}")
    if reach is not None:
        title_bits.append(f"reach {reach['area_km2']:,.0f} km2 at "
                          f"{reach['draft_m']:.1f} m draft")

    self.fig_wide.clear()
    ax = self.fig_wide.add_subplot(111)
    draw_map(
        ax, bbox, basemap=bm,
        radar=radar, radar_alpha=float(self.wide_radar_alpha.get()),
        depth=bathy if show["depth"] else None,
        reachable=(reach["mask"] if (reach is not None and show["reach"])
                   else None),
        layer_bbox=(bathy.bbox if bathy is not None else None),
        contacts=(res.detections if (res is not None and show["contacts"])
                  else ()),
        matched_ids=(set(res.matched) if res is not None else None),
        rejected=(res.rejected if (res is not None and show["rejected"])
                  else ()),
        cues=(self.wide_plan.cues if (self.wide_plan is not None
                                      and show["cues"]) else ()),
        start=(reach["snapped_to"] if reach is not None else None),
        title="   |   ".join(title_bits) or "run Tier 1 or Tier 2",
        label_contacts=show["labels"],
    )
    _draw_chips(self, ax)
    if show.get("track", True):
        _draw_track(self, ax)
    _attach_selector(self, ax)
    rb = getattr(self, "wide_route", None)
    if rb is not None:
        ax.plot(*rb["dest"], marker="P", color="#00ff66", mec="black", ms=14,
                zorder=9, label="destination")
        if rb.get("pinch_lonlat"):
            ax.plot(*rb["pinch_lonlat"], marker="X", color="#ff00ff",
                    mec="black", ms=14, zorder=9)
            ax.annotate(f"route pinches at {rb['bottleneck_depth_m']:.1f} m",
                        rb["pinch_lonlat"], xytext=(8, -12),
                        textcoords="offset points", color="white", fontsize=7,
                        bbox=dict(fc="#80008080", ec="none"), zorder=9)
    self.fig_wide.tight_layout()
    self.canvas_wide.draw_idle()


# ==========================================================================
# Planet chips: choose areas on the map, order them, classify them
# ==========================================================================
CHIP_COLOURS = {"new": "#ffffff", "found": "#00ff66", "none": "#999999",
                "ordered": "#ffd400", "classified": "#ff9f1a",
                "failed": "#ff3030"}


def _build_chip_panel(self, parent) -> None:
    box = ttk.LabelFrame(parent, text="Planet chips  (3 m imagery for "
                                      "classification)")
    box.pack(side="left", fill="y", padx=(0, 6))
    ttk.Label(box, text="Tick SELECT CHIPS above the map, then drag a box "
                        "or click a radar contact.",
              foreground="#555", wraplength=420, justify="left").pack(
        anchor="w", padx=4, pady=(2, 2))
    lb_frame = ttk.Frame(box)
    lb_frame.pack(fill="x", padx=4)
    self.chip_list = tk.Listbox(lb_frame, height=7, width=62, font=MONO,
                                selectmode="extended")
    sb = ttk.Scrollbar(lb_frame, orient="vertical",
                       command=self.chip_list.yview)
    self.chip_list.configure(yscrollcommand=sb.set)
    self.chip_list.pack(side="left", fill="x")
    sb.pack(side="left", fill="y")
    self.chip_total = tk.StringVar(value="no chips yet")
    ttk.Label(box, textvariable=self.chip_total, foreground="#246").pack(
        anchor="w", padx=4)

    opts = ttk.Frame(box)
    opts.pack(fill="x", padx=4, pady=2)
    ttk.Label(opts, text="context").pack(side="left")
    self.chip_kind = tk.StringVar(value="anchorage")
    ttk.Combobox(opts, textvariable=self.chip_kind, state="readonly",
                 width=11, values=("anchorage", "base", "gate",
                                   "commercial")).pack(side="left", padx=2)
    ttk.Label(opts, text=" window +/-").pack(side="left")
    self.chip_window = tk.DoubleVar(value=24.0)
    ttk.Spinbox(opts, from_=2, to=240, increment=2, width=5,
                textvariable=self.chip_window).pack(side="left")
    ttk.Label(opts, text="h  cloud <=").pack(side="left")
    self.chip_cloud = tk.DoubleVar(value=0.3)
    ttk.Spinbox(opts, from_=0.0, to=1.0, increment=0.05, width=5,
                textvariable=self.chip_cloud).pack(side="left")

    row = ttk.Frame(box)
    row.pack(fill="x", padx=4, pady=2)
    ttk.Button(row, text="Remove", command=lambda: _chip_remove(self)
               ).pack(side="left")
    ttk.Button(row, text="Clear", command=lambda: _chip_clear(self)
               ).pack(side="left", padx=2)
    ttk.Button(row, text="Add planned cues",
               command=lambda: _chip_from_cues(self)).pack(side="left")
    row0 = ttk.Frame(box)
    row0.pack(fill="x", padx=4, pady=(2, 0))
    ttk.Button(row0, text="Classify from downloaded imagery (FREE, no quota)",
               command=lambda: _chip_local(self)).pack(fill="x")
    ttk.Button(row0, text="Classify from Planet TILES (experimental: colour "
                          "only, ~150 tiles/chip)",
               command=lambda: _chip_tiles(self)).pack(fill="x", pady=(2, 0))
    row2 = ttk.Frame(box)
    row2.pack(fill="x", padx=4, pady=(2, 4))
    ttk.Button(row2, text="1. Find Planet images (free)",
               command=lambda: _chip_find(self)).pack(side="left",
                                                     fill="x", expand=True)
    ttk.Button(row2, text="2. Order + classify (quota)",
               command=lambda: _chip_order(self)).pack(side="left",
                                                      fill="x", expand=True,
                                                      padx=(2, 0))


def _chip_target_time(self):
    res = self.wide_dark
    if res is not None and res.acquired is not None:
        return res.acquired
    try:
        return datetime.strptime(self.wide_date.get(), "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=15, minute=30)
    except ValueError:
        return datetime.now(timezone.utc) - timedelta(days=45)


def _chip_refresh(self) -> None:
    from .aoi import MONTHLY_QUOTA_KM2
    self.chip_list.delete(0, "end")
    for i, c in enumerate(self.wide_chips):
        self.chip_list.insert("end", c.label(i))
        self.chip_list.itemconfig("end", foreground={
            "found": "#0a7a2a", "none": "#888", "classified": "#b35c00",
            "failed": "#c00000"}.get(c.status, "#000"))
    tot = sum(c.area_km2 for c in self.wide_chips)
    found = [c for c in self.wide_chips if c.scene is not None]
    cost = sum(c.billable_km2 for c in found)
    self.chip_total.set(
        f"{len(self.wide_chips)} chip(s), {tot:.1f} km2 drawn"
        + (f"  |  ordering {len(found)} would cost ~{cost:.0f} km2 "
           f"(100 km2 min per scene)" if found else
           "  |  classify free, or find Planet images to order"))


def _chip_remove(self) -> None:
    sel = sorted(self.chip_list.curselection(), reverse=True)
    for i in sel:
        del self.wide_chips[i]
    _chip_refresh(self)
    _wide_draw(self)


def _chip_clear(self) -> None:
    self.wide_chips = []
    _chip_refresh(self)
    _wide_draw(self)


def _chip_from_cues(self) -> None:
    from .chips import chip_from_bbox
    plan = self.wide_plan
    if plan is None or not plan.cues:
        messagebox.showinfo("navsim", "plan cues first (Tier 3 panel)",
                            parent=self)
        return
    for c in plan.cues:
        ch = chip_from_bbox(c.bbox(), c.when, c.reason)
        ch.source = "cue"
        self.wide_chips.append(ch)
    _chip_refresh(self)
    _wide_draw(self)


def _attach_selector(self, ax) -> None:
    """Drag-a-box / click-a-ship chip selection, only in select mode."""
    self._wide_selector = None
    if not self.wide_select.get():
        return
    from matplotlib.widgets import RectangleSelector

    def toolbar_busy() -> bool:
        mode = getattr(self.toolbar_wide, "mode", "")
        return str(getattr(mode, "value", mode) or "") != ""

    def onselect(eclick, erelease):
        if toolbar_busy() or eclick.xdata is None or erelease.xdata is None:
            return
        from .chips import chip_from_bbox, chip_from_contact
        x0, y0, x1, y1 = eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
        lat = 0.5 * (y0 + y1)
        dx_km = abs(x1 - x0) * 111.32 * math.cos(math.radians(lat))
        dy_km = abs(y1 - y0) * 110.54
        when = _chip_target_time(self)
        if self.wide_click.get() == "track":
            if max(dx_km, dy_km) >= 0.25:
                self.status.set("in track mode, click a single radar contact")
                return
            res = self.wide_dark
            best, bd = None, 3.0
            for d in (res.detections if res is not None else []):
                ex = d.extra or {}
                if ex.get("lon") is None:
                    continue
                dk = math.hypot((ex["lon"] - x0) * 111.32 *
                                math.cos(math.radians(lat)),
                                (ex["lat"] - y0) * 110.54)
                if dk < bd:
                    best, bd = d, dk
            if best is None:
                self.status.set("no radar contact within 3 km of that click")
                return
            _track_seed(self, best)
            return
        if max(dx_km, dy_km) < 0.25:
            # A click: snap to the nearest radar contact if one is close.
            best, bd = None, 2.0
            res = self.wide_dark
            for d in (res.detections if res is not None else []):
                ex = d.extra or {}
                if ex.get("lon") is None:
                    continue
                dk = math.hypot((ex["lon"] - x0) * 111.32 *
                                math.cos(math.radians(lat)),
                                (ex["lat"] - y0) * 110.54)
                if dk < bd:
                    best, bd = d, dk
            if best is not None:
                chip = chip_from_contact(best, when)
            else:
                half = 1.0 / 111.32 / max(math.cos(math.radians(y0)), 0.05)
                chip = chip_from_bbox((x0 - half, y0 - 1.0 / 110.54,
                                       x0 + half, y0 + 1.0 / 110.54), when,
                                      "clicked point")
        else:
            chip = chip_from_bbox((x0, y0, x1, y1), when)
        self.wide_chips.append(chip)
        _chip_refresh(self)
        self.status.set(f"chip {len(self.wide_chips)} added: "
                        f"{chip.area_km2:.1f} km2 ({chip.reason})")
        _wide_draw(self)

    self._wide_selector = RectangleSelector(
        ax, onselect, useblit=False, button=[1], minspanx=0, minspany=0,
        spancoords="data", interactive=False,
        props=dict(facecolor="#00ff66", edgecolor="#00ff66", alpha=0.18,
                   fill=True))


def _draw_chips(self, ax) -> None:
    from .chips import chip_findings
    show = {k: v.get() for k, v in self.wide_show.items()}
    if show.get("cues", True):
        for i, c in enumerate(self.wide_chips):
            w, s_, e, n = c.bbox
            col = CHIP_COLOURS.get(c.status, "#ffffff")
            ax.plot([w, e, e, w, w], [s_, s_, n, n, s_], "-" if c.status
                    != "new" else "--", color=col, lw=1.6, zorder=10)
            ax.text(w, n, f" {i + 1}", color=col, fontsize=8,
                    fontweight="bold", va="bottom", zorder=10,
                    bbox=dict(fc="black", ec="none", alpha=0.45, pad=0.5))
    if show.get("findings", True):
        for c in self.wide_chips:
            for f in chip_findings(c):
                if f["rejected"]:
                    continue
                if f["outline"] is not None:
                    ax.plot(*f["outline"], "-", color=f["colour"], lw=1.4,
                            zorder=11)
                else:
                    ax.plot(f["lon"], f["lat"], "o", mfc="none",
                            mec=f["colour"], ms=8, zorder=11)
                if show.get("labels", True):
                    ax.annotate(f["label"].split("  ", 1)[-1][:40],
                                (f["lon"], f["lat"]), xytext=(6, -10),
                                textcoords="offset points", fontsize=6,
                                color=f["colour"], zorder=11,
                                bbox=dict(fc="black", ec="none", alpha=0.5))


def _chip_find(self) -> None:
    from .chips import find_images
    if not self.wide_chips:
        messagebox.showinfo("navsim", "no chips yet: tick SELECT CHIPS and "
                            "drag a box on the map", parent=self)
        return
    chips = self.wide_chips
    window = float(self.chip_window.get())
    cloud = float(self.chip_cloud.get())

    def work():
        return find_images(chips, window_hours=window, max_cloud=cloud,
                           progress=lambda i, n, m: self.say(
                               m, i / max(n, 1)))

    def done(plan):
        self.chip_plan = plan
        _chip_refresh(self)
        set_text(self.txt_wide, plan.report())
        _wide_draw(self)
        self.status.set(f"{len(plan.orderable)} of {len(chips)} chips have a "
                        f"Planet image ({plan.total_km2:.1f} km2)")

    self._run_async(work, done, "searching Planet for each chip (free)...")


def _chip_local(self) -> None:
    """Classify every chip from imagery already on disk.  No quota."""
    from .chips import chip_findings, classify_local, local_index
    if not self.wide_chips:
        messagebox.showinfo("navsim", "no chips yet: tick SELECT ON MAP and "
                            "drag a box over an area you've downloaded",
                            parent=self)
        return
    chips = self.wide_chips
    kind = self.chip_kind.get()

    def work():
        self.say("indexing downloaded imagery...", 0.02)
        idx = local_index()
        for i, c in enumerate(chips):
            self.say(f"chip {i + 1} of {len(chips)}: reading local images",
                     0.05 + 0.9 * i / max(len(chips), 1))
            classify_local(c, idx, aoi_kind=kind)
        return chips, len({x.item_id for x in idx})

    def done(out):
        chips_, n_images = out
        _chip_refresh(self)
        lines = [f"Classified from {n_images} downloaded images -- no quota "
                 f"used.", ""]
        real_scenes, real_anns = [], {}
        for i, c in enumerate(chips_):
            lines.append(f"chip {i + 1}: {c.status} -- {c.note}")
            for f in chip_findings(c):
                if f["rejected"] or (f["length_m"] or 0) < 25:
                    continue
                pn = "-" if f["p_naval"] is None else f"{f['p_naval']:.2f}"
                lines.append(f"    {f['lat']:+.4f},{f['lon']:+.4f}  "
                             f"{(f['length_m'] or 0):5.0f} m  "
                             f"{f['domain']:8s} P(naval)={pn}  "
                             f"{f['category'] or ''}")
            sc = c.result.get("scene")
            if sc is not None and c.status == "classified":
                real_scenes.append(sc)
                real_anns[sc.scene_id] = c.result.get("annotations") or []
        if real_scenes:
            self.real_scenes = real_scenes
            self.real_annotations = real_anns
            ids = [x.scene_id for x in real_scenes]
            self.real_combo.configure(values=ids)
            self.real_pick.set(ids[0])
            _real_show(self)
            lines += ["", "Hull outlines for each chip are also on the Real "
                          "imagery tab."]
        none = [c for c in chips_ if c.status == "none"]
        if none:
            lines += ["", f"{len(none)} chip(s) are outside every downloaded "
                          f"image. Free classification only works where you "
                          f"already have pictures (currently the Norfolk "
                          f"pier area)."]
        set_text(self.txt_wide, "\n".join(lines))
        _wide_draw(self)
        self.status.set(f"{sum(1 for c in chips_ if c.status == 'classified')}"
                        f" chip(s) classified for free")

    self._run_async(work, done, "classifying chips from downloaded imagery...")


def _chip_tiles(self) -> None:
    """Classify chips from Planet scene tiles: the tile allowance, not km2."""
    from .chips import chip_findings, classify_tiles
    from .tiles import tiles_used
    if not self.wide_chips:
        messagebox.showinfo("navsim", "no chips yet: tick SELECT ON MAP and "
                            "drag a box", parent=self)
        return
    chips = self.wide_chips
    kind = self.chip_kind.get()
    window = float(self.chip_window.get())
    before = tiles_used()

    def work():
        for i, c in enumerate(chips):
            classify_tiles(c, aoi_kind=kind, window_hours=window,
                           progress=lambda f, n, m, i=i: self.say(
                               f"chip {i + 1} of {len(chips)}: {m}",
                               (i + f) / max(len(chips), 1)))
        return chips

    def done(chips_):
        used = tiles_used()
        _chip_refresh(self)
        lines = [f"Planet scene tiles used this run: {used - before}  "
                 f"(this program's running total {used:,} of the 100,000 "
                 f"allowance; the authoritative count is on your Planet "
                 f"account page)", ""]
        real_scenes, real_anns = [], {}
        for i, c in enumerate(chips_):
            lines.append(f"chip {i + 1}: {c.status} -- {c.note}")
            for f in chip_findings(c):
                if f["rejected"] or (f["length_m"] or 0) < 40:
                    continue
                pn = "-" if f["p_naval"] is None else f"{f['p_naval']:.2f}"
                lines.append(f"    {f['lat']:+.4f},{f['lon']:+.4f}  "
                             f"{(f['length_m'] or 0):5.0f} m  "
                             f"{f['domain']:8s} P(naval)={pn}  "
                             f"{f['category'] or ''}")
            sc = c.result.get("scene")
            if sc is not None and c.status == "classified":
                real_scenes.append(sc)
                real_anns[sc.scene_id] = c.result.get("annotations") or []
        if real_scenes:
            self.real_scenes = real_scenes
            self.real_annotations = real_anns
            ids = [x.scene_id for x in real_scenes]
            self.real_combo.configure(values=ids)
            self.real_pick.set(ids[0])
            _real_show(self)
            lines += ["", "Open the Real imagery tab to inspect the hulls on "
                          "the colour image.",
                      "",
                      "EXPERIMENTAL. Tiles are colour only, with no near-"
                      "infrared, and two failures are known:",
                      "  - dark-decked ships (carriers) barely differ from "
                      "dark water in colour and can be missed;",
                      "    at Newport News two clearly visible carriers were "
                      "not detected.",
                      "  - ships moored for months (shipyards) can be "
                      "mistaken for piers.",
                      "Use tiles to LOOK; trust the automatic labels only "
                      "after checking them by eye."]
        set_text(self.txt_wide, "\n".join(lines))
        _wide_draw(self)
        self.status.set(f"{sum(1 for c in chips_ if c.status == 'classified')}"
                        f" chip(s) classified from tiles")

    self._run_async(work, done, "classifying chips from Planet tiles...")


def _chip_order(self) -> None:
    from .chips import ChipPlan, order_and_classify
    plan = getattr(self, "chip_plan", None)
    if plan is None or not plan.orderable:
        messagebox.showinfo("navsim", "run step 1 (Find Planet images) first; "
                            "only chips with an image can be ordered",
                            parent=self)
        return
    plan = ChipPlan([c for c in self.wide_chips])
    root = Path.home() / "navsim_data" / "chips"
    from .planet import quota_status
    qs = quota_status()
    if qs is not None and qs["remaining_km2"] < plan.total_km2:
        if confirm_dialog(
                self, "Not enough Planet quota",
                f"These chips would cost about {plan.total_km2:.0f} km2 "
                f"(Planet bills at least 100 km2 per scene).\n"
                f"The account has used {qs['used_km2']:,.0f} of "
                f"{qs['quota_km2']:,.0f} km2, so Planet will reject the "
                f"order.\n\nClassify the chips from imagery you have "
                f"already downloaded instead? (free)",
                ok_text="Classify for free"):
            _chip_local(self)
        return
    ok = confirm_dialog(
        self, "Confirm  -  this spends download quota",
        f"Chips:    {len(plan.orderable)} (one image each)\n"
        f"Download: {plan.total_km2:.1f} km2\n"
        f"That is {plan.total_km2 / MONTHLY_QUOTA_KM2 * 100:.2f}% of the "
        f"monthly quota.\n\n"
        f"Each chip is ordered, downloaded and classified in turn.\n"
        f"Saving to {root}", ok_text="Order and classify")
    if not ok:
        self.status.set("cancelled")
        return
    kind = self.chip_kind.get()

    def work():
        return order_and_classify(
            plan, root, aoi_kind=kind, confirm=True,
            progress=lambda i, n, m: self.say(m, i / max(n, 1)),
            on_state=lambda st, el: self.say(
                f"Planet is processing ({st}, {el / 60:.1f} min)"),
            on_file=lambda i, n, nm, d=0, t=0: self.say(
                f"downloading {i}/{n}: {nm[:40]}"))

    def done(chips):
        from .chips import chip_findings
        _chip_refresh(self)
        lines = []
        real_scenes, real_anns = [], {}
        for i, c in enumerate(chips):
            if c.scene is None:
                continue
            lines.append(f"chip {i + 1}: {c.status}"
                         + (f"  -  {c.note}" if c.note else ""))
            for f in chip_findings(c):
                if f["rejected"] or (f["length_m"] or 0) < 25:
                    continue
                lines.append(
                    f"    {f['lat']:+.4f},{f['lon']:+.4f}  "
                    f"{(f['length_m'] or 0):5.0f} m  {f['domain']:8s} "
                    f"P(naval)={f['p_naval'] if f['p_naval'] is None else round(f['p_naval'], 2)}"
                    f"  {f['category'] or ''}")
            sc = c.result.get("scene")
            if sc is not None:
                real_scenes.append(sc)
                real_anns[sc.scene_id] = c.result.get("annotations") or []
        if real_scenes:
            self.real_scenes = real_scenes
            self.real_annotations = real_anns
            ids = [x.scene_id for x in real_scenes]
            self.real_combo.configure(values=ids)
            self.real_pick.set(ids[0])
            _real_show(self)
            lines += ["", "Each chip is also on the Real imagery tab with full "
                          "hull outlines and labels."]
        lines += ["", "Classification uses DEFAULT measurement uncertainty: "
                      "there is no AIS calibration for a chip.",
                  "Treat P(naval) as a ranking, not a probability, until the "
                  "error model has been calibrated."]
        set_text(self.txt_wide, "\n".join(lines) or "nothing classified")
        _wide_draw(self)
        n_ok = sum(1 for c in chips if c.status == "classified")
        failed = [c for c in chips if c.status == "failed"]
        self.status.set(f"{n_ok} chip(s) classified, {len(failed)} failed")
        if failed:
            reasons = sorted({c.note for c in failed})
            messagebox.showerror(
                "navsim", f"{len(failed)} chip(s) failed:\n\n"
                + "\n\n".join(r[:300] for r in reasons[:3]), parent=self)

    self._run_async(work, done, "ordering and classifying chips...")


# ==========================================================================
# Track a single ship across radar passes
# ==========================================================================
TRACKS_DIR = Path.home() / "navsim_data" / "tracks"


def _build_track_panel(self, parent) -> None:
    box = ttk.Frame(parent)
    box.pack(fill="both", padx=4, pady=4)
    ttk.Label(box, text="Set 'click makes' to track, tick SELECT ON MAP, "
                        "then click a radar contact.", foreground="#555",
              wraplength=420, justify="left").pack(anchor="w")
    self.track_info = tk.StringVar(value="no track yet")
    ttk.Label(box, textvariable=self.track_info, font=MONO,
              justify="left", foreground="#246").pack(anchor="w", pady=2)

    kn = ttk.Frame(box)
    kn.pack(fill="x", pady=2)
    ttk.Label(kn, text="bow heading").pack(side="left")
    self.track_bow = tk.StringVar()
    ttk.Entry(kn, textvariable=self.track_bow, width=6).pack(side="left")
    ttk.Button(kn, text="Set", width=4,
               command=lambda: _track_set_bow(self)).pack(side="left")
    ttk.Label(kn, text="  speed kn").pack(side="left")
    self.track_speed = tk.StringVar()
    ttk.Entry(kn, textvariable=self.track_speed, width=8).pack(side="left")
    ttk.Button(kn, text="Set", width=4,
               command=lambda: _track_set_speed(self)).pack(side="left")
    ttk.Label(kn, text="(e.g. 16 or 16+-3)", foreground="#777").pack(
        side="left")

    row = ttk.Frame(box)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="look ahead").pack(side="left")
    self.track_days = tk.DoubleVar(value=7.0)
    ttk.Spinbox(row, from_=0.5, to=30, increment=0.5, width=5,
                textvariable=self.track_days).pack(side="left")
    ttk.Label(row, text="days").pack(side="left")
    ttk.Button(row, text="1. Find next radar looks (free)",
               command=lambda: _track_plan(self)).pack(side="left", padx=4)

    self.track_list = tk.Listbox(box, height=5, width=66, font=MONO)
    self.track_list.pack(fill="x")
    self.track_list.bind("<<ListboxSelect>>",
                         lambda e: _track_show_look(self))

    row2 = ttk.Frame(box)
    row2.pack(fill="x", pady=2)
    ttk.Button(row2, text="2. Search selected look",
               command=lambda: _track_search(self)).pack(side="left")
    ttk.Button(row2, text="3. Accept best match",
               command=lambda: _track_accept(self)).pack(side="left", padx=2)
    ttk.Button(row2, text="Planet near prediction",
               command=lambda: _track_planet(self)).pack(side="left")
    row3 = ttk.Frame(box)
    row3.pack(fill="x", pady=(0, 2))
    ttk.Button(row3, text="Where is it heading?",
               command=lambda: _track_arrivals(self)).pack(side="left")
    ttk.Button(row3, text="Add top 3 port checks as chips",
               command=lambda: _track_port_chips(self)).pack(side="left",
                                                            padx=2)
    ttk.Button(row3, text="Save track", command=lambda: _track_save(self)
               ).pack(side="left")
    ttk.Button(row3, text="Load track", command=lambda: _track_load(self)
               ).pack(side="left", padx=2)
    ttk.Button(row3, text="Clear", command=lambda: _track_clear(self)
               ).pack(side="left")


def _track_land(self, days: float):
    """Coastline for the track's region, so impossible paths are pruned.

    Fetched once per track and reused.  Near a coast this alone often
    resolves bow from stern: the reversed direction of a ship leaving port
    runs inland and every sample of it is killed.
    """
    from .bathymetry import fetch_noaa_dem
    from .pursuit import land_from_bathymetry
    st = self.track
    reach_deg = min(max(st.speed_kn + 2 * st.speed_sd_kn, 5.0) * 1.852
                    * 24 * days / 111.0, 12.0)
    key = (round(st.last.lon, 2), round(st.last.lat, 2), round(reach_deg, 1))
    if getattr(self, "_track_land_key", None) == key:
        return self._track_land_fn
    box = (st.last.lon - reach_deg, max(st.last.lat - reach_deg, -80),
           st.last.lon + reach_deg, min(st.last.lat + reach_deg, 80))
    try:
        bathy = fetch_noaa_dem(box, target_px=1200)
        fn = land_from_bathymetry(bathy)
    except Exception:
        fn = None
    self._track_land_key, self._track_land_fn = key, fn
    return fn


def _track_refresh(self) -> None:
    st = self.track
    if st is None:
        self.track_info.set("no track yet")
        return
    f = st.last
    amb = "UNKNOWN (both ways)" if st.ambiguous else f"{st.heading_deg:.0f} deg"
    self.track_info.set(
        f"{len(st.fixes)} fix(es); last {f.t:%m-%d %H:%MZ} "
        f"{f.lat:.3f},{f.lon:.3f}\n"
        f"course {amb}  speed {st.speed_kn:.1f}+-{st.speed_sd_kn:.1f} kn "
        f"({st.speed_source[:22]})\n"
        f"size {st.length_m:.0f}+-{st.length_sd:.0f} m   chain confidence "
        f"{st.chain_confidence:.2f}")


def _track_seed(self, det) -> None:
    from .pursuit import seed_from_detection
    st = seed_from_detection(det, label=f"track {det.det_id[-5:]}")
    self.track, self.track_pred, self.track_looks = st, None, []
    self.track_assoc = None
    self.track_list.delete(0, "end")
    _track_refresh(self)
    set_text(self.txt_wide, st.summary() + "\n\n"
             "Bow and stern are not known from one radar image, so both "
             "directions are\npredicted until a later look decides. If you "
             "know the heading (a Planet hull\nfit, a visible wake, or "
             "common sense near a coast) set it above: it halves\nthe search.")
    self.status.set("track started; press 'Find next radar looks'")
    _wide_draw(self)


def _track_set_bow(self) -> None:
    if self.track is None:
        return
    try:
        h = float(self.track_bow.get()) % 360.0
    except ValueError:
        messagebox.showerror("navsim", "heading must be a number of degrees",
                             parent=self)
        return
    self.track.heading_deg, self.track.ambiguous = h, False
    self.track.heading_sd = min(self.track.heading_sd, 6.0)
    self.track.log.append(f"bow set to {h:.0f} deg by the user")
    _track_refresh(self)
    _wide_draw(self)


def _track_set_speed(self) -> None:
    if self.track is None:
        return
    txt = self.track_speed.get().replace(" ", "")
    try:
        if "+-" in txt:
            a, b = txt.split("+-")
            v, sd = float(a), float(b)
        else:
            v, sd = float(txt), max(0.15 * float(txt), 1.0)
    except ValueError:
        messagebox.showerror("navsim", "speed like 16 or 16+-3", parent=self)
        return
    self.track.speed_kn, self.track.speed_sd_kn = v, sd
    self.track.speed_source = "set by user"
    _track_refresh(self)
    _wide_draw(self)


def _track_plan(self) -> None:
    from .pursuit import plan_looks
    st = self.track
    if st is None:
        messagebox.showinfo("navsim", "start a track first", parent=self)
        return
    days = float(self.track_days.get())
    t0 = st.last.t + timedelta(hours=1)

    def work():
        self.say("loading the coastline so paths over land are ruled out...",
                 0.1)
        land = _track_land(self, days)
        return plan_looks(st, t0, t0 + timedelta(days=days), land=land,
                          progress=lambda i, n, m: self.say(m, 0.3))

    def done(looks):
        self.track_looks = looks
        self.track_list.delete(0, "end")
        for lk in looks:
            self.track_list.insert("end", str(lk))
        if looks:
            self.track_list.selection_set(0)
            _track_show_look(self)
        set_text(self.txt_wide,
                 f"{len(looks)} Sentinel-1 pass(es) in the next {days:g} days "
                 f"could contain the ship.\n\n"
                 + "\n".join(str(lk) for lk in looks[:12])
                 + ("\n\nPick one and press 'Search selected look'. Early "
                    "looks are best: the\nprediction grows with time, and so "
                    "does the number of look-alikes in it." if looks else
                    "\n\nNo radar pass covers the predicted area in that "
                    "window. Look further ahead,\nor the ship has left the "
                    "radar's coverage."))
        self.status.set(f"{len(looks)} candidate radar looks")

    self._run_async(work, done, "planning radar looks (free)...")


def _track_selected_look(self):
    sel = self.track_list.curselection()
    if not sel or not self.track_looks:
        return None
    return self.track_looks[sel[0]]


def _track_show_look(self) -> None:
    lk = _track_selected_look(self)
    if lk is None or self.track is None:
        return
    from .pursuit import predict
    self.track_pred = predict(self.track, lk.t,
                              land=getattr(self, "_track_land_fn", None))
    self.track_assoc = None
    set_text(self.txt_wide, self.track_pred.summary()
             + f"\n\nthis pass: {lk}")
    _wide_draw(self)


def _track_search(self) -> None:
    from .pursuit import search_look
    lk = _track_selected_look(self)
    if lk is None:
        messagebox.showinfo("navsim", "find looks first, then pick one",
                            parent=self)
        return
    st = self.track

    def work():
        return search_look(st, lk, land=getattr(self, "_track_land_fn", None),
                           progress=lambda i, n, m: self.say(
                               m, 0.05 + 0.9 * i / max(n, 1)))

    def done(out):
        assoc, dets = out
        self.track_assoc = assoc
        self._track_look_used = lk
        set_text(self.txt_wide, assoc.report())
        self.status.set(f"{assoc.n_contacts} contacts; best "
                        f"{(assoc.best.prob if assoc.best else 0) * 100:.0f}%")
        _wide_draw(self)

    self._run_async(work, done, "searching the radar pass...")


def _track_accept(self) -> None:
    from .pursuit import update
    a = self.track_assoc
    if a is None or a.best is None:
        messagebox.showinfo("navsim", "search a look first", parent=self)
        return
    b = a.best
    # Simulation showed 17-20% wrong links in the 50-80% band and almost none
    # above 80%, so anything below 80% needs an explicit decision.
    if b.prob < 0.8 and not confirm_dialog(
            self, "Not a confident match",
            f"The best contact is {b.prob * 100:.0f}% likely to be the ship.\n"
            f"In simulation, links in the 50-80% band were wrong about one "
            f"time in five,\nand a wrong link steers every later prediction "
            f"off course.\n\nConsider confirming with a Planet chip first. "
            f"Accept anyway?", ok_text="Accept anyway"):
        return
    lk = getattr(self, "_track_look_used", None)
    self.track = update(self.track, b, lk.t if lk else b.det.extra.get(
        "acquired"), lk.scene_id if lk else None)
    self.track_assoc, self.track_pred = None, None
    self.track_list.delete(0, "end")
    self.track_looks = []
    _track_refresh(self)
    set_text(self.txt_wide, self.track.summary() + "\n\nLinked. Course and "
             "speed now come from the two positions, so the next\nprediction "
             "is much tighter. Find the next looks to continue.")
    _wide_draw(self)


def _track_planet(self) -> None:
    from .chips import Chip
    from .pursuit import planet_looks
    st = self.track
    if st is None:
        return
    days = float(self.track_days.get())
    t0 = st.last.t

    def work():
        return planet_looks(st, t0, t0 + timedelta(days=days),
                            land=_track_land(self, days))

    def done(looks):
        if not looks:
            set_text(self.txt_wide, "No Planet image covers the predicted "
                     "area. Planet barely images open\nwater (none 100 km off "
                     "Norfolk in 60 days): it can confirm the ship near a\n"
                     "coast or in port, not at sea.")
            return
        best = looks[0]
        cells = best.prediction.cells(0.8)
        w = min(c[0] for c in cells)
        s_ = min(c[1] for c in cells)
        e = max(c[2] for c in cells)
        n = max(c[3] for c in cells)
        ch = Chip((w, s_, e, n), best.t, f"track prediction "
                  f"{best.t:%m-%d %H:%MZ}", "track")
        self.wide_chips.append(ch)
        _chip_refresh(self)
        set_text(self.txt_wide,
                 "\n".join(str(lk) for lk in looks[:8])
                 + f"\n\nAdded a chip covering 80% of the prediction for the "
                 f"best Planet image\n({ch.area_km2:,.0f} km2). Order it from "
                 f"the Planet chips page to classify\nwhat is there. If that "
                 f"is large, link another radar fix first: it shrinks.")
        _wide_draw(self)

    self._run_async(work, done, "searching Planet near the prediction...")


def _track_arrivals(self) -> None:
    from .pursuit import arrival_windows
    if self.track is None:
        messagebox.showinfo("navsim", "start a track first", parent=self)
        return
    days = max(float(self.track_days.get()), 3.0)
    self.track_arrivals = arrival_windows(self.track, horizon_days=days)
    lines = [f"Ports it could reach within {days:g} days, ranked by how well "
             f"they lie on its course:", ""]
    lines += [f"  {a}" for a in self.track_arrivals[:12]]
    lines += ["",
              "Why look for the arrival instead of the ship at sea: offshore, "
              "radar passes are",
              "6-16 days apart and a multi-day prediction holds thousands of "
              "look-alikes. In port,",
              "Planet images almost daily at 3 m, so size AND class can be "
              "matched. Sea routes are",
              f"approximated as great circles x{1.15:.2f}; extend the port list "
              f"in navsim/pursuit.py."]
    set_text(self.txt_wide, "\n".join(lines))
    self.status.set(f"{len(self.track_arrivals)} candidate ports")


def _track_port_chips(self) -> None:
    from .pursuit import port_chip
    arr = getattr(self, "track_arrivals", None)
    if not arr:
        _track_arrivals(self)
        arr = getattr(self, "track_arrivals", None)
    if not arr:
        return
    for a in arr[:3]:
        self.wide_chips.append(port_chip(a))
    _chip_refresh(self)
    self.status.set("added 3 port-arrival chips; use the Planet chips page to "
                    "find images and classify")
    _wide_draw(self)


def _track_save(self) -> None:
    if self.track is None:
        return
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    path = filedialog.asksaveasfilename(
        initialdir=str(TRACKS_DIR), defaultextension=".json",
        initialfile=f"track_{self.track.last.t:%Y%m%d}.json",
        filetypes=[("track", "*.json")])
    if path:
        Path(path).write_text(self.track.to_json())
        self.status.set(f"saved {path}")


def _track_load(self) -> None:
    from .pursuit import TrackState
    path = filedialog.askopenfilename(initialdir=str(TRACKS_DIR),
                                      filetypes=[("track", "*.json")])
    if not path:
        return
    self.track = TrackState.from_json(Path(path).read_text())
    self.track_pred = self.track_assoc = None
    _track_refresh(self)
    set_text(self.txt_wide, self.track.summary())
    _wide_draw(self)


def _track_clear(self) -> None:
    self.track = self.track_pred = self.track_assoc = None
    self.track_looks = []
    self.track_list.delete(0, "end")
    _track_refresh(self)
    _wide_draw(self)


def _track_extent(self, fallback):
    lons = [f.lon for f in self.track.fixes]
    lats = [f.lat for f in self.track.fixes]
    if self.track_pred is not None:
        for c in self.track_pred.cells(0.97):
            lons += [c[0], c[2]]
            lats += [c[1], c[3]]
    w, e = min(lons), max(lons)
    s_, n = min(lats), max(lats)
    if self.track_pred is None and len(lons) == 1:
        return fallback
    pad_x = max((e - w) * 0.12, 0.15)
    pad_y = max((n - s_) * 0.12, 0.12)
    return (w - pad_x, s_ - pad_y, e + pad_x, n + pad_y)


def _draw_track(self, ax) -> None:
    st = self.track
    if st is None:
        return
    pr = self.track_pred
    if pr is not None:
        ww = pr.w / max(pr.w.sum(), 1e-12)
        ax.scatter(pr.lon, pr.lat, s=2, c=ww, cmap="plasma", alpha=0.35,
                   zorder=5, linewidths=0)
        for c in pr.cells(0.95):
            ax.plot([c[0], c[2], c[2], c[0], c[0]],
                    [c[1], c[1], c[3], c[3], c[1]], "-", color="#ffcc00",
                    lw=0.6, alpha=0.7, zorder=5)
    xs = [f.lon for f in st.fixes]
    ys = [f.lat for f in st.fixes]
    ax.plot(xs, ys, "-", color="#ff00ff", lw=2, zorder=12)
    for i, f in enumerate(st.fixes):
        ax.plot(f.lon, f.lat, "D", color="#ff00ff", mec="white", ms=8,
                zorder=12)
        ax.annotate(f"fix {i + 1} {f.t:%m-%d %H:%MZ}"
                    + (f" P={f.link_prob:.2f}" if i else ""),
                    (f.lon, f.lat), xytext=(7, 7), textcoords="offset points",
                    color="white", fontsize=7, zorder=12,
                    bbox=dict(fc="#80008080", ec="none"))
    a = self.track_assoc
    if a is not None:
        for c in a.candidates[:6]:
            ax.plot(c.lon, c.lat, "o", mfc="none", mec="#00ffff", ms=12,
                    mew=2, zorder=13)
            ax.annotate(f"{c.prob * 100:.0f}%", (c.lon, c.lat),
                        xytext=(8, -12), textcoords="offset points",
                        color="#00ffff", fontsize=8, fontweight="bold",
                        zorder=13, bbox=dict(fc="black", ec="none",
                                             alpha=0.5))


NavsimApp._build_wide = _build_wide
