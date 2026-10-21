"""Naval Base Watch: the desktop application.

Layout
------
A base list on the left; on the right, five views of the selected base:

* **Overview** -- the headline numbers, the activity chart and the latest
  notable changes.
* **Berths** -- which berths held a hull on each pass, as a grid.  This is
  the raw material every other view is built from.
* **Events** -- every arrival, departure and anomaly, filterable, each with
  a structured explanation: what, when, where, size reading, evidence,
  confidence.
* **Radar** -- the radar image behind any pass or event, with hulls outlined.
* **Compare** -- several bases on one time axis.

Work runs on one background thread; the window never blocks, every job
shows progress and can be cancelled, and finished analyses stay in memory.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
import traceback
from datetime import datetime, timezone
from tkinter import filedialog, messagebox, ttk

import numpy as np

from . import charts, credentials, pursuit, radar, store
from .bases import BASES, BY_KEY, REGIONS
from .monitor import BaseReport, Event, compare, interpret, normal_range

PERIODS = {"6 months": 6, "9 months": 9, "12 months": 12, "18 months": 18,
           "24 months": 24}

# ---------------------------------------------------------------- theme
# Quiet and expensive-looking rather than loud: a slate sidebar, a warm
# off-white workspace, white cards with hairline borders, one ink-blue
# accent, large tabular numbers, and no decoration that does not carry
# meaning.
APP_NAME = "Keelix"
BG = "#f6f5f2"            # workspace
PANEL = "#ffffff"         # cards
SIDE = "#161b22"          # sidebar
SIDE_TEXT = "#e6e8eb"
SIDE_MUTED = "#8b949e"
INK = "#1b1f24"
MUTED = "#6e7681"
LINE = "#e3e1dc"
ACCENT = "#0b5fa5"
RED = "#b42318"
GREEN = "#1a7f37"
AMBER = "#9a6700"

SEVERITY_LABEL = {3: "Headline", 2: "Notable", 1: "Routine", 0: "Minor"}


def _apply_theme(root: tk.Tk) -> dict:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    base = tkfont.nametofont("TkDefaultFont")
    fam = base.cget("family")
    base.configure(size=12)
    fonts = {
        "body": base,
        "small": tkfont.Font(family=fam, size=10),
        "caps": tkfont.Font(family=fam, size=9, weight="bold"),
        "title": tkfont.Font(family=fam, size=17, weight="bold"),
        "brand": tkfont.Font(family=fam, size=14, weight="bold"),
        "kpi": tkfont.Font(family=fam, size=26, weight="bold"),
        "label": tkfont.Font(family=fam, size=11, weight="bold"),
        "mono": tkfont.Font(family="Menlo", size=10),
    }
    style.configure(".", background=BG, foreground=INK, font=base,
                    borderwidth=0, focuscolor=BG)
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Side.TFrame", background=SIDE)
    style.configure("TLabel", background=BG, foreground=INK)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("Muted.TLabel", foreground=MUTED, font=fonts["small"])
    style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED,
                    font=fonts["small"])
    style.configure("Caps.TLabel", background=PANEL, foreground=MUTED,
                    font=fonts["caps"])
    style.configure("Title.TLabel", font=fonts["title"])
    style.configure("Brand.TLabel", background=SIDE, foreground=SIDE_TEXT,
                    font=fonts["brand"])
    style.configure("SideMuted.TLabel", background=SIDE, foreground=SIDE_MUTED,
                    font=fonts["small"])
    style.configure("KPI.TLabel", background=PANEL, font=fonts["kpi"])
    style.configure("KPIcap.TLabel", background=PANEL, foreground=MUTED,
                    font=fonts["small"])
    style.configure("TButton", padding=(12, 6), background="#ffffff",
                    bordercolor=LINE, lightcolor="#ffffff", darkcolor="#ffffff",
                    relief="flat")
    style.map("TButton", background=[("active", "#f0efeb")])
    style.configure("Accent.TButton", foreground="white", background=ACCENT,
                    bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                    padding=(14, 6))
    style.map("Accent.TButton", background=[("active", "#094b84"),
                                             ("disabled", "#9cbbd8")])
    style.configure("Nav.TButton", background=SIDE, foreground=SIDE_MUTED,
                    bordercolor=SIDE, lightcolor=SIDE, darkcolor=SIDE,
                    padding=(14, 8), anchor="w", font=fonts["label"])
    style.map("Nav.TButton", background=[("active", "#1f262e")],
              foreground=[("active", SIDE_TEXT)])
    style.configure("NavOn.TButton", background="#232b35",
                    foreground=SIDE_TEXT, bordercolor="#232b35",
                    lightcolor="#232b35", darkcolor="#232b35",
                    padding=(14, 8), anchor="w", font=fonts["label"])
    style.map("NavOn.TButton", background=[("active", "#232b35")])
    style.layout("Plain.TNotebook.Tab", [])
    style.configure("Plain.TNotebook", background=BG, borderwidth=0)
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    rowheight=26, borderwidth=0, font=base,
                    bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
    style.configure("Treeview.Heading", font=fonts["caps"], background=PANEL,
                    foreground=MUTED, relief="flat", padding=(6, 6))
    style.map("Treeview", background=[("selected", "#e8f0f9")],
              foreground=[("selected", INK)])
    style.configure("Side.Treeview", background=SIDE, fieldbackground=SIDE,
                    foreground=SIDE_TEXT, rowheight=26, borderwidth=0,
                    bordercolor=SIDE, lightcolor=SIDE, darkcolor=SIDE)
    style.configure("Side.Treeview.Heading", background=SIDE,
                    foreground=SIDE_MUTED, font=fonts["caps"], relief="flat")
    style.map("Side.Treeview", background=[("selected", "#2b3440")],
              foreground=[("selected", "#ffffff")])
    style.configure("Side.TEntry", fieldbackground="#232b35",
                    foreground=SIDE_TEXT, bordercolor="#232b35",
                    lightcolor="#232b35", darkcolor="#232b35",
                    insertcolor=SIDE_TEXT, padding=6)
    style.configure("TEntry", fieldbackground=PANEL, bordercolor=LINE,
                    lightcolor=PANEL, darkcolor=PANEL, padding=5)
    style.configure("TCombobox", fieldbackground=PANEL, bordercolor=LINE,
                    arrowcolor=MUTED, padding=4)
    style.configure("TSpinbox", fieldbackground=PANEL, bordercolor=LINE,
                    arrowcolor=MUTED, padding=4)
    style.configure("TProgressbar", troughcolor=LINE, background=ACCENT,
                    thickness=4, borderwidth=0)
    style.configure("TCheckbutton", background=BG)
    style.configure("Panel.TCheckbutton", background=PANEL)
    style.configure("Vertical.TScrollbar", background=LINE, troughcolor=PANEL,
                    bordercolor=PANEL, arrowcolor=MUTED, width=10)
    style.configure("Side.Vertical.TScrollbar", background="#2b3440",
                    troughcolor=SIDE, bordercolor=SIDE, arrowcolor=SIDE_MUTED,
                    width=10)
    root.configure(background=BG)
    return fonts


def _short_country(c: str) -> str:
    return {"USA": "US", "Russia (Crimea)": "RU", "Russia": "RU",
            "China (Djibouti)": "CN", "China": "CN", "UK": "UK",
            "France": "FR", "Japan": "JP", "Germany": "DE", "Italy": "IT",
            "Spain": "ES", "Greece": "GR", "Turkey": "TR", "Canada": "CA",
            "Australia": "AU", "India": "IN", "Pakistan": "PK",
            "South Korea": "KR", "Netherlands": "NL", "Syria": "SY",
            "Israel": "IL", "Bahrain": "BH", "Iran": "IR"}.get(c, c)


def _autowrap(label: ttk.Label, margin: int = 24) -> None:
    """Wrap a label to the width of its container.

    Measured from the parent, not the label itself: a label that wraps
    gets narrower, which would shrink its own next measurement, and so on
    down to one word per line.
    """
    parent = label.master
    last = {"w": 0}

    def on_conf(e):
        w = parent.winfo_width() - margin
        if w > 60 and abs(w - last["w"]) > 4:
            last["w"] = w
            label.configure(wraplength=w)
    parent.bind("<Configure>", on_conf, add=True)


def _scrolled_text(master, font, height=10, **kw) -> tk.Text:
    """A borderless Text with a scrollbar that appears only when needed."""
    frame = ttk.Frame(master, style="Panel.TFrame")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    t = tk.Text(frame, wrap="word", relief="flat", background=PANEL,
                foreground=INK, insertbackground=INK,
                selectbackground="#d6e4f5", selectforeground=INK,
                font=font, padx=6, pady=6, height=height, bd=0,
                highlightthickness=0, **kw)
    sb = ttk.Scrollbar(frame, orient="vertical", command=t.yview)

    def on_scroll(lo, hi):
        sb.set(lo, hi)
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            sb.grid_remove()
        else:
            sb.grid(row=0, column=1, sticky="ns")
    t.configure(yscrollcommand=on_scroll)
    t.grid(row=0, column=0, sticky="nsew")
    sb.grid(row=0, column=1, sticky="ns")
    t.frame = frame
    return t


def _placeholder(entry: ttk.Entry, var: tk.StringVar, text: str) -> None:
    """Grey hint text in an empty entry, cleared on focus."""
    state = {"on": False}

    def show():
        if not var.get():
            state["on"] = True
            entry.configure(foreground=SIDE_MUTED)
            var.set(text)

    def hide(_e=None):
        if state["on"]:
            state["on"] = False
            var.set("")
            entry.configure(foreground=SIDE_TEXT)

    def blur(_e=None):
        if not var.get():
            show()
    entry.bind("<FocusIn>", hide)
    entry.bind("<FocusOut>", blur)
    entry.configure(foreground=SIDE_TEXT)
    show()
    entry.placeholder_active = lambda: state["on"]


def _figure(master, size=(8, 4)):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=100, facecolor="white")
    canvas = FigureCanvasTkAgg(fig, master=master)
    w = canvas.get_tk_widget()
    w.configure(highlightthickness=0, bd=0)
    return fig, canvas, w


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1600, int(sw * 0.94))}x{min(1000, int(sh * 0.9))}"
                      f"+{int(sw * 0.03)}+{int(sh * 0.04)}")
        self.minsize(1200, 760)
        self.fonts = _apply_theme(self)
        self.reports: dict[str, BaseReport] = {}
        self.current: str | None = None
        self.checked: set[str] = set()
        self.selected_event: Event | None = None
        self.view_pass: int = -1
        self.jobs: "queue.Queue" = queue.Queue()
        self.ui: "queue.Queue" = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self._photos: dict = {}
        self._build()
        threading.Thread(target=self._worker, daemon=True).start()
        self.after(100, self._poll)
        self._fill_list()
        self.after(250, self._startup)

    # =============================================================== layout
    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_sidebar()
        self._build_toolbar()
        self.nb = ttk.Notebook(self, style="Plain.TNotebook")
        self.nb.grid(row=1, column=1, sticky="nsew", padx=(18, 18),
                     pady=(0, 6))
        self._build_overview()
        self._build_berths()
        self._build_events()
        self._build_radar()
        self._build_compare()
        self._build_carriers()
        self._build_method()
        self._build_statusbar()
        self._build_nav()
        self._select_view(0)

    VIEWS = ("Overview", "Berths", "Events", "Radar", "Compare",
             "Sea search", "Method")

    def _build_nav(self) -> None:
        self.nav_btns = []
        for i, name in enumerate(self.VIEWS):
            b = ttk.Button(self.nav, text=name, style="Nav.TButton",
                           command=lambda i=i: self._select_view(i))
            b.pack(fill="x", pady=1)
            self.nav_btns.append(b)
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._sync_nav())

    def _select_view(self, i: int) -> None:
        self.nb.select(i)
        self._sync_nav()

    def _sync_nav(self) -> None:
        cur = self.nb.index(self.nb.select()) if self.nb.tabs() else 0
        for i, b in enumerate(self.nav_btns):
            b.configure(style="NavOn.TButton" if i == cur else "Nav.TButton")

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(18, 16, 18, 10))
        bar.grid(row=0, column=1, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.base_title = tk.StringVar(value="Select a base")
        self.base_sub = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.base_title, style="Title.TLabel"
                  ).grid(row=0, column=0, sticky="w")
        subf = ttk.Frame(bar)
        subf.grid(row=1, column=0, sticky="ew")
        sub = ttk.Label(subf, textvariable=self.base_sub, style="Muted.TLabel",
                        justify="left")
        sub.pack(anchor="w", fill="x")
        _autowrap(sub, margin=12)
        right = ttk.Frame(bar)
        right.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(right, text="Period", style="Muted.TLabel").pack(
            side="left", padx=(0, 6))
        self.period = tk.StringVar(value="12 months")
        cb = ttk.Combobox(right, textvariable=self.period, values=list(PERIODS),
                          state="readonly", width=10)
        cb.pack(side="left", padx=(0, 12))
        cb.bind("<<ComboboxSelected>>", lambda e: self._period_changed())
        ttk.Button(right, text="Update base", style="Accent.TButton",
                   command=self._update_current).pack(side="left")
        ttk.Button(right, text="Update ticked",
                   command=self._update_checked).pack(side="left", padx=6)
        ttk.Button(right, text="Settings", command=self._settings).pack(
            side="left")

    def _build_sidebar(self) -> None:
        side = ttk.Frame(self, style="Side.TFrame", padding=(16, 18, 16, 14))
        side.grid(row=0, column=0, rowspan=3, sticky="ns")
        side.rowconfigure(3, weight=1)
        ttk.Label(side, text=APP_NAME, style="Brand.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(side, text="naval base activity from radar",
                  style="SideMuted.TLabel").grid(row=0, column=0, sticky="w",
                                                 pady=(26, 0))
        self.nav = ttk.Frame(side, style="Side.TFrame")
        self.nav.grid(row=1, column=0, sticky="ew", pady=(22, 18))
        self.search = tk.StringVar()
        self.search.trace_add("write", lambda *_: self._fill_list())
        ent = ttk.Entry(side, textvariable=self.search, width=30,
                        style="Side.TEntry")
        ent.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        _placeholder(ent, self.search, "Search bases")
        lst = ttk.Frame(side, style="Side.TFrame")
        lst.grid(row=3, column=0, sticky="nsew")
        lst.rowconfigure(0, weight=1)
        lst.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(lst, columns=("tick", "state"),
                                 show="tree headings", selectmode="browse",
                                 style="Side.Treeview")
        self.tree.heading("#0", text="BASE")
        self.tree.heading("tick", text="CMP")
        self.tree.heading("state", text="LATEST")
        self.tree.column("#0", width=236, stretch=True)
        self.tree.column("tick", width=44, anchor="center", stretch=False)
        self.tree.column("state", width=112, stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tsb = ttk.Scrollbar(lst, orient="vertical", command=self.tree.yview,
                            style="Side.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-1>", self._on_click_tree, add=True)
        row = ttk.Frame(side, style="Side.TFrame")
        row.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(row, text="Tick region", style="Nav.TButton",
                   command=self._tick_region).pack(side="left")
        ttk.Button(row, text="Clear", style="Nav.TButton",
                   command=self._clear_ticks).pack(side="left", padx=4)
        ttk.Button(row, text="Tour", style="Nav.TButton",
                   command=lambda: Tour(self)).pack(side="right")
        self.key_status = tk.StringVar()
        ttk.Label(side, textvariable=self.key_status, style="SideMuted.TLabel"
                  ).grid(row=5, column=0, sticky="w", pady=(10, 0))
        self._refresh_key_status()

    def _panel(self, parent, **grid) -> ttk.Frame:
        """A white card with a hairline border."""
        outer = tk.Frame(parent, background=LINE, bd=0,
                         highlightthickness=0)
        outer.grid(**grid)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        f = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        f.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        return f

    # --------------------------------------------------------------- overview
    def _build_overview(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Overview")
        f.columnconfigure(0, weight=5)
        f.columnconfigure(1, weight=2)
        f.rowconfigure(1, weight=1)
        kpis = ttk.Frame(f)
        kpis.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.kpi: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        for k, (key, cap) in enumerate((
                ("now", "SHIPS IN PORT"), ("usual", "USUAL LEVEL"),
                ("delta", "SINCE LAST PASS"), ("carriers", "CARRIERS"),
                ("metres", "HULL ALONGSIDE"), ("quality", "LATEST PASS"))):
            outer = tk.Frame(kpis, background=LINE)
            outer.grid(row=0, column=k, sticky="ew", padx=(0, 10))
            kpis.columnconfigure(k, weight=1)
            card = ttk.Frame(outer, style="Panel.TFrame", padding=(16, 12))
            card.pack(fill="both", expand=True, padx=1, pady=1)
            v = tk.StringVar(value="–")
            c = tk.StringVar(value=cap)
            d = tk.StringVar(value="")
            ttk.Label(card, textvariable=c, style="Caps.TLabel").pack(
                anchor="w")
            ttk.Label(card, textvariable=v, style="KPI.TLabel").pack(
                anchor="w", pady=(2, 0))
            dl = ttk.Label(card, textvariable=d, style="PanelMuted.TLabel",
                           justify="left")
            dl.pack(anchor="w", fill="x")
            _autowrap(dl, margin=34)
            self.kpi[key] = (v, c, d)
        chart = self._panel(f, row=1, column=0, sticky="nsew", padx=(0, 8))
        chart.rowconfigure(1, weight=1)
        chart.columnconfigure(0, weight=1)
        ttk.Label(chart, text="ACTIVITY", style="Caps.TLabel").grid(
            row=0, column=0, sticky="w")
        hint = ttk.Label(chart, text="Large ships in port on every radar "
                                     "pass. Click a marker to read what "
                                     "happened; down is a departure, up an "
                                     "arrival, size follows the ship.",
                         style="PanelMuted.TLabel", justify="left")
        hint.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        _autowrap(hint, margin=34)
        chart.rowconfigure(2, weight=1)
        self.fig, self.canvas, w = _figure(chart, (9, 4.8))
        w.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.canvas.mpl_connect("pick_event", self._on_pick)
        recent = self._panel(f, row=1, column=1, sticky="nsew")
        recent.rowconfigure(1, weight=1)
        recent.columnconfigure(0, weight=1)
        ttk.Label(recent, text="LATEST NOTABLE CHANGES", style="Caps.TLabel"
                  ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.recent = _scrolled_text(recent, self.fonts["body"], height=10,
                                     cursor="arrow", width=34)
        self.recent.frame.grid(row=1, column=0, sticky="nsew")
        self.recent.tag_configure("h", font=self.fonts["label"], foreground=INK)
        self.recent.tag_configure("d", foreground=MUTED,
                                  font=self.fonts["small"])
        self.recent.tag_configure("sev3", foreground=RED)
        self.recent.tag_configure("link", foreground=ACCENT, underline=True)
        self.recent.tag_bind("link", "<Button-1>", self._recent_click)
        self.notes_var = tk.StringVar()
        nl = ttk.Label(f, textvariable=self.notes_var, style="Muted.TLabel",
                       justify="left")
        nl.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        _autowrap(nl)

    # ----------------------------------------------------------------- berths
    def _build_berths(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Berths")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)
        head = ttk.Frame(f)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(head, text="Berth occupancy, pass by pass").pack(anchor="w")
        hl = ttk.Label(head, style="Muted.TLabel", text=(
            "Each row is a berth (numbered north to south), each column a "
            "radar pass. Click a cell to see that pass in the radar view."))
        hl.pack(anchor="w", fill="x")
        _autowrap(hl)
        panel = self._panel(f, row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        self.bfig, self.bcanvas, w = _figure(panel, (10, 6))
        w.grid(row=0, column=0, sticky="nsew")
        self.bcanvas.mpl_connect("button_press_event", self._on_berth_click)
        self.berth_note = tk.StringVar()
        bl = ttk.Label(f, textvariable=self.berth_note, style="Muted.TLabel",
                       justify="left")
        bl.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        _autowrap(bl)

    # ----------------------------------------------------------------- events
    def _build_events(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Events")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=3)
        f.columnconfigure(1, weight=2)
        flt = ttk.Frame(f)
        flt.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(flt, text="Show", style="Muted.TLabel").pack(side="left")
        self.sev_vars = {}
        for sev, lab in ((3, "Headlines"), (2, "Notable"), (1, "Routine"),
                         (0, "Minor")):
            v = tk.BooleanVar(value=sev >= 1)
            self.sev_vars[sev] = v
            ttk.Checkbutton(flt, text=lab, variable=v,
                            command=self._fill_events).pack(side="left",
                                                            padx=(8, 0))
        ttk.Label(flt, text="   Type", style="Muted.TLabel").pack(side="left")
        self.kind_filter = tk.StringVar(value="all")
        kcb = ttk.Combobox(flt, textvariable=self.kind_filter, state="readonly",
                           width=26, values=(
                               "all", "arrivals and departures",
                               "groups and level changes",
                               "long stays", "single sightings"))
        kcb.pack(side="left", padx=(6, 0))
        kcb.bind("<<ComboboxSelected>>", lambda e: self._fill_events())
        ttk.Button(flt, text="Show in radar view",
                   command=self._show_event_radar).pack(side="right")
        ttk.Button(flt, text="Save report…",
                   command=self._save_report).pack(side="right", padx=6)
        left = self._panel(f, row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        cols = ("when", "sev", "what", "conf")
        self.ev_tree = ttk.Treeview(left, columns=cols, show="headings",
                                    selectmode="browse")
        for c, txt, wd in (("when", "When", 140), ("sev", "Level", 72),
                           ("what", "What happened", 230),
                           ("conf", "Confidence", 84)):
            self.ev_tree.heading(c, text=txt)
            self.ev_tree.column(c, width=wd, anchor="w",
                                stretch=(c == "what"))
        sb = ttk.Scrollbar(left, orient="vertical", command=self.ev_tree.yview)
        self.ev_tree.configure(yscrollcommand=sb.set)
        self.ev_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.ev_tree.bind("<<TreeviewSelect>>", self._on_event_select)
        self.ev_tree.tag_configure("sev3", foreground=RED)
        self.ev_tree.tag_configure("sev2", foreground=INK)
        self.ev_tree.tag_configure("sev1", foreground="#374151")
        self.ev_tree.tag_configure("sev0", foreground=MUTED)
        right = self._panel(f, row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.detail = _scrolled_text(right, self.fonts["body"], height=12)
        self.detail.frame.grid(row=0, column=0, sticky="nsew")
        self.detail.tag_configure("title", font=self.fonts["title"])
        self.detail.tag_configure("field", font=self.fonts["label"],
                                  foreground=MUTED)
        self.detail.tag_configure("body", font=self.fonts["body"])
        self.detail.tag_configure("red", foreground=RED)
        self.detail.tag_configure("green", foreground=GREEN)
        self.detail.tag_configure("amber", foreground=AMBER)

    # ------------------------------------------------------------------ radar
    def _build_radar(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Radar")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)
        ctl = ttk.Frame(f)
        ctl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(ctl, text="◀ Previous pass",
                   command=lambda: self._step_pass(-1)).pack(side="left")
        ttk.Button(ctl, text="Next pass ▶",
                   command=lambda: self._step_pass(1)).pack(side="left",
                                                            padx=(4, 12))
        self.radar_title = tk.StringVar(value="")
        ttk.Label(ctl, textvariable=self.radar_title).pack(side="left")
        opts = ttk.Frame(ctl)
        opts.pack(side="right")
        self.opt_change = tk.BooleanVar(value=False)
        self.opt_struct = tk.BooleanVar(value=False)
        self.opt_berths = tk.BooleanVar(value=True)
        self.opt_photo = tk.BooleanVar(value=False)
        for txt, var in (("change vs empty state", self.opt_change),
                         ("structure", self.opt_struct),
                         ("berth numbers", self.opt_berths),
                         ("satellite photo (reference, undated)",
                          self.opt_photo)):
            ttk.Checkbutton(opts, text=txt, variable=var,
                            command=self._draw_radar).pack(side="left",
                                                           padx=(12, 0))
        panel = self._panel(f, row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.rowconfigure(1, weight=0)
        panel.columnconfigure(0, weight=1)
        self.rfig, self.rcanvas, w = _figure(panel, (10, 6))
        w.grid(row=0, column=0, sticky="nsew")
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        tbf = ttk.Frame(panel, style="Panel.TFrame")
        tbf.grid(row=1, column=0, sticky="ew")
        tb = NavigationToolbar2Tk(self.rcanvas, tbf, pack_toolbar=False)
        tb.configure(background=PANEL)
        tb.pack(side="left")
        rl = ttk.Label(f, style="Muted.TLabel", justify="left",
                  text=("Bright = strong radar echo. Red outlines are hulls "
                        "found on that pass; the green outline is the hull an "
                        "event refers to; dashed orange are ships present on "
                        "every pass of the record. 'Change vs empty state' "
                        "shows each pixel against its own emptiest state, "
                        "which is what the detector actually looks at.")
                  )
        rl.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        _autowrap(rl)

    # ---------------------------------------------------------------- compare
    def _build_compare(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Compare")
        f.rowconfigure(1, weight=3)
        f.rowconfigure(2, weight=1)
        f.columnconfigure(0, weight=1)
        head = ttk.Frame(f)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(head, text="Ticked bases on one time axis").pack(side="left")
        ttk.Button(head, text="Refresh", command=self._compare).pack(
            side="right")
        panel = self._panel(f, row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        self.cfig, self.ccanvas, w = _figure(panel, (10, 5.5))
        w.grid(row=0, column=0, sticky="nsew")
        txt = self._panel(f, row=2, column=0, sticky="nsew", pady=(8, 6))
        txt.rowconfigure(0, weight=1)
        txt.columnconfigure(0, weight=1)
        self.ctext = _scrolled_text(txt, self.fonts["body"], height=7)
        self.ctext.frame.grid(row=0, column=0, sticky="nsew")

    # --------------------------------------------------------------- carriers
    def _build_carriers(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Sea search")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=2)
        f.columnconfigure(1, weight=3)
        head = ttk.Frame(f)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(head, text="Sea search · experimental size candidates").pack(
            anchor="w")
        sl = ttk.Label(head, style="Muted.TLabel", text=(
            "Pick a carrier-sized departure to inspect nearby radar strips "
            "for large, wide targets. Size and compatible timing do not "
            "establish vessel identity; nearby targets are not verified "
            "escorts. Coverage and request limits leave gaps. A search "
            "usually takes 10-40 minutes."))
        sl.pack(anchor="w", fill="x")
        _autowrap(sl)
        left = self._panel(f, row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.rowconfigure(3, weight=2)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="CARRIER-SIZED DEPARTURES AT THIS BASE",
                  style="Caps.TLabel").grid(row=0, column=0, sticky="w",
                                            pady=(0, 8))
        self.car_tree = ttk.Treeview(left, columns=("when", "size", "conf"),
                                     show="headings", selectmode="browse",
                                     height=6)
        for c, txt, wd in (("when", "Left between", 170),
                           ("size", "Hull on radar", 120),
                           ("conf", "Confidence", 90)):
            self.car_tree.heading(c, text=txt)
            self.car_tree.column(c, width=wd)
        self.car_tree.grid(row=1, column=0, sticky="nsew")
        ctl = ttk.Frame(left, style="Panel.TFrame")
        ctl.grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Label(ctl, text="days after", style="PanelMuted.TLabel").pack(
            side="left")
        self.car_days = tk.IntVar(value=10)
        ttk.Spinbox(ctl, from_=2, to=30, textvariable=self.car_days,
                    width=4).pack(side="left", padx=(4, 10))
        ttk.Label(ctl, text="radius km", style="PanelMuted.TLabel").pack(
            side="left")
        self.car_radius = tk.IntVar(value=700)
        ttk.Spinbox(ctl, from_=200, to=1500, increment=100,
                    textvariable=self.car_radius, width=5).pack(
            side="left", padx=(4, 10))
        ttk.Button(ctl, text="Inspect radar candidates", style="Accent.TButton",
                   command=self._pursue).pack(side="left")
        self.car_text = _scrolled_text(left, self.fonts["body"], height=12)
        self.car_text.frame.grid(row=3, column=0, sticky="nsew")
        right = self._panel(f, row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.pfig, self.pcanvas, w = _figure(right, (8, 6))
        w.grid(row=0, column=0, sticky="nsew")
        charts.blank(self.pfig, "Select a carrier-sized departure and press "
                                "'Inspect radar candidates'.")
        self.pursuits: dict[str, pursuit.Pursuit] = {}

    def _fill_carriers(self, rep: BaseReport) -> None:
        self.car_tree.delete(*self.car_tree.get_children())
        self._car_ids = {}
        for k, e in enumerate(sorted(pursuit.carrier_departures(rep),
                                     key=lambda e: e.t1, reverse=True)):
            iid = f"c{k}"
            self.car_tree.insert("", "end", iid=iid, values=(
                f"{e.t0:%d %b} – {e.t1:%d %b %Y}",
                f"{e.length_m:.0f} × {(e.width_m or 0):.0f} m", e.confidence))
            self._car_ids[iid] = e
        if not self._car_ids:
            self.car_text.delete("1.0", "end")
            self.car_text.insert("1.0", "No carrier-sized departure in this "
                                        "period at this base.")

    def _pursue(self) -> None:
        rep = self.reports.get(self.current or "")
        sel = self.car_tree.selection()
        if rep is None or not sel:
            messagebox.showinfo("Pick a departure",
                                "Select a carrier-sized departure first.")
            return
        if not radar.have_credentials():
            self._settings()
            return
        e = self._car_ids[sel[0]]
        b = BY_KEY[rep.key]
        days, radius = float(self.car_days.get()), float(self.car_radius.get())
        key = f"{rep.key}:{e.t1:%Y%m%d}"
        self._select_view(5)
        self.car_text.delete("1.0", "end")
        self.car_text.insert("1.0", "Searching… each radar strip is fetched "
                                    "at 50 m in 60 km tiles and scanned for a "
                                    "hull at least 200 m long; the best "
                                    "candidates are re-measured at 10 m "
                                    "output spacing from the same acquisition. "
                                    "This does not confirm vessel identity. "
                                    "Copernicus throttles free accounts, so "
                                    "this takes 10-40 minutes. You can cancel "
                                    "and resume later: fetched tiles are "
                                    "kept.")

        def job():
            P = pursuit.search(rep.key, b.name, b.centre, e.t0, e.t1,
                               e.length_m, e.width_m or 80.0, days, radius,
                               self._progress, self.cancel)
            pursuit.link_arrivals(P, self.reports)
            try:
                from .basemap import fetch_basemap
                bm = fetch_basemap(P.region, "ocean", 1000)
            except Exception:
                bm = None
            return P, bm

        def done(out):
            P, bm = out
            self.pursuits[key] = P
            self.car_text.delete("1.0", "end")
            self.car_text.insert("1.0", P.summary())
            charts.draw_pursuit(self.pfig, P, bm)
            self.pcanvas.draw_idle()
            self.status.set(f"Sea search done: {len(P.chain)} tentative "
                            f"association(s), {len(P.searched)} passes attempted")

        self._submit("Inspecting experimental sea-search candidates…", job, done)

    def _build_method(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Method & limits")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        panel = self._panel(f, row=0, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        t = _scrolled_text(panel, self.fonts["body"], height=20)
        t.configure(padx=14, pady=10)
        t.frame.grid(row=0, column=0, sticky="nsew")
        t.tag_configure("h", font=self.fonts["title"])
        for block in METHOD.split("\n\n"):
            if block.startswith("# "):
                t.insert("end", block[2:] + "\n\n", "h")
            else:
                t.insert("end", block + "\n\n")
        t.configure(state="disabled")

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, padding=(18, 4, 18, 10))
        bar.grid(row=2, column=1, sticky="ew")
        bar.columnconfigure(1, weight=1)
        self.bar = ttk.Progressbar(bar, length=220, maximum=1.0)
        self.bar.grid(row=0, column=0, sticky="w")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.status, style="Muted.TLabel").grid(
            row=0, column=1, sticky="w", padx=10)
        self.cancel_btn = ttk.Button(bar, text="Cancel", command=self._cancel,
                                     state="disabled")
        self.cancel_btn.grid(row=0, column=2, sticky="e")

    # ============================================================ base list
    def _fill_list(self) -> None:
        if not hasattr(self, "tree"):
            return                      # placeholder fires before the list
        q = self.search.get().strip().lower()
        if q == "search bases":
            q = ""
        self.tree.delete(*self.tree.get_children())
        for region in REGIONS:
            items = [b for b in BASES if b.region == region and (
                not q or q in b.name.lower() or q in b.country.lower()
                or q in b.key)]
            if not items:
                continue
            rid = f"region:{region}"
            self.tree.insert("", "end", iid=rid, text=region, open=True)
            for b in items:
                self.tree.insert(rid, "end", iid=b.key,
                                 text=f"{b.name}  ·  {_short_country(b.country)}",
                                 values=("☑" if b.key in self.checked else "☐",
                                         self._state_text(b.key)))
        if self.current and self.tree.exists(self.current):
            self.tree.selection_set(self.current)
            self.tree.see(self.current)

    def _state_text(self, key: str) -> str:
        rep = self.reports.get(key)
        if rep is not None and rep.passes:
            last = rep.passes[-1]
            pct = (f" · {last.in_port / rep.usual_in_port:.0%}"
                   if rep.usual_in_port else "")
            return f"{last.in_port} ships{pct}"
        meta = store.load_meta(BY_KEY[key])
        if meta.get("summary"):
            return meta["summary"].replace("~", "")
        n = len(meta.get("passes", []))
        return f"{n} passes" if n else "not downloaded"

    def _update_row(self, key: str) -> None:
        if self.tree.exists(key):
            self.tree.item(key, values=("☑" if key in self.checked else "☐",
                                        self._state_text(key)))

    def _on_click_tree(self, ev) -> None:
        col = self.tree.identify_column(ev.x)
        iid = self.tree.identify_row(ev.y)
        if col == "#1" and iid in BY_KEY:
            self.checked.symmetric_difference_update({iid})
            self._update_row(iid)
            return "break"

    def _on_select(self, _ev=None) -> None:
        sel = self.tree.selection()
        if not sel or sel[0] not in BY_KEY:
            return
        key = sel[0]
        if key == self.current and key in self.reports:
            return
        self.current = key
        self.selected_event = None
        self.view_pass = -1
        self._open_base(key)

    def _tick_region(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        region = (iid.split(":", 1)[1] if iid.startswith("region:")
                  else BY_KEY[iid].region)
        self.checked.update(b.key for b in BASES if b.region == region)
        self._fill_list()

    def _clear_ticks(self) -> None:
        self.checked.clear()
        self._fill_list()

    # ================================================================= jobs
    def _months(self) -> float:
        return float(PERIODS.get(self.period.get(), 12))

    def _submit(self, label: str, fn, done=None) -> None:
        if self.busy:
            messagebox.showinfo("Busy", "Another job is running. Wait for it "
                                        "or press Cancel.")
            return
        self.busy = True
        self.cancel.clear()
        self.cancel_btn.configure(state="normal")
        self.status.set(label)
        self.bar["value"] = 0.0
        self.jobs.put((fn, done))

    def _worker(self) -> None:
        while True:
            fn, done = self.jobs.get()
            try:
                out = fn()
                self.ui.put(("done", done, out))
            except store.Cancelled:
                self.ui.put(("cancelled", None, None))
            except Exception as exc:
                traceback.print_exc()
                self.ui.put(("error", None, exc))

    def _progress(self, frac: float, msg: str) -> None:
        self.ui.put(("progress", frac, msg))

    def _poll(self) -> None:
        try:
            while True:
                kind, a, b = self.ui.get_nowait()
                if kind == "progress":
                    self.bar["value"] = max(0.0, min(1.0, a))
                    self.status.set(b)
                elif kind == "call":
                    a()
                else:
                    self.busy = False
                    self.cancel_btn.configure(state="disabled")
                    if kind == "done":
                        self.bar["value"] = 0.0
                        if a is not None:
                            a(b)
                    elif kind == "cancelled":
                        self.status.set("Cancelled")
                        self.bar["value"] = 0.0
                    else:
                        self.status.set(f"Error: {b}")
                        messagebox.showerror("Something went wrong", str(b))
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _cancel(self) -> None:
        self.cancel.set()
        self.status.set("Cancelling…")

    # =============================================================== actions
    def _startup(self) -> None:
        if not _ui_state().get("tour_done"):
            self.after(600, lambda: Tour(self))
        if not radar.have_credentials():
            self.status.set("Add your free Copernicus key in Settings to "
                            "download radar passes.")
        first = next((b.key for b in BASES
                      if store.load_meta(b).get("passes")), "norfolk")
        if self.tree.exists(first):
            self.tree.selection_set(first)
            self.tree.see(first)

    def _period_changed(self) -> None:
        self.reports.clear()
        if self.current:
            key = self.current
            self.current = None
            self.tree.selection_set(key)
            self._on_select()

    def _open_base(self, key: str) -> None:
        b = BY_KEY[key]
        self.base_title.set(f"{b.name}")
        self.base_sub.set(f"{b.country}  ·  {b.note}")
        if key in self.reports:
            self._show_report(self.reports[key])
            return
        self._clear_views(f"No analysis yet for {b.name}.")
        if len(store.known_passes(b)) < 4:
            self._clear_views(
                f"No radar passes downloaded for {b.name} yet.\n\nPress "
                f"'Update base' to download them (free, one to three minutes).")
            return
        months = self._months()
        self._submit(f"Analysing {b.name}…",
                     lambda: store.analyse_base(b, months, self._progress,
                                                self.cancel),
                     self._analysed)

    def _analysed(self, rep: BaseReport) -> None:
        self.reports[rep.key] = rep
        b = BY_KEY[rep.key]
        meta = store.load_meta(b)
        last = rep.passes[-1]
        pct = (f" · {last.in_port / rep.usual_in_port:.0%}"
               if rep.usual_in_port else "")
        meta["summary"] = f"{last.in_port} ships{pct}"
        store.save_meta(b, meta)
        self._update_row(rep.key)
        self.status.set(rep.headline)
        if rep.key == self.current:
            self._show_report(rep)

    def _update_current(self) -> None:
        if not self.current:
            messagebox.showinfo("Pick a base", "Select a base in the list.")
            return
        self._update_keys([self.current])

    def _update_checked(self) -> None:
        keys = [b.key for b in BASES if b.key in self.checked]
        if not keys:
            messagebox.showinfo("Nothing ticked",
                                "Tick bases in the Compare column first.")
            return
        self._update_keys(keys)

    def _update_keys(self, keys: list[str]) -> None:
        if not radar.have_credentials():
            self._settings()
            return
        months = self._months()

        def job():
            out = []
            for k, key in enumerate(keys):
                b = BY_KEY[key]
                store.update(b, months,
                             lambda f, m, k=k: self._progress(
                                 (k + 0.8 * f) / len(keys), m), self.cancel)
                rep = store.analyse_base(
                    b, months, lambda f, m, k=k: self._progress(
                        (k + 0.8 + 0.2 * f) / len(keys), m), self.cancel)
                self.ui.put(("call", (lambda r=rep: self._analysed(r)), None))
                out.append(rep)
            return out

        def done(reps):
            self.status.set(f"Updated {len(reps)} base(s). "
                            + (reps[-1].headline if reps else ""))
            if len(keys) > 1:
                self._compare()

        self._submit(f"Updating {len(keys)} base(s)…", job, done)

    # ================================================================ views
    def _clear_views(self, msg: str) -> None:
        for fig, canvas in ((self.fig, self.canvas), (self.bfig, self.bcanvas),
                            (self.rfig, self.rcanvas)):
            charts.blank(fig, msg)
            canvas.draw_idle()
        for key, (v, _c, d) in self.kpi.items():
            v.set("–")
            d.set("")
        self.recent.configure(state="normal")
        self.recent.delete("1.0", "end")
        self.recent.configure(state="disabled")
        self.ev_tree.delete(*self.ev_tree.get_children())
        self.detail.delete("1.0", "end")
        self.notes_var.set("")
        self.berth_note.set("")

    def _show_report(self, rep: BaseReport) -> None:
        self._fill_kpis(rep)
        self._pick_map = charts.draw_activity(self.fig, rep,
                                              self._filtered(rep),
                                              self.selected_event)
        self.canvas.draw_idle()
        self._fill_recent(rep)
        self.notes_var.set("  ·  ".join(rep.notes))
        charts.draw_berths(self.bfig, rep)
        self.bcanvas.draw_idle()
        st = rep.static_hulls
        self.berth_note.set(
            f"{len(rep.berths)} berths. "
            + (f"{len(st)} ship-shaped object(s) bright on every pass "
               f"({', '.join(f'{h.length_m:.0f} m' for h in st)}) are "
               f"counted as present throughout; they do not appear in this "
               f"grid because they never changed." if st else "")
            + " A berth is a place beside a pier or quay that was empty on "
              "some passes and held a hull on others.")
        self._fill_events()
        self._fill_carriers(rep)
        self.view_pass = len(rep.passes) - 1
        self._draw_radar()

    def _fill_kpis(self, rep: BaseReport) -> None:
        P = rep.passes
        last = P[-1]
        prev = next((p for p in reversed(P[:-1]) if p.good), None)
        usual = rep.usual_in_port
        lo, hi = normal_range(rep)
        k = self.kpi
        k["now"][0].set(f"{last.in_port}")
        k["now"][2].set(f"on {last.when:%d %b %Y}"
                        + (f" · {last.static_n} present throughout"
                           if last.static_n else ""))
        k["usual"][0].set(f"{usual:.0f}")
        k["usual"][2].set(f"normal range {lo:.0f}–{hi:.0f}")
        if prev is not None:
            d = last.in_port - prev.in_port
            k["delta"][0].set(f"{d:+d}")
            k["delta"][2].set(f"since {prev.when:%d %b} "
                              f"({(last.when - prev.when).days} days)")
        else:
            k["delta"][0].set("–")
            k["delta"][2].set("")
        k["carriers"][0].set(f"{last.carriers}")
        k["carriers"][2].set("hulls both long and wide")
        k["metres"][0].set(f"{last.occupied_m / 1000:.1f} km")
        k["metres"][2].set("measured, steadier than a count")
        q = ("good" if last.good else
             "rough water" if last.rough else
             "bad image" if last.poor else "partial view")
        k["quality"][0].set(q)
        k["quality"][2].set(f"{last.when:%d %b %H:%M} UTC · view "
                            f"{last.p.geometry}")

    def _fill_recent(self, rep: BaseReport) -> None:
        t = self.recent
        t.configure(state="normal")
        t.delete("1.0", "end")
        evs = [e for e in rep.events if e.severity >= 2 and e.kind != "visit"]
        evs.sort(key=lambda e: (e.t1, e.severity), reverse=True)
        self._recent_map = {}
        if not evs:
            t.insert("end", "Nothing notable in this period.\n", "d")
        for k, e in enumerate(evs[:12]):
            tag = f"ev{k}"
            self._recent_map[tag] = e
            t.insert("end", f"{e.t1:%d %b}  ", "d")
            t.insert("end", e.title + "\n", ("h", "link", tag)
                     + (("sev3",) if e.severity >= 3 else ()))
            t.insert("end", f"{charts.kind_label(e.kind)} · "
                            f"confidence {e.confidence}\n\n", "d")
        t.configure(state="disabled")

    def _recent_click(self, ev) -> None:
        idx = self.recent.index(f"@{ev.x},{ev.y}")
        tags = self.recent.tag_names(idx)
        e = next((self._recent_map[t] for t in tags
                  if t in getattr(self, "_recent_map", {})), None)
        if e is not None:
            self._select_event(e)
            self._select_view(2)

    # --------------------------------------------------------------- events
    def _filtered(self, rep: BaseReport) -> list[Event]:
        sev_ok = {s for s, v in self.sev_vars.items() if v.get()}
        kf = self.kind_filter.get()
        groups = {
            "all": None,
            "arrivals and departures": ("arrived", "left", "left?",
                                        "part-left", "part-arrived"),
            "groups and level changes": charts.HEADLINE_KINDS,
            "long stays": ("long-stay",),
            "single sightings": ("visit",),
        }[kf]
        out = []
        for e in rep.events:
            if max(0, min(3, e.severity)) not in sev_ok:
                continue
            if groups is not None and e.kind not in groups:
                continue
            out.append(e)
        return out

    def _fill_events(self) -> None:
        self.ev_tree.delete(*self.ev_tree.get_children())
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        self._ev_ids = {}
        evs = self._filtered(rep)
        for k, e in enumerate(sorted(evs, key=lambda e: e.t1, reverse=True)):
            iid = f"e{k}"
            when = (f"{e.t1:%d %b %Y}" if e.t0 == e.t1 else
                    f"{e.t0:%d %b} – {e.t1:%d %b %Y}")
            sev = max(0, min(3, e.severity))
            self.ev_tree.insert("", "end", iid=iid, values=(
                when, SEVERITY_LABEL[sev], e.title, e.confidence),
                tags=(f"sev{sev}",))
            self._ev_ids[iid] = e
        self._pick_map = charts.draw_activity(self.fig, rep, evs,
                                              self.selected_event)
        self.canvas.draw_idle()

    def _select_event(self, e: Event) -> None:
        self.selected_event = e
        for iid, ee in getattr(self, "_ev_ids", {}).items():
            if ee is e:
                self.ev_tree.selection_set(iid)
                self.ev_tree.see(iid)
                break
        self._show_detail(e)
        rep = self.reports.get(self.current or "")
        if rep is not None:
            self._pick_map = charts.draw_activity(self.fig, rep,
                                                  self._filtered(rep), e)
            self.canvas.draw_idle()

    def _on_event_select(self, _ev=None) -> None:
        sel = self.ev_tree.selection()
        if not sel:
            return
        e = self._ev_ids.get(sel[0])
        if e is not None and e is not self.selected_event:
            self._select_event(e)

    def _on_pick(self, ev) -> None:
        e = getattr(self, "_pick_map", {}).get(ev.artist)
        if e is not None:
            self._select_event(e)
            self._select_view(2)

    def _show_detail(self, e: Event) -> None:
        rep = self.reports.get(self.current or "")
        d = self.detail
        d.delete("1.0", "end")
        d.insert("end", e.title + "\n\n", "title")

        def field(name, text, tag="body"):
            d.insert("end", name.upper() + "\n", "field")
            d.insert("end", text + "\n\n", tag)

        when = (f"{e.t1:%d %B %Y}" if e.t0 == e.t1 else
                f"between the passes of {e.t0:%d %B} and {e.t1:%d %B %Y}")
        field("What", e.text)
        field("When", when + ". Radar passes are 6–12 days apart, so the "
                      "exact day is not known.")
        if e.lon is not None:
            field("Where", f"{abs(e.lat):.4f}°{'N' if e.lat >= 0 else 'S'}, "
                           f"{abs(e.lon):.4f}°{'E' if e.lon >= 0 else 'W'}"
                           + (f" · berth {e.berth}" if e.berth else
                              " · open water"))
        if e.length_m:
            what = interpret(e.length_m, e.width_m or 40.0)
            field("Size reading", f"{e.length_m:.0f} m long"
                  + (f", {e.width_m:.0f} m wide" if e.width_m else "")
                  + f" on radar (±{0.12 * e.length_m + 15:.0f} m). "
                  + what["text"][:1].upper() + what["text"][1:] + ".")
        if rep is not None and e.before >= 0 and e.after >= 0:
            field("Evidence", f"The pass of {rep.passes[e.before].when:%d %b}"
                              f" shows the 'before' state and "
                              f"{rep.passes[e.after].when:%d %b} the 'after' "
                              f"state. Press 'Show in radar view' to see "
                              f"both with the hull outlined.")
        tag = {"high": "green", "medium": "amber", "low": "red"}.get(
            e.confidence, "body")
        field("Confidence", f"{e.confidence}" + (f" — {e.why}" if e.why
                                                 else ""), tag)

    # ---------------------------------------------------------------- berths
    def _on_berth_click(self, ev) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None or ev.inaxes is None or ev.xdata is None:
            return
        j = int(round(ev.xdata))
        if 0 <= j < len(rep.passes):
            self.selected_event = None
            self.view_pass = j
            self._draw_radar()
            self._select_view(3)

    # ----------------------------------------------------------------- radar
    def _show_event_radar(self) -> None:
        if self.selected_event is None:
            messagebox.showinfo("Pick an event", "Select an event first.")
            return
        self._select_view(3)
        self._draw_radar()

    def _step_pass(self, d: int) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        self.selected_event = None
        self.view_pass = int(np.clip(self.view_pass + d, 0,
                                     len(rep.passes) - 1))
        self._draw_radar()

    def _photo(self, rep: BaseReport):
        if rep.key in self._photos:
            return self._photos[rep.key]
        try:
            from .basemap import fetch_basemap
            from .radar import to_lonlat
            H, W = rep.shape
            a, _b, c0, _d, e, f0 = rep.transform[:6]
            f = to_lonlat(rep.crs)
            lon0, lat0 = f(c0, f0 + e * H)
            lon1, lat1 = f(c0 + a * W, f0)
            bm = fetch_basemap((lon0, lat0, lon1, lat1), "satellite", 1200)
        except Exception:
            bm = None
        self._photos[rep.key] = bm
        return bm

    def _draw_radar(self) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        e = self.selected_event
        if e is not None and e.before >= 0 and e.after >= 0:
            panels = [(e.before, f"before · {rep.passes[e.before].when:%d %b %Y}"
                                 f" · {rep.passes[e.before].in_port} ships"),
                      (e.after, f"after · {rep.passes[e.after].when:%d %b %Y}"
                                f" · {rep.passes[e.after].in_port} ships")]
            self.radar_title.set(f"{e.title}  ({e.t0:%d %b} → {e.t1:%d %b})")
        else:
            i = self.view_pass if self.view_pass >= 0 else len(rep.passes) - 1
            p = rep.passes[i]
            panels = [(i, f"{p.when:%d %b %Y %H:%M} UTC · view "
                          f"{p.p.geometry} · {p.in_port} ships"
                          + (" · rough water" if p.rough else ""))]
            self.radar_title.set(f"Pass {i + 1} of {len(rep.passes)}")
        charts.draw_radar(self.rfig, rep, panels,
                          photo=self._photo(rep) if self.opt_photo.get()
                          else None,
                          event=e, show_berths=self.opt_berths.get(),
                          show_structure=self.opt_struct.get(),
                          show_change=self.opt_change.get())
        self.rcanvas.draw_idle()

    # --------------------------------------------------------------- compare
    def _compare(self) -> None:
        keys = [b.key for b in BASES if b.key in self.checked]
        missing = [k for k in keys if k not in self.reports
                   and len(store.known_passes(BY_KEY[k])) >= 4]
        months = self._months()

        def job():
            for n, k in enumerate(missing):
                rep = store.analyse_base(
                    BY_KEY[k], months, lambda f, m, n=n: self._progress(
                        (n + f) / max(len(missing), 1), m), self.cancel)
                self.ui.put(("call", (lambda r=rep: self._analysed(r)), None))
            return keys

        self._select_view(4)
        if missing:
            self._submit("Analysing ticked bases…", job,
                         lambda _k: self._draw_compare(keys))
        else:
            self._draw_compare(keys)

    def _draw_compare(self, keys: list[str]) -> None:
        reps = [self.reports[k] for k in keys if k in self.reports]
        charts.draw_compare(self.cfig, reps)
        self.ccanvas.draw_idle()
        self.ctext.delete("1.0", "end")
        if not reps:
            self.ctext.insert("1.0", "Tick bases in the Compare column, then "
                                     "press Refresh. Bases without radar "
                                     "passes on disk need 'Update ticked' "
                                     "first.")
            return
        lines = compare(reps)
        self.ctext.insert("1.0", "Red lines: group departures, level falls "
                                 "or unusually empty periods. Green: the "
                                 "opposite.\n\n"
                          + ("\n\n".join(lines) if lines else
                             "Nothing happened at several bases in the same "
                             "week."))

    # ------------------------------------------------------------------ misc
    def _save_report(self) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{rep.key}_{datetime.now(timezone.utc):%Y%m%d}.txt")
        if not path:
            return
        try:
            write_report(path, rep)
        except OSError as exc:
            messagebox.showerror("Could not save the report",
                                 f"{path}\n\n{exc}")
            return
        self.status.set(f"Saved {path}")

    def _refresh_key_status(self) -> None:
        self.key_status.set("Copernicus key: "
                            + ("set" if radar.have_credentials()
                               else "not set — see Settings"))

    def _settings(self) -> None:
        d = tk.Toplevel(self)
        d.title("Settings")
        d.transient(self)
        d.configure(background=BG)
        fr = ttk.Frame(d, padding=16)
        fr.pack(fill="both", expand=True)
        ttk.Label(fr, text="Copernicus Data Space (free)",
                  style="Title.TLabel").grid(row=0, column=0, columnspan=2,
                                             sticky="w")
        ttk.Label(fr, wraplength=480, style="Muted.TLabel", text=(
            "Register at dataspace.copernicus.eu, open Sentinel Hub → User "
            "settings → OAuth clients, create a client, and paste its id and "
            "secret here. Stored only on this computer, in your user "
            "profile.")).grid(row=1, column=0, columnspan=2, sticky="w",
                          pady=(4, 10))
        cid = tk.StringVar()
        sec = tk.StringVar()
        ttk.Label(fr, text="Client id").grid(row=2, column=0, sticky="w")
        ttk.Entry(fr, textvariable=cid, width=50).grid(row=2, column=1,
                                                       pady=2)
        ttk.Label(fr, text="Client secret").grid(row=3, column=0, sticky="w")
        ttk.Entry(fr, textvariable=sec, width=50, show="•").grid(row=3,
                                                                 column=1,
                                                                 pady=2)
        cur = tk.StringVar()

        def refresh():
            cur.set(f"Currently: id "
                    f"{credentials.masked('copernicus', 'client_id')}, secret "
                    f"{credentials.masked('copernicus', 'client_secret')}")
        refresh()
        ttk.Label(fr, textvariable=cur, style="Muted.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=6)
        msg = tk.StringVar()

        def save():
            fields = {}
            if cid.get().strip():
                fields["client_id"] = cid.get().strip()
            if sec.get().strip():
                fields["client_secret"] = sec.get().strip()
            if fields:
                try:
                    credentials.set_credentials("copernicus", **fields)
                except OSError as exc:
                    msg.set(str(exc))
                    return
                radar._TOKEN["value"] = None
            refresh()
            self._refresh_key_status()
            msg.set(f"Saved to {credentials.CREDENTIALS_PATH}.")

        def test():
            try:
                radar.get_token(force=True)
                msg.set("Connected to Copernicus: the key works.")
            except Exception as exc:
                msg.set(str(exc))

        bf = ttk.Frame(fr)
        bf.grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Button(bf, text="Save", style="Accent.TButton",
                   command=save).pack(side="left")
        ttk.Button(bf, text="Test connection", command=test).pack(
            side="left", padx=6)
        ttk.Button(bf, text="Close", command=d.destroy).pack(side="left")
        ttk.Label(fr, textvariable=msg, wraplength=480).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=6)


# ==========================================================================
# text report
# ==========================================================================
def write_report(path, rep: BaseReport) -> None:
    """Save the report as UTF-8.

    Explicit, because Windows otherwise writes in the locale's code page
    (cp1252), which cannot encode the arrows in the text: the file was
    created and the write then failed, leaving it empty.  The text is built
    before the file is opened so a failure never leaves a half-written file.
    """
    text = report_text(rep)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def report_text(rep: BaseReport) -> str:
    out = [rep.name, "=" * len(rep.name), rep.headline, ""]
    out += ["Notes:"] + [f"  - {n}" for n in rep.notes] + [""]
    if rep.regimes:
        out.append("Level periods:")
        for r in rep.regimes:
            out.append(f"  {rep.passes[r['start']].when:%Y-%m-%d} to "
                       f"{rep.passes[r['end']].when:%Y-%m-%d}: about "
                       f"{r['level']:.0f} large ships ({r['n']} passes)")
        out.append("")
    out.append("Events, newest first:")
    for e in sorted(rep.events, key=lambda e: e.t1, reverse=True):
        if e.kind == "visit":
            continue
        out.append(f"\n[{e.t0:%Y-%m-%d} -> {e.t1:%Y-%m-%d}] {e.title}  "
                   f"(confidence {e.confidence})")
        out.append(f"  {e.text}")
        if e.why:
            out.append(f"  Why: {e.why}")
    out.append("\nPer pass:")
    for p in rep.passes:
        out.append(f"  {p.when:%Y-%m-%d %H:%M}Z  view {p.p.geometry:8s} "
                   f"{p.in_port:3d} large ships, {p.carriers} carrier(s), "
                   f"{p.occupied_m / 1000:.1f} km of hull"
                   + ("  rough water" if p.rough else ""))
    return "\n".join(out) + "\n"


METHOD = """# What this program does

It watches naval bases with Sentinel-1, a free European radar satellite. Every 6 to 12 days the satellite passes over a base at the same time of day, from the same angle, and takes a radar picture. Radar sees through cloud and works at night, so no pass is lost to weather.

# How ships are found

On radar, calm water is dark and a steel hull is very bright, but water beside a pier is never dark: the pier's own echo lifts it by 5 to 12 dB even when the berth is empty. So nothing is compared with open water. Instead the program learns, for every 10 m pixel, the emptiest it has ever looked across all the passes (the 3rd percentile), after removing each viewing angle's overall brightness offset and each pass's overall gain. A ship is a stretch of pixels that is at least 6 dB brighter than its own emptiest state, with a core at least 10 dB brighter and a peak of at least 18 dB over open water. Neighbouring ships are kept apart by giving the faint halo around each bright core to the nearest core, so two ships at one pier do not merge into one.

Every one of those numbers was set against PlanetScope photographs of Norfolk taken within hours of a radar pass (four clear same-day pairs, 60 ships): hull recall 0.65 at precision 0.95, and a correlation of 0.73 between radar and photo berth occupancy. The first version, which compared pixels with open water, scored 0.21 / 0.41 / 0.10 on the same photos.

# What a berth is

A berth is a place beside a pier or quay that was empty on some passes and held a hull on others. Piers, quays, cranes and land are "structure": bright even at their emptiest, and never counted as ships. A ship that never left during the whole record is bright at its emptiest too, so to this method it looks like structure. Where such a stretch is as bright as a hull (18 dB and up; piers are 9 to 15) and shaped like one, it is reported as "present throughout" and counted as a ship, with the caveat that a crane or shed can look the same.

# How sizes are read

Radar measures length to about 10-15%, so the program never claims an exact class. It uses size and shape: a carrier is long AND wide (its flight deck is ~75 m across); two destroyers moored bow to stern are long but narrow; two ships side by side are short but wide. Every explanation says which reading it chose and why.

# How anomalies are found

Level periods: the series of ships-in-port is cut into periods of steady level by exact optimal partitioning (every possible set of cuts is scored; the cost is each period's spread around its median, with a penalty per cut scaled to the series' own noise). A fall from 24 to 15 that holds for twelve passes is a level shift; one odd pass is not. Periods outside the normal range (the usual level ± 2.5 robust standard deviations) are flagged as unusually empty or full.

Group movements: more berths emptying between two passes than this base normally sees. A carrier-sized hull leaving with two or more escorts in the same interval is reported as a carrier group sailing.

Long stays: a ship at a berth for twice the typical stay seen at that same berth.

Confidence: every event says what it rests on. A departure is "high" when the ship was seen on two or more passes and the berth then imaged empty on two; "low" when only one pass has shown it empty. An arrival seen once is "medium" until the next pass.

# Experimental sea search

After a carrier-sized departure, the sea search inspects available radar products around the base within its time, area and request limits. It fetches 60 km tiles at 50 m output spacing and compares bright targets with a local sea background. Candidates at least 200 m long may be re-measured at 10 m output spacing from the same acquisition. A successful re-measurement is a size check, not an independent observation or identification: the underlying IW GRD resolution is about 20 by 22 m.

Large, wide targets are only size candidates. The current association rule accepts the first size candidate with compatible timing, then checks later candidates against it; it does not match dimensions to the original departure. Similar dimensions, nearby smaller targets and compatible timing cannot establish that two detections are the same vessel or that nearby vessels are escorts. Dashed map links show tentative timing associations, not a verified route. The search has sparse coverage, especially offshore; missing data, request limits and unavailable checks are reported. White map boxes bound usable pixels in fetched tiles and may contain gaps. The berth accuracy figures above do not validate this experimental search.

# Where a ship can float

Radar cannot tell a moored ship from a stack of shipping containers. Measured over a year of passes at two ports, the two are indistinguishable on every radar feature the detector has: median brightness 13.2 vs 13.4 dB, emptiest state 1.4 vs 0.7 dB, swing 18.0 vs 18.7 dB. Both are metal boxes on a flat surface. Elevation models cannot settle it either, because reclaimed container terminals read as 0 m in every free model and are mapped as open sea.

So Keelix asks a land-cover map instead: ESA WorldCover, free and 10 m, with an explicit permanent-water class. A detection must overlap mapped water to count. This keeps 100% of the real ships at the validated location and removes 63% of a container yard's blocks. WorldCover is a 2021 product, so land reclaimed after that still reads as water, and the notes say when no land-cover map could be fetched.

# What it cannot do

It cannot name a ship or tell close classes apart. Ships moored side by side or end to end merge into one hull; the ship count is an estimate (metres of hull alongside is the steadier number). A ship that arrives and leaves between two passes is never seen. Arrival and departure dates are only known to the gap between passes. Submarines sit low in the water and echo weakly. Outside the US the free elevation data is coarser, so the land mask is more forgiving and the depth check may say it cannot trace the exit route.

# Data

Radar: Copernicus Sentinel-1, free (needs a free Copernicus account - Settings). Land and depth: NOAA global elevation mosaic, free, no account. The satellite photo under the radar view is Esri World Imagery, for orientation only; it is not from the same date.
"""


# ==========================================================================
# first-run tour
# ==========================================================================
UI_STATE = credentials.CONFIG_DIR / "ui.json"

TOUR_STEPS = [
    ("Welcome", None,
     "This program watches naval bases with free Sentinel-1 radar. Every "
     "6-12 days the satellite photographs each base from the same angle; "
     "steel hulls are bright, water is dark, so the program can count the "
     "large ships in port, follow each berth, and explain what changed.\n\n"
     "This tour takes a minute and walks through every view. You can replay "
     "it any time with the Tour button at the bottom of the sidebar."),
    ("1 · Your free key", None,
     "Radar comes from the Copernicus Data Space. It is free: register at "
     "dataspace.copernicus.eu, open Sentinel Hub → User settings → OAuth "
     "clients, create one, and paste its id and secret into Settings (top "
     "right). The key is stored only on this computer.\n\n"
     "Without it the program can still open bases that are already on disk."),
    ("2 · Pick a base", None,
     "The sidebar lists 45 bases grouped by region; the box above filters "
     "them. Click a base to open it. The first time, press Update base: "
     "twelve months of radar takes one to three minutes, and after that the "
     "base opens instantly. The Latest column shows ships in port and how "
     "that compares with the base's usual level.\n\n"
     "The CMP column ticks a base for the Compare view and for 'Update "
     "ticked'. Tick region ticks every base in the selected region."),
    ("3 · Overview", 0,
     "Six numbers first: ships in port now, the usual level, the change "
     "since the last pass, carriers, metres of hull alongside, and the "
     "quality of the latest pass.\n\n"
     "The chart shows every radar pass. The band is the normal range, the "
     "black line the trend, red triangles departures, green arrivals; marker "
     "size follows the ship. Shaded periods are unusually empty or full. "
     "Click any marker to read what happened. On the right, the latest "
     "notable changes; click one to open it."),
    ("4 · Berths", 1,
     "The raw material. Each row is a berth beside a pier or quay, each "
     "column a radar pass, and the colour is how much of the berth held a "
     "hull. Amber columns are passes with rough water or a partial view. "
     "Click a cell to open that pass in the Radar view.\n\n"
     "Ships that never moved during the whole period are listed under the "
     "grid as 'present throughout'."),
    ("5 · Events", 2,
     "Every arrival, departure, group movement, level shift and long stay, "
     "newest first. Filter by level (Headlines down to Minor) and by type. "
     "Each event opens a structured explanation: what, when, where, the "
     "size reading, the evidence passes, and a confidence with the reason "
     "for it.\n\n"
     "Press 'Show in radar view' to see the before and after images with "
     "the hull outlined. 'Save report' writes everything as text."),
    ("6 · Radar", 3,
     "The image behind any claim. Step through passes, or arrive here from "
     "an event to see before and after side by side. Red outlines are hulls "
     "found on that pass; the green one is the hull the event refers to; "
     "dashed orange ones were present on every pass.\n\n"
     "'Change vs empty state' shows what the detector actually looks at: "
     "each pixel against its own emptiest state. The toolbar zooms and pans."),
    ("7 · Compare", 4,
     "Ticked bases on one time axis, each as a percentage of its own usual "
     "level, so a small base and Norfolk can be read together. Red lines "
     "mark group departures, level falls and unusually empty periods; green "
     "the opposite. Below the chart, weeks in which several bases changed "
     "together are called out."),
    ("8 · Sea search", 5,
     "Pick a carrier-sized departure to inspect available radar strips "
     "around the base. Large, wide targets are size candidates; their "
     "identity and any relationship to nearby vessels remain unverified.\n\n"
     "Dashed links are tentative timing associations, not a verified "
     "route. White boxes bound usable pixels and may contain gaps. Sparse "
     "coverage and request limits can leave candidates unchecked. A search "
     "usually takes 10-40 minutes and can be cancelled."),
    ("9 · What to trust", 6,
     "Measured against photographs of Norfolk: about two thirds of ships "
     "are found, nine in ten detections are real, and ship counts per pass "
     "match the photos to within one or two. The method cannot name ships, "
     "cannot tell close classes apart, merges ships moored together, and "
     "only dates events to the gap between passes.\n\n"
     "The Method view has the full account. Every event says what its "
     "confidence rests on; read that line before acting on it."),
]


def _ui_state() -> dict:
    try:
        return json.loads(UI_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ui_state(d: dict) -> None:
    try:
        UI_STATE.parent.mkdir(parents=True, exist_ok=True)
        UI_STATE.write_text(json.dumps(d), encoding="utf-8")
    except OSError:
        pass


class Tour(tk.Toplevel):
    """A small guided walk through every view; switches the view as it goes."""

    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app
        self.step = 0
        self.title("Tour")
        self.configure(background=PANEL)
        self.resizable(False, False)
        if app.winfo_viewable():
            self.transient(app)
        fr = ttk.Frame(self, style="Panel.TFrame", padding=(22, 18))
        fr.pack(fill="both", expand=True)
        self.v_step = tk.StringVar()
        self.v_title = tk.StringVar()
        ttk.Label(fr, textvariable=self.v_step, style="Caps.TLabel").pack(
            anchor="w")
        ttk.Label(fr, textvariable=self.v_title, style="Panel.TLabel",
                  font=app.fonts["title"]).pack(anchor="w", pady=(4, 10))
        self.body = tk.Text(fr, wrap="word", relief="flat", background=PANEL,
                            foreground=INK, insertbackground=INK,
                            selectbackground="#d6e4f5", selectforeground=INK,
                            font=app.fonts["body"], width=58, height=12,
                            bd=0, highlightthickness=0)
        self.body.pack(fill="both", expand=True)
        row = ttk.Frame(fr, style="Panel.TFrame")
        row.pack(fill="x", pady=(14, 0))
        self.b_back = ttk.Button(row, text="Back", command=self.back)
        self.b_back.pack(side="left")
        ttk.Button(row, text="Skip tour", command=self.finish).pack(
            side="left", padx=8)
        self.b_next = ttk.Button(row, text="Next", style="Accent.TButton",
                                 command=self.next)
        self.b_next.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.finish)
        self.bind("<Return>", lambda e: self.next())
        self.bind("<Escape>", lambda e: self.finish())
        self.show()
        if app.winfo_viewable():
            self.after(50, self._place)

    def _place(self) -> None:
        self.update_idletasks()
        ax, ay = self.app.winfo_rootx(), self.app.winfo_rooty()
        aw, ah = self.app.winfo_width(), self.app.winfo_height()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{ax + aw - w - 40}+{ay + ah - h - 60}")
        self.lift()

    def show(self) -> None:
        title, view, text = TOUR_STEPS[self.step]
        self.v_step.set(f"TOUR · {self.step + 1} OF {len(TOUR_STEPS)}")
        self.v_title.set(title)
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")
        if view is not None:
            self.app._select_view(view)
        self.b_back.configure(state="normal" if self.step else "disabled")
        self.b_next.configure(text="Done" if self.step == len(TOUR_STEPS) - 1
                              else "Next")

    def next(self) -> None:
        if self.step >= len(TOUR_STEPS) - 1:
            self.finish()
            return
        self.step += 1
        self.show()

    def back(self) -> None:
        if self.step:
            self.step -= 1
            self.show()

    def finish(self) -> None:
        d = _ui_state()
        d["tour_done"] = True
        _save_ui_state(d)
        self.app._select_view(0)
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
