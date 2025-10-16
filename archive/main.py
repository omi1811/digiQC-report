import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import TclError


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def default_candidates(*names: str) -> list[str]:
    return [os.path.join(SCRIPT_DIR, n) for n in names]


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("digiQC Reports Launcher")
        self.geometry("760x520")

        # State
        self.eqc_file = tk.StringVar(value=self._find_first_existing(default_candidates("Combined_EQC.csv", "Combined_EQC.scv")))
        self.ins_file = tk.StringVar(value=self._find_first_existing(default_candidates("Combined_Instructions.csv")))
        self.mode = tk.StringVar(value="weekly")
        self.action = tk.StringVar(value="Dashboard")
        self.output_dir = tk.StringVar(value=os.path.join(SCRIPT_DIR, "Activity-wise_instructions"))
        self.update_history = tk.BooleanVar(value=False)

        # UI
        self._build_ui()

        # Process
        self.proc = None

    def _find_first_existing(self, paths: list[str]) -> str:
        for p in paths:
            if p and os.path.exists(p):
                return p
        return paths[0] if paths else ""

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # Files frame
        files = ttk.LabelFrame(self, text="Inputs")
        files.pack(fill=tk.X, **pad)

        # EQC file row
        ttk.Label(files, text="Combined_EQC file:").grid(row=0, column=0, sticky=tk.W, **pad)
        ttk.Entry(files, textvariable=self.eqc_file, width=70).grid(row=0, column=1, sticky=tk.W, **pad)
        ttk.Button(files, text="Browse", command=self._pick_eqc).grid(row=0, column=2, **pad)

        # Instructions file row
        ttk.Label(files, text="Combined_Instructions file:").grid(row=1, column=0, sticky=tk.W, **pad)
        ttk.Entry(files, textvariable=self.ins_file, width=70).grid(row=1, column=1, sticky=tk.W, **pad)
        ttk.Button(files, text="Browse", command=self._pick_ins).grid(row=1, column=2, **pad)

        # Output dir (for Activity-wise)
        ttk.Label(files, text="Output dir (Activity-wise):").grid(row=2, column=0, sticky=tk.W, **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=70).grid(row=2, column=1, sticky=tk.W, **pad)
        ttk.Button(files, text="Browse", command=self._pick_out).grid(row=2, column=2, **pad)

        # Actions frame
        actions = ttk.LabelFrame(self, text="Action")
        actions.pack(fill=tk.X, **pad)

        ttk.Label(actions, text="Select action:").grid(row=0, column=0, sticky=tk.W, **pad)
        action_combo = ttk.Combobox(actions, textvariable=self.action, state="readonly", values=[
            "Dashboard",
            "EQC Report",
            "Instructions Activity-wise",
            "Open EQC GUI",
            "Run Futura_main",
            "Run Futura_modified",
        ])
        action_combo.grid(row=0, column=1, sticky=tk.W, **pad)
        action_combo.bind("<<ComboboxSelected>>", lambda e: self._update_mode_visibility())

        ttk.Label(actions, text="Mode:").grid(row=0, column=2, sticky=tk.E, **pad)
        self.mode_combo = ttk.Combobox(actions, textvariable=self.mode, state="readonly", values=["weekly", "monthly", "cumulative"])
        self.mode_combo.grid(row=0, column=3, sticky=tk.W, **pad)

        # Update history checkbox (only for Instructions Activity-wise)
        self.update_history_chk = ttk.Checkbutton(actions, text="Update history (append)", variable=self.update_history)
        self.update_history_chk.grid(row=1, column=1, sticky=tk.W, **pad)

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, **pad)
        ttk.Button(btns, text="Run", command=self._run).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Stop", command=self._stop).pack(side=tk.LEFT, padx=8)

        # Output
        self.output = tk.Text(self, height=18)
        self.output.pack(fill=tk.BOTH, expand=True, **pad)

        self._update_mode_visibility()

    def _pick_eqc(self):
        p = filedialog.askopenfilename(initialdir=SCRIPT_DIR, title="Select Combined_EQC file",
                                       filetypes=[("CSV/TSV", "*.csv *.scv *.tsv"), ("All", "*.*")])
        if p:
            self.eqc_file.set(p)

    def _pick_ins(self):
        p = filedialog.askopenfilename(initialdir=SCRIPT_DIR, title="Select Combined_Instructions file",
                                       filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.ins_file.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory(initialdir=SCRIPT_DIR, title="Select output directory")
        if p:
            self.output_dir.set(p)

    def _update_mode_visibility(self):
        act = self.action.get()
        # Mode is relevant for EQC Report and Instructions Activity-wise
        if act in ("EQC Report", "Instructions Activity-wise"):
            self.mode_combo.configure(state="readonly")
        else:
            self.mode_combo.configure(state="disabled")
        # Update-history checkbox enabled only for Instructions Activity-wise
        if act == "Instructions Activity-wise":
            self.update_history_chk.state(["!disabled"])
        else:
            self.update_history_chk.state(["disabled"])

    def _append(self, text: str):
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def _run(self):
        if self.proc is not None:
            self._append("A process is already running. Stop it first.\n")
            return
        act = self.action.get()
        cmd = None
        if act == "Dashboard":
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "build_dashboard.py"), "--eqc", self.eqc_file.get(), "--issues", self.ins_file.get()]
        elif act == "EQC Report":
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "Weekly_report.py"), "-i", self.eqc_file.get(), "-m", self.mode.get()]
        elif act == "Instructions Activity-wise":
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "activity_instructions_report.py"), "-i", self.ins_file.get(), "-m", self.mode.get(), "-o", self.output_dir.get()]
            if self.update_history.get():
                cmd.append("--update-history")
        elif act == "Open EQC GUI":
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "report_gui.py")]
        elif act == "Run Futura_main":
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "Futura_main.py")]
        elif act == "Run Futura_modified":
            # Pass the EQC combined file if provided; script defaults to Combined_EQC.csv otherwise
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "Futura_modified.py"), "-i", self.eqc_file.get()]
        else:
            self._append("Unknown action selected.\n")
            return

        self._append(f"$ {' '.join(cmd)}\n")
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            self._append(f"Failed to start process: {e}\n")
            self.proc = None
            return

        threading.Thread(target=self._pump_output, daemon=True).start()

    def _pump_output(self):
        assert self.proc is not None
        for line in self.proc.stdout:
            self._append(line)
        code = self.proc.wait()
        self._append(f"\nProcess exited with code {code}\n")
        self.proc = None

    def _stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self._append("Sent terminate signal.\n")
            except Exception as e:
                self._append(f"Failed to terminate: {e}\n")


def main():
    # Try GUI; if no display, fall back to CLI menu
    try:
        app = Launcher()
        app.mainloop()
        return
    except TclError:
        pass

    # Headless CLI fallback
    print("No display detected. Falling back to CLI mode.\n")
    eqc_default = os.path.join(SCRIPT_DIR, "Combined_EQC.csv")
    eqc_default_alt = os.path.join(SCRIPT_DIR, "Combined_EQC.scv")
    ins_default = os.path.join(SCRIPT_DIR, "Combined_Instructions.csv")
    out_default = os.path.join(SCRIPT_DIR, "Activity-wise_instructions")

    def inp(prompt: str, default: str | None = None) -> str:
        if default:
            s = input(f"{prompt} [{default}]: ").strip()
            return s or default
        return input(f"{prompt}: ").strip()

    def choose_mode() -> str:
        while True:
            s = inp("Mode (weekly/monthly/cumulative)", "weekly").lower()
            if s in ("weekly", "monthly", "cumulative"):
                return s
            print("Invalid mode.")

    def run_and_stream(cmd: list[str]) -> int:
        print("$ "+" ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
            return proc.wait()
        except Exception as e:
            print(f"Failed to start: {e}")
            return 1

    def resolve_path_input(prompt_label: str, default_path: str) -> str:
        raw = inp(prompt_label, default_path)
        # Build candidate tokens by splitting on common delimiters and stripping quotes
        pieces = []
        for sep in ("'", '"', " "):
            for tok in raw.split(sep):
                tok = tok.strip().strip("'\"")
                if tok:
                    pieces.append(tok)
        cleaned = raw.strip().strip("'\"")
        pieces.append(cleaned)
        # Deduplicate preserving order
        seen = set()
        cand = []
        for p in pieces:
            if p not in seen:
                seen.add(p)
                cand.append(p)
        for p in cand:
            if os.path.exists(p):
                return p
        return cleaned

    while True:
        print("Select action:")
        print("  1) Dashboard")
        print("  2) EQC Report")
        print("  3) Instructions Activity-wise")
        print("  4) Run Futura_main")
        print("  5) Run Futura_modified")
        print("  6) Quit")
        choice = input("Enter choice [1-6]: ").strip()
        if choice == "6":
            print("Exiting.")
            break
        elif choice == "1":
            eqc_path = resolve_path_input("Path to Combined_EQC", eqc_default if os.path.exists(eqc_default) else (eqc_default_alt if os.path.exists(eqc_default_alt) else eqc_default))
            ins_path = resolve_path_input("Path to Combined_Instructions", ins_default if os.path.exists(ins_default) else ins_default)
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "build_dashboard.py"), "--eqc", eqc_path, "--issues", ins_path]
            run_and_stream(cmd)
        elif choice == "2":
            eqc_path = resolve_path_input("Path to Combined_EQC", eqc_default if os.path.exists(eqc_default) else (eqc_default_alt if os.path.exists(eqc_default_alt) else eqc_default))
            mode = choose_mode()
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "Weekly_report.py"), "-i", eqc_path, "-m", mode]
            run_and_stream(cmd)
        elif choice == "3":
            ins_path = resolve_path_input("Path to Combined_Instructions", ins_default if os.path.exists(ins_default) else ins_default)
            mode = choose_mode()
            out_dir = inp("Output directory", out_default)
            upd = inp("Update history? (y/N)", "N").lower().startswith("y")
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "activity_instructions_report.py"), "-i", ins_path, "-m", mode, "-o", out_dir]
            if upd:
                cmd.append("--update-history")
            run_and_stream(cmd)
        elif choice == "4":
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "Futura_main.py")]
            run_and_stream(cmd)
        elif choice == "5":
            eqc_path = inp("Path to Combined_EQC", eqc_default if os.path.exists(eqc_default) else (eqc_default_alt if os.path.exists(eqc_default_alt) else eqc_default))
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "Futura_modified.py"), "-i", eqc_path]
            run_and_stream(cmd)
        else:
            print("Invalid choice.\n")


if __name__ == "__main__":
    main()
