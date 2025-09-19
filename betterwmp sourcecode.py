import os
if os.name != "nt":
   print("not a chance")
   sys.exit(1)
import sys
import time
import json
import ctypes
import shutil
import random
import base64
import traceback
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy.io import wavfile
from io import BytesIO
from PIL import Image, ImageTk 
from pydub.utils import which 
from playsound import playsound 
import sounddevice as sd 
from tkinterdnd2 import TkinterDnD, DND_FILES
import faulthandler

FAULT_LOG = os.path.expandvars(r"%localappdata%\betterwmpconf\fault.log")
CONF_DIR = os.path.expandvars(r"%localappdata%\betterwmpconf")
INSTALL_POINTER = os.path.join(CONF_DIR, "installpointer.conf")
fault_file = open(FAULT_LOG, "w")
faulthandler.enable(file=fault_file)
def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=fault_file)
    fault_file.flush()
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
sys.excepthook = excepthook
def show_native_messagebox(title, message):
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
def _read_install_dir():
    try:
        with open(INSTALL_POINTER, "r", encoding="utf-8") as f:
            p = f.read().strip().strip('"')
            return p if p else None
    except Exception as e:
        with open(FAULT_LOG, "a", encoding="utf-8") as lf:
            lf.write("Read failed:\n")
            traceback.print_exc(file=lf)
            lf.flush()
        return None
def _candidate_ffmpeg_paths():
    install_dir = _read_install_dir()
    if install_dir:
        yield os.path.join(install_dir, "ffmpeg.exe")
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        yield path_ffmpeg
def resolve_ffmpeg() -> str | None:
    for cand in _candidate_ffmpeg_paths():
        if cand and os.path.isfile(cand):
            return cand
    return None
SkinInfo = {}
ffmpeg_exe = resolve_ffmpeg()
from pydub import AudioSegment
AudioSegment.converter = ffmpeg_exe or which("ffmpeg")
if AudioSegment.converter is None:
    show_native_messagebox(
        "FFmpeg Not Found",
        "FFmpeg was not found. Non-WAV files will not open.\n"
        "Reinstall the newest version of BetterWMP to solve this issue."
    )
    with open(FAULT_LOG, "a", encoding="utf-8") as lf:
        lf.write("FFmpeg not found at startup.\n")
        lf.flush()
else:
    print(AudioSegment.converter)
    with open(FAULT_LOG, "a", encoding="utf-8") as lf:
        lf.write(f"Using FFmpeg at: {AudioSegment.converter}\n")
        lf.flush()
def tkinter_exception_handler(self, exc, val, tb):
    if EmergencyStop:
        raise exc.with_traceback(tb)
    print("Tkinter exception handler called!")
    print(f"Exception: {exc.__name__}: {val}")
    traceback.print_tb(tb)
    with open(FAULT_LOG, "a", encoding="utf-8") as lf:
        lf.write("Tkinter exception:\n")
        traceback.print_exception(exc, val, tb, file=lf)
        lf.flush()
def to_float32(audio):
    if audio.dtype == np.float32:
        return audio
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    audio = audio.astype(np.float32)
    maxv = max(1.0, np.max(np.abs(audio)))
    return audio / maxv

ICON = r"data:image/x-icon;base64,AAABAAEAICAAAAEAIACoEAAAFgAAACgAAAAgAAAAQAAAAAEAIAAAAAAAAAAAAMQOAADEDgAAAAAAAAAAAAD+/vX/nYxh/495Rv9gRQ7/rp90/7mogv/CsI3/w7KM/8W0kP/FtY7/xbSJ/8Gsfv/GsYL/x7GF/9nRt/+ab7T/nmux/6Jsrv+LZI3/Kiwn/ykpKf8oKCj/JSUl/yQkJP8kJSb/qWSG/7llk/++Y5H/u2eM//z//f////////////3///+olW//m4VU/8Ctef+/q4H/xbWO/8q7l//KuZX/zLuX/8u6lv/Mu5f/yLWI/8e1hv+Ncsr/j3DA/0A2R/8xMTH/MDAw/y8vL/8uLi7/LCws/ysrK/8qKir/KSkp/ygoKP8mJib/JSUl/yQkJP9zSGX/uGWV/7lklv/+/v///f///62cdv/KuI7/wax+/8e0h//Jupb/y72b/8/AnP/Qvpv/zryY/9C+m//7/vr/fnfQ/4R20P85OTn/NjY2/zQ0NP80NDT/MTEx/+Pj4//j4+P/LCws/ysrK/8sLCz/Kysr/yoqKv8pKSn/KCgo/ygoKP8qJif/rWib/7FlmP/9//3/rJyG/9LCm//Rvpb/uqmD/8SxhP/Pw5z/0cCj/9LBnf/7/P7/gnfM/3J93f93fdz/PT09/zs7O/86Ojr/OTk5/zk5Of80NDT/6enp/+jo6P/n5+f/5OTk/+Tk5P8qKir/LCws//8Js///CbP//wmz/ysrK/8qKir/p2mi/////f+hmoD/0caj/8q8mv+5p4P/0cCa/8q5k//Wx6P/08Wj//f9/f91fNn/a4Hj/2yA6P9DQkL//wmz//8Js///CbP//wmz//8Js//6Wcj/7+/v/+zs7P/r6+v/6urq/+jo6P/l5eX//wmz/zAwMP//CbP//wmz//8Js///CbP//////8zJt//MxqD/0Mek/9XMqf/Syaj/1smn/9fFo//9/vv/aIPo/2uB4v9cg+7/ZInq/0ZGRv9FRUX/RERE/0JCQv9AQED/PT09//f39//29vb/8/Pz//Ly8v/v7+//MzMz/zY2Nv//CbP//wmz//8Js/8zMzP/NDUz/5dvvP///////////9LJtP/Wzar/1cyp/9jNqv/Vyqj/08en/3R93P/r9Pv/YIfo/zZASf9ZjfL/WYrw/0hJR/9HSEX/RUVF/0VFRf9BQUH/+vr6//v7+/87Ozv/PT09/z4+Pv87Ozv/Ozs7/zo6Ov//CbP/ODg4/zc3Nv+KdMr/iXDF///////////////7/9rQtv/bzrL/2c6r/9jMrP/Vy6v/bILi/3GA4/9Wje//WYjw/+Xl5f9Skfb/VY72/ztIXf9KS0n/SUlJ/0hISP9GRkb/RUVF/0RERP9ERET/QkJC/0FBQf8/Pz//Pj4+/z09Pf9APF//e3rV/3940f+6w9j////////////////////9/9bOu//h0Lf/2tKy/9XKrP9hgfL/ZIjp/0VFRf9Zjfv/VI72/1iT6v+rq6v/S5D6/02Q+f9Qjvj/Q2KJ/0hKTP9LSkv/RklI/0lISP9GRkb/REZG/1BtmP9qg+T/b4Hi/3CC3//z9Pj/QWSe/0ZnmP///////////////////////v7+/wICAv8BAQH/UjFE//3+//9bivD/bIvo/0lISP9GRkb/TpH+/1GP+P9Sj/r/SnCn/0pLSf/y+/r/UpD3/06P9v9Ujvb/VYz1/1qM8/9ah+T/+vz5//z//P/Oz+z/i3DC/26Qvf/F0+X/YHKV/////////////////0xMTP9OL0H/UjFE/1IxRP9SMUT/w7eL//b//f9TkPb/VYz1/09MTv9LS0v/SUlJ/1FRUf/0+vv/UY/9/0qP/v9XkPX/VY31/1SN8v9Xie//VoTv//n7/v/IyMj/f3jS//7//f/+/v7/j6jE//3//f/7/f3/////////////////NSAs/1IxRP9SMUT/UjFE/1EwQ////vn/AAAA/6Sprf/z//r/S4/7/1CP+f9PjvX/TExN/0hLSv9JSUr/SEhI/0ZGRv9FRkT/REVH/ztAT/9sheH/bX/g/3GB4v////3/r7nR/8PM0//m4rr/xsim//7+/v9UVFT/oqKi/wkGCP9SMUT/UjFE/1IxRP9SMUT/GxAW//D3/f/w9/3/8Pf9//D3/f/w9/3/8Pf9//D3/f9olen/UZH6/1GQ9/9TjfX/VYz1/1uM8f9bie3/v8Lq//z//P/9//3/8+jM/+bcvv/9//3/7+LF/+nq2v///////////1IxRP9QMEL/HhIZ/1IxRP9SMUT/UjFE/1IxRP8RDhD/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3/Z2tu//D3/f/l6/H/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3/IyQl/+Lk4f/9//n/+fHn/97Vxv/8//v//f//////////////UjFE/1IxRP8eEhn/UjFE/1IxRP9SMUT/UjFE/wAAAP/w9/3/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3/z9Xa//D3/f/w9/3/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3///////7+/v/+/v7//f39////////////AAAA/wAAAP9SMUT/UjFE/y4cJv9SMUT/UjFE/1IxRP9SMUT/AAAA//D3/f/w9/3/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3/8Pf9/8nP1P///////////////////////////wAAAP8AAAD//////1IxRP9SMUT/Sy0+/1IxRP9SMUT/UjFE/1IxRP8EAwT/8Pf9/+/2/P8BAQH/tbq+//D3/f/w9/3/8Pf9//D3/f/w9/3/8Pf9/3yAg/8AAAD/8Pf9//D3/f9SVVj/OyQx////////////////////////////AAAA////////////UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/yEUG//w9/3/8Pf9/wAAAP8jJCX/8Pf9//D3/f/w9/3/8Pf9//D3/f/w9/3/2N7j/woLC//w9/3/AAAA/1IxRP9SMUT///////////////////////////8AAAD///////////9SMUT/UjFE/1IxRP88JDL/UjFE/1IxRP9SMUT/Ty9C/0tNT//w9/3/8Pf9//D3/f/w9/3/6O/1//D3/f/w9/3/8Pf9//D3/f/w9/3/ERES/wcHB/8FBQX/UjFE/08vQf///////////wAAAP///////////wAAAP///////////xgUFv9SMUT/UjFE/wAAAP9SMUT/UjFE/1IxRP9SMUT/Ty9C/wcEBv/w9/3/8Pf9/wAAAP9SMUT/UjFE/1EwQ/9QMEL/vcPH//D3/f/w9/3/8Pf9/w0ODv9SMUT/Mh4q////////////AAAA/wAAAP//////AAAA////////////EhIS/1IxRP9SMUT/JRYf/1IxRP9SMUT/UjFE/1IxRP8MBwr/UjFE/+Xr8f9jZmj/UjFE/1IxRP9SMUT/UjFE/1IxRP8BAQH/8Pf9/2NmaP/w9/3/AAAA/1IxRP8AAAD//////wAAAP//////AAAA//////8AAAD////////////9/f3/UjFE/1IxRP9SMUT/Sy0+/1IxRP9SMUT/UjFE/wAAAP9SMUT/AAAA//D3/f9RMUP/UjFE/1IxRP9SMUT/UjFE/1ExQ//w9/3/8Pf9/7vBxf8AAAD/UjFE/wEBAf8AAAD/AAAA//////8AAAD//f39/wAAAP////////////////8KCQn/UjFE/1IxRP8FBAT/UjFE/1IxRP9SMUT/NB8r/1IxRP9SMUT/7vX7/zwkMv9SMUT/UjFE/1IxRP9SMUT/UjFE/ykqK//w9/3/CwoL/1IxRP9SMUT/v7+//wAAAP8AAAD//////wQEBP8AAAD/AAAA/////////////////1NTU/9SMUT/UjFE/1IxRP9NLj//UjFE/1IxRP9SMUT/UjFE/1IxRP8AAAD/FBIX/1IxRP9SMUT/UjFE/1IxRP9SMUT/AgIC//D3/f9QMEP/UjFE/1IxRP///////////////////////////wAAAP8AAAD//////////////////////x8XG/9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP8AAAD/UjFE/1IxRP9SMUT/UjFE/1IxRP8pGCL/AAAA/1IxRP9SMUT/EQwP////////////////////////////AAAA/wAAAP//////////////////////MDAw/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/0wtP/9QMEL/UjFE/1IxRP8DAwP///////////////////////////8AAAD/AAAA////////////////////////////Ihkf/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE///////////////////////////////////////19fX////////////////////////////c3Nz/UTFD/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP8AAAD///////////////////////////////////////////////////////////////////////////9HR0f/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/SCs8//////////////////////////////////////////////////////////////////////////////////////8FBQX/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/z8mNP9cXFz///////////////////////////////////////////////////////////////////////////////////////////9vb2//NSUt/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/1IxRP9SMUT/UjFE/zgiL/8GBgb////////////////////////////////////////////////////////////////////////////////////////////////////////////9////AAAA/ykZIv9SMUT/UjFE/1IxRP9SMUT/Sy0+/w0LDP8kJCT//f/9//////////////////////////////7///3//f/////////////////////////////+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

class AudioEngine:
    def __init__(self, sr=44100):
        self.sr = sr
        self.left = np.zeros(1, dtype=np.float32)
        self.right = np.zeros(1, dtype=np.float32)
        self.n = 1
        self.idx = 0
        self._playing = threading.Event()
        self._crash_simulation = False
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype="float32",
            callback=self._callback,
            blocksize=0
        )
        self._stream.start()
        self.on_track_end = None
    def is_playing(self):
        return self._playing.is_set()
    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            idx, n = self.idx, self.n
            left, right = self.left, self.right
        if not self._playing.is_set():
            outdata.fill(0)
            return
        end = idx + frames
        if end <= n:
            outdata[:, 0] = left[idx:end]
            outdata[:, 1] = right[idx:end]
            self.idx = end
        else:
            remain = n - idx
            if remain > 0:
                outdata[:remain, 0] = left[idx:]
                outdata[:remain, 1] = right[idx:]
            outdata[remain:, :].fill(0)
            self._playing.clear()
            self.idx = 0
            if self.on_track_end:
                try:
                    self.on_track_end()
                except Exception as e:
                    print("on_track_end error:", e)
                    with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                        lf.write("Track end callback failed:\n")
                        traceback.print_exc(file=lf)
                        lf.flush()
    def play(self): self._playing.set()
    def pause(self): self._playing.clear()
    def toggle_play(self):
        if self._playing.is_set():
            self.pause()
        else:
            self.play()
        return self._playing.is_set()
    def seek_seconds(self, t):
        with self._lock:
            self.idx = int(t * self.sr)
    def current_seconds(self):
        return self.idx / float(self.sr)
    def duration_seconds(self):
        return self.n / float(self.sr)
    def load_track(self, sr, left, right):
        global EmergencyStop
        left = left.astype(np.float32, copy=False)
        right = right.astype(np.float32, copy=False)
        self.pause()
        with self._lock:
            self.sr = sr
            self.left = left
            self.right = right
            self.n = len(left)
            self.idx = 0
        if self._crash_simulation:
            self._stream.stop()
            self._stream.close()
            time.sleep(0.016)
    def close(self):
        try:
            if self._stream.active:
                self._stream.stop()
            self._stream.close()
        except Exception as e:
            print("Error closing AudioEngine:", e)
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("AudioEngine close failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
        finally:
            self._playing.clear()
            self.idx = 0

class Visualizer:
    def __init__(self, left, right, sr):
        self.left = np.asarray(left, dtype=np.float32)
        self.right = np.asarray(right, dtype=np.float32)
        self.mid = (self.left + self.right) * 0.5
        self.side = (self.left - self.right) * 0.5
        self.sr = int(sr)
    def list_for_fft(self, t_sec: float, channel: str, buffer_samples: int) -> np.ndarray:
        end_sample = int(round(t_sec * self.sr))
        start_sample = end_sample - buffer_samples
        if start_sample < 0:
            pad = np.zeros((-start_sample,), dtype=np.float32)
            data = self.mid[: max(0, end_sample)] if channel == 'm' else self.side[: max(0, end_sample)]
            return np.concatenate([pad, data])
        data = self.mid if channel == 'm' else self.side
        end_sample = min(end_sample, data.shape[0])
        start_sample = max(0, start_sample)
        return data[start_sample:end_sample]
    def fft_db(self, samples: np.ndarray, zero_pad_factor: int) -> tuple[np.ndarray, np.ndarray]:
        if samples is None or samples.size == 0:
            n = 512 
            N = int(n * max(1, zero_pad_factor))
            freqs = np.fft.rfftfreq(N, d=1.0 / self.sr)
            db = np.full(freqs.shape, -120.0)
            return freqs, db
        n = int(samples.shape[0])
        if n <= 0:
            n = 512
        if n <= 0:
            return np.array([]), np.array([])
        window = np.hanning(n)
        samples = samples * window
        N = int(n * max(1, zero_pad_factor))
        fft_result = np.fft.rfft(samples, n=N)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.sr)
        mags = np.abs(fft_result) * (2.0 / max(1, n))
        db = 20.0 * np.log10(mags + 1e-12)
        return freqs, db
    def spectrum_to_polyline(self, freqs, db_vals, fmin, fmax, db_min, db_max, width, height):
        freqs = np.maximum(freqs, 1e-6)
        logf = np.log10(freqs)
        log_fmin = np.log10(max(fmin, 1e-6))
        log_fmax = np.log10(max(fmax, fmin + 1))
        x_float = (logf - log_fmin) / (log_fmax - log_fmin)
        x_float = np.clip(x_float, 0.0, 1.0)
        x_pix = (x_float * (width - 1)).astype(int)
        db_vals = np.clip(db_vals, db_min, db_max)
        y_float = (db_vals - db_min) / (db_max - db_min)
        y_pix = (1.0 - y_float) * (height - 1)
        y_pix = y_pix.astype(int)
        max_db_per_x = {}
        for x, y, dbv in zip(x_pix, y_pix, db_vals):
            prev = max_db_per_x.get(x)
            if prev is None or dbv > prev[1]:
                max_db_per_x[x] = (y, dbv)
        points = []
        for x in range(0, width):
            if x in max_db_per_x:
                y = max_db_per_x[x][0]
                points.append((x, int(y)))
        return points

class BetterWMP(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg=SkinInfo["tkinter"]["bg"])
        icon_bytes = base64.b64decode(ICON.split(",")[1])
        image = Image.open(BytesIO(icon_bytes))
        photo = ImageTk.PhotoImage(image)
        self.iconphoto(True, photo)
        self.title("BetterWMP - *")
        self.geometry("750x600")
        self.minsize(750, 500)
        self.playlist = []
        self.playlist_listbox = None
        self.appdata_dir = os.path.expandvars(r"%localappdata%\betterwmpfiles")
        shutil.rmtree(self.appdata_dir, ignore_errors=True)
        os.makedirs(self.appdata_dir, exist_ok=True)
        self.title("BetterWMP: *")
        style = ttk.Style()
        style.configure('Custom.TMenubutton',
            background=SkinInfo["tkinter"].get("dropdownbg", "#444444"), highlightthickness=0, borderwidth=0, relief="flat",
            foreground=SkinInfo["tkinter"].get("dropdownfg", "#ffffff"))
        self.audio: AudioEngine | None = None
        self.vis: Visualizer | None = None
        self.displayname = tk.StringVar(value="<No file>")
        self.buffer_var = tk.IntVar(value=4096)
        self.zp_var = tk.IntVar(value=1)
        self.is_dragging = False
        self.was_playing_before_drag = False
        self.display_time = 0.0
        self.drag_target_time = 0.0
        self._pending_seek = None
        self._pending_play = False
        self.current_wav = None
        self.audio: AudioEngine | None = None
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self._on_file_drop)
        self.fmin = 10.0
        self.db_min, self.db_max = -80.0, 0.0
        self.bg = SkinInfo["fft"]["bg"]
        self.fg = SkinInfo["fft"]["fg"]
        self.mid_color = SkinInfo["fft"]["mid"]
        self.side_color = SkinInfo["fft"]["side"]
        self._build_ui()
        self._last_frame_time = time.perf_counter()
        if self._is_minimized():
            frame_delay = 100
        else:
            frame_delay = 16
        self.after(frame_delay, self._update_loop)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    def _is_minimized(self) -> bool:
        try:
            return (self.state() == 'iconic') or (not self.winfo_viewable())
        except Exception:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("Read minimization state failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            return False
    def _build_ui(self):
        global DEBUG
        tk_colors = SkinInfo.get("tkinter", {})
        top = tk.Frame(self, bg=tk_colors.get("bg", "#1a1a1a"))
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=8)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Flat.TButton",
            background=SkinInfo["tkinter"].get("buttonbg", "#333333"),
            foreground=SkinInfo["tkinter"].get("buttonfg", "#ffffff"),
            relief="flat", borderwidth=0
        )
        style.map("Flat.TButton",
            background=[
                ("pressed", SkinInfo["tkinter"].get("buttonpressbg", "#222222")),
                ("active", SkinInfo["tkinter"].get("buttonactivebg", "#505050")),
                ("disabled", SkinInfo["tkinter"].get("buttonbg", "#333333"))
            ],
            foreground=[
                ("pressed", SkinInfo["tkinter"].get("buttonpressfg", "#ffffff")),
                ("active", SkinInfo["tkinter"].get("buttonactivefg", "#d88aff")),
                ("disabled", SkinInfo["tkinter"].get("buttondisabledfg", "#777777"))
            ]
        )
        self.play_btn = ttk.Button(
            top, text="▶", command=self._toggle_play, takefocus=0, width=5, style="Flat.TButton"
        )
        self.play_btn.pack(side=tk.LEFT, padx=(5, 0))
        self.play_btn.state(["disabled"])
        self.title(f"BetterWMP: {self.displayname.get()}")
        self.buffer_var = tk.IntVar(value=4096)
        self.buffer_label = tk.StringVar(value=f"{self.buffer_var.get()}  ▾")
        tk.Label(top, text="Buffer", fg=tk_colors.get("label", "#ffffff"), bg=tk_colors.get("bg", "#1a1a1a")).pack(side=tk.LEFT, padx=(10, 2))
        buf_menu = tk.OptionMenu(top, self.buffer_label, "")
        buf_menu.config(
            indicatoron=0, 
            bg=tk_colors.get("dropdownbg", "#555555"),
            fg=tk_colors.get("dropdownfg", "#ffffff"),
            activebackground=tk_colors.get("dropdownactivebg", "#505050"),
            activeforeground=tk_colors.get("dropdownactivefg", "#ffffff"),
            relief="flat", borderwidth=0, highlightthickness=0
        )
        buf_menu["menu"].config(
            bg=tk_colors.get("dropdownbg", "#555555"),
            fg=tk_colors.get("dropdownfg", "#ffffff"),
            activebackground=tk_colors.get("dropdownactivebg", "#505050"),
            activeforeground=tk_colors.get("dropdownactivefg", "#ffffff")
        )
        buf_menu.pack(side=tk.LEFT)
        buf_menu["menu"].delete(0, "end")
        for opt in [512, 1024, 2048, 4096, 8192]:
            buf_menu["menu"].add_radiobutton(
                label=str(opt),
                variable=self.buffer_var,
                value=opt,
                command=lambda v=opt: self.set_buffer(v)
            )
        tk.Label(top, text="Zero-pad", fg=tk_colors.get("label", "#ffffff"), bg=tk_colors.get("bg", "#1a1a1a")).pack(side=tk.LEFT, padx=(10, 2))
        self.zp_label = tk.StringVar(value=f"{self.zp_var.get()}  ▾")
        zp_menu = tk.OptionMenu(top, self.zp_label, "")
        zp_menu.config(
            indicatoron=0, 
            bg=tk_colors.get("dropdownbg", "#555555"),
            fg=tk_colors.get("dropdownfg", "#ffffff"),
            activebackground=tk_colors.get("dropdownactivebg", "#505050"),
            activeforeground=tk_colors.get("dropdownactivefg", "#ffffff"),
            relief="flat", borderwidth=0, highlightthickness=0
        )
        zp_menu["menu"].config(
            bg=tk_colors.get("dropdownbg", "#555555"),
            fg=tk_colors.get("dropdownfg", "#ffffff"),
            activebackground=tk_colors.get("dropdownactivebg", "#505050"),
            activeforeground=tk_colors.get("dropdownactivefg", "#ffffff")
        )
        zp_menu.pack(side=tk.LEFT)
        zp_menu["menu"].delete(0, "end")
        for opt in [1, 2, 4]:
            zp_menu["menu"].add_radiobutton(
            label=str(opt),
            variable=self.zp_var,
            value=opt,
            command=lambda v=opt: self.set_zp(v)
            )
        tk.Label(top, text="Repeat", fg=tk_colors.get("label", "#ffffff"), bg=tk_colors.get("bg", "#1a1a1a")).pack(side=tk.LEFT, padx=(10, 5))
        self.repeat_label = tk.StringVar(value=f"{'track once'}  ▾")
        repeat_options = ["track once", "playlist once", "track repeat", "playlist repeat"]
        repeat_menu = tk.OptionMenu(top, self.repeat_label, "")
        repeat_menu.config(
            indicatoron=0, 
            bg=tk_colors.get("dropdownbg", "#555555"),
            fg=tk_colors.get("dropdownfg", "#ffffff"),
            activebackground=tk_colors.get("dropdownactivebg", "#505050"),
            activeforeground=tk_colors.get("dropdownactivefg", "#ffffff"),
            relief="flat", borderwidth=0, highlightthickness=0
        )
        repeat_menu["menu"].config(
            bg=tk_colors.get("dropdownbg", "#555555"),
            fg=tk_colors.get("dropdownfg", "#ffffff"),
            activebackground=tk_colors.get("dropdownactivebg", "#505050"),
            activeforeground=tk_colors.get("dropdownactivefg", "#ffffff")
        )
        repeat_menu.pack(side=tk.LEFT)
        self.repeat_mode = tk.StringVar(value="track once")
        repeat_menu["menu"].delete(0, "end")
        for opt in repeat_options:
            repeat_menu["menu"].add_radiobutton(
            label=opt,
            variable=self.repeat_mode,
            value=opt,
            command=lambda v=opt: self._on_repeat_mode_change(v)
            )
        self.prev_btn = ttk.Button(
            top, text="←", command=self._playlist_prev,
            takefocus=0, width=3, style="Flat.TButton"
        )
        self.prev_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.next_btn = ttk.Button(
            top, text="→", command=self._playlist_next,
            takefocus=0, width=3, style="Flat.TButton"
        )
        self.next_btn.pack(side=tk.LEFT, padx=(2, 0))
        self.restart_btn = ttk.Button(
            top, text="\u21BA", command=self._playlist_restart,
            takefocus=0, width=3, style="Flat.TButton"
        )
        self.restart_btn.pack(side=tk.LEFT, padx=(2, 0))
        self.skin_btn = ttk.Button(
            top, text="🔧", command=self._change_skin_pointer,
            takefocus=0, width=3, style="Flat.TButton"
        )
        self.skin_btn.pack(side=tk.LEFT, padx=(2, 0))
        self.timestamp_label = tk.Label(top, text="", fg=tk_colors.get("label", "#ffffff"), bg=tk_colors.get("bg", "#1a1a1a"))
        self.timestamp_label.pack(side=tk.LEFT, padx=(10, 0))
        self._update_nav_buttons()
        self.prog = tk.Canvas(self, height=24, background=SkinInfo["prog"]["bg"], highlightthickness=0)
        self.prog.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(4, 8))
        self.prog.bind("<Button-1>", self._on_prog_press)
        self.prog.bind("<B1-Motion>", self._on_prog_motion)
        self.prog.bind("<ButtonRelease-1>", self._on_prog_release)
        playlist_section = ttk.Frame(self)
        playlist_section.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=4)
        style = ttk.Style()
        style.theme_use("clam")
        scrollbar = ttk.Scrollbar(playlist_section, orient=tk.VERTICAL)
        self.playlist_listbox = tk.Listbox(playlist_section, yscrollcommand=scrollbar.set,
            bg=tk_colors.get("listboxbg", "#000000"), fg=tk_colors.get("listboxfg", "#ffffff"),
            selectbackground=tk_colors.get("listboxselbg", "#402e50"), selectforeground=tk_colors.get("listboxselfg", "#ffffff"),
            height = 13)
        scrollbar.config(command=self.playlist_listbox.yview)
        self.playlist_listbox.configure(activestyle="none")
        btns = [
            ("LOAD", self._playlist_play_selected),
            ("ADD", self._playlist_append),
            ("Insert", self._playlist_insert),
            ("Remove", self._playlist_remove),
            ("Clear", self._playlist_clear),
            ("Move Up", self._playlist_up),
            ("Move Down", self._playlist_down),
            ("Shuffle", self._playlist_shuffle),
        ]
        btn_frame = tk.Frame(playlist_section, bg=tk_colors.get("bg", "#1a1a1a"))
        self.playlist_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y)
        for txt, cmd in btns:
            b = ttk.Button(
                btn_frame,
                text=txt,
                command=cmd,
                takefocus=0,
                style="Flat.TButton"
            )
            b.pack(side=tk.TOP, fill=tk.X, pady=1)
        self.playlist_listbox.bind('<Double-Button-1>', lambda e: self._playlist_play_selected())
        self.bind_class("TButton", "<Return>", lambda e: "break")
        self.bind_class("TButton", "<space>", lambda e: "break")
        self.bind("<Control-o>", lambda e: self._playlist_append())
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("<Return>", lambda e: self._playlist_play_selected())
        self.bind("<Delete>", lambda e: self._playlist_remove())
        self.bind("<Up>", lambda e: self._highlight_loaded())
        self.bind("<Down>", lambda e: self._highlight_loaded())
        if DEBUG:
            self.bind("<Control-Shift-Alt-c>", lambda e: self._simulate_crash())
            self.bind("<Control-Shift-Alt-C>", lambda e: self._simulate_crash())
            self.bind("<Control-Shift-Alt-x>", lambda e: self._simulate_tkinterlevel_crash())
            self.bind("<Control-Shift-Alt-X>", lambda e: self._simulate_tkinterlevel_crash())
            self.bind("<Control-Shift-Alt-z>", lambda e: self._call_emergency_stop())
            self.bind("<Control-Shift-Alt-Z>", lambda e: self._call_emergency_stop())
        self.playlist_listbox.bind("<Double-Button-1>", self._on_double_click)
        self.playlist_listbox.bind("<Button-1>", self._on_single_click, add="+")
        self.playlist_listbox.bind("<ButtonPress-1>", self._on_single_click, add="+")
        self.viz = tk.Canvas(self, background=self.bg, highlightthickness=0)
        self.viz.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure('Custom.TMenubutton',
            background=SkinInfo["tkinter"].get("dropdownbg", "#444444"), 
            highlightthickness=0, borderwidth=0, relief="flat",
            foreground=SkinInfo["tkinter"].get("dropdownfg", "#ffffff"),
            activebackground=SkinInfo["tkinter"].get("dropdownactivebg", "#505050"),
            activeforeground=SkinInfo["tkinter"].get("dropdownactivefg", "#ffffff"))
    def _call_emergency_stop(self):
        global EmergencyStop
        response = ctypes.windll.user32.MessageBoxW(
            0,
            "This will initate an emergency stop.\nIf you continue, the app will crash.",
            "Continue",
            0x04 | 0x30
        )
        with open(FAULT_LOG, "a", encoding="utf-8") as lf:
            lf.write("NOT MY FAULT\n")
            lf.flush()
        if response == 6:
            EmergencyStop = True
    def _simulate_crash(self):
        response = ctypes.windll.user32.MessageBoxW(
            0,
            "This will initiate a PortAudio crash.\nIf you continue, the app will crash upon track-end handling. You can undo by clearing the playlist.",
            "Continue",
            0x04 | 0x30
        )
        with open(FAULT_LOG, "a", encoding="utf-8") as lf:
            lf.write("NOT MY FAULT\n")
            lf.flush()
        if response == 6:
            if self.audio is not None:
                self.audio._crash_simulation = True
    def _simulate_tkinterlevel_crash(self):
        response = ctypes.windll.user32.MessageBoxW(
            0,
            "This will initiate a Tkinter crash.\nIf you continue, the app will log an error. This will not cause a crash.",
            "Continue",
            0x04 | 0x30
        )
        with open(FAULT_LOG, "a", encoding="utf-8") as lf:
            lf.write("NOT MY FAULT\n")
            lf.flush()
        if response == 6:
            raise RuntimeError("User requested the crash.")
    def _change_skin_pointer(self):
        conf_dir = os.path.expandvars(r"%localappdata%\\betterwmpconf")
        pointer_path = os.path.join(conf_dir, "skinpointer.conf")
        if not os.path.isfile(pointer_path):
            messagebox.showinfo("Skin Pointer", "skinpointer.conf not found.")
            return
        new_file = filedialog.askopenfilename(
            title="Select Skin File",
            filetypes=[("BetterWMP Skin Files", "*.bwmpskin")],
            initialdir=conf_dir
        )
        if not new_file:
            return
        required_structure = {
            "fft": ["bg", "fg", "mid", "side", "line", "text"],
            "prog": ["bg", "left", "thumb"],
            "tkinter": [
                "bg", "label",
                "buttonbg", "buttonfg",
                "buttonactivebg", "buttonactivefg",
                "buttonpressbg", "buttonpressfg",
                "buttondisabledfg",
                "listboxbg", "listboxfg",
                "listboxselbg", "listboxselfg",
                "dropdownbg", "dropdownfg",
                "dropdownactivebg", "dropdownactivefg",
                "loadedfg", "doublefg"
            ]
        }
        try:
            with open(new_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for section, keys in required_structure.items():
                if section not in data:
                    raise ValueError(f"Missing section: {section}")
                for key in keys:
                    if key not in data[section]:
                        raise ValueError(f"Missing key: {section}.{key}")
        except Exception as e:
            messagebox.showerror("Invalid Skin", f"Invalid skin file:\n{e}")
            return
        try:
            with open(pointer_path, "w", encoding="utf-8") as f:
                f.write(os.path.basename(new_file))
            messagebox.showinfo("Skin Pointer", f"Skin pointer set to {os.path.basename(new_file)}\nYou can restart to see changes.")
        except Exception as e:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("Update Skin Pointer failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            messagebox.showerror("Error", f"Could not update skinpointer.conf:\n{e}")
    def _on_single_click(self, event):
        self._highlight_loaded()
        lb = self.playlist_listbox
        size = lb.size()
        if size == 0:
            return "break"
        index = lb.nearest(event.y)
        bbox = lb.bbox(index)
        if bbox is None or not (bbox[1] <= event.y < bbox[1] + bbox[3]):
            lb.selection_clear(0, tk.END)
            return "break"
        self._highlight_loaded()
    def _on_double_click(self, event):
        index = event.widget.nearest(event.y)
        bbox = event.widget.bbox(index)
        if bbox is None or not (bbox[1] <= event.y < bbox[1] + bbox[3]):
            self._playlist_append()
            return
        self._playlist_play_selected()
        self._highlight_loaded()
    def set_buffer(self, val):
        self.buffer_var.set(val)
        self.buffer_label.set(f"{val}  ▾")
    def set_zp(self, val):
        self.zp_var.set(val)
        self.zp_label.set(f"{val}  ▾")
    def _update_nav_buttons(self):
        if len(self.playlist) > 1:
            self.prev_btn.configure(state=tk.NORMAL)
            self.next_btn.configure(state=tk.NORMAL)
        else:
            self.prev_btn.configure(state=tk.DISABLED)
            self.next_btn.configure(state=tk.DISABLED)
        if len(self.playlist) > 0:
            self.restart_btn.configure(state=tk.NORMAL)
        else:
            self.restart_btn.configure(state=tk.DISABLED)
    def _playlist_next(self):
        was_playing = self.audio.is_playing() if self.audio else False
        idxs = [i for i, entry in enumerate(self.playlist) if entry['wav'] == getattr(self, "current_wav", None)]
        idx = idxs[0] if idxs else 0
        if len(self.playlist) == 0:
            return
        if idx + 1 < len(self.playlist):
            new_idx = idx + 1
        else:
            new_idx = 0
        self.playlist_listbox.selection_clear(0, tk.END)
        self.playlist_listbox.selection_set(new_idx)
        entry = self.playlist[new_idx]
        self._open_file(entry['wav'])
        if self.audio and was_playing:
            self.audio.seek_seconds(0.0)
            self._set_play(True)
        else:
            self.audio.seek_seconds(0.0)
            self._set_play(False)
        self._highlight_loaded()
    def _playlist_prev(self):
        was_playing = self.audio.is_playing() if self.audio else False
        idxs = [i for i, entry in enumerate(self.playlist) if entry['wav'] == getattr(self, "current_wav", None)]
        idx = idxs[0] if idxs else 0
        if len(self.playlist) == 0:
            return
        if idx <= 0:
            new_idx = len(self.playlist) - 1
        else:
            new_idx = idx - 1
        self.playlist_listbox.selection_clear(0, tk.END)
        self.playlist_listbox.selection_set(new_idx)
        entry = self.playlist[new_idx]
        self._open_file(entry['wav'])
        if self.audio and was_playing:
            self.audio.seek_seconds(0.0)
            self._set_play(True)
        else:
            self.audio.seek_seconds(0.0)
            self._set_play(False)
        self._highlight_loaded()
    def _playlist_restart(self):
        was_playing = self.audio.is_playing() if self.audio else False
        idxs = [i for i, entry in enumerate(self.playlist) if entry['wav'] == getattr(self, "current_wav", None)]
        idx = idxs[0] if idxs else 0
        if 0 <= idx < len(self.playlist):
            entry = self.playlist[idx]
            self._open_file(entry['wav'])
            if self.audio and was_playing:
                self.audio.seek_seconds(0.0)
                self._set_play(True)
            else:
                self.audio.seek_seconds(0.0)
                self._set_play(False)
        self._highlight_loaded()
    def _on_file_drop(self, event):
        def parse_paths(data):
            paths = []
            i = 0
            while i < len(data):
                if data[i] == '{':
                    j = data.find('}', i)
                    if j == -1:
                        break
                    paths.append(data[i+1:j])
                    i = j + 1
                else:
                    j = data.find(' ', i)
                    if j == -1:
                        paths.append(data[i:])
                        break
                    paths.append(data[i:j])
                    i = j + 1
                while i < len(data) and data[i] == ' ':
                    i += 1
            return paths
        file_paths = parse_paths(event.data.strip())
        valid_files = [f for f in file_paths if os.path.isfile(f) and f.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus"))]
        if valid_files:
            self._playlist_append_sysargv(valid_files)
            self._update_nav_buttons()
        self._highlight_loaded()
    def _open_file(self, path):
        already_in_playlist = any(entry['wav'] == path for entry in self.playlist)
        if not already_in_playlist:
            self.add_file_to_playlist(path)
        entry = next((e for e in self.playlist if e['wav'] == path), None)
        if entry is None:
            messagebox.showerror("Error", f"File not in playlist: {path}")
            return
        output_path = entry['wav']
        try:
            if output_path.lower().endswith(".wav"):
                sr, data = wavfile.read(output_path)
            else:
                audio = AudioSegment.from_file(output_path)
                audio.export(output_path, format="wav")
                sr, data = wavfile.read(output_path)
        except Exception as e:
            traceback.print_exc()
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("File open failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            self.after(0, lambda: messagebox.showerror("Open failed", f"Could not read file:\n{e}"))
            return
        data = to_float32(np.asarray(data))
        if data.ndim == 1:
            left = right = data
        else:
            left, right = data[:, 0], data[:, 1]
        if self.audio is None:
            self.audio = AudioEngine(sr)
            self.audio.on_track_end = self._handle_track_end
        self.audio.load_track(sr, left, right)
        self.vis = Visualizer(left, right, sr)
        self.displayname.set(os.path.basename(entry['orig']))
        self.title(f"BetterWMP: {self.displayname.get()}")
        self.play_btn.configure(text="▶", state="normal")
        self.display_time = 0.0
        self.drag_target_time = 0.0
        self.is_dragging = False
        self.current_wav = entry['wav']
        self._highlight_loaded()
    def _highlight_loaded(self):
        for i in range(self.playlist_listbox.size()):
            self.playlist_listbox.itemconfig(
                i, fg=SkinInfo["tkinter"].get("listboxfg", "#ffffff")
            )
        if not hasattr(self, "current_wav"):
            return
        for i, entry in enumerate(self.playlist):
            if entry['wav'] == self.current_wav:
                self.playlist_listbox.itemconfig(
                    i,
                    fg=SkinInfo["tkinter"].get("loadedfg", "#fcb1ff")
                )
                break
        selected = self.playlist_listbox.curselection()
        if selected:
            idx = selected[0]
            if self.playlist[idx]['wav'] == self.current_wav:
                self.playlist_listbox.config(
                    selectforeground=SkinInfo["tkinter"].get("doublefg", "#ffc1f6")
                )
            else:
                self.playlist_listbox.config(
                    selectforeground=SkinInfo["tkinter"].get("listboxselfg", "#ffffff")
                )
    def _convert_and_play(self, input_path):
        output_path = os.path.expandvars(r"%localappdata%\\betterwmpfiles\\temp.wav")
        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            print(f"Converted and saved WAV to: {output_path}")
        except Exception as e:
            print(f"Error converting file: {e}")
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("File conversion failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            messagebox.showerror("File Conversion Error", f"Could not convert file to WAV format:\n{e}")
            return
        try:
            playsound(output_path)
        except Exception as e:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("File playback failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            print(f"Error playing WAV file: {e}")
            messagebox.showerror("Playback Error", f"Could not play the converted WAV file:\n{e}")
    def show_progress_window(self, total_files):
        self.progress_win = tk.Toplevel(self)
        self.progress_win.title("Processing Files")
        self.progress_label = tk.Label(self.progress_win, text="Processing files...")
        self.progress_label.pack(padx=20, pady=10)
        self.progress_bar = ttk.Progressbar(self.progress_win, length=200, maximum=total_files)
        self.progress_bar.pack(padx=20, pady=10)
        self.progress_win.transient(self)
        self.progress_win.grab_set()
        self.progress_win.update()
    def update_progress(self, current, total, filename):
        self.progress_label.config(text=f"Preparing {filename}\n({current}/{total})")
        self.progress_bar['value'] = current
        self.progress_win.update()
    def close_progress_window(self):
        self.progress_win.destroy()
    def add_file_to_playlist(self, orig_path, insert_at=None):
        name = os.path.basename(orig_path)
        wav_path = os.path.join(self.appdata_dir, f"{int(time.time()*1000)}_{name}.wav")
        was_empty = len(self.playlist) == 0
        ffmpeg_exe = AudioSegment.converter or "ffmpeg"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", orig_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            wav_path
        ]
        try:
            subprocess.run(cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as e:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("File conversion failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            print(f"Conversion failed: {e}")
            return
        entry = {'orig': orig_path, 'wav': wav_path, 'name': name}
        if insert_at is None:
            self.playlist.append(entry)
            self.playlist_listbox.insert(tk.END, name)
        else:
            self.playlist.insert(insert_at, entry)
            self.playlist_listbox.insert(insert_at, name)
        if was_empty:
            self._set_play(False)
        self._update_nav_buttons()
    def _playlist_remove(self):
        idx = self.playlist_listbox.curselection()
        if idx:
            self._remove_file_from_playlist(idx[0])
            self.playlist_listbox.selection_clear(0, tk.END)
            if self.playlist:
                self.playlist_listbox.selection_set(min(idx[0], len(self.playlist) - 1))
        self._update_nav_buttons()
        self._highlight_loaded()
    def _remove_file_from_playlist(self, idx):
        entry = self.playlist.pop(idx)
        self.playlist_listbox.delete(idx)
        try:
            os.remove(entry['wav'])
        except Exception as e:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("File removal failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            print(f"Error removing file: {e}")
        if entry['wav'] == getattr(self, "current_wav", None):
            if self.audio is not None:
                self.audio.close()
                self.audio = None
            if not self.playlist: 
                self._set_play(False)
                self.displayname.set("<No file>")
                self.viz.delete("all")
                self.vis = None
                self.play_btn.configure(text="▶", state="disabled")
                self.title("BetterWMP: <No file>")
                self._update_nav_buttons()
                return
            self._set_play(False)
            self.displayname.set("<No file>")
            self.viz.delete("all")
            self.vis = None
            self.play_btn.configure(text="▶", state="disabled")
            self.title("BetterWMP: <No file>")
            self._update_nav_buttons()
        self._highlight_loaded()
    def _playlist_append(self):
        files = filedialog.askopenfilenames(title="Append files", filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac *.aac *.m4a *.wma *.opus")])
        if not files:
            return
        def worker():
            self.show_progress_window(len(files))
            for i, f in enumerate(files, start=1):
                if os.path.isfile(f):
                    self.add_file_to_playlist(f)
                self.after(0, lambda i=i, f=f: self.update_progress(i, len(files), os.path.basename(f)))
            self.after(0, self.close_progress_window)
        threading.Thread(target=worker, daemon=True).start()
        self._update_nav_buttons()
        self._highlight_loaded()
    def _playlist_append_sysargv(self, files):
        if not files:
            return
        def worker():
            self.show_progress_window(len(files))
            for i, f in enumerate(files, start=1):
                if os.path.isfile(f):
                    self.add_file_to_playlist(f)
                self.after(0, lambda i=i, f=f: self.update_progress(i, len(files), os.path.basename(f)))
            self.after(0, self.close_progress_window)
        threading.Thread(target=worker, daemon=True).start()
        self._update_nav_buttons()
        self._highlight_loaded()
    def _playlist_insert(self):
        files = filedialog.askopenfilenames(
            title="Insert files",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac *.aac *.m4a *.wma *.opus")]
        )
        idxs = self.playlist_listbox.curselection()
        insert_at = idxs[0] + 1 if idxs else len(self.playlist)
        try:
            for f in files:
                if os.path.isfile(f):
                    name = os.path.basename(f)
                    wav_path = os.path.join(self.appdata_dir, f"{int(time.time()*1000)}_{name}.wav")
                    ffmpeg_exe = AudioSegment.converter or "ffmpeg"
                    cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-i", f,
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ar", "44100",
                        "-ac", "2",
                        wav_path
                    ]
                    try:
                        subprocess.run(cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    except Exception as e:
                        print(f"Conversion failed: {e}")
                        with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                            lf.write("Conversion failed:\n")
                            traceback.print_exc(file=lf)
                            lf.flush()
                        continue
                    entry = {'orig': f, 'wav': wav_path, 'name': name}
                    self.playlist.insert(insert_at, entry)
                    self.playlist_listbox.insert(insert_at, name)
                    insert_at += 1
            if files:
                self.playlist_listbox.selection_clear(0, tk.END)
                self.playlist_listbox.selection_set(insert_at - 1)
        except Exception as e:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("Insert failed:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            print(e)
        self._update_nav_buttons()
        self._highlight_loaded()
    def _playlist_up(self):
        idx = self.playlist_listbox.curselection()
        if idx and idx[0] > 0:
            i = idx[0]
            self.playlist[i-1], self.playlist[i] = self.playlist[i], self.playlist[i-1]
            self.playlist_listbox.delete(0, tk.END)
            for entry in self.playlist:
                self.playlist_listbox.insert(tk.END, entry['name'])
            self.playlist_listbox.selection_set(i-1)
            self.playlist_listbox.see(i-1)
        self._highlight_loaded()
    def _playlist_down(self):
        idx = self.playlist_listbox.curselection()
        if idx and idx[0] < len(self.playlist) - 1:
            i = idx[0]
            self.playlist[i], self.playlist[i+1] = self.playlist[i+1], self.playlist[i]
            self.playlist_listbox.delete(0, tk.END)
            for entry in self.playlist:
                self.playlist_listbox.insert(tk.END, entry['name'])
            self.playlist_listbox.selection_set(i+1)
            self.playlist_listbox.see(i+1)
        self._highlight_loaded()
    def _playlist_play_selected(self):
        was_playing = self.audio.is_playing() if self.audio else False
        idx = self.playlist_listbox.curselection()
        if idx:
            entry = self.playlist[idx[0]]
            self._open_file(entry['wav'])
            self._set_play(False)
            self.play_btn.configure(state=tk.NORMAL)
        if was_playing:
            self._set_play(True)
        self._highlight_loaded()
    def _playlist_shuffle(self):
        idxs = self.playlist_listbox.curselection()
        if idxs:
            idx = idxs[0]
        else:
            idx = None
        if idx is not None and 0 <= idx < len(self.playlist):
            current = self.playlist[idx]
            rest = self.playlist[:idx] + self.playlist[idx+1:]
            random.shuffle(rest)
            self.playlist = [current] + rest
            self.playlist_listbox.delete(0, tk.END)
            for entry in self.playlist:
                self.playlist_listbox.insert(tk.END, entry['name'])
            self.playlist_listbox.selection_clear(0, tk.END)
            self.playlist_listbox.selection_set(0)
        else:
            random.shuffle(self.playlist)
            self.playlist_listbox.delete(0, tk.END)
            for entry in self.playlist:
                self.playlist_listbox.insert(tk.END, entry['name'])
            self.playlist_listbox.selection_clear(0, tk.END)
        self._update_nav_buttons()
        self._highlight_loaded()
    def _playlist_clear(self):
        if self.audio is not None:
            self.audio.close()
            self.audio = None
            self._update_nav_buttons()
        for entry in self.playlist:
            try:
                os.remove(entry['wav'])
            except Exception:
                with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                    lf.write("Remove failed:\n")
                    traceback.print_exc(file=lf)
                    lf.flush()
                pass
        self.playlist.clear()
        self.playlist_listbox.delete(0, tk.END)
        self._set_play(False)
        self.displayname.set("<No file>")
        self.vis = None
        self.viz.delete("all")
        self.play_btn.configure(state="disabled")
        self.title("BetterWMP: <No file>")
        self._highlight_loaded()
    def _on_repeat_mode_change(self, val):
        self.repeat_mode.set(val)
        self.repeat_label.set(f"{val}  ▾")
    def _handle_track_end(self):
        mode = self.repeat_mode.get()
        idxs = [i for i, entry in enumerate(self.playlist) if entry['wav'] == getattr(self, "current_wav", None)]
        if not idxs:
            idx = 0
        else:
            idx = idxs[0]
        playlist_len = len(self.playlist)
        if mode == "track once":
            self._set_play(False)
            self.display_time = 0.0
            if self.audio is not None:
                self.audio.seek_seconds(0.0)
        elif mode == "track repeat":
            self._set_play(False)
            self.display_time = 0.0
            if self.audio is not None:
                self.audio.seek_seconds(0.0)
            self._set_play(True)
        elif mode == "playlist once":
            if idx + 1 < playlist_len:
                idx = idx + 1
                self._open_file(self.playlist[idx]['wav'])
                self.display_time = 0.0
                if self.audio is not None:
                    self.audio.seek_seconds(0.0)
                self._pending_play = True
            else:
                self._set_play(False)
                self.display_time = 0.0
                if self.audio is not None:
                    self.audio.seek_seconds(0.0)
        elif mode == "playlist repeat":
            if playlist_len > 0:
                idx = (idx + 1) % playlist_len
                self._open_file(self.playlist[idx]['wav'])
                self.display_time = 0.0
                if self.audio is not None:
                    self.audio.seek_seconds(0.0)
                self._pending_play = True
            else:
                idx = 0
                self._open_file(self.playlist[idx]['wav'])
                self.display_time = 0.0
                if self.audio is not None:
                    self.audio.seek_seconds(0.0)
                self._pending_play = True
        self._highlight_loaded()
    def _format_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ds = int((seconds % 1) * 10)
        if h > 0:
            return f"{h}:{m:02}:{s:02}.{ds}"
        return f"{m}:{s:02}.{ds}"
    def _set_play(self, flag: bool):
        if self.audio is None:
            return
        if flag:
            self.audio.play()
            self.play_btn.configure(text="⏸")
            self._pending_play = False
            self._user_paused = False
        else:
            self.audio.pause()
            self.play_btn.configure(text="▶")
            self._user_paused = True
            self._pending_play = False
    def _toggle_play(self):
        if self.audio is None:
            return
        is_now_playing = self.audio.toggle_play()
        self.play_btn.configure(text="⏸" if is_now_playing else "▶")
    def _mouse_x_to_time(self, event_x):
        if self.audio is None:
            return 0.0
        w = max(1, int(self.prog.winfo_width()))
        x = np.clip(event_x, 0, w - 1)
        frac = x / float(w - 1)
        return frac * self.audio.duration_seconds()
    def _on_prog_press(self, e):
        if self.audio is None:
            return
        self.was_playing_before_drag = self.audio.is_playing()
        self._set_play(False)
        self.is_dragging = True
        self.play_btn.configure(state="disabled")
        self.drag_target_time = self._mouse_x_to_time(e.x)
    def _on_prog_motion(self, e):
        if self.audio is None or not self.is_dragging:
            return
        self.drag_target_time = self._mouse_x_to_time(e.x)
    def _on_prog_release(self, e):
        if self.audio is None or not self.is_dragging:
            return
        self.is_dragging = False
        self.play_btn.configure(state="normal")
        self.audio.seek_seconds(self.display_time)
        if self.was_playing_before_drag:
            self._pending_seek = self.display_time
            self._set_play(True)
    def _draw_progress(self):
        self.prog.delete("all")
        w = max(1, int(self.prog.winfo_width()))
        h = max(1, int(self.prog.winfo_height()))
        self.prog.create_rectangle(0, 0, w, h, fill=SkinInfo["prog"]["bg"], outline="")
        if self.audio is None:
            return
        dur = max(1e-9, self.audio.duration_seconds())
        frac = np.clip(self.display_time / dur, 0.0, 1.0)
        self.prog.create_rectangle(0, 0, int(frac * w), h, fill=SkinInfo["prog"]["left"], outline="")
        x = int(frac * (w - 1))
        self.prog.create_rectangle(x - 2, 0, x + 2, h, fill=SkinInfo["prog"]["thumb"], outline="")
    def _draw_axes(self):
        self.viz.delete("grid")
        w = max(10, int(self.viz.winfo_width()))
        h = max(10, int(self.viz.winfo_height()))
        for db in range(int(self.db_min), int(self.db_max) + 1, 10):
            y = (1.0 - (db - self.db_min) / (self.db_max - self.db_min)) * (h - 1)
            self.viz.create_line(0, y, w, y, fill=SkinInfo["fft"]["line"], tags="grid")
            self.viz.create_text(4, y - 2, anchor='sw', fill=SkinInfo["fft"]["text"], text=f"{db} dB", tags="grid")
        if self.vis is not None:
            fmax = max(1000.0, self.vis.sr / 2.0)
        else:
            fmax = 22050.0
        decades = [20, 30, 40, 50, 60, 70, 80, 90, 
                   100, 200, 300, 400, 500, 600, 700, 800, 900, 
                   1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 
                   10000, 20000, 100000]
        for f in decades:
            if f < self.fmin or f > fmax:
                continue
            x = (np.log10(f) - np.log10(self.fmin)) / (np.log10(fmax) - np.log10(self.fmin)) * (w - 1)
            self.viz.create_line(x, 0, x, h, fill=SkinInfo["fft"]["line"], tags="grid")
            if str(f)[0] in '124':
                if f >= 1000:
                    if f < 15000:
                        self.viz.create_text(x + 2, 12, anchor='nw', fill=SkinInfo["fft"]["text"], text=f"{f // 1000} kHz", tags="grid")
                else:
                    self.viz.create_text(x + 2, 12, anchor='nw', fill=SkinInfo["fft"]["text"], text=f"{f} Hz", tags="grid")
    def _draw_spectrum(self, freqs, mid_db, side_db):
        self.viz.delete("spec")
        w = max(5, int(self.viz.winfo_width()))
        h = max(5, int(self.viz.winfo_height()))
        fmax = (self.vis.sr / 2.0) if self.vis else 22050.0
        mid_pts = self.vis.spectrum_to_polyline(freqs, mid_db, self.fmin, fmax, self.db_min, self.db_max, w, h)
        side_pts = self.vis.spectrum_to_polyline(freqs, side_db, self.fmin, fmax, self.db_min, self.db_max, w, h)
        if len(side_pts) > 1:
            self.viz.create_line(*sum(side_pts, ()), fill=self.side_color, width=1, tags="spec")
        if len(mid_pts) > 1:
            self.viz.create_line(*sum(mid_pts, ()), fill=self.mid_color, width=1, tags="spec")
    def _update_loop(self):
        global frames, EmergencyStop
        t0 = time.perf_counter()
        try:
            if not EmergencyStop:
                if self.audio is not None:
                    was_playing = self.audio.is_playing()
                    if self.is_dragging:
                        self.display_time += (self.drag_target_time - self.display_time) / 3.0
                        self.display_time = float(np.clip(self.display_time, 0.0, self.audio.duration_seconds()))
                    elif was_playing:
                        self.display_time = self.audio.current_seconds()
                    if self.audio is not None and self.vis is not None:
                        cur = self._format_time(self.display_time)
                        total = self._format_time(self.audio.duration_seconds())
                        self.timestamp_label.config(text=f"{cur} / {total}")
                    else:
                        self.timestamp_label.config(text="")
                    if hasattr(self, '_pending_play') and self._pending_play:
                        if not self.audio.is_playing():
                            self._set_play(True)
                            self.audio.play()
                if self._pending_seek is not None:
                    self.audio.seek_seconds(self._pending_seek)
                    self._pending_seek = None
                self._draw_progress()
                if self.vis is not None and self.audio is not None and not self._is_minimized():
                    buffer_n = int(self.buffer_var.get())
                    zp = int(self.zp_var.get())
                    mids = self.vis.list_for_fft(self.display_time, 'm', buffer_n)
                    sides = self.vis.list_for_fft(self.display_time, 's', buffer_n)
                    freqs, mid_db = self.vis.fft_db(mids, zp)
                    _, side_db = self.vis.fft_db(sides, zp)
                    mask = (freqs >= self.fmin) & (freqs <= (self.vis.sr / 2.0))
                    freqs = freqs[mask]
                    mid_db = mid_db[mask]
                    side_db = side_db[mask]
                    self._draw_axes()
                    self._draw_spectrum(freqs, mid_db, side_db)
            else:
                raise Exception("EMERGENCY STOP")
        except Exception as e:
            if EmergencyStop:
                raise e
            traceback.print_exc()
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("Exception in update loop:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            print(f"Error in update loop: {e}")
            pass
        try:
            self._highlight_loaded()
            if frames % 7 == 0:
                self._update_nav_buttons()
            if frames % 83 == 0:
                self.bind("<Map>", lambda e: self.drop_target_register(DND_FILES))
            if frames % 293 == 0:
                self.drop_target_register(DND_FILES)
        except Exception:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("Exception in periodic tasks:\n")
                traceback.print_exc(file=lf)
                lf.flush()
            pass
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        if self._is_minimized():
            delay_minuend = 100
        else:
            delay_minuend = 16
        delay = max(0, int(delay_minuend - elapsed_ms))
        frames += 1
        self.after(delay, self._update_loop)
    def _on_close(self):
        if self.audio is not None:
            self.audio.pause()
        shutil.rmtree(self.appdata_dir, ignore_errors=True)
        with open(FAULT_LOG, "a", encoding="utf-8") as lf:
            lf.write("The last instance of BetterWMP has been successful.\n")
            lf.flush()
        self.destroy()

def setup_skin_json():
    conf_dir = os.path.expandvars(r"%localappdata%\betterwmpconf")
    os.makedirs(conf_dir, exist_ok=True)
    skin_path = os.path.join(conf_dir, "default.bwmpskin")
    pointer_path = os.path.join(conf_dir, "skinpointer.conf")
    if not os.path.isfile(pointer_path):
        with open(pointer_path, "w", encoding="utf-8") as pf:
            os.remove(skin_path)
            pf.write("default.bwmpskin")
    if not os.path.isfile(skin_path):
        print("Reset")
        with open(FAULT_LOG, "a", encoding="utf-8") as lf:
            lf.write("Skin reset to default.\n")
            lf.flush()
        skin_data = {
                    "fft":
                    {
                        "bg": "#0f0f0f",
                        "fg": "#aaaaaa",
                        "mid": "#b25aff",
                        "side": "#5f405d",
                        "line": "#232333",
                        "text": "#777777"
                    },
                    "prog":
                    {
                        "bg": "#1a1a1a",
                        "left": "#2d1d3d",
                        "thumb": "#d88aff"
                    },
                    "tkinter":
                    {
                        "bg": "#333333",
                        "label": "#ffffff",
                        "buttonbg": "#555555",
                        "buttonfg": "#ffffff",
                        "buttonactivebg": "#685868",
                        "buttonactivefg": "#ffffff",
                        "buttonpressbg": "#222222",
                        "buttonpressfg": "#ffffff",
                        "buttondisabledfg": "#777777",
                        "listboxbg": "#111111",
                        "listboxfg": "#ffffff",
                        "listboxselbg": "#342742",
                        "listboxselfg": "#ffffff",
                        "loadedfg": "#fca4ff",
                        "doublefg": "#ffc1f6",
                        "dropdownbg": "#555555",
                        "dropdownfg": "#ffffff",
                        "dropdownactivebg": "#685868",
                        "dropdownactivefg": "#ffffff"
                    }
                }
        with open(skin_path, "w", encoding="utf-8") as f:
            json.dump(skin_data, f, indent=4)
    if not os.path.isfile(pointer_path):
        with open(pointer_path, "w", encoding="utf-8") as pf:
            pf.write("default.bwmpskin")

def get_skin():
    conf_dir = os.path.expandvars(r"%localappdata%\betterwmpconf")
    pointer_path = os.path.join(conf_dir, "skinpointer.conf")
    if not os.path.isfile(pointer_path):
        return {}
    with open(pointer_path, "r", encoding="utf-8") as pf:
        skin_file = pf.read().strip()
    skin_path = os.path.join(conf_dir, skin_file)
    if not os.path.isfile(skin_path):
        return {}
    with open(skin_path, "r", encoding="utf-8") as sf:
        return json.load(sf)
    
SkinInfo = {}
app = None
frames = 1
EmergencyStop = False
DEBUG = False
def main():
    global SkinInfo, app, frames, EmergencyStop, DEBUG
    setup_skin_json()
    SkinInfo = get_skin()
    app = None
    try:
        app = BetterWMP()
        app.report_callback_exception = tkinter_exception_handler.__get__(app, type(app))
    except Exception as e:
        if 'tkinter' in str(e).lower():
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                    lf.write("Skin failed:\n")
                    traceback.print_exc(file=lf)
                    lf.flush()
            ctypes.windll.user32.MessageBoxW(0,"The skin configuration is probably corrupted.\nSkin pointer will reset upon next launch.", "Error from BetterWMP", 0x10)
            pointer = r"%localappdata%\betterwmpconf\skinpointer.conf"
            os.remove(os.path.expandvars(pointer))
            sys.exit(1)
        else:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                    lf.write("Startup failed:\n")
                    traceback.print_exc(file=lf)
                    lf.flush()
            traceback.print_exc()
            show_native_messagebox("Error from BetterWMP", f"An error occurred:\n{e}")
            sys.exit(1)
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        try:
            app._playlist_append_sysargv(files)
        except Exception:
            with open(FAULT_LOG, "a", encoding="utf-8") as lf:
                lf.write("Exception in command line argument processing:\n")
                traceback.print_exc(file=lf)
                lf.flush()
    try:
        app.mainloop()
    except Exception as e:
        with open(FAULT_LOG, "a") as lf:
            traceback.print_exc(file=lf)
        with open(FAULT_LOG, "r") as lf:
            fault_text = lf.read()
        show_native_messagebox("BetterWMP has crashed", fault_text)
        sys.exit(1)

if __name__ == "__main__":
    main()
