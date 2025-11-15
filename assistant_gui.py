import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import threading
import queue
import time
import datetime
import sqlite3
import requests
import webbrowser
import re
import traceback

from dotenv import load_dotenv
load_dotenv()  # loads .env if present

# Third-party libraries
try:
    import pyttsx3
except Exception:
    raise ImportError("pyttsx3 required: pip install pyttsx3")

try:
    import speech_recognition as sr
except Exception:
    sr = None  # optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

import dateparser
import wikipedia
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog

# Keep an OpenAI import harmless (not used here) to avoid breaking older flows
try:
    from openai import OpenAI  # optional, unused
except Exception:
    OpenAI = None

# ---------------- Config ----------------
ASSISTANT_NAME = "SUNDA.I"
DEFAULT_CITY = "New Delhi"
DB_PATH = "assistant_reminders.db"

# ---------------- TTS ----------------
engine = pyttsx3.init()
voices = engine.getProperty("voices")
default_rate = engine.getProperty("rate")


def set_tts_voice(index: int):
    try:
        engine.setProperty("voice", voices[index].id)
    except Exception:
        pass


def set_tts_rate(rate: int):
    try:
        engine.setProperty("rate", int(rate))
    except Exception:
        pass


# ---------------- Queues ----------------
chat_queue = queue.Queue()
log_queue = queue.Queue()


def append_chat(text: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    chat_queue.put((ts, text))


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    log_queue.put(f"[{ts}] {msg}")


def speak(text: str):
    append_chat(f"{ASSISTANT_NAME}: {text}")
    def _tts():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            log(f"TTS error: {e}")
    threading.Thread(target=_tts, daemon=True).start()


# ---------------- DB & Scheduler ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    time_iso TEXT NOT NULL,
    fired INTEGER DEFAULT 0
)
""")
conn.commit()

scheduler = BackgroundScheduler()
scheduler.start()


def _fire_reminder(reminder_id: int, message: str):
    append_chat(f"Reminder: {message}")
    try:
        cur.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
        conn.commit()
    except Exception as e:
        log(f"DB update error: {e}")


def schedule_reminder(dt: datetime.datetime, message: str, reminder_id: int):
    try:
        scheduler.add_job(lambda: _fire_reminder(reminder_id, message),
                          trigger="date", run_date=dt, id=f"rem_{reminder_id}")
        log(f"Scheduled reminder {reminder_id} at {dt.isoformat()}")
    except Exception as e:
        log(f"Schedule error: {e}")


def load_and_schedule_pending_reminders():
    try:
        cur.execute("SELECT id, message, time_iso FROM reminders WHERE fired = 0")
        rows = cur.fetchall()
        now = datetime.datetime.now()
        for rid, msg, time_iso in rows:
            try:
                dt = datetime.datetime.fromisoformat(time_iso)
            except Exception:
                dt = dateparser.parse(time_iso)
            if not dt:
                continue
            if dt <= now:
                threading.Thread(target=_fire_reminder, args=(rid, msg), daemon=True).start()
            else:
                try:
                    if not scheduler.get_job(f"rem_{rid}"):
                        schedule_reminder(dt, msg, rid)
                except Exception as e:
                    log(f"Schedule load error: {e}")
    except Exception as e:
        log(f"Load reminders error: {e}")


load_and_schedule_pending_reminders()

# ---------------- Weather (Open-Meteo) ----------------
def get_current_weather(city: str = DEFAULT_CITY) -> str:
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=8).json()
        if "results" not in geo_resp or len(geo_resp["results"]) == 0:
            return f"City '{city}' not found."
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        cname = geo_resp["results"][0]["name"]
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_resp = requests.get(weather_url, params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True
        }, timeout=8).json()
        if "current_weather" not in weather_resp:
            return f"Weather data unavailable for {cname}"
        cw = weather_resp["current_weather"]
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        code = cw.get("weathercode")
        weather_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Rain showers", 81: "Moderate rain showers", 82: "Violent rain showers"
        }
        desc = weather_map.get(code, "Unknown weather")
        return f"Weather in {cname}: {desc}. Temperature {temp}°C. Wind {wind} km/h."
    except Exception as e:
        return f"Weather error: {e}"

# ---------------- Fetch text from URL ----------------
def fetch_text_from_url(url: str, max_chars: int = 15000) -> str:
    try:
        headers = {"User-Agent": f"Mozilla/5.0 (compatible; {ASSISTANT_NAME}/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for s in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            s.decompose()
        article = soup.find("article")
        if article:
            text = article.get_text(separator="\n", strip=True)
        else:
            parts = []
            for tag in soup.find_all(["h1", "h2", "h3", "p"]):
                t = tag.get_text(separator=" ", strip=True)
                if t and len(t.split()) > 3:
                    parts.append(t)
            text = "\n\n".join(parts)
            if not text:
                text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n\s+\n", "\n\n", text)
        return text[:max_chars]
    except Exception as e:
        return f"Fetch error: {e}"

# ---------------- Gemini via SDK (SAFE) ----------------
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # change if your account uses another name

def get_gemini_api_key():
    # prefer GEMINI_API_KEY, fall back to GOOGLE_API_KEY if user set that
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None

# We only use the official SDK path. If it's not installed or the key missing/invalid, we fallback to Wikipedia.
try:
    from google import genai  # optional SDK
    _genai_available = True
except Exception:
    genai = None
    _genai_available = False

def answer_with_gemini_chat(question: str, context: str = None, model: str = DEFAULT_GEMINI_MODEL, max_output_tokens: int = 512) -> str:
    """
    Use google-genai SDK if available and GEMINI_API_KEY present.
    If SDK or key is missing, raise an exception so the caller can fallback.
    This avoids making REST calls that may 404 for incorrect endpoints.
    """
    prompt = question if not context else f"Context:\n{context}\n\nQuestion: {question}"
    api_key = get_gemini_api_key()

    if not _genai_available:
        raise RuntimeError("google-genai SDK not installed. Install with: pip install google-genai")

    # If SDK is available but no key set, try letting the SDK pick credentials from environment.
    try:
        # Create client with explicit key if provided, else default constructor.
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
    except Exception as e:
        # Fail early and clearly
        raise RuntimeError(f"Could not construct Gemini SDK client: {e}")

    # Try a single SDK pattern (models.generate_content). This matches current genai SDKs.
    try:
        # Some SDK versions accept `max_output_tokens` keyword, others may not.
        try:
            resp = client.models.generate_content(model=model, contents=prompt, max_output_tokens=max_output_tokens)
        except TypeError:
            resp = client.models.generate_content(model=model, contents=prompt)

        # Best-effort extraction of text
        # SDK response shapes vary; try common attributes
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "output") and resp.output:
            # join list fragments if present
            parts = []
            for item in resp.output:
                if isinstance(item, dict):
                    parts.append(item.get("content") or item.get("text") or str(item))
                else:
                    # object-like
                    parts.append(getattr(item, "content", None) or getattr(item, "text", None) or str(item))
            if parts:
                return "\n".join(parts)
        # fallback to stringification
        return str(resp)
    except Exception as e:
        # Surface a clear error to allow graceful fallback upstream
        raise RuntimeError(f"Gemini SDK call failed: {e}")

# ---------------- NLP (Gemini-first, safe fallback to Wikipedia) ----------------
def nlp_answer(question: str, context: str = None) -> str:
    # Try Gemini SDK if available and configured
    try:
        # attempt only if SDK present
        if _genai_available:
            try:
                # if nothing wrong, return Gemini answer
                return answer_with_gemini_chat(question, context=context)
            except Exception as e:
                log(f"Gemini attempt failed: {e}")
        else:
            log("google-genai SDK not installed; skipping Gemini attempt.")
    except Exception as e:
        # Catch-all safety
        log(f"Unexpected error in Gemini path: {e}")

    # Wikipedia fallback for definitional queries
    ql = question.strip().lower()
    if ql.startswith(("what is", "who is", "tell me about", "define")):
        try:
            topic = re.sub(r'^(what is|who is|tell me about|define)\s+', '', ql)
            if topic:
                try:
                    return wikipedia.summary(topic, sentences=2)
                except Exception:
                    return wikipedia.summary(topic.split(" ")[0], sentences=2)
        except Exception as e:
            log(f"Wikipedia fallback failed: {e}")

    # Generic fallback message (no crash)
    return ("Sorry — I couldn't use Gemini (SDK missing or key not set). "
            "Try installing the google-genai SDK (`pip install google-genai`) and set GEMINI_API_KEY in the Settings, "
            "or ask a Wikipedia-style question (e.g. 'What is X?').")

# ---------------- Short-circuit and parsing ----------------
REMINDER_PATTERNS = [
    re.compile(r"^set reminder (?:to )?(?P<msg>.+?) at (?P<time>.+)$", re.I),
    re.compile(r"^remind me to (?P<msg>.+?) (?:at|on|by) (?P<time>.+)$", re.I),
    re.compile(r"^remind me to (?P<msg>.+)$", re.I),
]

def parse_reminder_from_text(text: str):
    text = text.strip()
    for pat in REMINDER_PATTERNS:
        m = pat.match(text)
        if m:
            gd = m.groupdict()
            msg = gd.get("msg") or ""
            time_txt = gd.get("time") or ""
            return msg.strip(), time_txt.strip()
    return None, None

# ---------------- Voice helpers ----------------
if sr is not None:
    recognizer = sr.Recognizer()
else:
    recognizer = None

def recognize_audio_short(timeout: int = 3, phrase_time_limit: int = 3):
    if recognizer is None:
        return None
    try:
        with sr.Microphone() as source:
            recognizer.pause_threshold = 0.3
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            return recognizer.recognize_google(audio, language="en-in").lower()
    except Exception:
        return None

def take_command_voice(timeout: int = 6, phrase_time_limit: int = 8):
    if recognizer is None:
        return None
    try:
        with sr.Microphone() as source:
            append_chat("<listening...>")
            recognizer.pause_threshold = 0.7
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            append_chat("<recognizing...>")
            cmd = recognizer.recognize_google(audio, language="en-in")
            append_chat(f"You: {cmd}")
            return cmd.lower()
    except Exception as e:
        append_chat(f"Voice input failed: {e}")
        return None

# ---------------- Hotword ----------------
hotword_enabled = False
hotword_phrase = "hey sunda i"

def hotword_listener():
    global hotword_enabled
    log("Hotword listener started.")
    while hotword_enabled:
        try:
            text = recognize_audio_short(timeout=2, phrase_time_limit=2)
            if text:
                log(f"(hotword) heard: {text}")
                if hotword_phrase in text or "hey sunda" in text or "sunda i" in text:
                    speak("Yes?")
                    cmd = take_command_voice(timeout=6, phrase_time_limit=10)
                    if cmd:
                        handle_query_text(cmd)
        except Exception as e:
            log(f"Hotword error: {e}")
        time.sleep(0.2)
    log("Hotword listener stopped.")

# ---------------- Core handler ----------------
def handle_query_text(query: str, gui_context: str = None):
    if not query:
        return
    q = query.strip()
    ql = q.lower()

    # identity & small greetings
    if any(kw in ql for kw in ("your name", "who are you", "what is your name")):
        speak(f"My name is {ASSISTANT_NAME}.")
        return
    if ql in ("hello", "hi", "hey"):
        speak(f"Hello — I am {ASSISTANT_NAME}. How can I help?")
        return

    # direct reminder parsing
    msg, time_txt = parse_reminder_from_text(q)
    if msg and time_txt:
        dt = dateparser.parse(time_txt, settings={"PREFER_DATES_FROM": "future"})
        if not dt:
            speak("I couldn't parse the time. Please provide a clearer time.")
            return
        try:
            cur.execute("INSERT INTO reminders (message, time_iso) VALUES (?, ?)", (msg, dt.isoformat()))
            conn.commit()
            rid = cur.lastrowid
            schedule_reminder(dt, msg, rid)
            speak(f"Reminder set for {dt.strftime('%Y-%m-%d %H:%M')} (id {rid})")
        except Exception as e:
            speak(f"Failed to set reminder: {e}")
        return

    # weather
    if "weather" in ql:
        city = DEFAULT_CITY
        m = re.search(r"weather(?:.*in\s+([a-zA-Z\s]+))", ql)
        if m and m.group(1):
            city = m.group(1).strip()
        else:
            parts = q.split()
            if len(parts) > 1:
                idxs = [i for i, p in enumerate(parts) if p.lower().startswith("weather")]
                if idxs:
                    rest = parts[idxs[0] + 1: idxs[0] + 4]
                    if rest:
                        city = " ".join(rest).strip()
        speak(f"Fetching weather for {city}")
        wx = get_current_weather(city)
        speak(wx)
        return

    # wikipedia
    if "wikipedia" in ql:
        try:
            s = ql.replace("wikipedia", "").strip()
            if not s:
                speak("What should I search on Wikipedia?")
                return
            result = wikipedia.summary(s, sentences=3)
            speak(result)
        except Exception as e:
            speak("Wikipedia error: " + str(e))
        return

    # interactive reminder
    if ql in ("set reminder", "remind me"):
        speak("What should I remind you about?")
        rmsg = take_command_voice()
        if not rmsg:
            speak("Please type the reminder in the GUI.")
            return
        speak("When should I remind you?")
        rtime = take_command_voice()
        if not rtime:
            speak("Please type the time in the GUI.")
            return
        dt = dateparser.parse(rtime, settings={"PREFER_DATES_FROM": "future"})
        if not dt:
            speak("Couldn't parse the time. Please type it in the GUI.")
            return
        cur.execute("INSERT INTO reminders (message, time_iso) VALUES (?, ?)", (rmsg, dt.isoformat()))
        conn.commit()
        rid = cur.lastrowid
        schedule_reminder(dt, rmsg, rid)
        speak(f"Reminder set for {dt.strftime('%Y-%m-%d %H:%M')} (id {rid})")
        return

    # list reminders
    if "list reminders" in ql:
        try:
            cur.execute("SELECT id, message, time_iso, fired FROM reminders ORDER BY time_iso")
            rows = cur.fetchall()
            if not rows:
                speak("No reminders.")
                return
            speak("Listing reminders in logs.")
            out = []
            for r in rows:
                status = "fired" if r[3] else "pending"
                out.append(f"[{r[0]}] {r[2]} — {r[1]} ({status})")
            log("\n".join(out))
        except Exception as e:
            speak("Could not list reminders: " + str(e))
        return

    # cancel reminder
    if "cancel reminder" in ql or ql.startswith("cancel"):
        digits = "".join(ch for ch in q if ch.isdigit())
        if digits:
            rid = int(digits)
            try:
                try:
                    scheduler.remove_job(f"rem_{rid}")
                except JobLookupError:
                    pass
                cur.execute("DELETE FROM reminders WHERE id = ?", (rid,))
                conn.commit()
                speak(f"Cancelled reminder {rid}")
            except Exception as e:
                speak("Cancel error: " + str(e))
        else:
            speak("Please provide the reminder id to cancel.")
        return

    # NLP / QA
    if any(kw in ql for kw in ("ask", "explain", "question", "what is", "who is", "how to", "why is")) or ql.endswith("?"):
        context = gui_context
        if not context and recognizer is not None:
            speak("Do you want to provide context (webpage or text)? Say yes or no.")
            ans = take_command_voice(timeout=4, phrase_time_limit=4)
            if ans and "yes" in ans.lower():
                speak("Please speak or paste the context.")
                ctx = take_command_voice(timeout=10, phrase_time_limit=40)
                if ctx:
                    context = ctx
        speak("Searching for answer...")
        ans_text = nlp_answer(q, context)
        speak(ans_text)
        return

    # fallback web search
    speak("I didn't understand. Should I search the web for that?")
    ans = take_command_voice(timeout=4, phrase_time_limit=4)
    if ans and "yes" in ans.lower():
        webbrowser.open("https://www.google.com/search?q=" + requests.utils.requote_uri(query))
    else:
        speak("Okay.")

# ---------------- GUI ----------------
class AssistantGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{ASSISTANT_NAME} — Control Panel")
        self.root.geometry("980x680")
        self.gui_context_text = None
        self.create_widgets()
        self.update_loop()

    def create_widgets(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        self.start_btn = ttk.Button(top, text="Start Hotword", command=self.toggle_hotword)
        self.start_btn.pack(side="left")
        ttk.Button(top, text="Run Command (voice)", command=self.run_voice_command).pack(side="left", padx=6)
        ttk.Button(top, text="Run Command (type)", command=self.run_typed_command).pack(side="left", padx=6)
        ttk.Button(top, text="List Reminders", command=self.gui_list_reminders).pack(side="left", padx=6)
        ttk.Button(top, text="Open Logs Folder", command=self.open_logs_folder).pack(side="left", padx=6)
        ttk.Button(top, text="Settings (API Key)", command=self.open_settings).pack(side="left", padx=6)

        voice_frame = ttk.Frame(self.root)
        voice_frame.pack(fill="x", pady=6, padx=8)
        ttk.Label(voice_frame, text="Voice:").pack(side="left")
        voice_names = [getattr(v, "name", getattr(v, "id", str(i))) for i, v in enumerate(voices)]
        self.voice_combo = ttk.Combobox(voice_frame, values=voice_names, state="readonly", width=40)
        try:
            self.voice_combo.current(0)
        except Exception:
            pass
        self.voice_combo.pack(side="left", padx=6)
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_change)
        ttk.Label(voice_frame, text="Rate:").pack(side="left", padx=6)
        self.rate_var = tk.IntVar(value=default_rate)
        self.rate_slider = ttk.Scale(voice_frame, from_=80, to=250, variable=self.rate_var, command=self.on_rate_change)
        self.rate_slider.pack(side="left", padx=6, fill="x", expand=True)

        url_frame = ttk.Frame(self.root, padding=6)
        url_frame.pack(fill="x", padx=8)
        ttk.Label(url_frame, text="URL for context:").pack(side="left")
        self.url_entry = ttk.Entry(url_frame)
        self.url_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(url_frame, text="Fetch & Use", command=self.fetch_url_context).pack(side="left")

        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)

        chat_frame = ttk.Frame(panes, width=600)
        panes.add(chat_frame, weight=3)
        ttk.Label(chat_frame, text="Chat").pack(anchor="w")
        self.chat_area = scrolledtext.ScrolledText(chat_frame, state="disabled", wrap="word", height=20)
        self.chat_area.pack(fill="both", expand=True)
        chat_entry_frame = ttk.Frame(chat_frame)
        chat_entry_frame.pack(fill="x", pady=6)
        self.chat_entry = ttk.Entry(chat_entry_frame)
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(chat_entry_frame, text="Send", command=self.send_chat).pack(side="left")

        log_frame = ttk.Frame(panes, width=350)
        panes.add(log_frame, weight=1)
        ttk.Label(log_frame, text="Logs").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled", wrap="word")
        self.log_area.pack(fill="both", expand=True)

        bottom = ttk.Frame(self.root, padding=6)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Speak greeting", command=lambda: speak(f"Hello, I am {ASSISTANT_NAME}")).pack(side="left")
        ttk.Button(bottom, text="Open Google", command=lambda: webbrowser.open("https://google.com")).pack(side="left", padx=6)
        ttk.Button(bottom, text="Exit", command=self.on_exit).pack(side="right")

    def update_loop(self):
        while not chat_queue.empty():
            ts, text = chat_queue.get_nowait()
            self.chat_area.configure(state="normal")
            self.chat_area.insert("end", f"[{ts}] {text}\n")
            self.chat_area.see("end")
            self.chat_area.configure(state="disabled")
        while not log_queue.empty():
            item = log_queue.get_nowait()
            self.log_area.configure(state="normal")
            self.log_area.insert("end", item + "\n")
            self.log_area.see("end")
            self.log_area.configure(state="disabled")
        self.root.after(200, self.update_loop)

    def toggle_hotword(self):
        global hotword_enabled
        hotword_enabled = not hotword_enabled
        if hotword_enabled:
            self.start_btn.config(text="Stop Hotword")
            threading.Thread(target=hotword_listener, daemon=True).start()
        else:
            self.start_btn.config(text="Start Hotword")

    def run_voice_command(self):
        speak("Listening for your command now.")
        cmd = take_command_voice(timeout=8, phrase_time_limit=12)
        if not cmd:
            speak("I couldn't hear a command.")
            return
        handle_query_text(cmd, gui_context=self.gui_context_text)

    def run_typed_command(self):
        cmd = self.chat_entry.get().strip()
        if not cmd:
            speak("Type a command first.")
            return
        append_chat(f"You: {cmd}")
        self.chat_entry.delete(0, "end")
        handle_query_text(cmd, gui_context=self.gui_context_text)

    def fetch_url_context(self):
        url = self.url_entry.get().strip()
        if not url:
            speak("Paste a URL first.")
            return
        speak("Fetching page. This may take a few seconds.")
        txt = fetch_text_from_url(url)
        if txt.startswith("Fetch error:"):
            speak(txt)
            return
        self.gui_context_text = txt
        preview = txt[:400].replace("\n", " ") + ("..." if len(txt) > 400 else "")
        log(f"Fetched context preview: {preview}")
        speak("Context fetched and ready for NLP questions.")

    def send_chat(self):
        q = self.chat_entry.get().strip()
        if not q:
            return
        append_chat(f"You: {q}")
        self.chat_entry.delete(0, "end")
        lowered = q.lower()
        if any(lowered.startswith(k) for k in ("set reminder", "remind me", "weather", "wikipedia", "list reminders", "cancel reminder", "open google")):
            handle_query_text(q, gui_context=self.gui_context_text)
        else:
            append_chat(f"{ASSISTANT_NAME}: thinking...")
            def do_nlp():
                ans = nlp_answer(q, self.gui_context_text)
                append_chat(f"{ASSISTANT_NAME}: {ans}")
                speak(ans)
            threading.Thread(target=do_nlp, daemon=True).start()

    def gui_list_reminders(self):
        try:
            cur.execute("SELECT id, message, time_iso, fired FROM reminders ORDER BY time_iso")
            rows = cur.fetchall()
            if not rows:
                messagebox.showinfo("Reminders", "No reminders.")
                return
            out = "\n".join([f"[{r[0]}] {r[2]} — {r[1]} ({'fired' if r[3] else 'pending'})" for r in rows])
            messagebox.showinfo("Reminders", out)
        except Exception as e:
            speak("Could not read reminders: " + str(e))

    def open_logs_folder(self):
        folder = os.getcwd()
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform.startswith("darwin"):
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception:
            speak("Could not open folder.")

    def on_voice_change(self, *args):
        idx = self.voice_combo.current()
        try:
            set_tts_voice(idx)
            speak("Voice changed.")
        except Exception as e:
            log(f"Voice set error: {e}")

    def on_rate_change(self, val):
        set_tts_rate(float(val))

    def open_settings(self):
        # Simple dialog to input and optionally save GEMINI key to .env
        def do_save(key_text, save_to_env):
            key_text = key_text.strip()
            if not key_text:
                messagebox.showinfo("API Key", "No key entered.")
                return
            os.environ["GEMINI_API_KEY"] = key_text
            messagebox.showinfo("API Key", "Gemini API key set for this session.")
            if save_to_env:
                try:
                    with open(".env", "a", encoding="utf-8") as f:
                        f.write(f"\nGEMINI_API_KEY={key_text}\n")
                    messagebox.showinfo("API Key", "Saved to .env file.")
                except Exception as e:
                    messagebox.showerror("Save error", str(e))
            settings_win.destroy()

        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings — GEMINI API Key")
        settings_win.geometry("520x160")
        ttk.Label(settings_win, text="Enter Gemini API Key (or leave empty to use environment):").pack(anchor="w", padx=8, pady=6)
        key_entry = ttk.Entry(settings_win, width=80)
        key_entry.pack(fill="x", padx=8)
        save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_win, text="Save key to .env (local file)", variable=save_var).pack(anchor="w", padx=8, pady=6)
        btn_frame = ttk.Frame(settings_win)
        btn_frame.pack(fill="x", pady=6, padx=8)
        ttk.Button(btn_frame, text="Set Key (session)", command=lambda: do_save(key_entry.get(), False)).pack(side="left")
        ttk.Button(btn_frame, text="Set & Save (.env)", command=lambda: do_save(key_entry.get(), True)).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Close", command=settings_win.destroy).pack(side="right")

    def on_exit(self):
        if messagebox.askokcancel("Exit", "Stop SUNDA.I and exit?"):
            try:
                scheduler.shutdown(wait=False)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            try:
                self.root.destroy()
            except Exception:
                pass
            os._exit(0)


# ---------------- Startup ----------------
def init_defaults():
    set_tts_rate(default_rate)
    try:
        set_tts_voice(0)
    except Exception:
        pass

def start_gui_app():
    init_defaults()
    root = None
    try:
        root = tk.Tk()
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.after(200, lambda: root.attributes("-topmost", False))
        except Exception:
            pass
        app = AssistantGUI(root)
        def greet():
            time.sleep(0.6)
            speak(f"Hello, I am {ASSISTANT_NAME}. The assistant is ready.")
        threading.Thread(target=greet, daemon=True).start()
        try:
            root.mainloop()
        except KeyboardInterrupt:
            log("GUI interrupted by user.")
        except Exception:
            tb = traceback.format_exc()
            log("Unhandled exception in GUI mainloop:\n" + tb)
            try:
                messagebox.showerror(ASSISTANT_NAME, "An unexpected error occurred. See logs for details.")
            except Exception:
                pass
    finally:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            if root:
                root.destroy()
        except Exception:
            pass

def cli_loop():
    speak(f"{ASSISTANT_NAME} command-line mode. Type 'help' for commands.")
    while True:
        try:
            cmd = input(">> ").strip()
        except Exception:
            log("Input not available in this environment. Exiting CLI loop.")
            break
        if cmd in ("exit", "quit", "bye"):
            speak("Goodbye")
            break
        if cmd == "help":
            print("Commands: weather <city>, remind, list, cancel <id>, ask, fetch <url>, exit")
            continue
        if cmd.startswith("weather"):
            parts = cmd.split(maxsplit=1)
            city = parts[1] if len(parts) > 1 else DEFAULT_CITY
            print(get_current_weather(city))
            continue
        if cmd.startswith("fetch "):
            url = cmd.split(" ", 1)[1]
            print(fetch_text_from_url(url)[:2000])
            continue
        if cmd == "list":
            try:
                cur.execute("SELECT id, message, time_iso, fired FROM reminders ORDER BY time_iso")
                rows = cur.fetchall()
                for r in rows:
                    print(f"[{r[0]}] {r[2]} - {r[1]} ({'fired' if r[3] else 'pending'})")
            except Exception as e:
                print("DB error:", e)
            continue
        print("Unknown command. Type help.")

if __name__ == "__main__":
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            print("Detected Jupyter — run this script from a terminal to open the GUI.")
        else:
            if os.environ.get("SUNDA_CLI", "0") == "1":
                init_defaults()
                cli_loop()
            else:
                start_gui_app()
    except Exception:
        try:
            start_gui_app()
        except Exception as e:
            log(f"Failed to start GUI: {e}")
            try:
                scheduler.shutdown(wait=False)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass