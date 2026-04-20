import math
import os
import json
import html
import re
from datetime import datetime
from typing import Optional

import httpx
import streamlit as st
import streamlit.runtime as st_runtime
from google import genai
from google.genai import types


def _load_local_env(path: str = ".env") -> None:
    """Simple .env loader: does not overwrite keys that already exist in the environment."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.lower().startswith("export "):
                    key = key[7:].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
                    value = value[1:-1]
                if not key:
                    continue
                # Let .env values override older environment variables for Gemini keys.
                if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "CHAT_GEMINI_API_KEY", "RECO_GEMINI_API_KEY"):
                    os.environ[key] = value
                elif key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Do not stop the app if .env cannot be read.
        pass


def _normalize_api_key(raw: str) -> str:
    key = (raw or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ('"', "'"):
        key = key[1:-1].strip()
    return key


def _has_streamlit_secrets_file() -> bool:
    # Check whether the secrets file exists before reading st.secrets; otherwise Streamlit prints a console warning.
    home = os.path.expanduser("~")
    cwd = os.getcwd()
    candidates = [
        os.path.join(home, ".streamlit", "secrets.toml"),
        os.path.join(cwd, ".streamlit", "secrets.toml"),
        f"{home}.streamlit\\secrets.toml",
        f"{cwd}.streamlit\\secrets.toml",
    ]
    return any(os.path.isfile(path) for path in candidates)


def _resolve_named_gemini_api_key(primary_names: tuple[str, ...], fallback_names: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")) -> str:
    for env_name in primary_names:
        env_key = _normalize_api_key(os.getenv(env_name, ""))
        if env_key:
            return env_key
    for env_name in fallback_names:
        env_key = _normalize_api_key(os.getenv(env_name, ""))
        if env_key:
            return env_key
    if _has_streamlit_secrets_file():
        try:
            for secret_name in primary_names:
                secret_key = _normalize_api_key(st.secrets.get(secret_name, ""))
                if secret_key:
                    return secret_key
            for secret_name in fallback_names:
                secret_key = _normalize_api_key(st.secrets.get(secret_name, ""))
                if secret_key:
                    return secret_key
        except Exception:
            pass
    return ""


def _build_gemini_client(api_key: str) -> Optional[genai.Client]:
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def _looks_like_gemini_key(api_key: str) -> bool:
    key = _normalize_api_key(api_key)
    return key.startswith("AIza") and len(key) >= 30


def _build_recommendation_hint(temperature, weather_code, wind_speed, humidity) -> str:
    if weather_code in (61, 63, 65, 80, 81, 82):
        return "Carry an umbrella."
    if weather_code in (71, 73, 75):
        return "Dress warmly."
    if temperature >= 30:
        return "Wear light colors and drink water."
    if 24 <= temperature < 30:
        return "Choose light clothes and keep water with you."
    if temperature <= 8:
        return "Wear a jacket."
    if wind_speed >= 35:
        return "Watch out for strong wind."
    if humidity >= 80 and temperature >= 22:
        return "It may feel muggy, wear breathable clothes."
    if weather_code in (0, 1, 2):
        return "Dress comfortably; it can be a good time to be outside."
    return "Give a short, simple, practical suggestion."


def _sanitize_ai_text(text: str, max_lines: int = 2) -> str:
    cleaned = (text or "").strip().replace("\r", "\n")
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]+\s*", "", line)
        line = re.sub(r"^\d+[\).]\s*", "", line)
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    result = "\n".join(lines[:max_lines]).strip()
    if result and result[-1] not in ".!?…":
        result += "."
    return result


def _strip_location_mentions(text: str, location: str = "") -> str:
    cleaned = text or ""
    if not location:
        return cleaned
    location_name = location.split(",", 1)[0].strip()
    if not location_name:
        return cleaned
    patterns = [
        re.escape(location_name),
        re.escape(location_name.replace(" ", "")),
        re.escape(location_name.split()[0]),
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern + r"(?:'?[a-zA-ZçğıöşüÇĞİÖŞÜ]*)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^([,.;:\-\s]+)", "", cleaned)
    cleaned = re.sub(r"([,.;:\-\s]+)$", "", cleaned)
    return cleaned


def _format_recommendation_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0].lower().startswith("recommendation") and lines[1].lower().startswith("reason"):
        return "\n".join(lines[:2])
    if len(lines) == 1:
        line = lines[0]
        if line.lower().startswith("recommendation:"):
            return line
        return f"Recommendation: {line}\nReason: This fits the current weather conditions."
    first = lines[0]
    second = lines[1] if len(lines) > 1 else "This fits the current weather conditions."
    if not first.lower().startswith("recommendation:"):
        first = f"Recommendation: {first}"
    if not second.lower().startswith("reason:"):
        second = f"Reason: {second}"
    return f"{first}\n{second}"


def _plain_recommendation_text(text: str) -> str:
    cleaned = _strip_location_mentions(_sanitize_ai_text(text, max_lines=4))
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    merged = " ".join(lines)
    merged = re.sub(r"^(recommendation|reason)\s*:\s*", "", merged, flags=re.IGNORECASE).strip()
    merged = re.sub(r"\s+", " ", merged).strip()

    # Keep the AI recommendation card short: show only the first complete sentence.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", merged) if s.strip()]
    if not sentences:
        return ""

    selected = sentences[0].strip()
    selected = re.sub(r"\b(and|but|because|with|like)$", "", selected, flags=re.IGNORECASE).strip()
    if selected and selected[-1] not in ".!?…":
        selected += "."
    return selected


def _polish_chat_reply(text: str) -> str:
    cleaned = _strip_location_mentions(_sanitize_ai_text(text, max_lines=4))
    if not cleaned:
        return ""

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and cleaned[-1] not in ".!?…":
        cleaned += "."

    return cleaned


# Also load environment variables from the .env file.
_load_local_env()

# Plain Python execution triggers noisy Streamlit context warnings.
# Exit early with a clear instruction instead.
if __name__ == "__main__" and not st_runtime.exists():
    print("This app runs with Streamlit. Use this command:")
    print("python -m streamlit run main.py")
    raise SystemExit(0)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherWise — Your Weather Assistant",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Config ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = _resolve_named_gemini_api_key(("GEMINI_API_KEY", "GOOGLE_API_KEY"), fallback_names=())
CHAT_GEMINI_API_KEY = _resolve_named_gemini_api_key(("CHAT_GEMINI_API_KEY",))
RECO_GEMINI_API_KEY = _resolve_named_gemini_api_key(("RECO_GEMINI_API_KEY",))
GEMINI_MODEL   = "gemini-2.5-flash"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_BASE  = "https://nominatim.openstreetmap.org"

# Gemini clients — separate key support for different purposes
_chat_gemini_client: Optional[genai.Client] = _build_gemini_client(CHAT_GEMINI_API_KEY or GEMINI_API_KEY)
_reco_gemini_client: Optional[genai.Client] = _build_gemini_client(RECO_GEMINI_API_KEY or GEMINI_API_KEY)

WEATHER_CODES = {
    0:  {"desc": "Clear",              "icon": "☀️"},
    1:  {"desc": "Mostly Clear",       "icon": "🌤️"},
    2:  {"desc": "Partly Cloudy",      "icon": "⛅"},
    3:  {"desc": "Overcast",           "icon": "☁️"},
    45: {"desc": "Foggy",              "icon": "🌫️"},
    48: {"desc": "Rime Fog",           "icon": "🌫️"},
    51: {"desc": "Light Drizzle",      "icon": "🌦️"},
    53: {"desc": "Drizzle",            "icon": "🌦️"},
    55: {"desc": "Dense Drizzle",      "icon": "🌧️"},
    61: {"desc": "Light Rain",         "icon": "🌧️"},
    63: {"desc": "Rain",               "icon": "🌧️"},
    65: {"desc": "Heavy Rain",         "icon": "⛈️"},
    71: {"desc": "Light Snow",         "icon": "🌨️"},
    73: {"desc": "Snow",               "icon": "❄️"},
    75: {"desc": "Heavy Snow",         "icon": "❄️"},
    80: {"desc": "Rain Showers",       "icon": "🌦️"},
    81: {"desc": "Showers",            "icon": "🌧️"},
    82: {"desc": "Violent Showers",    "icon": "⛈️"},
    95: {"desc": "Thunderstorm",       "icon": "⛈️"},
    99: {"desc": "Hail Thunderstorm",  "icon": "⛈️"},
}


def _default_chat_history() -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": "Hi! 👋 What would you like to know about today's weather?"}
    ]


def _chat_location_key(lat: float, lon: float) -> str:
    return f"{lat:.3f}|{lon:.3f}"

# ── Session State Init ────────────────────────────────────────────────────────
if "weather_data"       not in st.session_state: st.session_state.weather_data       = None
if "location_display"   not in st.session_state: st.session_state.location_display   = "Diyarbakir, TR"
if "chat_history"       not in st.session_state: st.session_state.chat_history       = _default_chat_history()
if "lat"                not in st.session_state: st.session_state.lat                = 37.9144
if "lon"                not in st.session_state: st.session_state.lon                = 40.2306
if "ai_recommendation"  not in st.session_state: st.session_state.ai_recommendation  = None
if "liked_tips"         not in st.session_state: st.session_state.liked_tips         = set()
if "weather_signature"  not in st.session_state: st.session_state.weather_signature  = None
if "api_key_override"   not in st.session_state: st.session_state.api_key_override   = ""
if "last_chat_request_ts" not in st.session_state: st.session_state.last_chat_request_ts = 0.0
if "chat_cooldown_seconds" not in st.session_state: st.session_state.chat_cooldown_seconds = 12
if "chat_histories" not in st.session_state: st.session_state.chat_histories = {}
if "current_chat_key" not in st.session_state:
    st.session_state.current_chat_key = _chat_location_key(st.session_state.lat, st.session_state.lon)

# Transition from the old single-history model to a location-based model.
active_chat_key = _chat_location_key(st.session_state.lat, st.session_state.lon)
if st.session_state.current_chat_key != active_chat_key:
    st.session_state.current_chat_key = active_chat_key
if active_chat_key not in st.session_state.chat_histories:
    existing_history = st.session_state.chat_history if isinstance(st.session_state.chat_history, list) else []
    st.session_state.chat_histories[active_chat_key] = existing_history or _default_chat_history()
st.session_state.chat_history = st.session_state.chat_histories[active_chat_key]

# The user can temporarily enter a key from inside the app if needed.
st.sidebar.markdown("### Gemini API Settings")
st.sidebar.text_input(
    "GEMINI_API_KEY",
    key="api_key_override",
    type="password",
    help="This key is used only for this session if .env/secrets are missing and is never written to disk.",
)

ACTIVE_GEMINI_API_KEY = GEMINI_API_KEY or _normalize_api_key(st.session_state.api_key_override)
ACTIVE_CHAT_GEMINI_API_KEY = CHAT_GEMINI_API_KEY or ACTIVE_GEMINI_API_KEY
ACTIVE_RECO_GEMINI_API_KEY = RECO_GEMINI_API_KEY or ACTIVE_GEMINI_API_KEY

_chat_gemini_client = _build_gemini_client(ACTIVE_CHAT_GEMINI_API_KEY)
_reco_gemini_client = _build_gemini_client(ACTIVE_RECO_GEMINI_API_KEY)
if ACTIVE_CHAT_GEMINI_API_KEY and not _looks_like_gemini_key(ACTIVE_CHAT_GEMINI_API_KEY):
    _chat_gemini_client = None
    st.sidebar.error(
        "The chat key format looks invalid. Gemini keys usually start with AIza..."
    )
if ACTIVE_RECO_GEMINI_API_KEY and not _looks_like_gemini_key(ACTIVE_RECO_GEMINI_API_KEY):
    _reco_gemini_client = None
    st.sidebar.error(
        "The recommendation key format looks invalid. Gemini keys usually start with AIza..."
    )


def build_weather_signature(location_display: str, lat: float, lon: float, cur: dict) -> str:
    return "|".join([
        location_display,
        f"{lat:.4f}",
        f"{lon:.4f}",
        str(cur["temperature"]),
        str(cur["feels_like"]),
        str(cur["weather_code"]),
        str(cur["wind_speed"]),
        str(cur["humidity"]),
    ])


# ── Gemini error message helper ─────────────────────────────────────────────
def _gemini_error_message(e: Exception) -> str:
    """Convert Gemini / google-genai exceptions into user-friendly English messages."""
    err_str = str(e).lower()
    if "api_key" in err_str or "api key" in err_str or "401" in err_str or "unauthenticated" in err_str:
        return (
            "API key is invalid or missing. Please check GEMINI_API_KEY in .env or .streamlit/secrets.toml. "
            "Get a key at: https://aistudio.google.com/app/apikey"
        )
    if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
        return "API quota exceeded or rate limit reached. Please wait a bit and try again. Free tier has daily and per-minute limits."
    if "503" in err_str or "unavailable" in err_str:
        return "Gemini service is currently unavailable. Please try again shortly."
    if "timeout" in err_str:
        return "Request timed out. Check your internet connection and try again."
    if "400" in err_str or "invalid_argument" in err_str:
        return "Invalid request. Please try again."
    return f"Unexpected error: {str(e)[:120]}"


def _is_transient_gemini_error(e: Exception) -> bool:
    err_str = str(e).lower()
    return any(token in err_str for token in ("503", "unavailable", "timeout", "deadline exceeded", "connection reset", "service unavailable"))


def _generate_gemini_content(*, client: genai.Client, model, contents, config):
    last_error = None
    for attempt in range(2):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            if not _is_transient_gemini_error(e) or attempt == 1:
                break
    if last_error:
        raise last_error
    raise RuntimeError("Gemini failed to generate a response.")


def _find_rain_alert_hour(weather_data: Optional[dict], threshold: int = 45) -> Optional[dict]:
    if not weather_data:
        return None
    hourly = weather_data.get("hourly") or []
    for item in hourly:
        if int(item.get("precipitation_probability", 0)) >= threshold:
            return item
    return None


def _format_hour_for_chat(hour_value) -> str:
    try:
        return f"{int(hour_value) % 24:02d}:00"
    except Exception:
        return "later today"


def _is_rainy_condition(weather_code: int, precipitation_probability: int = 0) -> bool:
    rain_codes = (51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 99)
    return weather_code in rain_codes or precipitation_probability >= 45


def _build_clothing_guidance(
    temperature: int,
    feels_like: int,
    weather_code: int,
    wind_speed: int,
    humidity: int,
    precipitation_probability: int,
    rain_alert_hour: str = "",
) -> str:
    parts = []

    if temperature <= 7 or feels_like <= 7:
        parts.append("Go with a warm coat or thick jacket and a layered top")
    elif temperature <= 14 or feels_like <= 14:
        parts.append("A jacket or hoodie with long trousers will be comfortable")
    elif temperature <= 23:
        parts.append("A light layer like a thin jacket over a t-shirt should be enough")
    else:
        parts.append("Choose light, breathable clothes")

    if _is_rainy_condition(weather_code, precipitation_probability):
        if rain_alert_hour:
            parts.append(f"carry an umbrella because rain risk increases around {rain_alert_hour}")
        else:
            parts.append("carry an umbrella and prefer a water-resistant outer layer")

    if wind_speed >= 30:
        parts.append("a windproof outer layer will help in gusts")

    if humidity >= 80 and temperature >= 20:
        parts.append("pick breathable fabrics to stay comfortable in humidity")

    sentence = "; ".join(parts).strip()
    if sentence and sentence[-1] not in ".!?…":
        sentence += "."
    return sentence


def _fallback_chat_reply(
    user_text: str = "",
    weather_context: str = "",
    weather_snapshot: Optional[dict] = None,
    weather_data: Optional[dict] = None,
) -> str:
    query = (user_text or "").lower()
    rain_alert = _find_rain_alert_hour(weather_data, threshold=45)
    rain_hour_label = _format_hour_for_chat(rain_alert.get("hour")) if rain_alert else ""
    rain_hour_prob = int(rain_alert.get("precipitation_probability", 0)) if rain_alert else 0

    if weather_snapshot:
        desc = WEATHER_CODES.get(weather_snapshot.get("weather_code", 0), {}).get("desc", "variable")
        temp = weather_snapshot.get("temperature", 20)
        feels_like = weather_snapshot.get("feels_like", temp)
        weather_code = weather_snapshot.get("weather_code", 0)
        wind_speed = weather_snapshot.get("wind_speed", 10)
        humidity = weather_snapshot.get("humidity", 50)
        rain_now = int(weather_snapshot.get("precipitation_probability", 0))

        is_storm = weather_code in (95, 99)
        is_rainy = _is_rainy_condition(weather_code, rain_now) or bool(rain_alert)
        strong_wind = wind_speed >= 35

        asks_umbrella = any(k in query for k in ("umbrella", "take umbrella", "need umbrella"))
        asks_walk = any(k in query for k in ("walk", "stroll", "outdoor"))
        asks_wear = any(k in query for k in ("wear", "what to wear", "outfit", "clothes"))

        if asks_umbrella:
            if is_rainy:
                if rain_hour_label:
                    return f"Yes, take an umbrella today; rain chance rises to about %{rain_hour_prob} around {rain_hour_label}."
                return "Yes, carrying an umbrella is a good idea today because rain risk is noticeable."
            return "An umbrella is optional right now, but checking updates before heading out is still wise."

        if asks_walk:
            if is_storm:
                return "A walk is not a good idea right now because there is thunderstorm risk; staying indoors is safer."
            if is_rainy:
                if rain_hour_label:
                    return f"A short walk can work in drier periods, but keep it flexible because rain risk increases around {rain_hour_label}."
                return "A walk is possible, but conditions are wet enough that carrying rain protection is recommended."
            if strong_wind:
                return f"A walk is possible, but wind around {wind_speed} km/h may feel uncomfortable in open areas."
            return f"It looks reasonably good for a walk now with around {temp}°C and {desc.lower()} conditions."

        if asks_wear:
            return _build_clothing_guidance(
                temp,
                feels_like,
                weather_code,
                wind_speed,
                humidity,
                rain_now,
                rain_hour_label,
            )

        summary = f"Right now it looks {desc.lower()} with a temperature around {temp}°C."
        follow_up = _build_clothing_guidance(
            temp,
            feels_like,
            weather_code,
            wind_speed,
            humidity,
            rain_now,
            rain_hour_label,
        )
        return f"{summary} {follow_up}"

    if weather_context:
        return "Conditions look fairly calm right now. If you are heading out, a light extra layer can help."
    return "You can keep your plans flexible today; I can also give a quick weather-based tip."


def _extract_requested_hour(text: str) -> Optional[int]:
    query = (text or "").lower().strip()
    if not query:
        return None

    patterns = [
        r"\bsaat\s*([01]?\d|2[0-3])(?:[:.]00)?\s*(?:'?(?:de|da))?\b",
        r"\bhour\s*([01]?\d|2[0-3])(?:[:.]00)?\b",
        r"\bat\s*([01]?\d|2[0-3])(?:[:.]00)?\b",
        r"\b([01]?\d|2[0-3])[:.]00\b",
        r"\b([01]?\d|2[0-3])\s*'?(?:de|da)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, query)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def _build_chat_weather_context(weather_data: Optional[dict], user_text: str) -> str:
    if not weather_data:
        return ""

    cur = weather_data.get("current") or {}
    requested_hour = _extract_requested_hour(user_text)
    rain_alert = _find_rain_alert_hour(weather_data, threshold=45)
    rain_alert_text = ""
    if rain_alert:
        rain_alert_text = (
            f". Umbrella signal: around {rain_alert.get('hour', 0):02d}:00 rain chance can reach "
            f"%{rain_alert.get('precipitation_probability', 0)}"
        )

    if requested_hour is None:
        return (
            f"Current conditions: temperature {cur.get('temperature', '?')}°C, "
            f"feels like {cur.get('feels_like', '?')}°C, "
            f"condition {cur.get('description', 'Unknown')}, "
            f"humidity %{cur.get('humidity', '?')}, "
            f"wind {cur.get('wind_speed', '?')} km/h, "
            f"rain chance %{cur.get('precipitation_probability', 0)}"
            f"{rain_alert_text}"
        )

    hourly = weather_data.get("hourly") or []
    hour_item = next((h for h in hourly if h.get("hour") == requested_hour), None)
    if not hour_item:
        return (
            f"Requested hour not available. Current conditions: temperature {cur.get('temperature', '?')}°C, "
            f"condition {cur.get('description', 'Unknown')}, "
            f"humidity %{cur.get('humidity', '?')}, wind {cur.get('wind_speed', '?')} km/h, "
            f"rain chance %{cur.get('precipitation_probability', 0)}"
        )

    h_code = hour_item.get("weather_code")
    h_desc = WEATHER_CODES.get(h_code, {"desc": "Bilinmiyor"})["desc"]
    h_humidity = hour_item.get("humidity", cur.get("humidity", "?"))
    h_wind = hour_item.get("wind_speed", cur.get("wind_speed", "?"))
    return (
        f"Forecast for {requested_hour:02d}:00: temperature {hour_item.get('temperature', '?')}°C, "
        f"condition {h_desc}, humidity %{h_humidity}, wind {h_wind} km/h, "
        f"rain chance %{hour_item.get('precipitation_probability', 0)}"
        f"{rain_alert_text}"
    )


# ── API Helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_weather(lat: float, lon: float):
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current":   "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,visibility",
        "hourly":    "temperature_2m,weather_code,precipitation_probability,relative_humidity_2m,wind_speed_10m",
        "timezone":  "auto",
        "forecast_days": 1,
    }
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return None, str(e)

    cur  = data["current"]
    code = cur["weather_code"]
    winfo = WEATHER_CODES.get(code, {"desc": "Unknown", "icon": "🌡️"})

    current_hour  = datetime.now().hour
    hourly_temps  = data["hourly"]["temperature_2m"]
    hourly_codes  = data["hourly"]["weather_code"]
    hourly_precip = data["hourly"]["precipitation_probability"]
    hourly_humidity = data["hourly"].get("relative_humidity_2m", [])
    hourly_wind = data["hourly"].get("wind_speed_10m", [])
    current_precip_prob = hourly_precip[current_hour] if current_hour < len(hourly_precip) else 0

    hourly = []
    for i in range(12):
        idx = current_hour + i
        if idx < len(hourly_temps):
            h_code  = hourly_codes[idx]
            h_winfo = WEATHER_CODES.get(h_code, {"desc": "Unknown", "icon": "🌡️"})
            hourly.append({
                "hour":  (current_hour + i) % 24,
                "label": "Now" if i == 0 else f"{(current_hour + i) % 24:02d}:00",
                "temperature": round(hourly_temps[idx]),
                "weather_code": h_code,
                "icon": h_winfo["icon"],
                "precipitation_probability": hourly_precip[idx] if idx < len(hourly_precip) else 0,
                "humidity": round(hourly_humidity[idx]) if idx < len(hourly_humidity) else None,
                "wind_speed": round(hourly_wind[idx]) if idx < len(hourly_wind) else None,
            })

    return {
        "current": {
            "temperature":   round(cur["temperature_2m"]),
            "feels_like":    round(cur["apparent_temperature"]),
            "humidity":      cur["relative_humidity_2m"],
            "wind_speed":    round(cur["wind_speed_10m"]),
            "precipitation_probability": round(current_precip_prob),
            "visibility_km": round(cur.get("visibility", 0) / 1000, 1),
            "weather_code":  code,
            "description":   winfo["desc"],
            "icon":          winfo["icon"],
        },
        "hourly": hourly,
    }, None


@st.cache_data(ttl=3600)
def fetch_location(lat: float, lon: float):
    headers = {"User-Agent": "WeatherWise/1.0"}
    try:
        with httpx.Client(timeout=8) as client:
            resp = client.get(
                f"{NOMINATIM_BASE}/reverse",
                params={"lat": lat, "lon": lon, "format": "json", "accept-language": "en"},
                headers=headers,
            )
            data = resp.json()
            addr    = data.get("address", {})
            city    = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county") or "Unknown"
            country = addr.get("country_code", "").upper()
            return f"{city}, {country}"
    except Exception:
        return "Current Location"


def gemini_recommend(temperature, feels_like, weather_code, wind_speed, humidity, precipitation_probability, city, hour):
    """Generate an AI recommendation based on weather conditions."""
    if not _reco_gemini_client and not _chat_gemini_client:
        return _plain_recommendation_text(_fallback_ai_recommendation(temperature, feels_like, weather_code, wind_speed, humidity))

    winfo = WEATHER_CODES.get(weather_code, {"desc": "Unknown"})
    recommendation_hint = _build_recommendation_hint(temperature, weather_code, wind_speed, humidity)
    variety_tag = datetime.now().strftime("%H:%M:%S")

    primary_client = _reco_gemini_client or _chat_gemini_client
    secondary_client = _chat_gemini_client if (_reco_gemini_client and _chat_gemini_client) else None

    def _looks_generic(text: str) -> bool:
        low = (text or "").lower()
        generic_patterns = (
            "weather is fine",
            "you can comfortably make your plans",
            "it is a nice day to go outside",
            "the weather looks pleasant",
            "weather is fine",
            "you can proceed with your plans",
            "great day to go out",
        )
        return any(p in low for p in generic_patterns)

    # ── Short and practical prompt ──
    prompt = f"""You are WeatherWise, a warm and practical weather assistant.
Read the data below and write one short, friendly recommendation in English.
Do not use city or location names to address the user.
Do not write labels like Recommendation: or Reason:.
Avoid command-like tone; be kind, natural, and conversational.
Avoid generic boilerplate; prefer a fresh phrasing in each answer.

🌡️ Temperature: {temperature}°C (feels like {feels_like}°C)
🌤️ Condition: {winfo['desc']}
💧 Humidity: %{humidity}
🌧️ Rain chance: %{precipitation_probability}
💨 Wind: {wind_speed} km/h
🎯 Hint: {recommendation_hint}
🎲 Variation tag: {variety_tag}

Your task:
- Write exactly one short sentence.
- Keep it complete; never end mid-sentence.
- Keep it clear, warm, and practical.
- Mention rain, umbrella, light clothing, hydration, wind, or staying indoors when relevant.
- Avoid technical jargon.
- End with proper punctuation."""

    try:
        response = _generate_gemini_content(
            client=primary_client,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.9,
                top_k=40,
                top_p=0.9,
                max_output_tokens=260,
            ),
        )
        text = _plain_recommendation_text(response.text)
        if text and not _looks_generic(text):
            return text

        retry_prompt = (
            prompt
            + "\nExtra rule: Use wording different from the previous answer, stay tied to the data, and avoid clichés."
        )
        retry_client = secondary_client or primary_client
        retry_response = _generate_gemini_content(
            client=retry_client,
            model=GEMINI_MODEL,
            contents=retry_prompt,
            config=types.GenerateContentConfig(
                temperature=1.0,
                top_k=40,
                top_p=0.95,
                max_output_tokens=260,
            ),
        )
        retry_text = _plain_recommendation_text(retry_response.text)
        return retry_text if retry_text else _plain_recommendation_text(_fallback_ai_recommendation(temperature, feels_like, weather_code, wind_speed, humidity))

    except Exception as e:
        return _plain_recommendation_text(_fallback_ai_recommendation(temperature, feels_like, weather_code, wind_speed, humidity))


def gemini_chat(messages, weather_context="", city="", weather_snapshot=None, weather_data=None):
    """Generate AI response for chat messages."""
    if not _chat_gemini_client:
        return "🔑 Gemini API key is missing, but I can still help with weather-based guidance."

    variety_tag = datetime.now().strftime("%H:%M:%S")
    system_instruction = f"""You are WeatherWise, a warm and practical weather assistant.
Reply in natural, clear English.
Avoid technical jargon and long explanations.
Always write complete sentences; never cut a sentence midway.
Keep replies to 2-3 sentences.
Use the first sentence for a concise weather summary.
Use the next sentence(s) for polite, actionable advice.
If the user asks for a specific hour, answer only using that hour's forecast.
Do not address the user with city/location names.
Use clear suggestions such as umbrella, layers, hydration, wind caution, or staying indoors when needed.
Do not contradict temperature, rain chance, humidity, or wind data.
Avoid reassuring clichés like "weather is fine" during thunderstorm/high rain risk.
If any hour today has rain chance >=45%, suggest carrying an umbrella and cite one practical hour like "around 16:00".
Vary phrasing across replies.
Variation tag: {variety_tag}
{f"🌡️ Current weather: {weather_context}" if weather_context else ""}"""

    # Convert the chat history to Gemini format.
    filtered = [
        m for m in messages
        if not (m["role"] == "assistant" and ("today's weather" in m["content"].lower()))
    ]

    gemini_contents = []
    for m in filtered:
        role = "user" if m["role"] == "user" else "model"
        gemini_contents.append(
            types.Content(role=role, parts=[types.Part(text=m["content"])])
        )

    # Do not crash with an empty user message if there is no history.
    if not gemini_contents:
        return "How can I help you? 😊"

    latest_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user_text = m.get("content", "")
            break

    try:
        response = _generate_gemini_content(
            client=_chat_gemini_client,
            model=GEMINI_MODEL,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.75,
                top_k=40,
                top_p=0.9,
                max_output_tokens=500,
            ),
        )
        text = _polish_chat_reply(response.text)
        return text if text else _fallback_chat_reply(latest_user_text, weather_context, weather_snapshot, weather_data)

    except Exception as e:
        if "429" in str(e).lower() or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
            return _fallback_chat_reply(latest_user_text, weather_context, weather_snapshot, weather_data)
        if _is_transient_gemini_error(e):
            return _fallback_chat_reply(latest_user_text, weather_context, weather_snapshot, weather_data)
        return _gemini_error_message(e)


def _fallback_recommendation(t, code, wind):
    if t > 30:
        return f"🌡️ It is quite hot today at {t}°. Stay hydrated and avoid prolonged midday sun."
    if t < 5:
        return f"🧥 It is cold today at {t}°. Layered clothing is your best choice."
    if code in (61, 63, 65, 80, 81, 82):
        return "☂️ Rain is expected today. Bring an umbrella and wear something water-resistant."
    if 18 <= t <= 25:
        return f"✨ {t}° feels pleasant today. It is a good window for outdoor plans."
    return f"🌤️ Current condition is {WEATHER_CODES.get(code, {}).get('desc', 'moderate')}."


def _fallback_ai_recommendation(temperature, feels_like, weather_code, wind_speed, humidity) -> str:
    if weather_code in (95, 99):
        return "There is thunderstorm risk; staying indoors when possible is the safer choice."
    if weather_code in (61, 63, 65, 80, 81, 82):
        return "It is wise to carry an umbrella today; there is a real chance of getting wet outdoors."
    if temperature >= 30 or feels_like >= 30:
        return "Light clothing and steady hydration will help today; conditions are quite warm."
    if temperature <= 8 or feels_like <= 8:
        return "A warm outer layer is a good choice today; mornings and evenings can feel chilly."
    if wind_speed >= 35:
        return "Be cautious with strong wind today; lightweight items may be difficult to handle."
    if humidity >= 80 and temperature >= 22:
        return "Breathable fabrics are a better choice today; humidity may feel uncomfortable."
    if weather_code in (3, 45, 48):
        return "Cloudy and cool conditions are likely; a light extra layer can keep you comfortable."
    if 18 <= temperature <= 25:
        return "Conditions look pleasant; comfortable clothing should work well for being outdoors."
    return "Weather looks changeable today; check updates occasionally and keep your plans flexible."


def get_uv(lat, code):
    b  = 9 if abs(lat) < 30 else 7 if abs(lat) < 45 else 4
    m  = datetime.now().month
    uv = round(b + math.cos(((m - 6) / 6) * math.pi) * 2)
    if code in (45, 48, 61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 99):
        uv = max(1, uv - 3)
    return max(1, min(11, uv))


def get_outfit_tags(temp, code, wind, humidity, precipitation_probability=0, hourly=None):
    tags = []
    rain_alert = None
    if hourly:
        rain_alert = next((h for h in hourly if int(h.get("precipitation_probability", 0)) >= 45), None)

    if temp <= 0:
        tags += ["🧥 Heavy Coat", "🧤 Gloves", "🧣 Scarf"]
    elif temp <= 8:
        tags += ["🧥 Warm Jacket", "🧣 Scarf", "👟 Closed Shoes"]
    elif temp <= 14:
        tags += ["🧥 Light Jacket", "👖 Trousers"]
    elif temp <= 20:
        tags += ["🥻 Thin Jacket", "👕 Long Sleeve"]
    elif temp <= 26:
        tags += ["👕 T-Shirt", "👖 Light Trousers"]
    else:
        tags += ["👕 Breathable Top", "🩳 Light Bottoms", "🧴 Sunscreen"]

    if _is_rainy_condition(code, precipitation_probability) or rain_alert:
        tags += ["🧥 Rain Jacket", "☂️ Umbrella", "👢 Waterproof Shoes"]

    if code in (71, 73, 75):
        tags.append("🥾 Waterproof Boots")

    if wind >= 30:
        tags += ["🧥 Windproof Layer", "🧢 Wind Caution"]

    if humidity >= 80 and temp >= 20:
        tags += ["👕 Breathable Fabric", "💧 Stay Hydrated"]

    # Keep tag order stable while removing duplicates.
    tags = list(dict.fromkeys(tags))
    return tags


def get_tips(temp, code, wind, humidity):
    tips = []
    hour = datetime.now().hour
    if code in (61, 63, 65, 80, 81, 82):
        tips.append({"icon": "☂️", "color": "#e0f2fe", "cat": "Weather Alert",
                     "text": "Rain is expected. Do not forget your umbrella!"})
    if temp > 30:
        tips.append({"icon": "🌡️", "color": "#fef3c7", "cat": "Hot Weather",
                     "text": f"It is very warm at {temp}°. Stay hydrated and avoid strong midday sun."})
    elif temp < 5:
        tips.append({"icon": "❄️", "color": "#e0f2fe", "cat": "Cold Weather",
                     "text": f"{temp}° is quite cold. Layered clothing is the best strategy."})
    elif 18 <= temp <= 25:
        tips.append({"icon": "✨", "color": "#dcfce7", "cat": "Great Conditions",
                     "text": f"{temp}° is ideal for being outside. Great for a walk or biking."})
    if wind > 40:
        tips.append({"icon": "💨", "color": "#f3e8ff", "cat": "Wind Alert",
                     "text": f"Wind speed is {wind} km/h. Keep light items secure and dress accordingly."})
    if humidity > 75 and temp > 22:
        tips.append({"icon": "💦", "color": "#e0f2fe", "cat": "High Humidity",
                     "text": f"Humidity is %{humidity}, so it may feel muggy. Choose breathable fabrics."})
    if 6 <= hour <= 9 and code not in (61, 63, 65):
        tips.append({"icon": "🌅", "color": "#fef3c7", "cat": "Morning Tip",
                     "text": "Morning air looks clean — consider walking for part of your commute."})
    if not tips:
        tips.append({"icon": "🌤️", "color": "#e0f2fe", "cat": "General",
                     "text": "Weather is fairly stable today and suitable for daily activities."})
    return tips[:4]


# ── CSS Injection ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --sky: #0ea5e9;
  --sky-dark: #0369a1;
  --sky-pale: #e0f2fe;
  --warm: #f59e0b;
  --warm-pale: #fef3c7;
  --night: #0f172a;
  --cloud: #f1f5f9;
  --mist: #94a3b8;
  --sun: #fbbf24;
  --grass: #22c55e;
  --card-bg: rgba(255,255,255,0.92);
  --card-border: rgba(14,165,233,0.15);
  --radius: 20px;
  --shadow: 0 4px 24px rgba(14,165,233,0.10);
  --text: #0f172a;
  --text-hint: #94a3b8;
}

body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #0369a1 0%, #0ea5e9 45%, #38bdf8 100%) fixed !important;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0;
  background: radial-gradient(ellipse at 70% 20%, rgba(251,191,36,0.12) 0%, transparent 60%);
  pointer-events: none; z-index: 0;
}

[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], footer { display: none !important; }

[data-testid="stMain"] > div { padding-top: 0 !important; }
.block-container { padding: 20px 20px 100px !important; max-width: 1100px !important; }

/* Main card */
.main-card {
  background: var(--card-bg);
  border-radius: var(--radius);
  border: 1px solid var(--card-border);
  box-shadow: var(--shadow);
  overflow: hidden;
  backdrop-filter: blur(20px);
  margin-bottom: 16px;
}
.main-top {
  background: linear-gradient(135deg, #0369a1, #0ea5e9);
  padding: 24px 24px 20px; color: #fff; position: relative; overflow: hidden;
}
.main-top::after {
  content: ''; position: absolute; right: -20px; top: -20px;
  width: 140px; height: 140px; border-radius: 50%;
  background: rgba(255,255,255,0.08);
}
.temp-row { display: flex; align-items: flex-start; justify-content: space-between; }
.temp-big { font-family: 'DM Serif Display', serif; font-size: 72px; line-height: 1; letter-spacing: -3px; }
.weather-icon-big { font-size: 64px; line-height: 1; animation: float 4s ease-in-out infinite; }
@keyframes float { 0%,100%{ transform: translateY(0); } 50%{ transform: translateY(-8px); } }
.weather-desc { font-size: 18px; font-weight: 300; margin-top: 4px; opacity: 0.9; }
.feels-like { font-size: 13px; opacity: 0.7; margin-top: 2px; }

.stats-strip { display: flex; gap: 0; border-top: 1px solid rgba(255,255,255,0.15); margin-top: 16px; }
.stat-item { flex: 1; text-align: center; padding: 12px 8px; border-right: 1px solid rgba(255,255,255,0.1); }
.stat-item:last-child { border-right: none; }
.stat-icon { font-size: 16px; margin-bottom: 2px; }
.stat-val { font-size: 14px; font-weight: 500; }
.stat-lbl { font-size: 10px; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.5px; }

.ai-section { padding: 20px 24px; }
.ai-label {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 1px; color: var(--sky-dark); margin-bottom: 10px;
}
.ai-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: linear-gradient(135deg, var(--sky), var(--warm));
  animation: aiPulse 1.5s ease-in-out infinite; display: inline-block;
}
@keyframes aiPulse { 0%,100%{ transform: scale(1); opacity: 1; } 50%{ transform: scale(1.4); opacity: 0.6; } }
.ai-text { font-size: 16px; line-height: 1.6; color: var(--text); white-space: normal; overflow-wrap: anywhere; word-break: break-word; }

.outfit-section { padding: 0 24px 20px; border-top: 1px solid rgba(14,165,233,0.08); padding-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.tag {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--sky-pale); border: 1px solid rgba(14,165,233,0.2);
  border-radius: 50px; padding: 6px 14px; font-size: 13px;
  color: var(--sky-dark); font-weight: 500;
}

/* Hourly */
.hourly-scroll {
  display: flex; gap: 10px; overflow-x: auto;
  padding: 4px 0 16px; scrollbar-width: none;
}
.hourly-scroll::-webkit-scrollbar { display: none; }
.hour-card {
  flex-shrink: 0; background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.15); border-radius: 16px;
  padding: 12px 14px; text-align: center; min-width: 66px;
}
.hour-card.now { background: rgba(255,255,255,0.22); border-color: rgba(255,255,255,0.35); }
.hour-time { font-size: 11px; color: rgba(255,255,255,0.5); margin-bottom: 6px; }
.hour-icon { font-size: 20px; margin-bottom: 6px; }
.hour-temp { font-size: 14px; font-weight: 600; color: #fff; }
.hour-rain { font-size: 10px; color: #93c5fd; margin-top: 3px; }

/* Section titles */
.section-title { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.6); margin-bottom: 10px; }

/* Tip cards */
.tip-card {
  background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: var(--radius); padding: 16px 18px;
  box-shadow: var(--shadow); backdrop-filter: blur(20px);
  display: flex; gap: 14px; align-items: flex-start; margin-bottom: 12px;
}
.tip-icon { width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
.tip-category { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-hint); margin-bottom: 3px; }
.tip-text { font-size: 14px; line-height: 1.5; color: var(--text); white-space: normal; overflow-wrap: anywhere; word-break: break-word; }
.tip-time { font-size: 11px; color: var(--text-hint); margin-top: 8px; }

/* Logo */
.logo { font-family: 'DM Serif Display', serif; font-size: 22px; color: #fff; letter-spacing: -0.5px; }
.logo span { color: var(--sun); }

/* Header */
.app-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 0 16px; }
.location-badge {
  display: flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
  border-radius: 50px; padding: 6px 12px; font-size: 13px;
  color: rgba(255,255,255,0.9);
}
.live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--grass); animation: pulse 2s infinite; display: inline-block; }
@keyframes pulse { 0%,100%{ opacity:1; } 50%{ opacity:0.4; } }

/* Chat */
.chat-msg-user {
  display: flex; justify-content: flex-end; margin-bottom: 12px;
}
.chat-msg-ai {
  display: flex; justify-content: flex-start; margin-bottom: 12px;
}
.bubble-user {
  background: linear-gradient(135deg, #0369a1, #0ea5e9);
  color: #fff; padding: 10px 14px; border-radius: 18px 18px 4px 18px;
  font-size: 14px; line-height: 1.55; max-width: 80%;
}
.bubble-ai {
  background: #f8fafc; color: var(--text);
  border: 1px solid #e2e8f0;
  padding: 10px 14px; border-radius: 18px 18px 18px 4px;
  font-size: 14px; line-height: 1.55; max-width: 80%;
}
.api-status { display: flex; align-items: center; gap: 6px; font-size: 11px; color: rgba(255,255,255,0.4); padding: 4px 0 12px; }
.status-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--grass); display: inline-block; }

/* Streamlit element overrides */
.stTextInput > div > div > input {
  background: rgba(255,255,255,0.95) !important;
  border-radius: 50px !important;
  border: none !important;
  padding: 10px 18px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  color: var(--text) !important;
}
.stButton > button {
  background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
  color: #fff !important; border: none !important;
  border-radius: 50px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  transition: transform 0.2s !important;
}
.stButton > button:hover { transform: scale(1.02) !important; }
div[data-testid="column"] { padding: 0 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
data, err = fetch_weather(st.session_state.lat, st.session_state.lon)
if data:
    st.session_state.weather_data = data
    cur = data["current"]
    st.session_state.ai_recommendation = gemini_recommend(
        cur["temperature"],
        cur["feels_like"],
        cur["weather_code"],
        cur["wind_speed"],
        cur["humidity"],
        cur.get("precipitation_probability", 0),
        st.session_state.location_display,
        datetime.now().hour,
    )

cur = st.session_state.weather_data["current"] if st.session_state.weather_data else None

# ── API key warning ──────────────────────────────────────────────────────────
if not ACTIVE_CHAT_GEMINI_API_KEY and not ACTIVE_RECO_GEMINI_API_KEY:
    st.warning(
        "⚠️ **Gemini API key not found.** "
        "AI features are disabled. Add CHAT_GEMINI_API_KEY=... and RECO_GEMINI_API_KEY=... to .env "
        "or define them in .streamlit/secrets.toml. "
        "You can also enter a temporary key from the left sidebar. "
        "Free key: https://aistudio.google.com/app/apikey",
        icon="🔑",
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <div class="logo">Weather<span>Wise</span></div>
  <div class="location-badge">
    <span class="live-dot"></span>
        <span>{html.escape(st.session_state.location_display)}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    if cur:
        tags_html = "".join(
            f'<span class="tag">{t}</span>'
            for t in get_outfit_tags(
                cur["temperature"],
                cur["weather_code"],
                cur["wind_speed"],
                cur["humidity"],
                cur.get("precipitation_probability", 0),
                st.session_state.weather_data.get("hourly") if st.session_state.weather_data else None,
            )
        )

        st.markdown(f"""
<div class="main-card">
  <div class="main-top">
    <div class="temp-row">
      <div>
        <div class="temp-big">{cur['temperature']}<sup style="font-size:28px;vertical-align:super">°</sup></div>
        <div class="weather-desc">{cur['description']}</div>
        <div class="feels-like">Feels like: {cur['feels_like']}°C</div>
      </div>
      <div class="weather-icon-big">{cur['icon']}</div>
    </div>
    <div class="stats-strip">
    <div class="stat-item"><div class="stat-icon">💧</div><div class="stat-val">{cur['humidity']}%</div><div class="stat-lbl">Humidity</div></div>
    <div class="stat-item"><div class="stat-icon">🌧️</div><div class="stat-val">{cur.get('precipitation_probability', 0)}%</div><div class="stat-lbl">Rain</div></div>
    <div class="stat-item"><div class="stat-icon">💨</div><div class="stat-val">{cur['wind_speed']} km/h</div><div class="stat-lbl">Wind</div></div>
    <div class="stat-item"><div class="stat-icon">👁️</div><div class="stat-val">{cur['visibility_km']} km</div><div class="stat-lbl">Visibility</div></div>
      <div class="stat-item"><div class="stat-icon">🌅</div><div class="stat-val">{get_uv(st.session_state.lat, cur['weather_code'])}</div><div class="stat-lbl">UV</div></div>
    </div>
  </div>
  <div class="outfit-section">{tags_html}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.error("Could not fetch weather data.")

    # Change location
    st.markdown('<div class="section-title">📍 Change Location</div>', unsafe_allow_html=True)
    loc_col1, loc_col2 = st.columns([3, 1])
    with loc_col1:
        new_city = st.text_input(
            " ", placeholder="Enter a city name or lat,lon…",
            label_visibility="collapsed", key="city_input"
        )
    with loc_col2:
        if st.button("Search 🔍"):
            if new_city:
                if "," in new_city:
                    try:
                        parts = new_city.split(",")
                        st.session_state.lat = float(parts[0].strip())
                        st.session_state.lon = float(parts[1].strip())
                        st.session_state.location_display = f"{parts[0].strip()}, {parts[1].strip()}"
                    except Exception:
                        st.warning("Invalid coordinate format. Example: 37.91, 40.23")
                else:
                    try:
                        with httpx.Client(timeout=8) as hclient:
                            resp = hclient.get(
                                f"{NOMINATIM_BASE}/search",
                                params={"q": new_city, "format": "json", "limit": 1, "accept-language": "en"},
                                headers={"User-Agent": "WeatherWise/1.0"},
                            )
                            results = resp.json()
                            if results:
                                st.session_state.lat = float(results[0]["lat"])
                                st.session_state.lon = float(results[0]["lon"])
                                st.session_state.location_display = (
                                    results[0].get("display_name", new_city).split(",")[0] + ", TR"
                                )
                            else:
                                st.warning("City not found.")
                    except Exception:
                            st.warning("Search failed. Check your internet connection.")
                st.session_state.ai_recommendation = None
                new_chat_key = _chat_location_key(st.session_state.lat, st.session_state.lon)
                st.session_state.current_chat_key = new_chat_key
                if new_chat_key not in st.session_state.chat_histories:
                    st.session_state.chat_histories[new_chat_key] = _default_chat_history()
                fetch_weather.clear()
                st.rerun()

    # The chat section is now in the left column.
    st.markdown('<div class="section-title" style="margin-top:16px">💬 WeatherWise AI Chat</div>', unsafe_allow_html=True)

    qr_cols = st.columns(4)
    quick_questions = ["👕 What to Wear?", "☂️ Umbrella?", "🚶 Walk?", "🌙 Tonight?"]
    quick_map = {
        "👕 What to Wear?": "What should I wear today?",
        "☂️ Umbrella?":   "Should I take an umbrella?",
        "🚶 Walk?":       "Is it good for a walk?",
        "🌙 Tonight?":    "How will the weather be tonight?",
    }
    for i, (col, q) in enumerate(zip(qr_cols, quick_questions)):
        with col:
            if st.button(q, key=f"qr_{i}"):
                now_ts = datetime.now().timestamp()
                elapsed = now_ts - st.session_state.last_chat_request_ts
                cooldown = st.session_state.chat_cooldown_seconds
                if elapsed < cooldown:
                    st.warning(f"You are sending requests too fast. Please wait {math.ceil(cooldown - elapsed)} seconds.")
                else:
                    st.session_state.chat_history.append({"role": "user", "content": quick_map[q]})
                    context = _build_chat_weather_context(st.session_state.weather_data, quick_map[q])
                    reply = gemini_chat(
                        st.session_state.chat_history,
                        context,
                        st.session_state.location_display,
                        cur,
                        st.session_state.weather_data,
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.session_state.last_chat_request_ts = now_ts
                    st.rerun()

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            msg_safe = html.escape(msg["content"]).replace("\n", "<br>")
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user"><div class="bubble-user">{msg_safe}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-ai"><div class="bubble-ai">{msg_safe}</div></div>', unsafe_allow_html=True)

    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        user_msg = st.text_input(
            " ", placeholder="Ask something about the weather…",
            label_visibility="collapsed", key="chat_input_field"
        )
    with btn_col:
        if st.button("Send ✈️"):
            if user_msg.strip():
                now_ts = datetime.now().timestamp()
                elapsed = now_ts - st.session_state.last_chat_request_ts
                cooldown = st.session_state.chat_cooldown_seconds
                if elapsed < cooldown:
                    st.warning(f"You are sending requests too fast. Please wait {math.ceil(cooldown - elapsed)} seconds.")
                else:
                    st.session_state.chat_history.append({"role": "user", "content": user_msg.strip()})
                    context = _build_chat_weather_context(st.session_state.weather_data, user_msg.strip())
                    reply = gemini_chat(
                        st.session_state.chat_history,
                        context,
                        st.session_state.location_display,
                        cur,
                        st.session_state.weather_data,
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.session_state.last_chat_request_ts = now_ts
                    st.rerun()

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        active_key = _chat_location_key(st.session_state.lat, st.session_state.lon)
        st.session_state.chat_histories[active_key] = _default_chat_history()
        st.session_state.chat_history = st.session_state.chat_histories[active_key]
        st.session_state.last_chat_request_ts = 0.0
        st.rerun()

with col_right:
    # Hourly forecast
    if st.session_state.weather_data:
        hourly = st.session_state.weather_data["hourly"]
        st.markdown('<div class="section-title">🕐 Hourly Forecast</div>', unsafe_allow_html=True)
        hour_cards = "".join(
            (
                f"<div class=\"hour-card{' now' if i == 0 else ''}\">"
                f"<div class=\"hour-time\">{h['label']}</div>"
                f"<div class=\"hour-icon\">{h['icon']}</div>"
                f"<div class=\"hour-temp\">{h['temperature']}°</div>"
                + (
                    f"<div class=\"hour-rain\">💧{h['precipitation_probability']}%</div>"
                    if h['precipitation_probability'] > 20 else ""
                )
                + "</div>"
            )
            for i, h in enumerate(hourly)
        )
        st.markdown(f'<div class="hourly-scroll">{hour_cards}</div>', unsafe_allow_html=True)

    # AI recommendation is now in the right column (instead of Daily Tips).
    st.markdown('<div class="section-title" style="margin-top:16px">🤖 AI Recommendation</div>', unsafe_allow_html=True)
    ai_rec_right = html.escape(st.session_state.ai_recommendation or "").replace("\n", "<br>")
    st.markdown(f"""
<div class="tip-card">
  <div class="tip-icon" style="background:#e0f2fe">🤖</div>
  <div style="flex:1">
    <div class="tip-category">AI Recommendation</div>
    <div class="tip-text">{ai_rec_right}</div>
    <div class="tip-time">Updated just now</div>
  </div>
</div>
""", unsafe_allow_html=True)
