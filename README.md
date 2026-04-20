# WeatherWise 🌤️

> An AI-powered social weather app that turns weather data into action.

## Features

- **Real-Time Weather** — Open-Meteo API (free, no API key required)
- **AI Recommendations** — Clothing, activity, and alert suggestions powered by Google Gemini 2.5 Flash
- **Chat Bot** — AI chat for contextual weather questions
- **Hourly Forecast** — 12 hours of forward-looking hourly data
- **Social Feed** — Tip cards generated from current conditions
- **Fully Responsive** — Works on phones, tablets, and desktops
- **Streamlit** — Single-file Python application

## Quick Start

### 1. Clone the repo
```bash
git clone <repo-url>
cd weatherwise
```

### 2. Prepare the .env file
```bash
cp .env.example .env
```
Open the .env file and set the GEMINI_API_KEY value:
```
GEMINI_API_KEY=your-key-here
```
> To get an API key: https://aistudio.google.com/app/apikey

### 3. Run
```bash
chmod +x run.sh
./run.sh
```

Open in your browser: **http://localhost:8000**

---

## Manual Setup

```bash
pip install streamlit httpx google-genai
python -m streamlit run main.py --server.port 8000
```

---

## Project Structure

```
weatherwise/
├── main.py           # Streamlit app (single file)
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── run.sh            # One-command launcher
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|--------|-----------|
| Framework | Python 3.11+, Streamlit |
| AI | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| AI Client | `google-genai` (`from google import genai`) |
| Weather Data | Open-Meteo API |
| Geocoding | Nominatim (OpenStreetMap) |
| HTTP Client | httpx |
| Font | DM Sans + DM Serif Display |

---

## API Endpoints (internal)

The app makes all API calls directly from Python:
- Open-Meteo → weather data
- Nominatim → location name
- Gemini API (`google-genai` SDK) → AI recommendations and chat
