#!/bin/bash
# WeatherWise — Start with one command (Streamlit)
set -e

echo "╔══════════════════════════════════════╗"
echo "║        WeatherWise (Streamlit)       ║"
echo "╚══════════════════════════════════════╝"

# Load the .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✓ .env loaded"
else
  echo "⚠  .env not found — copy .env.example:"
  echo "   cp .env.example .env && nano .env"
fi

# Install dependencies (streamlit, httpx, google-genai)
echo ""
echo "📦 Installing dependencies…"
pip install -r requirements.txt -q

# Verify google-genai installation
echo "🤖 Checking google-genai…"
python -c "from google import genai" 2>/dev/null && echo "✓ google-genai ready" || {
  echo "⚠  google-genai not found, installing it separately…"
  pip install google-genai -q
}

# Start Streamlit
echo ""
echo "🚀 Streamlit starting → http://localhost:8000"
echo "   Press Ctrl+C to exit"
echo ""

python -m streamlit run main.py --server.port 8000
