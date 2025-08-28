# 🌱 Agronomy Deep Research Agent

An intelligent **multi-agent agronomy assistant** for **crop monitoring, soil health, and agricultural economics**.  
This project integrates **Google Earth Engine, Tavily Search, and LLM-powered expert agents** to provide actionable advice for farmers, researchers, and agronomists.

---

## 🚀 Features

- **Agronomy Q&A** – Ask agronomy-related questions and get tailored advice.
- **NDVI Crop Monitoring** – Compute vegetation health from **Sentinel-2** imagery via Google Earth Engine.
- **Agronomy Research** – Fetch latest research articles with Tavily search.
- **Expert Agents**  
  - 👨‍🌾 **SoilExpert** – Soil fertility, pH, and nutrients.  
  - 🌾 **CropExpert** – Yield prediction, crop rotation, NDVI monitoring.  
  - 📈 **AgriEconomist** – Market trends, profitability, pricing.  
  - 🚨 **Escalation Agent** – Urgent pest/disease handling with escalation option.  
- **Stateful Memory** – Builds on previous conversations to provide context-aware responses.
- **Escalation Handling** – Suggests immediate actions and offers escalation to a professional agronomist.

### Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows



## 🛠️ Installation

### 1. Clone the repository
git clone https://github.com/AleemaSaleem/AI_AGRONOMIST.git
cd agronomy-agent

### 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

### 3. Setup environment variables
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key

### 4. Authenticate Google Earth Engine
earthengine authenticate

### ▶️ Usage

Run the assistant:
python agronomist.py

Example interaction:

🌱 Agronomy Deep Research Agent ready! Type 'exit' to quit.
❓ Ask your agronomy question: Give NDVI for region [[74.0,31.5],[74.1,31.5],[74.1,31.6],[74.0,31.6]] between 2025-01-01 and 2025-01-31

💬 Thinking...

📊 Mean NDVI for 2025-01-01 to 2025-01-31: 0.672

### 🤝 Contributing

Contributions are welcome!

Fork the repo
Create a feature branch
Submit a Pull Request




