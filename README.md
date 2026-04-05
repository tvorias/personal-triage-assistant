# Patient Triage Assistant via Streamlit App

An interactive web application that provides personalized healthcare triage recommendations using an LLM (Llama via Ollama). For emergency cases, the app identifies the nearest hospitals based on the user's current location.

## Features

### Patient Input Collection
- **Demographics:** Age, sex, race
- **Physical Measurements:** Height (feet/inches), weight (lbs), auto-calculated BMI
- **Medical History:** Chronic conditions and comorbidities
- **Current:** Name, duration, and severity of each symptom
- **Onset Description:** How symptoms started

### AI-powered Triage Recommendation
- Uses Llama 3.2 via Ollama to analyze patient information
- Provides one of 6 triage levels:
    - **ER - Immediate**: Life-threatening, go to ER now
    - **Urgent Care - Within 24 hours**: Needs prompt attention today
    - **See Physician - Within Few Days**: Schedule appointment soon
    - **See Physician - Within Few Weeks**: Routine follow-up
    - **Monitor Symptoms**: Self-limiting, watch for changes
    - **Self-Care**: Minor issue, home treatment
 
### Emergency Hospital Finder
- For "ER - Immediate" cases only
- Uses Google Geolocation API to get user's current location
- Identifies 5 nearest hospital from database
- Shows hospital names, addresses, and distance in miles


## Prerequisites

1. **Python 3.8+**
2. **Ollama** - Local LLM runtime
    - Install from https://ollama.ai
    - After installation, pull the LLama model:
    ```bash
    ollama pull llama3.2
    ```
3. **Google Geolocation API Key** (optional, uses a dummy location if not configured)
- You will need to get your own API Key, create a .env file, and add in GOOGLE_GEOLOCATION_API_KEY=[insert your api key]


## Installation
1. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

2. Configure Google Geolocation API (optional)
     - Get an API key from [Google Cloud Console](https://console.cloud.google.com/)
     - enable the Geolocation API
     - create a .env file, and add in GOOGLE_GEOLOCATION_API_KEY=[insert your api key]
  
3. Ensure hospital database exists:
  - default location:  `us_hospital_locations.csv`
  - A dummt San Francisco dataset is included


## Usage

### Running the App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Enter Patient Information**:
  - Fill in demographics (age, sex, race)
  - Enter height and weight (BMI calculated automatically)
  - Add any chronic conditions if applicable
  - Describe how symptoms started
  - Add symptoms with duration and severity

2. **Get Recommendation**:
  - Click "Get Care Recommendation"
  - Wait for LLM analysis (5-10 seconds)
  - View triage level and reasoning

3. **Emergency Cases**:
  - If triage level is "ER - Immediate"
  - App automatically finds 5 nearest hospitals
  - Shows hospital names, addresses, and distances


### Location Settings

Without Google API key:
- Uses dummy location (San Francisco)

With Google API key:
- Gets actual user location via IP Address
- More accurate hospital distance calculations


## File Structure
```
|-- streamlit_app.py            # Main streamlit application
|-- us_hospital_locations.csv   # Hospital database
|-- requirements.txt            # Python dependencies


## Disclaimers

**Important Notes:**
- This app is for **educational and informational purposes only**
- Does NOT replace professional medical advice
- Always call 911 for true emergencies
- Patient data is NOT stored or transmitted (except to local Ollama)
- Location data is only used for hospital distance calculations
