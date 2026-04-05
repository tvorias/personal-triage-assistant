"""
Streamlit app for patient triage recommendation.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import requests
import ollama
import os
from dotenv import load_dotenv
from math import radians, sin, cos, sqrt, atan2

# Load environment variables from .env file
load_dotenv()

# Configuration
GOOGLE_GEOLOCATION_API_KEY = os.getenv("GOOGLE_GEOLOCATION_API_KEY")
HOSPITALS_CSV = 'us_hospital_locations.csv'

def calculate_bmi(weight_lbs, height_in):
    """Calculate BMI given weight in lbs and height in inches."""
    bmi = (weight_lbs / (height_in ** 2)) * 703
    return round(bmi, 1)

def feet_inches_to_inches(feet, inches):
    """Convert feet and inches to total inches."""
    return (feet * 12) + inches

def get_current_location():
    """Get user;s current location using Google Geolocation API.

    Returns:
        tuple: (latitude, longitude) of the user's current location or SF if location cannot be determined.
    """
    if not GOOGLE_GEOLOCATION_API_KEY:
        st.warning("Google Geolocation API key is not set. Unable to determine current location.")
        return 37.7749, -122.4194 # Default to San Francisco

    try:
        url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_GEOLOCATION_API_KEY}"
        #response = requests.post(url, json={})

        # if response.status_code == 200:
        #     data = response.json()
        #     location = data.get('location', {})
        #     return location.get('lat', 37.7749), location.get('lng', -122.4194)
        # else:
        #     return 37.7749, -122.4194
        IP = {'considerIP': True}
        google_request = requests.post(url, IP)
        google_data = json.loads(google_request.text)
        cur_latitude = google_data['location']['lat']
        cur_longitude = google_data['location']['lng']
        return cur_latitude, cur_longitude

    except Exception as e:
        st.error(f"Error fetching location: {e}")
        return 37.7749, -122.4194
    
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in miles

    Args:
        lat1 (float): Latitude of point 1
        lon1 (float): Longitude of point 1
        lat2 (float): Latitude of point 2
        lon2 (float): Longitude of point 2
    
    Returns:
        float: Distance in miles
    """
    R = 3958.8 # Radius of Earth in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance = R * c
    return distance

def find_nearest_hospitals(user_lat, user_lon, hospitals_df, top_n=5):
    """Find the nearest hospitals to the user's location.

    Args:
        user_lat (float): User's latitude
        user_lon (float): User's longitude
        hospitals_df (pd.DataFrame): DataFrame containing hospital locations
        top_n (int): Number of nearest hospitals to return
    
    Returns:
        pd.DataFrame: DataFrame of nearest hospitals with distance
    """
    # Calculate distance from user location to each hospital
    hospitals_df['distance'] = hospitals_df.apply(
        lambda row: haversine_distance(user_lat, user_lon, row['LATITUDE'], row['LONGITUDE']), axis=1)
    
    # Sort hospitals by distance and return the top N
    nearest_hospitals = hospitals_df.nsmallest(top_n, 'distance')
    return nearest_hospitals[['NAME', 'ADDRESS', 'CITY', 'STATE', 'ZIP', 'distance']]

def generate_triage_recommendation(patient_data):
    """Generate triage recommendation using LLM based on user-provided information.

    Args:
        patient_data (dict): Dictionary containing patient information and symptoms
    
    Returns:
        str: Triage recommendation with triage_level and reasoning
    """
    # Format symptoms for display
    symptoms_text = []
    
    for symptom in patient_data['symptoms']:
        symptoms_text.append(f"- {symptom['name']}: {symptom['severity']} severity, duration {symptom['duration']}")
    
    # Build prompt with user's actual symptoms and information
    prompt = f"""You are a medical triage assistant. Based on the following patient information, determine the appropriate level of care.

Patient Information:
- Age: {patient_data['age']} years old
- Sex: {patient_data['sex']}
- BMI: {patient_data['bmi']}
- Existing Conditions: {', '.join(patient_data['comorbidities']) if patient_data['comorbidities'] else 'None'}

Current symptoms:
{chr(10).join(symptoms_text)}

Onset: {patient_data['onset']}

Determine the appropriate triage level. Return ONLY valid JSON:
{{
    "triage_level": "one of: ER - Immediate, Urgent Care - Within 24 hours, See Physician - Within Few Days, See Physician - Within Few Weeks, Monitor Symptoms, Self-Care",
    "reasoning": "Brief explanation of why this triage level was chosen based on the symptoms and patient information."
}}

Triage Guidelines:
- ER - Immediate: Severe symptoms such as chest pain, difficulty breathing, severe bleeding, loss of consciousness, severe trauma, or any life-threatening condition.
- Urgent Care - Within 24 hours: Moderate symptoms such as high fever, persistent vomiting, severe abdominal pain, worsening symptoms, or any condition that requires prompt medical attention but is not immediately life-threatening.
- See Physician - Within Few Days: Mild symptoms such as mild fever, cough, sore throat, minor injuries, or any condition that can wait a few days but should be evaluated by a healthcare professional.
- See Physician - Within Few Weeks: Chronic symptoms, routine check-ups, or any condition that is not urgent but should be evaluated by a healthcare professional within a few weeks.
- Monitor Symptoms: Mild symptoms that do not require immediate medical attention but should be monitored for any changes or worsening.
- Self-Care: Minor symptoms that can be managed at home with over-the-counter treatments and do not require medical evaluation unless symptoms worsen.
"""

    # Get response from LLM
    try:
        response = ollama.chat(model="llama3.2", 
                               messages=[{"role": "user", "content": prompt}],
                               format='json',
                               options={"temperature": 0.3} # Lower temperature for more deterministic output
                            )
        
        content = response['message']['content']

        # Extract JSON from response
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end+1]
            
        result = json.loads(content)

        # Normalize triage level capitalization to match exact categories
        triage_categories = [
            "ER - Immediate", 
            "Urgent Care - Within 24 hours", 
            "See Physician - Within Few Days", 
            "See Physician - Within Few Weeks", 
            "Monitor Symptoms", 
            "Self-Care"
        ]

        triage_lower = result['triage_level'].lower()
        for canonical in triage_categories:
            if triage_lower == canonical.lower():
                result['triage_level'] = canonical
                break
        return result
    
    except Exception as e:
        st.error(f"Error generating triage recommendation: {e}")
        return {
            "triage_level": "See Physcian - Within Few Days",
            "reasoning": "Unable to generate recommendation. Please consult a healthcare professional."
        }

def main():
    st.set_page_config(
        page_title="Personal Triage Assistant",
        page_icon="🩺",
        layout="wide"
    )

    st.title('Patient Triage Assistant')
    st.markdown("Get personalized healthcare recommendation based on your symptoms.")

    # Create two columns
    col1, col2 = st.columns([2,1])

    with col1:
        st.header("Patient Information")

        # Demographics
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=120, value=30)
        sex = st.selectbox('Sex', options=["Male", "Female"])
        race = st.selectbox("Race/Ethnicity", options=[
                    "White", 
                    "Black or African American", 
                    "Asian", 
                    "Hispanic or Latino", 
                    "Other"
        ])

        # Physical Measurements
        st.subheader("Physical Measurements")
        col_height1, col_height2 = st.columns(2)
        with col_height1:
            height_feet = st.number_input("Height (feet)", min_value=3, max_value=8, value=5)
        with col_height2:
            height_inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=6)
        
        weight_lbs = st.number_input("Weight (pounds))", min_value=50, max_value=500, value=150)

        # Calculate BMI
        total_height_inches = feet_inches_to_inches(height_feet, height_inches)
        bmi = calculate_bmi(weight_lbs, total_height_inches)

        st.info(f"Calculated BMI: {bmi}")

        # Comorbidities
        st.subheader("Medical history")
        has_comorbidites = st.checkbox("Do you have any existing medical conditions?")
        comorbidities = []

        if has_comorbidites:
            comorbidities_text = st.text_area(
                "List your conditions (one per line)",
                placeholder="e.g. Diabetes\nHypertension\nAsthma"
            )
            comorbidities = [c.strip() for c in comorbidities_text.split('\n') if c.strip()]

        # Symptoms
        st.subheader("Current Symptoms")
        onset = st.text_input(
            'When or how did your symptoms start?',
            placeholder="e.g. sudden, gradual over 2 days, after exercise"
        )

        st.markdown("**Add your symptoms below:**")
        num_symptoms = st.number_input("Number of symptoms", min_value=1, max_value=10, value=1)

        symptoms = []
        for i in range(num_symptoms):
            st.markdown(f"**Symptom {i+1}:**")
            col_s1, col_s2, col_s3 = st.columns(3)

            with col_s1:
                symptom_name = st.text_input(
                    "Symptom {i+1} name", 
                    key=f"symptom_name_{i}",
                    placeholder="e.g. cough, chest pain"
                )

            with col_s2:
                duration = st.text_input(
                    "Duration", 
                    key=f"duration_{i}",
                    placeholder="e.g., 2 days, 3 hours"
                )
            
            with col_s3:
                severity = st.selectbox(
                    "Severity",
                    options=["Mild", "Moderate", "Severe"],
                    key=f"severity_{i}"
                )

            if symptom_name and duration:
                symptoms.append({
                    "name": symptom_name,
                    "duration": duration,
                    "severity": severity
                })
        
        # Submit button
        if st.button("Get Care Recommendation", type="primary"):
            if not symptoms:
                st.warning("Please enter at least one symptom.")
            elif not onset:
                st.warning("Please provide information about symptom onset.")
            else:
                # Store data in session state
                st.session_state.patient_data = {
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'weight_lbs': weight_lbs,
                    'height_in': total_height_inches,
                    'bmi': bmi,
                    'comorbidities': comorbidities if comorbidities else ['None'],
                    'symptoms': symptoms,
                    'onset': onset
                }

                st.session_state.show_recommendation = True
    
    with col2:
        st.header("Care Recommendation")
        if st.session_state.get('show_recommendation', False):
            recommendation = generate_triage_recommendation(st.session_state.patient_data)

            triage_level = recommendation.get('triage_level', 'Unknown')
            reasoning = recommendation.get('reasoning', 'No reasoning provided.')

            # Display triage level 
            if triage_level == 'ER - Immediate':
                st.error(f"**{triage_level}**")
            elif triage_level == 'Urgent Care - Within 24 hours':
                st.warning(f"**{triage_level}**")
            elif triage_level in ['See Physician - Within Few Days', 'See Physician - Within Few Weeks']:
                st.info(f"**{triage_level}**")
            elif triage_level == 'Monitor Symptoms':
                st.success(f"**{triage_level}**")
            elif triage_level == 'Self-Care':
                st.success(f"**{triage_level}**")
            else:
                st.info(f"**Triage Level: {triage_level}**")
            
            if reasoning:
                st.markdown(f"**Reasoning:** {reasoning}")

            # If ER, find nearest hospitals
            if triage_level == 'ER - Immediate':
                st.markdown("---")
                st.subheader("Nearest Hospitals")
                
                with st.spinner("Finding nearest hospitals..."):
                    # Get user's current location
                    user_lat, user_lon = get_current_location()

                    if user_lat and user_lon:
                        try:
                            # Load hospitals
                            hospital_df = pd.read_csv(HOSPITALS_CSV)
                            nearest_hospitals = find_nearest_hospitals(user_lat, user_lon, hospital_df)

                            st.success("Found nearby hospitals:")

                            for idx, hospital in nearest_hospitals.iterrows():
                                st.markdown(f"""
                                **{hospital['NAME']}** 
                                {hospital['ADDRESS']}, {hospital['CITY']}, {hospital['STATE']}, {hospital['ZIP']}  
                                {hospital['distance']:.2f} miles away
                                 """)
                                st.markdown("---")
                        except Exception as e:
                            st.error(f"Error loading hospital data: {e}")
                    else:
                        st.error("Unable to determine current location.")
                
                st.markdown("---")
                st.caption("This is a tool for informational purposes only and does replace medical advice.")
        else:
            st.info("Please enter your information and symptoms to get a care recommendation.")
    
if __name__ == "__main__":
    main()
