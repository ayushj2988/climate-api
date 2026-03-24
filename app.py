from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import time
from groq import Groq  
app = Flask(__name__)
CORS(app) 

# ==========================================
# ==========================================
client = Groq(api_key="sk-xxxx")

print("Loading ML Model...")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'climate_rf_model.pkl')
columns_path = os.path.join(current_dir, 'model_columns.pkl')

rf_model = joblib.load(model_path)
model_columns = joblib.load(columns_path)
print("Model Loaded Successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        country = data.get('country', 'India')
        co2_reduction = data.get('co2Reduction', 0)
        temp_reduction_slider = data.get('tempReduction', 0)
        needs_ai = data.get('needsAI', True)
        
        # --- ML DATA PREP ---
        input_data = {
            'co2_concentration_ppm': 450.0, 'annual_rainfall_mm': 1000.0,
            'sea_level_rise_mm': 5.0, 'sea_surface_temperature': 17.0,
            'heatwave_days': 40, 'drought_index': 3.0, 'flood_events_count': 6,
            'forest_cover_percent': 25.0, 'deforestation_rate': 2.0,
            'fossil_fuel_consumption': 80.0, 'renewable_energy_share': 20.0,
            'air_quality_index': 120, 'climate_risk_index': 50.0, 'year': 2030
        }
        
        if co2_reduction > 0:
            factor = co2_reduction / 100
            input_data['co2_concentration_ppm'] *= (1 - factor)
            input_data['air_quality_index'] *= (1 - (factor / 2))
            
        if temp_reduction_slider > 0:
            policy_factor = temp_reduction_slider / 30 
            input_data['fossil_fuel_consumption'] *= (1 - policy_factor)
            input_data['deforestation_rate'] *= (1 - policy_factor)
            input_data['renewable_energy_share'] += (policy_factor * 40)
            
        input_df = pd.DataFrame([input_data])
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 1 if col == f"country_{country}" else 0
        input_df = input_df[model_columns]
        
        # 1. Random Forest ML Prediction
        prediction = rf_model.predict(input_df)[0]
        anomaly_result = round(prediction, 3)

        # 2. GROQ AI MAGIC (Free & Extremely Fast)
        dynamic_insight = ""
        
        if needs_ai:
            try:
                # Delay taaki API limit safe rahe
                time.sleep(1.5) 
                
                ai_prompt = f"""
                Context: The user is looking at {country}. 
                They applied a {co2_reduction}% reduction in CO2 emissions and a strictness level of {temp_reduction_slider}/30.
                Our ML model predicts the temperature anomaly will be {anomaly_result}°C by 2050.
                
                Task: Write a punchy, 2-sentence insight. Make it urgent if the value is high (>1.2), and encouraging if it drops. Return only plain text without bolding.
                """
                
                # NAYA GROQ API CALL (Using Llama 3)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a professional climate data analyst."},
                        {"role": "user", "content": ai_prompt}
                    ],
                    model="llama-3.1-8b-instant", # Naya aur aur bhi zyada fast model!, # Ye model bohot fast aur free hai, # Ye model bohot fast aur free hai
                    temperature=0.7,
                    max_tokens=60,
                )
                
                dynamic_insight = chat_completion.choices[0].message.content.strip()
                
            except Exception as ai_error:
                print("🚨 GROQ ERROR:", str(ai_error))
                print("⚠️ AI API Failed! Using Fallback Text.")
                if anomaly_result > 1.2:
                    dynamic_insight = f"🚨 Critical: Projected anomaly is high ({anomaly_result}°C). Stricter policies required."
                else:
                    dynamic_insight = f"✅ Stabilizing: Projected anomaly is {anomaly_result}°C. Current interventions are working."

        return jsonify({
            "status": "success",
            "country": country,
            "applied_co2_reduction_percent": co2_reduction,
            "predicted_temperature_anomaly": anomaly_result,
            "ai_generated_insight": dynamic_insight
        })

    except Exception as e:
        print("🚨 Backend Error:", str(e))
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("🚀 Starting Flask Server on Port 5000...")
    app.run(debug=True, port=5000)