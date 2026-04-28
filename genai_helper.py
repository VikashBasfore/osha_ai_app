import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# Load API key
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def explain_prediction(row_data, prediction, confidence):
    try:
        prompt = f"""
        You are a professional workplace safety analyst.

        Analyze the following workplace incident and provide a structured explanation.

        Give answers in the following sections:

        1. Root Cause (4–6 lines)
        Explain why the incident happened in detail.

        2. Risk Factors (4–6 lines)
        Describe the key hazards and unsafe conditions involved.

        3. Chances of Recurrence (4–6 lines)
        Explain how likely this incident is to happen again and why.

        4. Prevention & Solutions (4–6 lines)
        Explain what actions the company should take to prevent this.

        5. Worker Treatment (4–6 lines)
        Explain what medical treatment the injured worker should receive.

        6. Long-term Safety Measures (4–6 lines)
        Explain what improvements should be made to avoid future incidents.

        Incident details:
        {row_data}

        Prediction: {prediction}
        Confidence: {confidence}

        Instructions:
        - Write each section clearly with its heading
        - Each section must have 4–6 lines
        - Use simple professional language
        - Highlight key terms using **bold**
        - Do NOT use HTML
        - Keep it structured and readable
        """

        # Retry logic for API overload
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                text = response.text
                return text.strip()

            except Exception as e:
                if "503" in str(e):
                    time.sleep(5)   # retry after delay
                elif "429" in str(e):
                    return "🚫 Daily AI limit reached. Please try again tomorrow."
                else:
                    return f"AI Error: {str(e)}"

        return "⚠ AI is busy. Please try again in a few seconds."

    except Exception as e:
        return f"AI Error: {str(e)}"
