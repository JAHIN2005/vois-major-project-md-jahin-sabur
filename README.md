# vois-major-project-md-jahin-sabur
AI-Based Network Intrusion Detection System using Machine Learning and Streamlit for VOIS Major Project submission.
# AI-Based Network Intrusion Detection System

This project is developed as a **Major Project submission** under the VOIS program.
It implements an AI-based Network Intrusion Detection System (NIDS) using Machine
Learning techniques to detect malicious network traffic and explain the results.

The system uses a Random Forest classifier trained on real-world network traffic data
and a Streamlit-based interactive dashboard. Additionally, Groq AI is integrated to
generate human-readable explanations for detected network intrusions.

---

## Features
- Detection of network intrusions such as DDoS attacks
- Machine learning based classification using Random Forest
- Real-time packet simulation from test data
- Interactive Streamlit dashboard
- AI-based explanation of detection results using Groq

---

## Technologies Used
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Random Forest Algorithm
- Groq Large Language Model (LLM)

---

## Dataset
CIC-IDS2017 – Friday Working Hours Afternoon DDoS dataset.

Due to GitHub file size limitations, the dataset is **not uploaded** to this repository.
The dataset is publicly available and can be downloaded from the official source:

https://www.unb.ca/cic/datasets/ids-2017.html

After downloading, place the CSV file in the same directory as `nids_main.py` before
running the application.

---

## How to Run the Project

1. Install the required dependencies:
pip install -r requirements.txt

2. Run the Streamlit application:
streamlit run nids_main.py

3. Open the application in your browser:
http://localhost:8501


---

## Project Structure
- `nids_main.py` – Main application source code
- `requirements.txt` – List of required Python libraries
- `screenshots/` – Project execution and result screenshots
- `certificates/` – Course completion certificates

---

## Author
Md Jahin Sabur  
