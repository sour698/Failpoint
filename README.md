🤖 AI Interview Failure Analyzer



An AI-powered web application that analyzes interview failures by comparing a candidate’s resume with a job description and self-reflection, providing actionable insights, confidence score, and skill gap analysis.



🚀 Live Demo: 

https://failpoint-eluxwmwgkg2gqmbb7qgrzh.streamlit.app/



📌 Features



🔐 Secure Sign In / Sign Up (Email \& Password)



📄 Resume upload (PDF) \& text input



📌 Job Description analysis



🧠 Self-reflection input



🤖 ML-based interview failure prediction



🎯 Confidence gauge (visual)



🔥 Missing skills with percentage gaps



📜 User-specific history tracking



🗑️ Clear history option



📥 Downloadable PDF report



🚪 Logout support



☁️ Deployed for public access



🛠️ Tech Stack



Frontend / UI: Streamlit



Backend Logic: Python



Machine Learning: Scikit-learn



Data Processing: Pandas, NumPy



Visualization: Matplotlib



PDF Handling: PyPDF2, ReportLab



Storage: JSON (users \& history)





📂 Project Structure



AI-Interview-Failure-Analyzer/

│

├── app.py                  # Main Streamlit app

├── auth.py                 # Authentication logic

├── history.py              # User history management

├── explain\_failure.py      # ML inference \& explanation

├── requirements.txt        # Dependencies

├── history.json            # User history storage

│

├── model/

│   ├── failure\_model.pkl   # Trained ML model

│   └── vectorizer.pkl      # Text vectorizer



⚙️ Installation (Local Setup)



1️⃣ Clone the repository



git clone  https://github.com/sour698/Failpoint.git



cd ai-interview-failure-analyzer



2️⃣ Install dependencies

pip install -r requirements.txt



3️⃣ Run the app

streamlit run app.py



🔐 Authentication Flow



App opens on the main page



Sign In / Sign Up available at the top-right corner



Users must log in to:



Upload resume



Enter job description



Run analysis



History is stored per user



Logout clears session securely



📊 Output Details



Failure Reason – predicted interview failure cause



Confidence Score – prediction confidence



Resume–JD Match – similarity percentage



Missing Skills – listed with gap percentages



Explanation – human-readable AI reasoning



PDF Report – downloadable summary



☁️ Deployment



The app is deployed using Streamlit Community Cloud, making it accessible to anyone via a public URL.



Steps used:



Pushed project to GitHub



Added requirements.txt



Deployed via share.streamlit.io



🎓 Academic Use



This project is suitable for:



Final year project



Mini project



AI / ML coursework



Interview demonstrations



📈 Future Enhancements



Database integration (MongoDB / Firebase)



Admin dashboard with analytics



Google OAuth login



Permanent cloud storage



Interview question recommendations



👤 Author



Sourav Das

AI \& ML Enthusiast



❤️ Acknowledgements



Built using Streamlit and Scikit-learn with explainable AI principles."# Failpoint" 



