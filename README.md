CODSOFT Data Science Internship – Task 3
🌸 Iris Flower Classification 🌸

This repository contains my work for the CODSOFT Data Science Internship.  
Task 3: Iris Flower Classification – a classic machine learning project to classify Iris flowers into three species (setosa, versicolor, virginica) using sepal and petal measurements.

📌 Project Overview
Classic beginner ML project that never gets old!  
Using the famous **Iris dataset** to build models that predict flower species from petal & sepal lengths/widths.

🚀 Objectives
- Perform beautiful EDA with pairplots, histograms & class distribution
- Train & compare 5 powerful classifiers
- Achieve near-perfect accuracy (because Iris is super clean!)
- Build an interactive Streamlit web app for real-time predictions 🎉

📊 Dataset
- **150 samples**, 4 features + species label
- Features: sepal_length, sepal_width, petal_length, petal_width
- Classes: Iris-setosa, Iris-versicolor, Iris-virginica (perfectly balanced!)
- Source: Standard UCI / Seaborn Iris dataset

⚙️ Technologies Used
- 🐍 Python  
- 🐼 Pandas & NumPy  
- 📈 Matplotlib & Seaborn (gorgeous visuals!)  
- 🤖 Scikit-learn (ML models & tools)  
- 🌐 Streamlit (interactive demo app)

🧠 Machine Learning Pipeline
1. Load & explore the data → pairplots & countplots  
2. Preprocess: encode labels, split train/test (80/20)  
3. Train 5 models:  
   - K-Nearest Neighbors (KNN)  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Support Vector Machine (SVM)  
4. Compare accuracies → all hit ~100%!  
5. Show confusion matrix for the winner  
6. Deploy fun Streamlit app with sliders → predict any flower instantly!

📁 Files in this Repository- 
| File                        | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `IRIS.csv`                  | The Iris dataset (150 samples, 4 features + species label)                  |
| `iris_csv_classifier.py`    | Full analysis: EDA, model training, comparison, confusion matrix            |
| `app.py`                    | Streamlit web app for interactive predictions                               |
| `requirements.txt`          | List of required Python libraries                                           |

🚀 How to Run Locally
1. Clone the repository
   ```bash
   git clone https://github.com/enigma-script/codsoft_task3.git
   cd codsoft_task3
2. (Recommended) Create a virtual environment
python -m venv venv
source venv\Scripts\activate
3. Install dependencies: 
pip install -r requirements.txt
4. Run the analysis script: 
python iris_csv_classifier.py→ Plots and model accuracies will appear.
5. Launch the interactive Streamlit app: 
streamlit run app.py→ Open http://localhost:8501 in your browser and play with the sliders!




About the Internship:
This is Task 3 of the CODSOFT Data Science Internship.
I'm working on completing at least 3 tasks and will add more repositories.

Connect with Me-
LinkedIn: www.linkedin.com/in/khadeejah-shaikh-741001395
GitHub: https://github.com/enigma-script

#codsoft #datascience #machinelearning #internship #python #streamlit

Feel free to reach out if you'd like to collaborate or discuss ML projects! 🚀

