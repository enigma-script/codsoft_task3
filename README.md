CODSOFT Data Science Internship – Task 3: Iris Flower Classification 🌸

This repository contains my work for the CODSOFT Data Science Internship.  
Task 3: Iris Flower Classification – a classic machine learning project to classify Iris flowers into three species (setosa, versicolor, virginica) using sepal and petal measurements.

Project Overview
- Performed Exploratory Data Analysis (EDA) with visualizations (pairplots, class distribution, histograms).
- Trained and compared 5 classification models:  
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
  → All models achieved ~100% accuracy on this well-separated dataset!
- Built an interactive web application using Streamlit for real-time species prediction with user-friendly sliders.

Files in this Repository (Task 3)
| File                        | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `IRIS.csv`                  | The Iris dataset (150 samples, 4 features + species label)                  |
| `iris_csv_classifier.py`    | Full analysis: EDA, model training, comparison, confusion matrix            |
| `app.py`                    | Streamlit web app for interactive predictions                               |
| `requirements.txt`          | List of required Python libraries                                           |

How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/CODSOFT.git
   cd CODSOFT
2. (Recommended) Create a virtual environment
python -m venv venv
source venv\Scripts\activate
3. Install dependencies: 
pip install -r requirements.txt
4. Run the analysis script: 
python iris_csv_classifier.py→ Plots and model accuracies will appear.
5. Launch the interactive Streamlit app: 
streamlit run app.py→ Open http://localhost:8501 in your browser and play with the sliders!


Results Highlights

Best Model: KNN (perfect classification on test set)
Confusion Matrix: No misclassifications
Interactive Demo: Enter any sepal/petal measurements → instant species prediction

About the Internship
This is Task 3 of the CODSOFT Data Science Internship.
I'm working on completing at least 3 tasks (as required) and will add more folders like task1_titanic, task2_movie_rating, etc., to this same repository.
Technologies Used

Python
Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn (for ML models)
Streamlit (for web app)

Connect with Me

LinkedIn: www.linkedin.com/in/khadeejah-shaikh-741001395
GitHub: https://github.com/enigma-script

#codsoft #datascience #machinelearning #internship #python #streamlit
Feel free to reach out if you'd like to collaborate or discuss ML projects! 🚀