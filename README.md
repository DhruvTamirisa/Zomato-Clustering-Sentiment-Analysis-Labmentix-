# Zomato Restaurant Analysis and Clustering

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20NLTK-orange.svg)
![ML](https://img.shields.io/badge/ML-Clustering%20%7C%20Sentiment%20Analysis-green.svg)

---
## Project Overview

This project performs a comprehensive analysis of Zomato's restaurant data in India. It begins with data cleaning and exploratory data analysis (EDA) to uncover key insights about the restaurant market. It then applies two types of machine learning: **unsupervised clustering** to segment restaurants into distinct groups and **supervised classification** to perform sentiment analysis on customer reviews. The ultimate goal is to transform raw data into actionable business intelligence for both Zomato and its users.

---
## Business Problem

For customers, the sheer number of restaurant choices on Zomato can be overwhelming. For Zomato, the vast amount of user-generated data is a valuable asset that needs to be leveraged for business growth. This project tackles these challenges by:
1.  Using clustering to create intuitive restaurant segments, helping users discover places that match their specific needs beyond simple filters.
2.  Using sentiment analysis to understand the *why* behind customer ratings, providing deep insights for quality control and restaurant improvement.

---
## Tech Stack & Libraries

This project is implemented in a Jupyter Notebook and utilizes the following Python libraries:
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Text Processing:** NLTK, WordCloud, Re
* **Machine Learning:** Scikit-learn

---
## Project Workflow

The project is broken down into a series of logical steps:

1.  **Data Cleaning:** Handled missing values, removed duplicate entries, and corrected data types (e.g., converting the 'Cost' column to a number).
2.  **Exploratory Data Analysis (EDA):** Created over 15 visualizations to uncover patterns. Key insights included identifying the most popular cuisines, the distribution of restaurant costs, and the relationship between cost and rating.
3.  **Feature Engineering:** Created new, valuable features from the existing data, such as `City`, `review_length`, and reviewer follower counts. Categorical features like `Cuisines` and `City` were encoded into a numerical format.
4.  **Text Preprocessing:** Cleaned and normalized the raw review text by converting it to lowercase, removing punctuation and stopwords, and performing lemmatization.
5.  **Model Implementation:**
    * **Model 1 (K-Means Clustering):** A baseline unsupervised model to group restaurants into `k` segments.
    * **Model 2 (Naive Bayes):** A supervised model to classify the sentiment of reviews as 'Positive' or 'Negative'.
    * **Model 3 (DBSCAN):** A more advanced unsupervised model that discovered clusters based on data density and identified outlier restaurants.
6.  **Model Evaluation:** Evaluated the models using appropriate metrics: **Silhouette Score** for clustering and **Accuracy/Classification Report** for sentiment analysis.

---
## How to Run This Project

To replicate this analysis, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-folder>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
    ```
3.  **Download NLTK data:**
    Run the following commands in a Python interpreter or the first cell of your notebook:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```
4.  **Place the datasets** (`Zomato Restaurant names and Metadata.csv` and `Zomato Restaurant reviews.csv`) in the root directory of the project.
5.  **Run the Jupyter Notebook:** `Sample_ML_Submission_Template.ipynb`.

---
## Key Insights

* **Cost does not equal Quality:** There is no significant correlation between a restaurant's cost and its average rating.
* **Budget-Friendly Market:** The vast majority of restaurants are in the budget to mid-range price category.
* **Granular Market Segments:** The DBSCAN model revealed 36 distinct, dense clusters, suggesting a highly segmented market with many niche categories.
* **High Sentiment Prediction Accuracy:** The Naive Bayes model was able to predict review sentiment with ~91% accuracy.
* **Key Sentiment Drivers:** The most important words for positive sentiment were related to food quality and good experiences (e.g., 'good', 'food', 'place', 'great'), while negative sentiment was driven by words related to poor service and bad food.