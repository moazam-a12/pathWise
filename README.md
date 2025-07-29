# Personalized Learning Path Recommendation System

## Overview

This project builds a **course recommendation system** to suggest personalized learning paths for students based on their past performance (`Final_Exam_Score`). Using **Singular Value Decomposition (SVD)** from the Surprise library, it predicts ratings for unrated courses and recommends the top-3 courses for randomly selected students, enriched with metadata (title, subject, level) from `edx.csv`.

### Key Features
- Processes 10,000 student-course interactions from `personalized_learning_dataset.csv`.
- Trains an SVD model with hyperparameter tuning (RMSE ~19.7444, Precision@3 ~0.3333, Recall@3 ~1.0000).
- Maps course names to `edx.csv` metadata using fuzzy matching.
- Generates professional recommendations for random students.

### Datasets
- `personalized_learning_dataset.csv`: Student interactions (`Student_ID`, `Course_Name`, `Final_Exam_Score`).
- `edx.csv`: Course metadata (`title`, `subject`, `level`).

## Setup

### Requirements
- Python 3.8+
- Jupyter Notebook
- pip

### Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Includes pandas, numpy, scikit-learn, scipy, scikit-surprise.
3. Place datasets in `data/`:
   - `personalized_learning_dataset.csv`
   - `edx.csv`

## Usage

1. Open `main.ipynb` in Jupyter Notebook.
2. Run cells in order:
   - **Cell 1**: Installs dependencies.
   - **Cell 2**: Loads and preprocesses data.
   - **Cell 3**: Trains SVD model, shows dataset stats and metrics.
   - **Cell 4**: Sets up fuzzy matching for course metadata.
   - **Cell 5**: Recommends top-3 courses for 3 random students.
3. Check outputs:
   - Cell 3: Model performance (e.g., RMSE: 19.7444).
   - Cell 5: Recommendations, e.g.:
     ```
     Selected random students: [123, 456, 789]
     === Course Recommendations for Random Students ===
     Student ID: 123
     1. Course: Data Science Tools
        Predicted Rating: 66.58
        Subject: Data Analysis & Statistics
        Level: Introductory
     ...
     ```

## Project Structure
```
your-repo-name/
├── data/
│   ├── personalized_learning_dataset.csv
│   ├── Course Content Metadata/
│       ├── edx.csv
├── utils/
│   ├── preprocess.py          # Data preprocessing
├── models/
│   ├── train_model.py         # SVD training
│   ├── recommend.py           # Recommendation generation
├── main.ipynb                 # Main workflow
├── requirements.txt           # Dependencies
├── README.md                  # This file
```

## Notes
- **Performance**: RMSE (~19.7444) is reasonable given data variability (std=20.1). Precision@3 (~0.3333) suggests moderate relevance.
- **Improvements**: Consider a hybrid model or rating normalization to boost personalization.
- **Issues**: Report problems via GitHub Issues.

## Author
Crafted with care by Moazam — NLP enthusiast & ML practitioner in this economy. 😮‍💨
>Transforming feedback into insights.
