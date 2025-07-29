import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(data_path, course_metadata_path):
    # Load datasets
    student_df = pd.read_csv(data_path)
    course_df = pd.read_csv(course_metadata_path)

    # Drop rows with missing values
    student_df.dropna(subset=['Student_ID', 'Course_Name', 'Final_Exam_Score'], inplace=True)

    # Encode categorical values
    user_encoder = LabelEncoder()
    course_encoder = LabelEncoder()
    student_df['user_id'] = user_encoder.fit_transform(student_df['Student_ID'])
    student_df['course_id'] = course_encoder.fit_transform(student_df['Course_Name'])

    # Use Final_Exam_Score as rating (already 0–100 scale)
    student_df['rating'] = student_df['Final_Exam_Score']

    # Create interaction matrix
    interaction_df = student_df[['user_id', 'course_id', 'rating']]

    return interaction_df, user_encoder, course_encoder, course_df