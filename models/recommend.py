import difflib

def get_recommendations(user_id, model, interaction_df, course_encoder, course_name_to_title, course_df, top_n=3):
    """
    Generate top-N course recommendations for a user.
    
    Parameters:
    - user_id: ID of the user to generate recommendations for
    - model: Trained SVD model
    - interaction_df: DataFrame with columns ['user_id', 'course_id', 'rating']
    - course_encoder: LabelEncoder for course IDs
    - course_name_to_title: Dict mapping course names to edX titles
    - course_df: DataFrame with course metadata
    - top_n: Number of recommendations to return (default: 3)
    
    Returns:
    - List of dicts with course_title, predicted_rating, and metadata
    """
    try:
        all_course_ids = interaction_df['course_id'].unique()
        rated_courses = interaction_df[interaction_df['user_id'] == user_id]['course_id'].values
        unrated_courses = [cid for cid in all_course_ids if cid not in rated_courses]
        
        if not unrated_courses:
            print(f"No unrated courses available for user {user_id}.")
            return []
        
        # Predict ratings
        predictions = [model.predict(user_id, cid) for cid in unrated_courses]
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        # Format recommendations
        recommendations = []
        for pred in predictions[:top_n]:
            course_name = course_encoder.inverse_transform([pred.iid])[0]
            mapped_title = course_name_to_title.get(course_name, course_name)
            course_info = course_df[course_df['title'] == mapped_title][['title', 'subject', 'level']]
            rec = {
                'course_title': mapped_title,
                'predicted_rating': round(pred.est, 2),
                'metadata': course_info.to_dict('records')[0] if not course_info.empty else {'title': mapped_title, 'subject': 'N/A', 'level': 'N/A'}
            }
            recommendations.append(rec)
        
        return recommendations
    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {str(e)}")
        return []