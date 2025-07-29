from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from collections import defaultdict
import numpy as np

def evaluate_top_n(predictions, n=3, threshold=70):
    """Evaluate precision@k and recall@k for top-N recommendations."""
    user_ratings = defaultdict(list)
    for pred in predictions:
        user_ratings[pred.uid].append((pred.iid, pred.r_ui, pred.est))
    
    precision, recall = [], []
    for uid, ratings in user_ratings.items():
        ratings.sort(key=lambda x: x[2], reverse=True)
        top_n = [iid for (iid, _, _) in ratings[:n]]
        relevant = [iid for (iid, r_ui, _) in ratings if r_ui >= threshold]
        if relevant:
            true_positives = len(set(top_n) & set(relevant))
            precision.append(true_positives / n)
            recall.append(true_positives / len(relevant))
    
    return np.mean(precision) if precision else 0, np.mean(recall) if recall else 0

def train_svd_model(interaction_df, rating_scale=(0, 100), test_size=0.2, random_state=42, tune_params=False, verbose=False):
    """
    Train an SVD model for collaborative filtering and evaluate RMSE, Precision@3, Recall@3.
    
    Parameters:
    - interaction_df: DataFrame with columns ['user_id', 'course_id', 'rating']
    - rating_scale: Tuple of (min, max) rating values (default: 0–100)
    - test_size: Proportion of data for test set (default: 0.2)
    - random_state: Seed for reproducibility (default: 42)
    - tune_params: If True, perform grid search for hyperparameter tuning
    - verbose: If True, print detailed training steps
    
    Returns:
    - model: Trained SVD model
    - trainset: Surprise trainset
    - testset: Surprise testset
    - rmse: Root Mean Squared Error on testset
    - precision: Precision@3
    - recall: Recall@3
    """
    if verbose:
        print(f"Training model on {len(interaction_df)} interactions...")
    
    # Load data into Surprise
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(interaction_df[['user_id', 'course_id', 'rating']], reader)

    # Hyperparameter tuning
    if tune_params:
        if verbose:
            print("Performing hyperparameter tuning...")
        param_grid = {
            'n_factors': [10, 20, 50, 100],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.02, 0.05, 0.1]
        }
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
        gs.fit(data)
        if verbose:
            print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
            print(f"Best parameters: {gs.best_params['rmse']}")
        model = SVD(**gs.best_params['rmse'], random_state=random_state)
    else:
        model = SVD(n_factors=20, random_state=random_state)

    # Split data
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
    if verbose:
        print(f"Training set: {trainset.n_ratings} ratings, Test set: {len(testset)} ratings")

    # Train model
    model.fit(trainset)
    if verbose:
        print("SVD model training completed.")

    # Evaluate model
    if verbose:
        print("Evaluating model...")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    precision, recall = evaluate_top_n(predictions, n=3, threshold=70)
    
    # Print summary
    print("\n=== Model Training Summary ===")
    print(f"Number of Interactions: {len(interaction_df)}")
    print(f"Training Set Size: {trainset.n_ratings} ratings")
    print(f"Test Set Size: {len(testset)} ratings")
    if tune_params:
        print(f"Best Parameters: {gs.best_params['rmse']}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision@3: {precision:.4f}")
    print(f"Recall@3: {recall:.4f}")
    print("=============================\n")

    return model, trainset, testset, rmse, precision, recall