from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def get_data_directories() -> Tuple[str, str]:
    """ Returns the raw and preprocessed data directories.
    
    Returns:
        RAW_DATA_DIR (str): Directory for raw data.
        PRE_DATA_DIR (str): Directory for preprocessed data.
    """
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PRE_DATA_DIR = DATA_DIR / "preprocessed"

    error_message = lambda x: f"Unable to find data directory {x}, please check the project structure."

    assert DATA_DIR.exists(), error_message(DATA_DIR)
    assert RAW_DATA_DIR.exists(), error_message(RAW_DATA_DIR)
    assert PRE_DATA_DIR.exists(), error_message(PRE_DATA_DIR)

    print("Data directories successfully set.")

    return RAW_DATA_DIR, PRE_DATA_DIR

def save_dataset_to_csv(dataset: pd.DataFrame, dataset_filename: str):
    """ Saves dataset to a CSV file in the preprocessed data directory.
    
    Args:
        dataset (pandas.DataFrame): The dataset to be saved.
        dataset_filename (str): File name of the dataset when saved.
    """
    data_directories = get_data_directories()
    output_path = data_directories[1] / f"{dataset_filename}.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved in {data_directories[1]}")

def preprocessor(data: pd.DataFrame):
    # Define columns
    numeric_features = ['age', 'study_hours_per_day', 'attendance_percentage', 
                        'sleep_hours', 'exercise_frequency', 'mental_health_rating', 
                        'total_screen_time']

    nominal_features = ['gender', 'part_time_job', 'internet_quality', 'extracurricular_participation']

    ordinal_features = ['diet_quality', 'parental_education_level']
    ordinal_categories = [
        ['Poor', 'Fair', 'Good'],  # diet_quality
        ['No education', 'High School', 'Bachelor', 'Master']  # parental education
    ]

    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Nominal pipeline
    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Ordinal pipeline
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=ordinal_categories))
    ])

    # Combine all
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('nom', nominal_transformer, nominal_features),
        ('ord', ordinal_transformer, ordinal_features)
    ])

    X = preprocessor.fit_transform(data)
    return X