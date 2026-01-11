from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_regression_pipeline(numeric_features, categorical_features):
    """
    Creates a professional ML pipeline:
    1. Scales numbers (StandardScaler)
    2. Encodes text (OneHotEncoder)
    3. Trains a Random Forest
    """
    # Preprocessing for numerical data
    numeric_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for both types of data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return model

from sklearn.ensemble import RandomForestClassifier

def get_classification_pipeline(numeric_features, categorical_features):
    """
    Creates a classification pipeline:
    1. Scaling and Encoding
    2. Random Forest Classifier
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    return model

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_clustering_pipeline(n_clusters=4):
    """
    Creates a simple K-Means model. 
    Note: Clustering usually happens after manual scaling.
    """
    return KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)

import tensorflow as tf
from tensorflow.keras import layers, models

def get_deep_learning_model(input_shape):
    """
    Creates a Neural Network using TensorFlow/Keras.
    """
    model = models.Sequential([
        # Layer 1: 64 neurons, learning patterns in the data
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2), # Randomly shuts off neurons to prevent 'memorization' (overfitting)
        
        # Layer 2: 32 neurons
        layers.Dense(32, activation='relu'),
        
        # Output Layer: 1 neuron with Sigmoid (converts output to a 0-1 probability)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model