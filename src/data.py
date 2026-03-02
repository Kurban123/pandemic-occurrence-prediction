import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import MinMaxScaler

class PandemicDataLoader:
    """
    Class for loading, preprocessing, and vectorizing pandemic dataset.
    Handles feature scaling and sequence generation for LSTM models.
    """
    
    def __init__(self, data_path: Path, look_back: int = 5):
        self.data_path = data_path
        self.look_back = look_back
        self.feature_cols = ['Interval', 'Severity', 'Duration', 'Population', 'Urbanization', 'Trade_Openness']
        self.target_cols = ['Interval', 'Severity', 'Duration']
        
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        self.df = None
        self.X = None
        self.Y = None
        self.scaled_features = None

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Loads CSV data and calculates the inter-pandemic interval.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        
        # Calculate year intervals; fill the first entry with a default historical baseline (35 years)
        self.df['Interval'] = self.df['Year'].diff().fillna(35.0)
        return self.df

    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes data and generates sliding window sequences for LSTM training.
        Returns:
            X: Input sequences (N, look_back, num_features)
            Y: Target values (N, num_targets)
        """
        features = self.df[self.feature_cols].values.astype('float32')
        targets = self.df[self.target_cols].values.astype('float32')

        # Fit and transform using MinMaxScaler (0 to 1 range)
        self.scaled_features = self.scaler_x.fit_transform(features)
        scaled_targets = self.scaler_y.fit_transform(targets)

        X_list, Y_list = [], []
        for i in range(len(self.scaled_features) - self.look_back):
            X_list.append(self.scaled_features[i:(i + self.look_back)])
            Y_list.append(scaled_targets[i + self.look_back])
            
        self.X = np.array(X_list)
        self.Y = np.array(Y_list)
        return self.X, self.Y

    def get_last_sequence(self) -> np.ndarray:
        """
        Extracts the most recent sequence to forecast the next event.
        """
        last_seq_features = self.scaled_features[-self.look_back:]
        return last_seq_features.reshape(1, self.look_back, len(self.feature_cols))