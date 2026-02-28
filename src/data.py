import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import MinMaxScaler

class PandemicDataLoader:
    """Класс для загрузки, предобработки и векторизации данных."""
    
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
        """Загрузка данных и создание признака 'Interval'."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден по пути: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        
        # Расчет интервала с дефолтным значением для первого события
        self.df['Interval'] = self.df['Year'].diff().fillna(35.0)
        return self.df

    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Нормализация и создание временных окон для LSTM."""
        features = self.df[self.feature_cols].values.astype('float32')
        targets = self.df[self.target_cols].values.astype('float32')

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
        """Получение последнего окна для предсказания будущего события."""
        last_seq_features = self.scaled_features[-self.look_back:]
        return last_seq_features.reshape(1, self.look_back, len(self.feature_cols))