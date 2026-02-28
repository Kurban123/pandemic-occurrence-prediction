import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List, Dict
import logging

def build_lstm_model(look_back: int, num_features: int, num_targets: int, reg_strength: float = 0.02) -> Model:
    """
    Построение архитектуры Shallow LSTM с MC Dropout.
    Упрощенная архитектура предотвращает переобучение на малом датасете (Desk Reject Prevention).
    """
    inputs = Input(shape=(look_back, num_features))
    
    # Layer 1: Shallow LSTM
    x = LSTM(16, return_sequences=False,
             kernel_regularizer=l2(reg_strength),
             recurrent_regularizer=l2(reg_strength))(inputs)
    
    # Bayessian Approximation (MC Dropout) - training=True активен во время inference
    x = Dropout(0.30)(x, training=True) 
    
    # Dense bottleneck
    x = Dense(8, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    outputs = Dense(num_targets)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='huber')
    return model

def run_walk_forward_validation(X: np.ndarray, Y: np.ndarray, 
                                scaler_y, 
                                look_back: int, 
                                num_features: int, 
                                num_targets: int,
                                test_steps: int = 6,
                                logger: logging.Logger = None) -> Dict[str, float]:
    """Сравнение LSTM с базовыми моделями на последних исторических событиях."""
    if logger:
        logger.info("--- ЗАПУСК TIME-SERIES WALK-FORWARD ВАЛИДАЦИИ ---")
        
    lstm_maes, ridge_maes, rf_maes = [], [], []
    lstm_mses, ridge_mses, rf_mses = [], [], []

    X_flat = X.reshape(X.shape[0], -1)

    for i in range(len(X) - test_steps, len(X)):
        X_train_lstm, Y_train = X[:i], Y[:i]
        X_test_lstm, Y_test = X[i:i+1], Y[i:i+1]
        X_train_flat, X_test_flat = X_flat[:i], X_flat[i:i+1]
        
        # 1. Train LSTM
        tf.keras.backend.clear_session()
        lstm = build_lstm_model(look_back, num_features, num_targets)
        lstm.fit(X_train_lstm, Y_train, epochs=150, batch_size=2, verbose=0)
        
        mc_preds = [lstm(X_test_lstm, training=True) for _ in range(50)]
        y_pred_lstm = np.mean(mc_preds, axis=0)
        
        # 2. Train Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_flat, Y_train)
        y_pred_ridge = ridge.predict(X_test_flat)
        
        # 3. Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_flat, Y_train)
        y_pred_rf = rf.predict(X_test_flat)
        
        # Денормализация 'Interval' (индекс 0) для получения ошибки в ГОДАХ
        true_interval = scaler_y.inverse_transform(Y_test)[0][0]
        pred_interval_lstm = scaler_y.inverse_transform(y_pred_lstm)[0][0]
        pred_interval_ridge = scaler_y.inverse_transform(y_pred_ridge)[0][0]
        pred_interval_rf = scaler_y.inverse_transform(y_pred_rf)[0][0]
        
        lstm_maes.append(abs(true_interval - pred_interval_lstm))
        ridge_maes.append(abs(true_interval - pred_interval_ridge))
        rf_maes.append(abs(true_interval - pred_interval_rf))
        
        lstm_mses.append((true_interval - pred_interval_lstm)**2)
        ridge_mses.append((true_interval - pred_interval_ridge)**2)
        rf_mses.append((true_interval - pred_interval_rf)**2)

    results = {
        'lstm_mae': np.mean(lstm_maes), 'ridge_mae': np.mean(ridge_maes), 'rf_mae': np.mean(rf_maes),
        'lstm_rmse': np.sqrt(np.mean(lstm_mses)), 'ridge_rmse': np.sqrt(np.mean(ridge_mses)), 'rf_rmse': np.sqrt(np.mean(rf_mses))
    }
    
    if logger:
        logger.info(f"LSTM MAE (Годы):           {results['lstm_mae']:.2f}")
        logger.info(f"Ridge Reg MAE (Годы):      {results['ridge_mae']:.2f}")
        logger.info(f"Random Forest MAE (Годы):  {results['rf_mae']:.2f}")
        logger.info(f"LSTM RMSE (Годы):          {results['lstm_rmse']:.2f}")
        logger.info(f"Ridge Reg RMSE (Годы):     {results['ridge_rmse']:.2f}")
        logger.info(f"Random Forest RMSE (Годы): {results['rf_rmse']:.2f}")
        
    return results

def monte_carlo_forecast(model: Model, last_seq: np.ndarray, scaler_y, n_sims: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Генерация вероятностного прогноза (MC Dropout + Aleatoric Noise)."""
    mc_preds = []
    for _ in range(n_sims):
        pred = model(last_seq, training=True)
        # Добавление алеаторного шума (неопределенности данных)
        pred += np.random.normal(0, 0.02, size=pred.shape) 
        real_pred = scaler_y.inverse_transform(pred)
        # Ограничение минимальной тяжести
        real_pred[0][1] = np.maximum(real_pred[0][1], 0.1) 
        mc_preds.append(real_pred)

    mc_preds = np.array(mc_preds).squeeze()
    mean_p = np.mean(mc_preds, axis=0)
    std_p = np.std(mc_preds, axis=0)
    
    return mc_preds, mean_p, std_p