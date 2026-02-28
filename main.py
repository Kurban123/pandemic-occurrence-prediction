import tensorflow as tf
from pathlib import Path

from src.utils import enforce_reproducibility, setup_logger
from src.data import PandemicDataLoader
from src.models import build_lstm_model, run_walk_forward_validation, monte_carlo_forecast
from src.visualization import generate_figure_1, generate_figure_2

def main():
    # 0. Настройка окружения
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / 'data' / 'raw' / 'data.csv'
    RESULTS_DIR = BASE_DIR / 'results'
    FIGURES_DIR = RESULTS_DIR / 'figures'
    LOGS_DIR = RESULTS_DIR / 'logs'
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(LOGS_DIR / 'execution.log')
    enforce_reproducibility(seed=42)
    logger.info("Pipeline started with fixed seed 42.")

    # 1. Загрузка данных
    loader = PandemicDataLoader(DATA_PATH, look_back=5)
    df = loader.load_and_preprocess()
    X, Y = loader.create_sequences()
    logger.info(f"Данные загружены. Форма X: {X.shape}, Y: {Y.shape}")

    # 2. Валидация
    num_features = len(loader.feature_cols)
    num_targets = len(loader.target_cols)
    run_walk_forward_validation(X, Y, loader.scaler_y, loader.look_back, num_features, num_targets, logger=logger)

    # 3. Обучение финальной модели
    logger.info("Обучение финальной модели на полном датасете...")
    tf.keras.backend.clear_session()
    model = build_lstm_model(loader.look_back, num_features, num_targets)
    model.fit(X, Y, epochs=300, batch_size=2, verbose=0)

    # 4. Прогнозирование с неопределенностью (Monte Carlo Dropout)
    last_seq = loader.get_last_sequence()
    mc_preds, mean_p, std_p = monte_carlo_forecast(model, last_seq, loader.scaler_y, n_sims=2000)

    next_year_val = df['Year'].iloc[-1] + mean_p[0]
    next_severity = mean_p[1]
    next_duration = mean_p[2]
    ci_low = int(next_year_val - std_p[0]*1.96)
    ci_high = int(next_year_val + std_p[0]*1.96)

    # 5. Научный отчет
    logger.info("--- SCIENTIFIC SUMMARY ---")
    logger.info(f"Data: Validated dataset (N={len(df)}).")
    logger.info(f"Architecture: Shallow LSTM + MC Dropout (Baseline validated).")
    logger.info("ESTIMATED OUTCOME:")
    logger.info(f"1. Onset Year Window: {int(next_year_val)} (95% CI: {ci_low} - {ci_high})")
    logger.info(f"2. Severity Index: {next_severity:.2f}")
    logger.info(f"3. Expected Duration: {next_duration:.2f} years")

    # 6. Генерация фигур
    logger.info("Генерация фигур для публикации...")
    generate_figure_1(df, mean_p, std_p, next_year_val, FIGURES_DIR, logger)
    generate_figure_2(df, mc_preds, mean_p, next_year_val, ci_low, ci_high, FIGURES_DIR)
    logger.info(f"Фигуры успешно сохранены в: {FIGURES_DIR}")

if __name__ == "__main__":
    main()