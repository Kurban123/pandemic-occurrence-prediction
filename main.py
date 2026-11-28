# Bayesian forecasting of inter-event intervals among historical global biological events
# Self-contained pipeline (updated version): PCA-based anthropogenic index,
# shallow LSTM with Gaussian NLL (aleatoric) + MC Dropout (epistemic) uncertainty,
# walk-forward validation against statistical baselines, causality-constrained MC forecast.
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.stats import weibull_min
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 0. HARDCORE REPRODUCIBILITY SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# Настройка стиля (Nature/Science)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 8,
    'figure.dpi': 300,
    'pdf.fonttype': 42
})

# --- 1. DATASET (SELF-CONTAINED, N=38) ---
data = [
    {'Year': 1492.0, 'Name': 'Smallpox in America', 'Severity': 0.1, 'Duration': 28.0, 'Population': 0.35, 'Urbanization': 5.5, 'Trade_Openness': 2.2},
    {'Year': 1519.0, 'Name': 'Mexico epidemic', 'Severity': 4.0, 'Duration': 1.0, 'Population': 0.4, 'Urbanization': 6.2, 'Trade_Openness': 2.7},
    {'Year': 1545.0, 'Name': 'Cocoliztli', 'Severity': 1.0, 'Duration': 5.0, 'Population': 0.43, 'Urbanization': 6.5, 'Trade_Openness': 3.8},
    {'Year': 1576.0, 'Name': 'Cocoliztli (2nd)', 'Severity': 0.3, 'Duration': 4.0, 'Population': 0.45, 'Urbanization': 7.0, 'Trade_Openness': 4.6},
    {'Year': 1596.0, 'Name': 'Spanish Plague', 'Severity': 0.1, 'Duration': 6.0, 'Population': 0.48, 'Urbanization': 7.2, 'Trade_Openness': 5.3},
    {'Year': 1629.0, 'Name': 'Italian Plague', 'Severity': 0.2, 'Duration': 2.0, 'Population': 0.5, 'Urbanization': 7.3, 'Trade_Openness': 5.5},
    {'Year': 1633.0, 'Name': 'Ming Plague', 'Severity': 0.1, 'Duration': 11.0, 'Population': 0.5, 'Urbanization': 7.3, 'Trade_Openness': 5.6},
    {'Year': 1647.0, 'Name': 'Plague of Seville', 'Severity': 0.1, 'Duration': 5.0, 'Population': 0.51, 'Urbanization': 7.3, 'Trade_Openness': 5.6},
    {'Year': 1656.0, 'Name': 'Naples Plague', 'Severity': 0.3, 'Duration': 2.0, 'Population': 0.51, 'Urbanization': 7.3, 'Trade_Openness': 5.6},
    {'Year': 1665.0, 'Name': 'Great Plague London', 'Severity': 0.1, 'Duration': 1.0, 'Population': 0.52, 'Urbanization': 7.3, 'Trade_Openness': 5.6},
    {'Year': 1675.0, 'Name': 'Central Europe Epi', 'Severity': 0.1, 'Duration': 8.0, 'Population': 0.52, 'Urbanization': 7.3, 'Trade_Openness': 5.6},
    {'Year': 1709.0, 'Name': 'Great Northern War', 'Severity': 0.1, 'Duration': 4.0, 'Population': 0.56, 'Urbanization': 7.4, 'Trade_Openness': 5.8},
    {'Year': 1720.4, 'Name': 'Plague of Marseille', 'Severity': 0.1, 'Duration': 2.0, 'Population': 0.59, 'Urbanization': 7.5, 'Trade_Openness': 6.2},
    {'Year': 1738.0, 'Name': 'Great Plague 1738', 'Severity': 0.1, 'Duration': 2.0, 'Population': 0.67, 'Urbanization': 7.7, 'Trade_Openness': 6.8},
    {'Year': 1770.0, 'Name': 'Russian Plague', 'Severity': 0.1, 'Duration': 2.0, 'Population': 0.8, 'Urbanization': 7.9, 'Trade_Openness': 7.7},
    {'Year': 1772.0, 'Name': 'Persian Plague', 'Severity': 1.0, 'Duration': 1.0, 'Population': 0.8, 'Urbanization': 8.0, 'Trade_Openness': 7.8},
    {'Year': 1789.0, 'Name': 'NSW Smallpox', 'Severity': 0.1, 'Duration': 1.0, 'Population': 0.88, 'Urbanization': 8.2, 'Trade_Openness': 8.5},
    {'Year': 1812.0, 'Name': 'Russia Typhus', 'Severity': 0.2, 'Duration': 1.0, 'Population': 0.97, 'Urbanization': 8.6, 'Trade_Openness': 9.3},
    {'Year': 1817.0, 'Name': '1st Cholera', 'Severity': 0.1, 'Duration': 6.0, 'Population': 0.99, 'Urbanization': 8.8, 'Trade_Openness': 9.4},
    {'Year': 1826.0, 'Name': '2nd Cholera', 'Severity': 0.1, 'Duration': 11.0, 'Population': 1.03, 'Urbanization': 9.0, 'Trade_Openness': 9.8},
    {'Year': 1846.0, 'Name': '3rd Cholera', 'Severity': 0.1, 'Duration': 14.0, 'Population': 1.09, 'Urbanization': 9.8, 'Trade_Openness': 10.7},
    {'Year': 1863.0, 'Name': '4th Cholera', 'Severity': 0.1, 'Duration': 12.0, 'Population': 1.17, 'Urbanization': 11.0, 'Trade_Openness': 11.4},
    {'Year': 1870.0, 'Name': 'Europe Smallpox', 'Severity': 0.1, 'Duration': 5.0, 'Population': 1.23, 'Urbanization': 11.6, 'Trade_Openness': 11.7},
    {'Year': 1881.0, 'Name': '5th Cholera', 'Severity': 0.1, 'Duration': 15.0, 'Population': 1.34, 'Urbanization': 12.8, 'Trade_Openness': 16.2},
    {'Year': 1889.0, 'Name': 'Russian Flu', 'Severity': 0.6, 'Duration': 1.0, 'Population': 1.45, 'Urbanization': 14.0, 'Trade_Openness': 16.8},
    {'Year': 1894.0, 'Name': '3rd Plague', 'Severity': 0.3, 'Duration': 26.0, 'Population': 1.5, 'Urbanization': 15.0, 'Trade_Openness': 15.5},
    {'Year': 1896.0, 'Name': 'Trypanosomiasis', 'Severity': 0.1, 'Duration': 10.0, 'Population': 1.51, 'Urbanization': 15.4, 'Trade_Openness': 16.8},
    {'Year': 1899.0, 'Name': '6th Cholera', 'Severity': 0.1, 'Duration': 24.0, 'Population': 1.55, 'Urbanization': 16.1, 'Trade_Openness': 17.6},
    {'Year': 1900.0, 'Name': 'Uganda Sleeping Sickness', 'Severity': 0.1, 'Duration': 20.0, 'Population': 1.56, 'Urbanization': 16.2, 'Trade_Openness': 17.2},
    {'Year': 1918.0, 'Name': 'Spanish Flu', 'Severity': 10.0, 'Duration': 2.0, 'Population': 1.80, 'Urbanization': 19.7, 'Trade_Openness': 18.7},
    {'Year': 1919.0, 'Name': 'Encephalitis', 'Severity': 0.1, 'Duration': 11.0, 'Population': 1.81, 'Urbanization': 19.9, 'Trade_Openness': 17.1},
    {'Year': 1939.0, 'Name': 'WWII Epidemics', 'Severity': 0.1, 'Duration': 6.0, 'Population': 2.38, 'Urbanization': 25.1, 'Trade_Openness': 10.11},
    {'Year': 1957.0, 'Name': 'Asian Flu', 'Severity': 2.3, 'Duration': 1.0, 'Population': 3.00, 'Urbanization': 31.4, 'Trade_Openness': 21.8},
    {'Year': 1968.0, 'Name': 'Hong Kong Flu', 'Severity': 0.6, 'Duration': 2.0, 'Population': 3.8, 'Urbanization': 35.4, 'Trade_Openness': 23.0},
    {'Year': 1977.0, 'Name': 'Russian Flu', 'Severity': 0.2, 'Duration': 2.0, 'Population': 4.46, 'Urbanization': 38.0, 'Trade_Openness': 33.1},
    {'Year': 1995.0, 'Name': 'HIV/AIDS (Peak)', 'Severity': 0.8, 'Duration': 15.0, 'Population': 5.97, 'Urbanization': 44.8, 'Trade_Openness': 42.9},
    {'Year': 2009.0, 'Name': 'Swine Flu', 'Severity': 0.1, 'Duration': 1.0, 'Population': 7.06, 'Urbanization': 50.4, 'Trade_Openness': 52.0},
    {'Year': 2020.0, 'Name': 'COVID-19', 'Severity': 1.3, 'Duration': 3.0, 'Population': 7.80, 'Urbanization': 55.4, 'Trade_Openness': 52.0}
]

df = pd.DataFrame(data)
df['Interval'] = df['Year'].diff().fillna(35.0)

# --- ШАГ 1: REFACTORING & PCA (Устранение мультиколлинеарности) ---
scaler_pca = StandardScaler()
pca_features = scaler_pca.fit_transform(df[['Population', 'Urbanization', 'Trade_Openness']])
pca = PCA(n_components=1)
df['Global_Anthropogenic_Index'] = pca.fit_transform(pca_features)

# Нормализуем индекс для лучшей сходимости нейросети и наглядности графиков (в диапазоне от 1 до 10)
scaler_index = MinMaxScaler(feature_range=(1, 10))
df['Global_Anthropogenic_Index'] = scaler_index.fit_transform(df[['Global_Anthropogenic_Index']])

# Обновляем массивы предикторов
feature_cols = ['Interval', 'Severity', 'Duration', 'Global_Anthropogenic_Index']
target_cols = ['Interval', 'Severity', 'Duration']

features = df[feature_cols].values.astype('float32')
targets = df[target_cols].values.astype('float32')

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaled_features = scaler_x.fit_transform(features)
scaled_targets = scaler_y.fit_transform(targets)

look_back = 5

def create_multivariate_sequences(data_x, data_y, look_back=5):
    X, Y = [], []
    for i in range(len(data_x) - look_back):
        X.append(data_x[i:(i + look_back)])
        Y.append(data_y[i + look_back])
    return np.array(X), np.array(Y)

X, Y = create_multivariate_sequences(scaled_features, scaled_targets, look_back)


# --- 3. MODEL ARCHITECTURE (SHALLOW LSTM + ALEATORIC UNCERTAINTY) ---
import tensorflow.keras.backend as K

def nll_loss(y_true, y_pred):
    """
    Gaussian Negative Log-Likelihood loss.
    y_pred contains [mu_1, mu_2, mu_3, log_var_1, log_var_2, log_var_3]
    """
    # Первые 3 нейрона - это средние значения (mu)
    mu = y_pred[:, :3]
    # Следующие 3 нейрона - это логарифм дисперсии (log_variance)
    log_var = y_pred[:, 3:]
    
    # Избегаем деления на ноль с помощью экспоненты
    precision = tf.exp(-log_var)
    
    # NLL = 0.5 * precision * (y - mu)^2 + 0.5 * log_var
    loss = 0.5 * precision * tf.square(y_true - mu) + 0.5 * log_var
    return tf.reduce_mean(loss)

def build_lstm_model():
    reg_strength = 0.02
    inputs = Input(shape=(look_back, len(feature_cols)))

    x = LSTM(16, return_sequences=False,
             kernel_regularizer=l2(reg_strength),
             recurrent_regularizer=l2(reg_strength))(inputs)
    
    # Epistemic Uncertainty (MC Dropout)
    x = Dropout(0.30)(x, training=True)
    
    x = Dense(8, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    
    # ВАЖНО: Теперь выходной слой имеет 6 нейронов (3 для mu, 3 для log_var)
    outputs = Dense(len(target_cols) * 2)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss=nll_loss)
    return model

# --- 4. BASELINE COMPARISON & CROSS-VALIDATION (ШАГ 4: Расширенная валидация) ---
print("--- RUNNING EXPANDED WALK-FORWARD VALIDATION (N=15) ---")
# Увеличиваем окно валидации до 15 событий
test_steps = 15 
lstm_maes, ridge_maes, rf_maes, nhpp_maes, weibull_maes = [], [], [], [], []
lstm_mses, ridge_mses, rf_mses, nhpp_mses, weibull_mses = [], [], [], [], []

X_flat = X.reshape(X.shape[0], -1)

for i in range(len(X) - test_steps, len(X)):
    X_train_lstm, Y_train = X[:i], Y[:i]
    X_test_lstm, Y_test = X[i:i+1], Y[i:i+1]
    X_train_flat, X_test_flat = X_flat[:i], X_flat[i:i+1]

    true_interval = scaler_y.inverse_transform(Y_test)[0][0]
    
    # 1. Train LSTM (Aleatoric + Epistemic)
    lstm = build_lstm_model()
    lstm.fit(X_train_lstm, Y_train, epochs=200, batch_size=2, verbose=0)
    
    mc_preds = [lstm(X_test_lstm, training=True) for _ in range(50)]
    mc_preds = np.array(mc_preds)
    
    # Извлекаем предсказанные средние (первые 3 столбца)
    mu_preds = mc_preds[:, 0, :3]
    y_pred_lstm = np.mean(mu_preds, axis=0).reshape(1, -1)
    pred_interval_lstm = scaler_y.inverse_transform(y_pred_lstm)[0][0]

    # 2. Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_flat, Y_train)
    y_pred_ridge = ridge.predict(X_test_flat)
    pred_interval_ridge = scaler_y.inverse_transform(y_pred_ridge)[0][0]

    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, Y_train)
    y_pred_rf = rf.predict(X_test_flat)
    pred_interval_rf = scaler_y.inverse_transform(y_pred_rf)[0][0]

    # Базовые модели (NHPP и Weibull)
    current_year = df['Year'].iloc[i + look_back - 1]
    train_years = df['Year'].iloc[1:i+look_back].values
    train_intervals = df['Interval'].iloc[1:i+look_back].values

    def exp_func(x, a, b, c): return a * np.exp(-b * (x - 1450)) + c
    try:
        popt_nhpp, _ = curve_fit(exp_func, train_years, train_intervals, p0=(40, 0.005, 1), maxfev=10000)
        pred_interval_nhpp = exp_func(current_year, *popt_nhpp)
    except:
        pred_interval_nhpp = np.mean(train_intervals)

    try:
        shape, loc, scale = weibull_min.fit(train_intervals, floc=0)
        pred_interval_weibull = weibull_min.mean(shape, loc=loc, scale=scale)
    except:
        pred_interval_weibull = np.mean(train_intervals)

    # Запись метрик
    lstm_maes.append(abs(true_interval - pred_interval_lstm))
    ridge_maes.append(abs(true_interval - pred_interval_ridge))
    rf_maes.append(abs(true_interval - pred_interval_rf))
    nhpp_maes.append(abs(true_interval - pred_interval_nhpp))
    weibull_maes.append(abs(true_interval - pred_interval_weibull))

    lstm_mses.append((true_interval - pred_interval_lstm)**2)
    ridge_mses.append((true_interval - pred_interval_ridge)**2)
    rf_mses.append((true_interval - pred_interval_rf)**2)
    nhpp_mses.append((true_interval - pred_interval_nhpp)**2)
    weibull_mses.append((true_interval - pred_interval_weibull)**2)

print(f"LSTM MAE (Years):           {np.mean(lstm_maes):.2f}")
print(f"Ridge Reg MAE (Years):      {np.mean(ridge_maes):.2f}")
print(f"Random Forest MAE (Years):  {np.mean(rf_maes):.2f}")
print(f"NHPP Baseline MAE (Years):  {np.mean(nhpp_maes):.2f}")
print(f"Weibull Survival MAE (Yrs): {np.mean(weibull_maes):.2f}")
print("-" * 40)
print(f"LSTM RMSE (Years):          {np.sqrt(np.mean(lstm_mses)):.2f}")


# --- 5. FINAL FULL-DATASET TRAINING & PREDICTION (WITH TRUNCATED SAMPLING) ---
model = build_lstm_model()
model.fit(X, Y, epochs=300, batch_size=2, verbose=0)

last_seq_features = scaled_features[-look_back:]
last_seq = last_seq_features.reshape(1, look_back, len(feature_cols))

last_event_year = df['Year'].iloc[-1] # 2020 (COVID-19)
current_year = 2026

n_sims = 3000
mc_preds = []

# Используем Rejection Sampling для соблюдения принципа причинности (Causality)
while len(mc_preds) < n_sims:
    # Модель возвращает [mu_1, mu_2, mu_3, log_var_1, log_var_2, log_var_3]
    pred_raw = model(last_seq, training=True).numpy()
    
    mu_scaled = pred_raw[0, :3]
    log_var_scaled = pred_raw[0, 3:]
    std_scaled = np.exp(0.5 * log_var_scaled)
    
    # Сэмплирование из некорректированного распределения
    sampled_scaled = np.random.normal(loc=mu_scaled, scale=std_scaled)
    
    # Обратная трансформация масштаба
    real_pred = scaler_y.inverse_transform(sampled_scaled.reshape(1, -1))[0]
    
    predicted_year = last_event_year + real_pred[0]
    
    # УСЛОВИЕ ПРИЧИННОСТИ: Прогнозный год должен быть строго >= 2026
    if predicted_year >= current_year:
        real_pred[1] = np.maximum(real_pred[1], 0.1) # Пол для Severity
        real_pred[2] = np.maximum(real_pred[2], 0.5) # Пол для Duration
        mc_preds.append(real_pred)

mc_preds = np.array(mc_preds)
mean_p = np.mean(mc_preds, axis=0)
std_p = np.std(mc_preds, axis=0)

next_year_val = last_event_year + mean_p[0]
next_severity = mean_p[1]
next_duration = mean_p[2]

# Процентные квантили (2.5% и 97.5%) вместо +-1.96*std для усеченных распределений
ci_low = int(np.percentile(last_event_year + mc_preds[:, 0], 2.5))
ci_high = int(np.percentile(last_event_year + mc_preds[:, 0], 97.5))

print("\n--- SCIENTIFIC SUMMARY (CAUSALITY CONSTRAINED) ---")
print(f"Data: Validated dataset (N={len(df)}). Baseline Anchor: COVID-19 ({last_event_year}).")
print(f"1. Onset Year Window: {int(next_year_val)} (95% CI: {ci_low} - {ci_high})")
print(f"2. Severity Index: {next_severity:.2f}")
print(f"3. Expected Duration: {next_duration:.2f} years")


# --- 6. VISUALIZATION ---
import matplotlib.gridspec as gridspec

fig1 = plt.figure(figsize=(6.7, 6.7))
gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.2, wspace=0.2)

# Panel A. Historical Acceleration
ax1 = fig1.add_subplot(gs1[0, 0])
sns.scatterplot(x=df['Year'], y=df['Interval'], s=40, color='#2E86AB', edgecolor='black', linewidth=0.7, zorder=3, alpha=0.85, ax=ax1, label='Historical Events')
def exp_fit(x, a, b, c): return a * np.exp(-b * (x - 1450)) + c
try:
    popt, _ = curve_fit(exp_fit, df['Year'][1:], df['Interval'][1:], p0=(40, 0.005, 1), maxfev=10000)
    x_axis = np.linspace(1460, 2040, 200)
    ax1.plot(x_axis, exp_fit(x_axis, *popt), 'r--', linewidth=1.5, label='Acceleration Trend (NHPP)')
except: pass
ax1.errorbar(next_year_val, mean_p[0], yerr=std_p[0]*1.96, fmt='o', markersize=6, color='#e74c3c', ecolor='#e74c3c', elinewidth=2, capsize=4, label='LSTM Estimate (95% CI)')
ax1.set_title('A', loc='left', fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Inter-pandemic Interval (Years)')
ax1.legend(frameon=True, fancybox=True, framealpha=0.9)


# Panel B. Anthropogenic Index vs Severity
ax2 = fig1.add_subplot(gs1[0, 1])

years_sev = df['Year']
severity = df['Severity']
bottom_base = 0.01
years_anthro = df['Year']
anthro_values = df['Global_Anthropogenic_Index']

ax2.fill_between(years_sev, bottom_base, severity, color='tab:red', alpha=0.25, label='Pandemic Severity')
ax2.plot(years_sev, severity, color='tab:red', linewidth=1.5)
ax2.set_ylabel('Pandemic Severity (Log Scale)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_yscale('log')
ax2.set_ylim(bottom_base, max(severity) * 1.5)

ax2_twin = ax2.twinx()
ax2_twin.plot(years_anthro, anthro_values, color='tab:purple', linestyle='--', linewidth=2, label='Global Anthropogenic Index (PCA)')
ax2_twin.set_ylabel('Anthropogenic Index (1-10)', color='tab:purple')
ax2_twin.tick_params(axis='y', labelcolor='tab:purple')
ax2_twin.grid(False)

ax2.set_xlabel('Year')
ax2.set_title('B', loc='left', fontweight='bold')
ax2.set_xlim(1490, 2030)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

# Panel C. Correlation Matrix (Обновленная)
ax3 = fig1.add_subplot(gs1[1, 0])
corr_matrix = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, ax=ax3, annot_kws={"size": 6})
ax3.set_title('C', loc='left', fontweight='bold')

# Panel D. Anthropogenic Index vs Duration/Severity
ax4 = fig1.add_subplot(gs1[1, 1])
sc = ax4.scatter(df['Global_Anthropogenic_Index'], df['Severity'],
                 s=df['Duration']*10 + 20, # Размер точки теперь зависит от длительности
                 c=df['Duration'], cmap='viridis_r',
                 edgecolors='black', alpha=0.7)
ax4.set_xlabel('Global Anthropogenic Index')
ax4.set_ylabel('Severity Index')
ax4.set_title('D', loc='left', fontweight='bold')
cbar = plt.colorbar(sc, ax=ax4, label='Duration (Years)')
cbar.ax.tick_params(labelsize=6)
ax4.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(pad=2.0)

fig2 = plt.figure(figsize=(6.7, 8)) 
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.2, wspace=0.2)

# Panel E. Forecast Distribution (MC Dropout)
ax5 = fig2.add_subplot(gs2[0, 0])
sim_years = df['Year'].iloc[-1] + mc_preds[:, 0]
sns.histplot(sim_years, kde=True, color='#2E86AB', element="step", alpha=0.3, ax=ax5, bins=30)
ax5.axvline(next_year_val, color='#2E86AB', linestyle='--', linewidth=2, label=f'Mean: {int(next_year_val)}')
ax5.axvspan(ci_low, ci_high, color='#2E86AB', alpha=0.1, label='95% CI')
ax5.set_title('E', loc='left', fontweight='bold')
ax5.set_xlabel('Predicted Year')
ax5.legend(frameon=True, loc='upper right')

# Panel F. Threat Profile (Radar Chart)
ax6 = fig2.add_subplot(gs2[0, 1], polar=True)
labels = ['Severity (Index)', 'Duration (Years)', 'Freq. Pressure']

def get_radar_stats_improved(sev, dur, interval):
    norm_sev = min(sev, 10)
    norm_dur = min(dur, 10)
    norm_freq = min(15 / (interval + 1), 10)
    return [norm_sev, norm_dur, norm_freq]

stats_spanish = get_radar_stats_improved(10.0, 3.0, 18)
stats_covid = get_radar_stats_improved(1.0, 4.0, 11)
stats_pred = get_radar_stats_improved(next_severity, next_duration, mean_p[0])

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
stats_spanish += stats_spanish[:1]
stats_covid += stats_covid[:1]
stats_pred += stats_pred[:1]

ax6.plot(angles, stats_spanish, linestyle=':', linewidth=2, color='#95a5a6', label='Spanish Flu (1918)')
ax6.plot(angles, stats_covid, linestyle='--', linewidth=2, color='#5dade2', label='COVID-19 (2020)')
ax6.plot(angles, stats_pred, linestyle='-', linewidth=3, color='#e59866', label='PREDICTION')
ax6.fill(angles, stats_pred, color='#e59866', alpha=0.2)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(labels, size=8)
ax6.set_title('F', loc='center', pad=20, fontweight='bold')
ax6.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
ax6.set_ylim(0, 10)

# Panel G. Cluster Analysis (Scatter Log-Log)
ax7 = fig2.add_subplot(gs2[1, :]) 
viz_df = df.copy()
scatter = ax7.scatter(viz_df['Duration'], viz_df['Severity'],
                      s=viz_df['Severity']*40 + 30,
                      c=viz_df['Severity'],
                      cmap='magma_r', alpha=0.75, edgecolors='black', linewidth=0.5)

ax7.scatter(next_duration, next_severity, color='#e74c3c', s=350, marker='*',
            edgecolor='black', label='PREDICTION', zorder=10)

ax7.set_xscale('log')
ax7.set_yscale('log')
ax7.set_ylim(0.08, 15)
ax7.set_xlim(0.9, 40)
ax7.grid(True, which="both", ls="--", alpha=0.3)

ax7.set_xlabel('Duration (Years)')
ax7.set_ylabel('Severity Index')
ax7.set_title('G', loc='left', fontweight='bold')

for i in range(len(viz_df)):
    if viz_df['Severity'].iloc[i] >= 0.3:
        ax7.text(viz_df['Duration'].iloc[i]*1.1, viz_df['Severity'].iloc[i],
                 viz_df['Name'].iloc[i], fontsize=6, alpha=1.0)

scatter_small = ax7.scatter([], [], s=50, c='gray', alpha=0.6, edgecolor='black', label='Low (< 0.3)')
scatter_med   = ax7.scatter([], [], s=100, c='gray', alpha=0.6, edgecolor='black', label='Moderate (0.3 - 0.6)')
scatter_large = ax7.scatter([], [], s=200, c='gray', alpha=0.6, edgecolor='black', label='High (> 0.6)')
scatter_pred  = ax7.scatter([], [], s=150, marker='*', c='#e74c3c', edgecolor='black', label='Model Prediction')

ax7.legend(handles=[scatter_small, scatter_med, scatter_large, scatter_pred],
           title='Severity Index',
           loc='upper right',
           frameon=True,
           fontsize=6,
           title_fontsize=7,
           borderpad=0.8,
           labelspacing=1.0)

plt.tight_layout(pad=3.0)

plt.savefig('Figure_2_HighRes_Updated.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

# --- REPORT ---
print("--- SCIENTIFIC SUMMARY ---")
print(f"Data: Validated dataset (N={len(df)}). PCA Index extracted.")
print(f"Architecture: Shallow LSTM + MC Dropout vs Survival Baselines.")
print("-" * 40)
print(f"ESTIMATED OUTCOME:")
print(f"1. Onset Year Window: {int(next_year_val)} (95% CI: {ci_low} - {ci_high})")
print(f"2. Severity Index: {next_severity:.2f}")
print(f"3. Expected Duration: {next_duration:.2f} years")