import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from pathlib import Path
import logging

def setup_plot_style():
    """Настройка стиля графиков в соответствии с требованиями Nature/Science."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'legend.title_fontsize': 7,
        'figure.dpi': 300,
        'pdf.fonttype': 42  # Важно для редактируемого текста в PDF
    })

def generate_figure_1(df: pd.DataFrame, mean_p: np.ndarray, std_p: np.ndarray, next_year_val: float, output_dir: Path, logger: logging.Logger):
    """Генерация Фигуры 1: Панели A, B, C, D."""
    setup_plot_style()
    fig1 = plt.figure(figsize=(6.7, 6.7)) # Ширина 170 мм (Nature two-column standard)
    gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.5, wspace=0.5)

    # Panel A. Historical Acceleration
    ax1 = fig1.add_subplot(gs1[0, 0])
    sns.scatterplot(x=df['Year'], y=df['Interval'], s=40, color='#2E86AB', edgecolor='black', linewidth=0.7, zorder=3, alpha=0.85, ax=ax1, label='Historical Events')
    
    def exp_fit(x, a, b, c): return a * np.exp(-b * (x - 1450)) + c
    try:
        popt, _ = curve_fit(exp_fit, df['Year'][1:], df['Interval'][1:], p0=(40, 0.005, 1), maxfev=10000)
        x_axis = np.linspace(1460, 2040, 200)
        ax1.plot(x_axis, exp_fit(x_axis, *popt), 'r--', linewidth=1.5, label='Acceleration Trend')
    except RuntimeError as e: 
        logger.warning(f"Не удалось подобрать кривую (Curve fit failed): {e}")
        
    ax1.errorbar(next_year_val, mean_p[0], yerr=std_p[0]*1.96, fmt='o', markersize=6, color='#e74c3c', ecolor='#e74c3c', elinewidth=2, capsize=4, label='LSTM Estimate (95% CI)')
    ax1.set_title('A. Acceleration of Pandemic Occurrences', loc='left')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Inter-pandemic Interval (Years)')
    ax1.legend(frameon=True, fancybox=True, framealpha=0.9)

    # Panel B. Correlation Matrix
    ax2 = fig1.add_subplot(gs1[0, 1])
    feature_cols = ['Interval', 'Severity', 'Duration', 'Population', 'Urbanization', 'Trade_Openness']
    corr_matrix = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, ax=ax2, annot_kws={"size": 6})
    ax2.set_title('B. Multivariate Correlations', loc='left')

    # Panel C. Trade vs Severity
    ax3 = fig1.add_subplot(gs1[1, 0])
    years_sev = df['Year']
    severity = df['Severity']
    bottom_base = 0.01 
    
    # ИСПРАВЛЕН БАГ: Отсутствующие переменные trade_values и years_trade
    years_trade = df['Year']
    trade_values = df['Trade_Openness']

    ax3.fill_between(years_sev, bottom_base, severity, color='tab:red', alpha=0.25, label='Pandemic Severity')
    ax3.plot(years_sev, severity, color='tab:red', linewidth=1.5)
    ax3.set_ylabel('Pandemic Severity (Log Scale)', color='tab:red')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    ax3.set_yscale('log')
    ax3.set_ylim(bottom_base, max(severity) * 1.5)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(years_trade, trade_values, color='tab:purple', linestyle='--', linewidth=2, label='Global Trade Openness')
    ax3_twin.set_ylabel('Trade Openness Index (%)', color='tab:purple')
    ax3_twin.tick_params(axis='y', labelcolor='tab:purple')
    ax3_twin.grid(False)

    ax3.set_xlabel('Year')
    ax3.set_title('C. Global Trade vs. Severity', loc='left')
    ax3.set_xlim(1490, 2030)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

    # Panel D. 3D Cluster
    ax4 = fig1.add_subplot(gs1[1, 1])
    sc = ax4.scatter(df['Population'], df['Urbanization'],
                     s=df['Severity']*40 + 20,
                     c=df['Severity'], cmap='magma_r',
                     edgecolors='black', alpha=0.7)
    ax4.set_xlabel('Population (Billions)')
    ax4.set_ylabel('Urbanization (%)')
    ax4.set_title('D. Pandemic Intensity (Pop vs Urb)', loc='left')
    cbar = plt.colorbar(sc, ax=ax4, label='Severity Index')
    cbar.ax.tick_params(labelsize=6)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(pad=2.0)
    
    fig1.savefig(output_dir / 'Figure_1_Panels_A_D.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / 'Figure_1_Panels_A_D.tiff', format='tiff', dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig1)

def generate_figure_2(df: pd.DataFrame, mc_preds: np.ndarray, mean_p: np.ndarray, next_year_val: float, ci_low: int, ci_high: int, output_dir: Path):
    """Генерация Фигуры 2: Панели E, F, G."""
    setup_plot_style()
    fig2 = plt.figure(figsize=(6.7, 8))
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)
    
    next_severity = mean_p[1]
    next_duration = mean_p[2]

    # Panel E. Forecast Distribution (MC Dropout)
    ax5 = fig2.add_subplot(gs2[0, 0])
    sim_years = df['Year'].iloc[-1] + mc_preds[:, 0]
    sns.histplot(sim_years, kde=True, color='#2E86AB', element="step", alpha=0.3, ax=ax5, bins=30)
    ax5.axvline(next_year_val, color='#2E86AB', linestyle='--', linewidth=2, label=f'Mean: {int(next_year_val)}')
    ax5.axvspan(ci_low, ci_high, color='#2E86AB', alpha=0.1, label='95% CI')
    ax5.set_title('E. Probabilistic Onset Window', loc='left')
    ax5.set_xlabel('Predicted Year')
    ax5.legend(frameon=True, loc='upper right')

    # Panel F. Threat Profile (Radar Chart)
    ax6 = fig2.add_subplot(gs2[0, 1], polar=True)
    labels = ['Severity (Index)', 'Duration (Years)', 'Freq. Pressure']

    def get_radar_stats(sev, dur, interval):
        return [min(sev, 10), min(dur, 10), min(15 / (interval + 1), 10)]

    stats_spanish = get_radar_stats(10.0, 3.0, 18)
    stats_covid = get_radar_stats(1.0, 4.0, 11)
    stats_pred = get_radar_stats(next_severity, next_duration, mean_p[0])

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
    ax6.set_xticklabels(labels, size=8, fontweight='bold')
    ax6.set_title('F. Comparative Threat Profile', loc='center', pad=20)
    ax6.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    ax6.set_ylim(0, 10)

    # Panel G. Cluster Analysis (Scatter Log-Log)
    ax7 = fig2.add_subplot(gs2[1, :]) 

    viz_df = df.copy()
    ax7.scatter(viz_df['Duration'], viz_df['Severity'],
                s=viz_df['Severity']*40 + 30, c=viz_df['Severity'],
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
    ax7.set_title('G. Event Clusters (Severity vs Duration)', loc='left')

    for i in range(len(viz_df)):
        if viz_df['Severity'].iloc[i] >= 0.3:
            ax7.text(viz_df['Duration'].iloc[i]*1.1, viz_df['Severity'].iloc[i],
                     viz_df['Name'].iloc[i], fontsize=6, alpha=1.0)

    scatter_small = ax7.scatter([], [], s=50, c='gray', alpha=0.6, edgecolor='black', label='Low (< 0.3)')
    scatter_med   = ax7.scatter([], [], s=150, c='gray', alpha=0.6, edgecolor='black', label='Moderate (0.3 - 0.6)')
    scatter_large = ax7.scatter([], [], s=400, c='gray', alpha=0.6, edgecolor='black', label='High (> 0.6)')
    scatter_pred  = ax7.scatter([], [], s=300, marker='*', c='#e74c3c', edgecolor='black', label='Model Prediction')

    ax7.legend(handles=[scatter_small, scatter_med, scatter_large, scatter_pred],
               title='Severity Index', loc='upper right', frameon=True,
               fontsize=6, title_fontsize=7, borderpad=0.8, labelspacing=1.0)

    plt.tight_layout(pad=3.0)
    
    fig2.savefig(output_dir / 'Figure_2_Panels_E_G.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / 'Figure_2_Panels_E_G.tiff', format='tiff', dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig2)