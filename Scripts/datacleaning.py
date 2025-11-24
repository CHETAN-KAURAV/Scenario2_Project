'''

Usage : python datacleaning.py --folder Data/AP01

'''



import os
import re
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from datetime import datetime

def parse_datetime(s, fallback_date=None):
    s = s.replace(",", ".").strip()
    try:
        return datetime.strptime(s, '%d.%m.%Y %H:%M:%S.%f')
    except ValueError:
        if fallback_date is not None:
            try:
                time_part = datetime.strptime(s, '%H:%M:%S.%f').time()
                return fallback_date.replace(hour=time_part.hour, minute=time_part.minute,
                                            second=time_part.second, microsecond=time_part.microsecond)
            except Exception:
                return None
        return None

def robust_read_signal_file(filename):
    # Only loads lines after the first proper data row, robust to headers/metadata
    data_started = False
    rows = []
    time_pattern = re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}[,\.]\d{3}")
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not data_started:
                if time_pattern.match(line.split(";")[0]):
                    data_started = True
                else:
                    continue
            if not line or ";" not in line:
                continue
            parts = line.split(';')
            if len(parts) < 2: continue
            dt = parse_datetime(parts[0])
            if dt is None: continue
            try:
                v = float(parts[1])
                rows.append((dt, v))
            except:
                continue
    df = pd.DataFrame(rows, columns=["datetime", "value"])
    return df

def bandpass_filter(signal, fs, lowcut=0.17, highcut=0.4, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

def plot_raw_cleaned(df, fs, title, savedir, fname_part):
    raw = df['value'].values
    filtered = df['filtered_value'].values
    time = df['datetime'].values
    plt.figure(figsize=(14,4))
    plt.plot(time, raw, color='grey', alpha=0.8, label='Raw')
    plt.plot(time, filtered, color='blue', alpha=0.8, label='Cleaned')
    plt.title(f'{title} - Raw vs. Cleaned')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.tight_layout()
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, f"{fname_part}_cleaning.png"))
    plt.close()

def save_cleaned_signal(df, out_file):
    # Save as: datetime ; filtered_value
    df_to_write = df[['datetime', 'filtered_value']]
    df_to_write['datetime'] = df_to_write['datetime'].dt.strftime('%d.%m.%Y %H:%M:%S.%f').str[:-3]
    df_to_write.to_csv(out_file, sep=';', header=True, index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=".", help="Participant folder containing txt files")
    parser.add_argument("--outdir", type=str, default="CleanedSignals", help="Where to write cleaned data")
    parser.add_argument("--plotdir", type=str, default="CleaningPlots", help="Where to save cleaning plots")
    args = parser.parse_args()

    folder = args.folder
    outdir = os.path.join(folder, args.outdir)
    plotdir = os.path.join(folder, args.plotdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading signals from {folder}")
    airflow = robust_read_signal_file(os.path.join(folder, "nasal_airflow.txt"))
    thoracic = robust_read_signal_file(os.path.join(folder, "thoracic_movements.txt"))
    spo2 = robust_read_signal_file(os.path.join(folder, "spo2.txt"))

    print("Cleaning nasal airflow...")
    airflow['filtered_value'] = bandpass_filter(airflow['value'].values, fs=32)
    plot_raw_cleaned(airflow, fs=32, title='Nasal Airflow', savedir=plotdir, fname_part='nasal_airflow')

    print("Cleaning thoracic movement...")
    thoracic['filtered_value'] = bandpass_filter(thoracic['value'].values, fs=32)
    plot_raw_cleaned(thoracic, fs=32, title='Thoracic Movement', savedir=plotdir, fname_part='thoracic_movement')

    print("Saving cleaned signals...")
    save_cleaned_signal(airflow, os.path.join(outdir, "nasal_airflow_cleaned.txt"))
    save_cleaned_signal(thoracic, os.path.join(outdir, "thoracic_movements_cleaned.txt"))

    # For SpO2, just copy raw (typically not filtered for breathing frequency)
    spo2_to_write = spo2.copy()
    spo2_to_write['filtered_value'] = spo2_to_write['value']
    save_cleaned_signal(spo2_to_write, os.path.join(outdir, "spo2_cleaned.txt"))

    print(f"✓ All cleaned signals saved to {outdir}")
    print(f"✓ Plots saved to {plotdir}")

if __name__ == "__main__":
    main()
