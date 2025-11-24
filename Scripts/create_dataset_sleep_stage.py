"""
create_dataset_sleep_stage.py -

Usage:
    python create_dataset_sleep_stage.py -in_dir Data -out_dir Dataset_SleepStage
"""
import os
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt


def parse_datetime(s):
    """Parse datetime from sleep profile format"""
    s = s.replace(",", ".").strip()
    try:
        return datetime.strptime(s, '%d.%m.%Y %H:%M:%S.%f')
    except:
        return None


def robust_read_signal_file(filename):
    """Read signal files (airflow, thoracic, spo2)"""
    data_started = False
    rows = []
    time_pattern = re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}[,\.]\d{3}")

    with open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not data_started:
                if ";" in line and time_pattern.match(line.split(";")[0]):
                    data_started = True
                else:
                    continue
            if not line or ";" not in line:
                continue

            parts = line.split(';')
            if len(parts) < 2:
                continue

            dt = parse_datetime(parts[0])
            if dt is None:
                continue

            try:
                v = float(parts[1])
                rows.append((dt, v))
            except:
                continue

    return pd.DataFrame(rows, columns=["datetime", "value"])


def read_sleep_profile(filename):
    """Read sleep stage annotations - FIXED to handle semicolon separator"""
    sleep_stages = []
    time_pattern = re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}[,\.]\d{3}")

    print(f"  Reading sleep profile: {filename}")

    with open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # FIXED: Use semicolon as separator
            if ";" in line:
                parts = line.split(';')
                if len(parts) >= 2:
                    datetime_str = parts[0].strip()
                    stage = parts[1].strip()

                    # Check if first part is a datetime
                    if time_pattern.match(datetime_str):
                        dt = parse_datetime(datetime_str)
                        if dt and stage in ['Wake', 'N1', 'N2', 'N3', 'N4', 'REM', 'Movement']:
                            sleep_stages.append({'datetime': dt, 'stage': stage})

    df = pd.DataFrame(sleep_stages)
    print(f"    Found {len(df)} sleep stage annotations")
    return df


def bandpass_filter(signal, fs=32, lowcut=0.17, highcut=0.4, order=4):
    """Apply bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def resample_uniform(df, target_fs=32):
    """Resample to uniform sampling rate"""
    df = df.sort_values('datetime').reset_index(drop=True)
    start = df['datetime'].iloc[0]
    end = df['datetime'].iloc[-1]
    duration = (end - start).total_seconds()
    num_samples = int(duration * target_fs)

    new_times = pd.date_range(start=start, end=end, periods=num_samples)
    df_resampled = pd.DataFrame({'datetime': new_times})
    df_resampled['value'] = np.interp(
        new_times.astype(np.int64),
        df['datetime'].astype(np.int64),
        df['value']
    )
    return df_resampled


def create_windows(signal_df, window_sec=30, overlap=0.5, fs=32):
    """Create overlapping windows"""
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1 - overlap))
    windows = []
    times = []

    for i in range(0, len(signal_df) - window_samples + 1, step_samples):
        window = signal_df.iloc[i:i + window_samples]
        windows.append(window['value'].values)
        times.append((window['datetime'].iloc[0], window['datetime'].iloc[-1]))

    return windows, times


def assign_sleep_stage_labels(window_times, sleep_profile_df):
    """Assign sleep stage based on window midpoint"""
    labels = []

    for win_start, win_end in window_times:
        win_mid = win_start + (win_end - win_start) / 2

        # Find closest sleep stage annotation
        sleep_profile_df['time_diff'] = abs((sleep_profile_df['datetime'] - win_mid).dt.total_seconds())
        closest_idx = sleep_profile_df['time_diff'].idxmin()

        # Only assign if within 60 seconds
        if sleep_profile_df.loc[closest_idx, 'time_diff'] <= 60:
            labels.append(sleep_profile_df.loc[closest_idx, 'stage'])
        else:
            labels.append('Unknown')

    return labels


def process_participant(participant_dir, participant_id):
    """Process one participant"""
    print(f"\nProcessing {participant_id}...")

    try:
        # Read signals
        airflow = robust_read_signal_file(os.path.join(participant_dir, "nasal_airflow.txt"))
        thoracic = robust_read_signal_file(os.path.join(participant_dir, "thoracic_movements.txt"))
        spo2 = robust_read_signal_file(os.path.join(participant_dir, "spo2.txt"))
        sleep_profile = read_sleep_profile(os.path.join(participant_dir, "Sleep profile.txt"))

        if len(sleep_profile) == 0:
            raise Exception("No sleep stages found in Sleep-profile.txt")

        print(f"  Sleep stage distribution: {sleep_profile['stage'].value_counts().to_dict()}")

        # Apply bandpass filter
        airflow['value'] = bandpass_filter(airflow['value'].values, fs=32)
        thoracic['value'] = bandpass_filter(thoracic['value'].values, fs=32)

        # Resample
        airflow = resample_uniform(airflow, target_fs=32)
        thoracic = resample_uniform(thoracic, target_fs=32)
        spo2 = resample_uniform(spo2, target_fs=4)

        # Create windows
        airflow_windows, window_times = create_windows(airflow, window_sec=30, overlap=0.5, fs=32)
        thoracic_windows, _ = create_windows(thoracic, window_sec=30, overlap=0.5, fs=32)
        spo2_windows, _ = create_windows(spo2, window_sec=30, overlap=0.5, fs=4)

        # Assign labels
        labels = assign_sleep_stage_labels(window_times, sleep_profile)

        # Create dataset
        dataset = []
        for i in range(len(labels)):
            if labels[i] != 'Unknown':
                dataset.append({
                    'participant_id': participant_id,
                    'window_id': i,
                    'start_time': window_times[i][0],
                    'end_time': window_times[i][1],
                    'airflow': airflow_windows[i],
                    'thoracic': thoracic_windows[i],
                    'spo2': spo2_windows[i],
                    'label': labels[i]
                })

        print(f"  ✓ Created {len(dataset)} windows")
        return pd.DataFrame(dataset)

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', required=True, help='Input directory with participant folders')
    parser.add_argument('-out_dir', required=True, help='Output directory for dataset')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    participants = sorted([d for d in os.listdir(args.in_dir)
                           if os.path.isdir(os.path.join(args.in_dir, d)) and d.startswith('AP')])

    print(f"\n{'=' * 60}")
    print(f"Creating Sleep Stage Dataset")
    print(f"{'=' * 60}")
    print(f"Participants: {', '.join(participants)}\n")

    all_data = []
    for participant_id in participants:
        participant_dir = os.path.join(args.in_dir, participant_id)
        df = process_participant(participant_dir, participant_id)
        if df is not None and len(df) > 0:
            all_data.append(df)

    if len(all_data) == 0:
        print("\n✗ ERROR: No data was successfully processed!")
        return

    final_dataset = pd.concat(all_data, ignore_index=True)

    output_file = os.path.join(args.out_dir, 'sleep_stage_dataset.parquet')
    final_dataset.to_parquet(output_file, engine='pyarrow', compression='snappy')

    print(f"\n{'=' * 60}")
    print(f"Dataset Created Successfully")
    print(f"{'=' * 60}")
    print(f"Total windows: {len(final_dataset)}")
    print(f"Participants: {final_dataset['participant_id'].nunique()}")
    print(f"\nSleep Stage Distribution:")
    print(final_dataset['label'].value_counts())
    print(f"\nSaved to: {output_file}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
