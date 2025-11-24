"""
create_dataset.py
Creates a machine learning dataset from preprocessed respiratory signals.

Usage:
    python create_dataset.py -in_dir Data -out_dir Dataset
"""
import os
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt


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


def read_events_file(filename):
    events = []
    time_pattern = re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}[,\.]\d{3}")
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line or "-" not in line:
                continue
            parts = line.split(';')
            if len(parts) < 3: continue
            start_end = parts[0].split('-')
            if len(start_end) != 2: continue
            if not time_pattern.match(start_end[0].strip()):
                continue
            start_dt = parse_datetime(start_end[0])
            end_dt = parse_datetime(start_end[1], fallback_date=start_dt)
            e_type = parts[2].strip()
            events.append({'start': start_dt, 'end': end_dt, 'type': e_type})
    return pd.DataFrame(events)


def bandpass_filter(signal, fs=32, lowcut=0.17, highcut=0.4, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def resample_uniform(df, target_fs=32):
    """Resample signal to uniform sampling rate"""
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
    """Create overlapping windows from signal"""
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1 - overlap))

    windows = []
    times = []

    for i in range(0, len(signal_df) - window_samples + 1, step_samples):
        window = signal_df.iloc[i:i + window_samples]
        windows.append(window['value'].values)
        times.append((window['datetime'].iloc[0], window['datetime'].iloc[-1]))

    return windows, times


def assign_labels(window_times, events_df):
    """Assign labels based on >50% overlap with events"""
    labels = []

    for win_start, win_end in window_times:
        win_duration = (win_end - win_start).total_seconds()

        # Find overlapping events
        overlapping = events_df[
            (events_df['end'] >= win_start) &
            (events_df['start'] <= win_end)
            ]

        if overlapping.empty:
            labels.append('Normal')
        else:
            max_overlap_ratio = 0
            max_label = 'Normal'

            for _, event in overlapping.iterrows():
                overlap_start = max(win_start, event['start'])
                overlap_end = min(win_end, event['end'])
                overlap_dur = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_dur / win_duration

                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    # Normalize label
                    if 'hypopnea' in event['type'].lower():
                        max_label = 'Hypopnea'
                    elif 'apnea' in event['type'].lower():
                        max_label = 'Obstructive Apnea'
                    else:
                        max_label = 'Normal'

            # Assign only if >50% overlap
            if max_overlap_ratio > 0.5:
                labels.append(max_label)
            else:
                labels.append('Normal')

    return labels


def process_participant(participant_dir, participant_id):
    """Process one participant's data"""
    print(f"Processing {participant_id}...")

    # Read signals
    airflow = robust_read_signal_file(os.path.join(participant_dir, "nasal_airflow.txt"))
    thoracic = robust_read_signal_file(os.path.join(participant_dir, "thoracic_movements.txt"))
    spo2 = robust_read_signal_file(os.path.join(participant_dir, "spo2.txt"))
    events = read_events_file(os.path.join(participant_dir, "flow_events.txt"))

    # Apply bandpass filter
    airflow['value'] = bandpass_filter(airflow['value'].values, fs=32)
    thoracic['value'] = bandpass_filter(thoracic['value'].values, fs=32)

    # Resample to uniform rate
    airflow = resample_uniform(airflow, target_fs=32)
    thoracic = resample_uniform(thoracic, target_fs=32)
    spo2 = resample_uniform(spo2, target_fs=4)

    # Create windows
    airflow_windows, window_times = create_windows(airflow, window_sec=30, overlap=0.5, fs=32)
    thoracic_windows, _ = create_windows(thoracic, window_sec=30, overlap=0.5, fs=32)
    spo2_windows, _ = create_windows(spo2, window_sec=30, overlap=0.5, fs=4)

    # Assign labels
    labels = assign_labels(window_times, events)

    # Create dataset rows
    dataset = []
    for i in range(len(labels)):
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

    return pd.DataFrame(dataset)


def main():
    parser = argparse.ArgumentParser(description='Create dataset from respiratory signals')
    parser.add_argument('-in_dir', required=True, help='Input directory containing participant folders')
    parser.add_argument('-out_dir', required=True, help='Output directory for dataset')
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find all participant folders
    participants = sorted([d for d in os.listdir(in_dir)
                           if os.path.isdir(os.path.join(in_dir, d)) and d.startswith('AP')])

    print(f"Found {len(participants)} participants: {', '.join(participants)}")

    all_data = []
    for participant_id in participants:
        participant_dir = os.path.join(in_dir, participant_id)
        try:
            df = process_participant(participant_dir, participant_id)
            all_data.append(df)
            print(f"  ✓ {participant_id}: {len(df)} windows created")
        except Exception as e:
            print(f"  ✗ {participant_id}: Error - {e}")

    # Combine all participants
    final_dataset = pd.concat(all_data, ignore_index=True)

    # Save as Parquet (efficient, preserves arrays)
    output_file = os.path.join(out_dir, 'respiratory_dataset.parquet')
    final_dataset.to_parquet(output_file, engine='pyarrow', compression='snappy')

    print(f"\n{'=' * 60}")
    print(f"Dataset Creation Complete")
    print(f"{'=' * 60}")
    print(f"Total windows: {len(final_dataset)}")
    print(f"Participants: {final_dataset['participant_id'].nunique()}")
    print(f"\nLabel distribution:")
    print(final_dataset['label'].value_counts())
    print(f"\nSaved to: {output_file}")
    print(f"\nFormat: Parquet")
    print(f"Why Parquet?")
    print(f"  - Efficient storage for large arrays (signal windows)")
    print(f"  - Fast read/write operations")
    print(f"  - Preserves data types (arrays, datetime)")
    print(f"  - Compatible with pandas, scikit-learn, PyTorch")
    print(f"  - Column-based format ideal for ML workflows")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
