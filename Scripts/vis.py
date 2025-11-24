
'''Usage:
            python vis.py --folder Data/AP01
'''


import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

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

def read_signal_file(filename):
    # Only loads lines after the first proper data row, robust to headers/metadata
    data_started = False
    rows = []
    time_pattern = re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}[,\.]\d{3}")
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # identify start of signal data
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
    # Handles start-end in 'date time' and end in 'time' only; skips headers/irrelevant lines
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
            events.append({'start': start_dt, 'end': end_dt, 'etype': e_type})
    return pd.DataFrame(events)

def plot_window(airflow, thoracic, spo2, events, tmin, tmax, page_idx):
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    ax1, ax2, ax3 = axes
    # Crop to window
    af = airflow[(airflow["datetime"] >= tmin) & (airflow["datetime"] <= tmax)]
    th = thoracic[(thoracic["datetime"] >= tmin) & (thoracic["datetime"] <= tmax)]
    sp = spo2[(spo2["datetime"] >= tmin) & (spo2["datetime"] <= tmax)]
    ev = events[(events['end'] >= tmin) & (events['start'] <= tmax)]

    ax1.plot(af["datetime"], af["value"], color="midnightblue", lw=1)
    ax1.set_ylabel("Nasal Flow (L/min)", fontsize=11)
    ax1.set_title(f"Nasal Airflow | Page {page_idx+1}", fontsize=14)
    for _, event in ev.iterrows():
        color = "red" if ("apnea" in event["etype"].lower()) else "orange"
        ax1.axvspan(event["start"], event["end"], color=color, alpha=0.27, lw=0)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    ax2.plot(th["datetime"], th["value"], color="teal", lw=1)
    ax2.set_ylabel("Resp. Amplitude (Thoracic)", fontsize=11)
    ax2.set_title("Thoracic Abdominal Resp.", fontsize=14)
    for _, event in ev.iterrows():
        color = "red" if ("apnea" in event["etype"].lower()) else "orange"
        ax2.axvspan(event["start"], event["end"], color=color, alpha=0.2, lw=0)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    ax3.plot(sp["datetime"], sp["value"], color="darkorange", lw=1)
    ax3.set_ylabel("SpO₂ (%)", fontsize=11)
    ax3.set_title("Oxygen Saturation (SpO₂)", fontsize=14)
    ax3.set_xlabel("Time", fontsize=12)
    for _, event in ev.iterrows():
        color = "red" if ("apnea" in event["etype"].lower()) else "orange"
        ax3.axvspan(event["start"], event["end"], color=color, alpha=0.15, lw=0)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    legend_patches = [Patch(color="red", alpha=0.27, label="Apnea"),
                      Patch(color="orange", alpha=0.27, label="Hypopnea")]
    ax1.legend(handles=legend_patches, loc="upper right", ncol=2, fontsize=11, framealpha=0.9)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=300, help="Window length in seconds (default 300 = 5min)")
    parser.add_argument("--step", type=int, default=150, help="Sliding step in seconds (default 150 = 2.5min overlap)")
    parser.add_argument("--folder", type=str, default=".", help="Participant folder (default: current directory)")
    args = parser.parse_args()

    folder = args.folder
    participant_name = os.path.basename(os.path.abspath(folder))

    airflow = read_signal_file(os.path.join(folder, "nasal_airflow.txt"))
    thoracic = read_signal_file(os.path.join(folder, "thoracic_movements.txt"))
    spo2 = read_signal_file(os.path.join(folder, "spo2.txt"))
    events = read_events_file(os.path.join(folder, "flow_events.txt"))

    start_time = max(df["datetime"].min() for df in [airflow, thoracic, spo2])
    end_time = min(df["datetime"].max() for df in [airflow, thoracic, spo2])
    window = timedelta(seconds=args.window)
    step = timedelta(seconds=args.step)

    outdir = "Visualizations"
    outfile = os.path.join(outdir, f"Visualizations_{participant_name}.pdf")
    os.makedirs(outdir, exist_ok=True)

    with PdfPages(outfile) as pdf:
        page_idx = 0
        t = start_time
        while t + window <= end_time:
            fig = plot_window(airflow, thoracic, spo2, events, t, t + window, page_idx)
            pdf.savefig(fig)
            plt.close(fig)
            t += step
            page_idx += 1
    print(f"✓ Saved multipage PDF: {outfile}")

if __name__ == "__main__":
    main()
