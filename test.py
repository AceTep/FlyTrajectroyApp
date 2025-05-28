import os
import sys
import pandas as pd
from musice_edgelist_sorted import CSVFilterApp, generate_fly_color
from PyQt5.QtWidgets import QApplication
from pathlib import Path

def run_automated_test():
    app = QApplication(sys.argv)
    
    # ===== SET YOUR PATHS HERE =====
    FLY_CSV_PATHS = [
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly1.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly2.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly3.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly4.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly5.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly6.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly7.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly8.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly9.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly10.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly11.csv",
        "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05/fly12.csv",

    ]
    EDGELIST_PATH = "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05/CsCh_A2_20_06_2023-12_05.csv"
    VIDEO_PATH = "./../primjer-podataka/c/CsCh_A2_20_06_2023-12_05-001.mp4"  # or None for blank background
    # ===============================
    
    # Create and configure the app
    window = CSVFilterApp()
    window.show()
    
    # Load the data programmatically
    try:
        # Load fly CSVs
        all_data = []
        window.fly_colors = {}
        for file_path in FLY_CSV_PATHS:
            df = pd.read_csv(file_path, usecols=["pos x", "pos y", "ori"])
            fly_id = os.path.splitext(os.path.basename(file_path))[0]
            df["fly_id"] = fly_id
            all_data.append(df)
            window.fly_colors[fly_id] = generate_fly_color(fly_id)
        window.all_flies_df = pd.concat(all_data, ignore_index=True)
        
        # Load edgelist
        window.edgelist_path = EDGELIST_PATH
        
        # Set video path
        window.video_path = VIDEO_PATH
        
        # Configure settings (modify these as needed)
        window.use_blank_background = VIDEO_PATH is None
        window.draw_boxes = True
        window.show_labels = True
        window.draw_arrows = True
        window.show_frame_counter = True
        window.scale_factor = 1.0
        window.edge_persistence_seconds = 2
        window.start_time_min = 0
        window.end_time_min = 1  # 5 minutes of video
        
        # Enable the generate button
        window.video_button.setEnabled(True)
        
        # Optional: Auto-start the video generation
        # window.generate_video()
        
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        sys.exit(1)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/path/to/your/qt/plugins/platforms"  # update only if needed

    run_automated_test()