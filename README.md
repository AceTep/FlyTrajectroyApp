# FlyTrajectoryApp

## About
This program visualizes and analyzes the trajectories and interactions of fruit flies from video and CSV data. It overlays fly trajectories and interaction edges on the video, allows real-time playback control, and generates processed output videos with detailed interaction visualization.

## Features
- Load and visualize fly trajectories from CSV files
- Overlay interaction edges from edgelist CSV on videos
- Play videos with real-time controls (play, pause, skip, fullscreen)
- Generate videos with overlays (bounding boxes, edges, graphs)
- Adjustable visual settings (box colors, graph plots)
- Timing controls for edge persistence and graph saving intervals
- Export processed videos with and without background overlays


## Installation

### Option 1: Run from Source
1. Clone this repository:
   ```bash
   git clone https://github.com/AceTep/FlyTrajectroyApp.git
   cd zavrsni_rad
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:
   ```bash
   python main.py
   ```

### Option 2: Build Executable
If you want to create a standalone executable:

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Build the executable with custom icon:
   ```bash
   pyinstaller --onefile --windowed --icon=icon.ico main.py
   ```

3. The executable will be created in the `dist` folder.

## How It Works
When you start the program, it opens a GUI that lets you load fly trajectory CSV files, edgelist CSVs (defining fly interactions), calibration file and the background video. The main processes are:

1. Data Loading: Reads CSV data for fly positions, interactions.

2. Visualization: Overlays bounding boxes and interaction edges on the video frames, color-coded by interaction size.

3. Playback Controls: Provides play, pause, skip, and fullscreen options with real-time updates.

4. Video Processing: Generates output videos with overlays, optionally saving graphs periodically based on user settings.

5. Output: Saves processed videos and optionally exports graphs of fly interactions and distances.

Technically, video processing and overlays are handled in a separate thread to maintain UI responsiveness. Visual and timing parameters can be adjusted dynamically.

## Usage
1. Launch the program: python `main.py` 
2. Load the required CSV files: fly trajectories, edgelist, distances, and angles.
3. Load the background video file.
4. Use playback controls to view trajectories and interactions overlaid on the video.
5. Adjust visual and timing settings as desired.
6. Use the "Generate Video" button to create processed output videos with overlays.
7. Optionally, save graphs of interactions that are automatically generated during video processing.

## File Structure
```
FlyTrajectroyApp/
├── app/
│   ├── __pycache__/          # Python cache directory (auto-generated)
│   ├── fly_grid.py           # Grid-related functionality
│   ├── ui_main.py            # Main UI components
│   ├── video_player.py       # Video playback handling
│   ├── video_processor.py    # Video processing logic
│   └── widgets.py           # Custom widget definitions
├── utils/
│   ├── __pycache__/          # Python cache directory (auto-generated)
│   ├── helpers.py            # Helper functions and utilities
│   └── theme.py             # Theme/stylesheet definitions
├── icon.ico                  # Application icon
├── main.py                   # Main program entry point
└── README.md                 # Project documentation
```


## Author
Mateo Trakoštanec

## License
CC BY
