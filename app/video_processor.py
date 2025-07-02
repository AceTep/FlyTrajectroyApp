import cv2
import numpy as np
import time
import csv
import os
import traceback
from collections import defaultdict, deque
from PyQt5.QtCore import QThread, pyqtSignal


class VideoProcessingThread(QThread):
    # PyQt signals for UI updates
    update_progress = pyqtSignal(str)
    update_progress_bar = pyqtSignal(int)
    video_saved = pyqtSignal(str, float)

    def __init__(self, all_flies_df, video_path, edgelist_path=None, fly_colors=None,
                 use_blank=False, draw_boxes=True, draw_labels=True, draw_arrows=True,
                 show_frame_counter=True, scale_factor=1.0, edge_persistence_seconds=0,
                 save_graphs=False, graph_interval_min=1, start_time_min=0, end_time_min=None, 
                 fly_size=10, draw_petri_circle=False, min_edge_duration=0, color_code_edges=False,
                 calibration_values=None): 
        super().__init__()

        # Store configuration parameters
        self.fly_size = fly_size
        self.all_flies_df = all_flies_df
        self.video_path = video_path
        self.edgelist_path = edgelist_path
        self.fly_colors = fly_colors or {}
        self.use_blank = use_blank
        self.draw_boxes = draw_boxes
        self.draw_labels = draw_labels
        self.draw_arrows = draw_arrows
        self.show_frame_counter = show_frame_counter
        self._is_cancelled = False
        self.scale_factor = scale_factor
        self.edge_persistence_seconds = edge_persistence_seconds
        self.save_graphs = save_graphs
        self.graph_interval_min = graph_interval_min
        self.start_time_min = start_time_min
        self.end_time_min = end_time_min
        self.draw_petri_circle = draw_petri_circle
        self.min_edge_duration = min_edge_duration
        self.color_code_edges = color_code_edges
        self.calibration_values = calibration_values
        self.font_scale = max(0.4, 0.6 * scale_factor)
        self.font_thickness = max(1, int(1 * scale_factor))
        self.edge_thickness = max(1, int(2 * scale_factor))
        self.box_thickness = max(1, int(2 * scale_factor))
        self.label_offset = int(10 * scale_factor)
        self.arrow_tip_length = int(20 * scale_factor)
        self.arrow_line_thickness = max(1, int(4 * scale_factor))
        self.bounding_box_margin = int(60 * scale_factor)
            

    def cancel(self):
        """Flag thread to stop early if cancelled by user."""
        self._is_cancelled = True

    def parse_edgelist(self, path):
        """
        Load and parse the edge list CSV file.
        Filters out interactions shorter than min_edge_duration.
        Returns sorted list of tuples: (start_frame, end_frame, fly_id_1, fly_id_2)
        """
        interactions = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = int(row["start_of_interaction"])
                    end = int(row["end_of_interaction"])
                    if (end - start) >= self.min_edge_duration:
                        node_1 = row["node_1"]
                        node_2 = row["node_2"]
                        interactions.append((start, end, node_1, node_2))
                except Exception as e:
                    print(f"Skipping row: {e}")
        return sorted(interactions, key=lambda x: x[0])

    def get_video_info(self, cap):
        """
        Extract video properties: frame width/height (scaled), FPS, total frame count.
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_factor)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, total_frames
    
    def draw_static_tip_arrow(self, frame, pt1, pt2, color, thickness=4, tip_length=20, duration=None):
        """
        Draws a directional arrow from pt1 to pt2.
        If color_code_edges is enabled and duration is provided, arrow color reflects duration.
        """
        if self.color_code_edges and duration is not None:
            if duration < 50:
                color = (0, 255, 0)      # Green
            elif duration < 150:
                color = (0, 255, 255)    # Yellow
            else:
                color = (0, 0, 255)      # Red

        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance == 0:
            return  # avoid division by zero

        # Offset arrow to avoid overlapping with fly icons
        offset = 10 + (40 * min(distance / 200, 1))  # Interpolated offset
        ux, uy = dx / distance, dy / distance
        start = int(pt1[0] + ux * offset), int(pt1[1] + uy * offset)
        end = int(pt2[0] - ux * offset), int(pt2[1] - uy * offset)

        cv2.line(frame, start, end, color, thickness)

        # Draw arrowhead
        base_x, base_y = end[0] - int(ux * tip_length), end[1] - int(uy * tip_length)
        nx, ny = -uy, ux  # Perpendicular vector
        wing_size = tip_length // 2
        wing1 = int(base_x + nx * wing_size), int(base_y + ny * wing_size)
        wing2 = int(base_x - nx * wing_size), int(base_y - ny * wing_size)

        cv2.fillConvexPoly(frame, np.array([end, wing1, wing2], dtype=np.int32), color)

    def draw_edges(self, frame, frame_idx, interactions, transformed_positions):
        """
        Draw arrows for all interactions active in this frame.
        Extended by edge_persistence_seconds after actual end time.
        """
        for (start, end, fly1, fly2) in interactions:
            extended_end = end + int(self.edge_persistence_seconds * self.fps)
            if start <= frame_idx <= extended_end:
                if (fly1 in transformed_positions and fly2 in transformed_positions and
                    frame_idx < len(transformed_positions[fly1]) and
                    frame_idx < len(transformed_positions[fly2])):

                    x1, y1, _ = transformed_positions[fly1][frame_idx]
                    x2, y2, _ = transformed_positions[fly2][frame_idx]
                    duration = end - start
                    self.draw_static_tip_arrow(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                            (0, 0, 0), thickness=self.arrow_line_thickness, tip_length=self.arrow_tip_length, duration=duration)


    def transform_fly_positions(self, frame_width, frame_height):
        """
        Transforms original fly coordinates to screen coordinates.
        Uses calibration values if provided, otherwise scales and centers raw data to fit frame.
        Returns: dict of fly_id -> array of [x, y, orientation] per frame
        """
        fly_data = {
            fly_id: df.reset_index(drop=True)
            for fly_id, df in self.all_flies_df.groupby("fly_id")
        }

        try:
            # Use calibration values if provided
            min_x = self.calibration_values['min_x']
            min_y = self.calibration_values['min_y']
            x_px_ratio = self.calibration_values['x_px_ratio']
            y_px_ratio = self.calibration_values['y_px_ratio']

            transformed = {}
            for fly_id, df in fly_data.items():
                scaled_x = (df["pos x"].values * x_px_ratio) + min_x
                scaled_y = (df["pos y"].values * y_px_ratio) + min_y
                scaled_x *= self.scale_factor
                scaled_y *= self.scale_factor
                ori = df["ori"].values
                transformed[fly_id] = np.stack([scaled_x, scaled_y, ori], axis=1)
            return transformed, min(len(df) for df in fly_data.values())

        except Exception as e:
            print("Calibration fallback activated:", e)

            # Fallback: fit all positions into frame
            all_x = self.all_flies_df["pos x"].values
            all_y = self.all_flies_df["pos y"].values
            min_x, max_x = all_x.min(), all_x.max()
            min_y, max_y = all_y.min(), all_y.max()

            scale = 0.9 * min(frame_width / (max_x - min_x), frame_height / (max_y - min_y))
            offset_x = (frame_width - scale * (max_x - min_x)) / 2
            offset_y = (frame_height - scale * (max_y - min_y)) / 2

            transformed = {}
            for fly_id, df in fly_data.items():
                x = (df["pos x"].values - min_x) * scale + offset_x
                y = (df["pos y"].values - min_y) * scale + offset_y
                ori = df["ori"].values
                transformed[fly_id] = np.stack([x, y, ori], axis=1)

            return transformed, min(len(df) for df in fly_data.values())


    def run(self):
        cap = None
        out = None
        try:
            start_time = time.time()  # Start timing the entire process

            # If using a blank canvas instead of a video
            if self.use_blank:
                cap = None
                frame_width, frame_height = 3052, 2304  # Default canvas size
                frame_width = int(frame_width * self.scale_factor)
                frame_height = int(frame_height * self.scale_factor)
                self.fps = 24  # Set fixed FPS for blank canvas
                transformed_positions, max_data_len = self.transform_fly_positions(frame_width, frame_height)
                total_frames = max_data_len  # Total available data points from CSV
            else:
                # Try to open the video file
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.update_progress.emit("Failed to open video.")
                    return

                # Extract video resolution, FPS and total frame count
                frame_width, frame_height, self.fps, total_frames = self.get_video_info(cap)

                # Transform fly positions based on video resolution and scaling
                transformed_positions, max_data_len = self.transform_fly_positions(frame_width, frame_height)

            # Calculate start and end frames from user-specified time interval (in minutes)
            start_frame = int(self.start_time_min * 60 * self.fps)
            end_frame = int(self.end_time_min * 60 * self.fps) if self.end_time_min is not None else None

            # Clip end frame to data length
            if end_frame is None or end_frame > max_data_len:
                end_frame = max_data_len
            if start_frame >= end_frame:
                self.update_progress.emit("Invalid time range selected.")
                return

            # Prepare output video filename and VideoWriter object
            base_name = os.path.splitext(os.path.basename(self.video_path if self.video_path else "blank"))[0]
            time_suffix = f"_{self.start_time_min}to{self.end_time_min}min" if self.end_time_min else ""
            output_filename = f"{base_name}_overlay_fly{time_suffix}.mp4"

            out = cv2.VideoWriter(
                output_filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (frame_width, frame_height)
            )
            if not out.isOpened():
                self.update_progress.emit("Failed to create output video file.")
                return

            # Parse interaction edges from edgelist CSV if provided
            interactions = self.parse_edgelist(self.edgelist_path) if self.edgelist_path else []

            frame_idx = start_frame
            last_screenshot_frame = -1  # To track last saved screenshot

            if not self.use_blank:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Seek video to start frame

            # Main loop over selected frames
            while frame_idx < end_frame:
                if self._is_cancelled:
                    break  # Early exit if user cancels

                # Generate a white frame if blank canvas is used
                if self.use_blank:
                    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
                    if self.draw_petri_circle:
                        center = (frame.shape[1] // 2, frame.shape[0] // 2)
                        radius = min(frame.shape[1], frame.shape[0]) // 2 - 50
                        cv2.circle(frame, center, radius, (200, 200, 200), 8)
                    ret = True
                else:
                    # Read frame from video
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.scale_factor != 1.0:
                        frame = cv2.resize(frame, (frame_width, frame_height))  # Resize if needed

                # Draw fly positions
                self.draw_flies(frame, frame_idx, transformed_positions)

                # Draw interaction edges (between fly pairs)
                self.draw_edges(frame, frame_idx, interactions, transformed_positions)

                # Draw group bounding boxes (for 2+ fly interactions)
                if self.draw_boxes:
                    self.draw_interaction_groups(frame, frame_idx, interactions, transformed_positions)

                # Optionally overlay a frame counter
                if self.show_frame_counter:
                    self.draw_frame_counter(frame, frame_idx)

                # Save screenshots every X frames, based on graph interval
                if self.save_graphs:
                    screenshot_every_frames = int(self.graph_interval_min * 60 * self.fps)
                    if frame_idx % screenshot_every_frames == 0 and frame_idx != last_screenshot_frame:
                        screenshot_path = f"screenshot_frame_{frame_idx}.png"
                        cv2.imwrite(screenshot_path, frame)
                        last_screenshot_frame = frame_idx

                # Write processed frame to output video
                out.write(frame)

                # Update progress bar
                progress = int(((frame_idx - start_frame) / (end_frame - start_frame)) * 100)
                self.update_progress_bar.emit(progress)

                frame_idx += 1  # Next frame

            if self._is_cancelled:
                self.update_progress.emit("Video generation cancelled.")
                return

            # Save transformed positions and calibration values as a .npz file
            np.savez(
                f"{os.path.splitext(output_filename)[0]}_data.npz",
                fly_positions=transformed_positions,
                calibration=self.calibration_values
            )

            elapsed = time.time() - start_time  # Total duration

            self.video_saved.emit(output_filename, elapsed)  # Notify UI of success

        except Exception as e:
            # Catch all errors and display stack trace in UI
            error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.update_progress.emit(error_message)
        finally:
            # Always release resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
    def draw_frame_counter(self, frame, frame_idx):
        # Create text label with current frame number
        text = f"FRAMES: {frame_idx}"
        position = (20, 40)  # Top-left position for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        # Black text on white canvas, white text on real video
        color = (0, 0, 0) if self.use_blank else (255, 255, 255)
        thickness = 2
        # Draw text on frame
        cv2.putText(frame, text, position, font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)


    def draw_offset_edge(self, frame, pos1, pos2, color, thickness=2, offset=50):
        # Convert points to NumPy arrays for math
        p1 = np.array(pos1, dtype=np.float32)
        p2 = np.array(pos2, dtype=np.float32)
        vec = p2 - p1
        norm = np.linalg.norm(vec)
        if norm == 0:
            return  # Skip drawing if both points are the same

        # Unit vector and offset along the line
        unit_vec = vec / norm
        offset_vec = unit_vec * offset

        # Apply offset from both ends
        start = tuple(np.round(p1 + offset_vec).astype(int))
        end = tuple(np.round(p2 - offset_vec).astype(int))

        # Draw the shortened line between offset points
        cv2.line(frame, start, end, color, thickness)



    def draw_flies(self, frame, frame_idx, transformed_positions):
        """
        Draws each fly on the given frame at its respective position for the current frame index.
        - Optionally draws orientation arrows or circles.
        - Optionally adds fly ID labels next to each fly.
        """
        # Set label color based on background type
        label_color = (0, 0, 0) if self.use_blank else (255, 255, 255)

        # Iterate through all flies and plot their position
        for fly_id, coords in transformed_positions.items():
            if frame_idx >= len(coords):
                continue  # Skip if data for current frame is missing

            # Extract and convert coordinates to integers
            x, y, ori = coords[frame_idx]
            x, y = int(x), int(y)

            # Determine fly color, falling back to white if unspecified
            color = self.fly_colors.get(fly_id, (255, 255, 255))

            # Determine fly size based on blank mode
            circle_size = self.fly_size * 2 if self.use_blank else self.fly_size

            if self.draw_arrows:
                # Draw directional triangle (indicating orientation)

                triangle_size = circle_size  # Scale with fly size

                # Front tip of triangle (direction of movement)
                front_x = x + int(triangle_size * 2 * np.cos(ori))
                front_y = y - int(triangle_size * 2 * np.sin(ori))

                # Back-left corner
                back_left_x = x + int(triangle_size * np.cos(ori + np.pi/2))
                back_left_y = y - int(triangle_size * np.sin(ori + np.pi/2))

                # Back-right corner
                back_right_x = x + int(triangle_size * np.cos(ori - np.pi/2))
                back_right_y = y - int(triangle_size * np.sin(ori - np.pi/2))

                # Combine points and draw filled triangle
                triangle_pts = np.array([[front_x, front_y],
                                        [back_left_x, back_left_y],
                                        [back_right_x, back_right_y]])
                cv2.fillConvexPoly(frame, triangle_pts, color)
            else:
                # Draw simple circle if arrow display is disabled
                cv2.circle(frame, (x, y), circle_size, color, -1)

            # Optionally draw fly ID label near its position
            if self.draw_labels:
                cv2.putText(frame, str(fly_id), (x + self.label_offset, y - self.label_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, label_color, self.font_thickness, cv2.LINE_AA)



    def draw_interaction_groups(self, frame, frame_idx, interactions, transformed_positions):
        """
        Detects and draws bounding boxes around groups of flies that are interacting
        based on current frame and the edge list interactions.
        - Groups are visualized with different colored boxes based on group size.
        """

        adjacency = defaultdict(set)

        # Build interaction adjacency list for flies connected at current frame
        for (start, end, fly1, fly2) in interactions:
            # Extend edge visibility duration by configured persistence
            extended_end = end + int(self.edge_persistence_seconds * self.fps)
            if start <= frame_idx <= extended_end:
                # Validate position data for both flies
                if (fly1 in transformed_positions and fly2 in transformed_positions and
                    frame_idx < len(transformed_positions[fly1]) and
                    frame_idx < len(transformed_positions[fly2])):

                    x1, y1, _ = transformed_positions[fly1][frame_idx]
                    x2, y2, _ = transformed_positions[fly2][frame_idx]

                    # Only consider edges within proximity threshold
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance <= 300:
                        adjacency[fly1].add(fly2)
                        adjacency[fly2].add(fly1)

        visited = set()
        components = []

        # Identify connected fly groups using BFS
        for fly in adjacency:
            if fly not in visited:
                group = set()
                queue = deque([fly])
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        group.add(current)
                        queue.extend(adjacency[current] - visited)
                components.append(group)

        # Draw bounding boxes around each detected group
        for group in components:
            xs, ys = [], []
            group_positions = []

            # Collect all fly positions in this group
            for fly_id in group:
                if fly_id in transformed_positions and frame_idx < len(transformed_positions[fly_id]):
                    x, y, _ = transformed_positions[fly_id][frame_idx]
                    xs.append(x)
                    ys.append(y)
                    group_positions.append((x, y))

            if not group_positions:
                continue  # Skip if no valid positions

            # Bounding box coordinates (with margin)
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            # Verify that group is tightly packed
            max_distance = 0
            for i in range(len(group_positions)):
                for j in range(i + 1, len(group_positions)):
                    x1, y1 = group_positions[i]
                    x2, y2 = group_positions[j]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance > max_distance:
                        max_distance = distance

            if max_distance <= 300:
                margin = self.bounding_box_margin   
                min_x_box = max(0, int(min_x - margin))
                max_x_box = min(frame.shape[1], int(max_x + margin))
                min_y_box = max(0, int(min_y - margin))
                max_y_box = min(frame.shape[0], int(max_y + margin))

                # Box color by group size
                group_size = len(group)
                if group_size == 2:
                    box_color = (0, 128, 0)  # Green
                elif group_size == 3:
                    box_color = (0, 255, 255)  # Yellow
                else:
                    box_color = (0, 0, 255)  # Red

                # Draw the rectangle for the group
                cv2.rectangle(frame, (min_x_box, min_y_box), (max_x_box, max_y_box), box_color, self.box_thickness)

