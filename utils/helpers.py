HARDCODED_FLY_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 255, 255),    # Cyan
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (128, 128, 0),    # Olive
    (0, 128, 128),    # Teal
    (128, 0, 0),      # Maroon
    (0, 0, 128),      # Navy
    (210, 105, 30),   # Chocolate
    (139, 69, 19),    # Saddle Brown
    (70, 130, 180),   # Steel Blue
    (255, 20, 147),   # Deep Pink
    (0, 191, 255),    # Deep Sky Blue
    (152, 251, 152),  # Pale Green
    (186, 85, 211),   # Medium Orchid
    (255, 215, 0),    # Gold
    (105, 105, 105),  # Dim Gray
    (199, 21, 133),   # Medium Violet Red
    (173, 255, 47),   # Green Yellow
    (0, 100, 0),      # Dark Green
    (255, 99, 71),     # Tomato
    (30, 144, 255),    # Dodger Blue
    (218, 112, 214),   # Orchid
    (46, 139, 87),     # Sea Green
    (240, 230, 140),   # Khaki
    (220, 20, 60),     # Crimson
    (72, 61, 139),     # Dark Slate Blue
    (255, 105, 180),   # Hot Pink
    (0, 206, 209),     # Dark Turquoise
    (148, 0, 211),     # Dark Violet
    (233, 150, 122),   # Dark Salmon
    (143, 188, 143),   # Dark Sea Green
    (147, 112, 219),   # Medium Purple
    (60, 179, 113),    # Medium Sea Green
    (250, 128, 114),   # Salmon
    (123, 104, 238),   # Medium Slate Blue
    (255, 140, 0),     # Dark Orange
    (154, 205, 50),    # Yellow Green
    (139, 0, 139),     # Dark Magenta
    (32, 178, 170),    # Light Sea Green
    (216, 191, 216),   # Thistle
    (255, 127, 80),    # Coral
    (100, 149, 237),   # Cornflower Blue
    (219, 112, 147),   # Pale Violet Red
]

def generate_fly_colors(fly_ids):
    """
    Assign distinct colors to each fly using a fixed, human-friendly palette.
    """
    colors = {}
    palette = HARDCODED_FLY_COLORS
    n = len(palette)

    for i, fly_id in enumerate(sorted(fly_ids)):
        colors[fly_id] = palette[i % n]  # Wrap around if > 24 flies

    return colors