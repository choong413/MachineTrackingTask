import cv2
import os
from paddleocr import PaddleOCR
import json
import numpy as np
import time

# Constants
FONT_SCALE = 0.6
FONT_BLUE_COLOR = (255, 0, 0)
FONT_RED_COLOR = (0, 0, 255)
FONT_GREEN_COLOR = (0, 255, 0)
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 2
MAPPED_RATIO_MIN = 0.7
MAPPED_RATIO_MAX = 1.0

def initialize_output_directory(output_dir):
    """Create the output directory if it doesn't exist.

    Args:
        output_dir (str): The path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)


def initialize_paddleocr():
    """Initializes PaddleOCR with default parameters.

    Returns:
        PaddleOCR: An instance of the PaddleOCR class.
    """
    return PaddleOCR(
        use_gpu=True,  # Default to GPU if available
        lang="en",  # Default language
        use_angle_cls=False,  # Default is no angle classification
        det_db_thresh=0.1,
        det_db_box_thresh=0.1,  # Default detection box threshold
        rec_algorithm="CRNN",  # Default recognition algorithm (CRNN or SVTR_LCNet)
        drop_score=0.5,  # Default minimum confidence score
        cpu_threads=14,  # Use all available CPU threads
        max_text_length=1,
        det_db_score_mode="slow"  # (slow or fast)
    )


def initialize_tracker():
    """Initialize a CSRT tracker instance.

    Returns:
        cv2.TrackerCSRT: An instance of the CSRT tracker class.
    """
    return cv2.TrackerCSRT_create()


def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon.

    Args:
        point (tuple): The (x, y) coordinates of the point.
        polygon (numpy.ndarray): The polygon defined as an array of points.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def calculate_adjusted_y_coordinate(bottom_right_y, image_height, max_offset):
    """Calculate the adjusted y-coordinate based on the vertical position ratio.

    Args:
        bottom_right_y (int): The y-coordinate of the bottom-right corner of the bounding box.
        image_height (int): The height of the image in pixels.
        max_offset (int): The maximum downward offset in pixels.

    Returns:
        int: The adjusted y-coordinate after applying the dynamic offset.
    """
    vertical_position_ratio = bottom_right_y / image_height
    mapped_ratio = MAPPED_RATIO_MIN + vertical_position_ratio * (MAPPED_RATIO_MAX - MAPPED_RATIO_MIN)
    adjusted_offset = max_offset * mapped_ratio
    return bottom_right_y + int(adjusted_offset)


def draw_bounding_box_and_line(frame, bbox, adjusted_bottom_right_y):
    """Draw the bounding box and adjusted line on the frame.

    Args:
        frame (numpy.ndarray): The image frame on which to draw.
        bbox (list): The bounding box coordinates as a list of points.
        adjusted_bottom_right_y (int): The adjusted y-coordinate for the line.
    """
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    cv2.rectangle(frame, top_left, bottom_right, BOX_COLOR, FONT_THICKNESS)
    cv2.line(frame, (bottom_right[0], bottom_right[1]), (bottom_right[0], adjusted_bottom_right_y), LINE_COLOR,
             LINE_THICKNESS)


def label_text_on_frame(frame, bbox, text, box_label):
    """Label the detected text on the frame.

    Args:
        frame (numpy.ndarray): The image frame on which to draw.
        bbox (list): The bounding box coordinates as a list of points.
        text (str): The text detected by OCR.
        box_label (str): The label associated with the bounding box.
    """
    label_text = f"{text} ({box_label})"
    label_position = tuple(map(int, bbox[0]))
    cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_BLUE_COLOR,
                FONT_THICKNESS)


def display_box_status(frame, box_status):
    """Display the status of boxes A1-A10 and B1-B10 in two columns at the top-right corner.

    Args:
        frame (numpy.ndarray): The image frame on which to display the box status.
        box_status (dict): The dictionary holding the status of each box (e.g., A1, B1, etc.).
    """
    # Define starting position for the text
    start_x = frame.shape[1] - 300  # Right side of the frame
    start_y = 30  # Top of the frame
    line_height = 30  # Vertical spacing between lines

    # Sort labels starting with "A" and "B" in ascending order
    a_boxes = sorted([label for label in box_status if label.startswith('A')])
    b_boxes = sorted([label for label in box_status if label.startswith('B')])

    # Display A Boxes status (grouped by label "A")
    cv2.putText(
        frame,
        "A Boxes:",
        (start_x, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_BLUE_COLOR,
        FONT_THICKNESS,
    )
    y_offset = start_y + line_height
    for label in a_boxes:
        status = box_status[label]
        # Set color based on the status value
        if status == "None":
            status_color = FONT_RED_COLOR
        else:
            status_color = FONT_GREEN_COLOR

        cv2.putText(
            frame,
            f"{label}: {status}",
            (start_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            status_color,
            FONT_THICKNESS,
        )
        y_offset += line_height

    # Display B Boxes status (grouped by label "B")
    cv2.putText(
        frame,
        "B Boxes:",
        (start_x + 150, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_BLUE_COLOR,
        FONT_THICKNESS,
    )
    y_offset = start_y + line_height
    for label in b_boxes:
        status = box_status[label]
        # Set color based on the status value
        if status == "None":
            status_color = FONT_RED_COLOR
        else:
            status_color = FONT_GREEN_COLOR

        cv2.putText(
            frame,
            f"{label}: {status}",
            (start_x + 150, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            status_color,
            FONT_THICKNESS,
        )
        y_offset += line_height


def process_first_frame(frame, ocr, polygons, box_status, image_height, max_offset):
    """Process the first frame using PaddleOCR to detect text, initialize trackers,
    and assign detected text to predefined polygons.

    Args:
        frame: The first video frame to process.
        ocr (PaddleOCR): An instance of the PaddleOCR class for text recognition.
        polygons (list of dict): A list of dictionaries, each containing:
            - 'label' (str): The identifier of the polygon.
            - 'points' (numpy.ndarray): A set of points defining the polygon.
        box_status (dict): A dictionary mapping polygon labels to detected text values.
        image_height (int): The height of the video frame in pixels.
        max_offset (int): The maximum downward offset in pixels for text alignment.

    Returns:
        list: A list of CSRT tracker instances, each initialized with a detected text bounding box.
        list: A list of tuples, each containing:
            - Detected text (str)
            - The corresponding polygon label (str) or "Unknown" if no match is found.
    """
    ocr_results = ocr.ocr(frame)
    if not ocr_results or not ocr_results[0]:
        print("No text detected in the first frame.")
        return [], []

    trackers = []
    box_labels = []

    for line in ocr_results[0]:
        bbox, (text, prob) = line
        x1, y1 = map(int, bbox[0])
        x2, y2 = map(int, bbox[2])
        w, h = x2 - x1, y2 - y1
        adjusted_bottom_right_y = calculate_adjusted_y_coordinate(y2, image_height, max_offset)

        box_label = "Unknown"
        for polygon in polygons:
            if is_point_in_polygon((x2, adjusted_bottom_right_y), polygon["points"]):
                box_label = polygon["label"]
                box_status[box_label] = text
                break

        tracker = initialize_tracker()
        tracker.init(frame, (x1, y1, w, h))
        trackers.append(tracker)
        box_labels.append((text, box_label))

    return trackers, box_labels


def track_objects(frame, trackers, box_labels, polygons, box_status, image_height, max_offset):
    """Track detected text across video frames using CSRT trackers and update polygon statuses.

    Args:
        frame: The current video frame being processed.
        trackers (list): A list of CSRT tracker instances.
        box_labels (list of tuples): A list of tuples containing:
            - Detected text (str)
            - The assigned polygon label (str).
        polygons (list of dict): A list of dictionaries, each containing:
            - 'label' (str): The identifier of the polygon.
            - 'points' (numpy.ndarray): A set of points defining the polygon.
        box_status (dict): A dictionary mapping polygon labels to detected text values.
        image_height (int): The height of the video frame in pixels.
        max_offset (int): The maximum downward offset in pixels for text alignment.
    """
    for box_name in box_status:
        box_status[box_name] = "None"

    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            adjusted_bottom_right_y = calculate_adjusted_y_coordinate(y + h, image_height, max_offset)
            box_label = "Unknown"

            for polygon in polygons:
                if is_point_in_polygon((x + w, adjusted_bottom_right_y), polygon["points"]):
                    box_label = polygon["label"]
                    box_status[box_label] = box_labels[i][0]
                    break

            draw_bounding_box_and_line(frame, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], adjusted_bottom_right_y)
            label_text_on_frame(frame, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], box_labels[i][0], box_label)

    display_box_status(frame, box_status)


def extract_frames_and_track(video_path, output_dir, ocr, max_offset, polygons, box_status):
    """Extract frames from a video file, apply OCR to detect text, and track detected objects.

    Args:
        video_path (str): The file path of the input video.
        output_dir (str): The directory where the processed video will be saved.
        ocr (PaddleOCR): An instance of the PaddleOCR class for text recognition.
        max_offset (int): The maximum downward offset in pixels for text alignment.
        polygons (list of dict): A list of dictionaries, each containing:
            - 'label' (str): The identifier of the polygon.
            - 'points' (numpy.ndarray): A set of points defining the polygon.
        box_status (dict): A dictionary mapping polygon labels to detected text values.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(output_dir, "tracked_output.mp4"), fourcc, frame_rate,
                          (frame_width, frame_height))

    print("Processing video...\n")

    ret, frame = cap.read()
    if not ret:
        raise IOError("Failed to read the first frame.")

    trackers, box_labels = process_first_frame(frame, ocr, polygons, box_status, frame_height, max_offset)
    track_objects(frame, trackers, box_labels, polygons, box_status, frame_height, max_offset)
    out.write(frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        track_objects(frame, trackers, box_labels, polygons, box_status, frame_height, max_offset)

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print("Tracking completed. Output saved.")


def main():
    """Main function to execute the video processing pipeline."""
    video_path = "camera10001-0250.mkv"  # Video filename
    output_dir = "output"
    max_offset = 110  # Maximum downward offset in pixels

    try:
        # Load the config JSON
        with open("config.json", "r") as f:
            config = json.load(f)

        # Parse polygons from config JSON
        polygons = []
        box_status = {}  # Initialize as empty dictionary

        # Dynamically populate the box_status dictionary with polygon labels
        for shape in config["shapes"]:
            label = shape["label"]
            polygons.append(
                {
                    "label": label,
                    "points": np.array(shape["points"], dtype=np.int32),
                }
            )
            box_status[label] = "None"  # Set initial value to "None"

        initialize_output_directory(output_dir)
        ocr = initialize_paddleocr()  # Initialize PaddleOCR

        start_time = time.time()

        extract_frames_and_track(
            video_path, output_dir, ocr, max_offset, polygons, box_status
        )

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time
        print(f"Process completed in {elapsed_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
