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
        PaddleOCR: An instance of the PaddleOCR class with default settings.
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
    mapped_ratio = MAPPED_RATIO_MIN + vertical_position_ratio * (
            MAPPED_RATIO_MAX - MAPPED_RATIO_MIN
    )
    adjusted_offset = max_offset * mapped_ratio
    return bottom_right_y + int(adjusted_offset)


def draw_bounding_box_and_line(frame, bbox, adjusted_bottom_right_y):
    """Draw the bounding box and adjusted line on the frame.

    Args:
        frame (numpy.ndarray): The image frame on which to draw.
        bbox (list): The bounding box coordinates as a list of points.
        adjusted_bottom_right_y (int): The adjusted y-coordinate for the line.
    """
    top_left = tuple(map(int, bbox[0]))  # Top-left corner
    bottom_right = tuple(map(int, bbox[2]))  # Bottom-right corner

    # Draw the original bounding box
    cv2.rectangle(frame, top_left, bottom_right, BOX_COLOR, FONT_THICKNESS)

    # Draw the adjusted line
    cv2.line(
        frame,
        (bottom_right[0], bottom_right[1]),
        (bottom_right[0], adjusted_bottom_right_y),
        LINE_COLOR,
        LINE_THICKNESS,
    )


def label_text_on_frame(frame, bbox, text, box_label):
    """Label the detected text on the frame.

    Args:
        frame (numpy.ndarray): The image frame on which to draw.
        bbox (list): The bounding box coordinates as a list of points.
        text (str): The text detected by OCR.
        box_label (str): The label associated with the bounding box.
    """
    label_text = f"{text} ({box_label})"
    label_position = tuple(map(int, bbox[0]))  # Use the first point as label position
    cv2.putText(
        frame,
        label_text,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_BLUE_COLOR,
        FONT_THICKNESS,
    )


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


def process_frame(frame, ocr, image_height, max_offset, polygons, box_status):
    """Process a single frame: detect text, adjust coordinates, and draw annotations.

    Args:
        frame (numpy.ndarray): The image frame to be processed.
        ocr (PaddleOCR): An instance of PaddleOCR to perform text recognition.
        image_height (int): The height of the image in pixels.
        max_offset (int): The maximum downward offset in pixels.
        polygons (list): A list of polygons used to detect text within specific areas.
        box_status (dict): A dictionary to update and track the status of boxes.

    Returns:
        numpy.ndarray: The processed frame with annotations (bounding boxes, lines, and text).
    """
    # Perform OCR detection using PaddleOCR
    ocr_results = ocr.ocr(frame)

    # Reset box status for the current frame (optional: not needed if you want the box to retain status across frames)
    for box_name in box_status:
        box_status[box_name] = "None"

    # Overlay detected text on the frame
    for line in ocr_results[0]:
        bbox, (text, prob) = line  # Unpack properly
        print(bbox, text, prob)

        # Get the bottom-right corner of the bounding box
        bottom_right_y = int(bbox[2][1])

        # Calculate the adjusted y-coordinate
        adjusted_bottom_right_y = calculate_adjusted_y_coordinate(
            bottom_right_y, image_height, max_offset
        )

        # Check which polygon contains the adjusted point
        box_label = "Unknown"
        for polygon in polygons:
            if is_point_in_polygon(
                    (int(bbox[2][0]), adjusted_bottom_right_y), polygon["points"]
            ):
                box_label = polygon["label"]
                box_status[box_label] = text  # Update box status
                break

        # Draw bounding box, adjusted line, and label
        draw_bounding_box_and_line(frame, bbox, adjusted_bottom_right_y)
        label_text_on_frame(frame, bbox, text, box_label)

    # Display box status at the top-right corner
    display_box_status(frame, box_status)

    return frame


def extract_frames_and_detect_text(
        video_path, output_dir, ocr, image_height, max_offset, polygons, box_status
):
    """Process each frame of the video, detect text, and save the output as a new video.

    Args:
        video_path (str): The path to the input video file.
        output_dir (str): The directory where the output video will be saved.
        ocr (PaddleOCR): An instance of the PaddleOCR class.
        image_height (int): The height of the image in pixels.
        max_offset (int): The maximum downward offset in pixels.
        polygons (list): A list of polygons defined in the config JSON.
        box_status (dict): A dictionary holding the status of each box (e.g., A1, B1, etc.).

    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Codec for .mp4
    out_video_path = os.path.join(output_dir, "output_video4.mp4")
    out = cv2.VideoWriter(
        out_video_path, fourcc, frame_rate, (frame_width, frame_height)
    )

    print("Processing video...\n")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(
            frame, ocr, image_height, max_offset, polygons, box_status
        )

        # Write the processed frame to the output video
        out.write(processed_frame)

    cap.release()
    out.release()
    print("\nProcessing complete. Output video saved as:", out_video_path)


def main():
    """Main function to execute the video processing pipeline."""
    video_path = "camera40001-0250.mkv"  # Video filename
    output_dir = "output"
    image_height = 1080  # Image height in pixels
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

        extract_frames_and_detect_text(
            video_path, output_dir, ocr, image_height, max_offset, polygons, box_status
        )

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time
        print(f"Process completed in {elapsed_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
