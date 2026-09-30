import os
import json
import sys
import math
import cv2
import numpy as np
from ultralytics import YOLO

# Core Configuration Parameters
REAL_WIDTH_CM = 5.0          # Physical width of target object in cm (e.g. Bluetooth case)
REAL_HEIGHT_CM = 4.0         # Physical height of target object in cm
DEFAULT_FOV_DEG = 60.0       # Fallback estimated horizontal FOV of webcam in degrees
ALPHA = 0.25                 # Exponential Moving Average (EMA) smoothing factor
CALIB_FILE = "calibration.json"

# Default calibration constants:
# Z: Z_calibrated = calib_A * Z_raw + calib_B
DEFAULT_CALIB_A = 1.5417
DEFAULT_CALIB_B = -6.25

# X & Y default calibration constants:
# X_calibrated = calib_Ax * X_raw + calib_Bx
# Y_calibrated = calib_Ay * Y_raw + calib_By
DEFAULT_CALIB_AX = 1.0
DEFAULT_CALIB_BX = 0.0
DEFAULT_CALIB_AY = 1.0
DEFAULT_CALIB_BY = 0.0


def load_calibration(width, default_fov_deg):
    """Load 3D calibration parameters from JSON or estimate defaults."""
    f_x = None
    f_y = None
    calib_A = DEFAULT_CALIB_A
    calib_B = DEFAULT_CALIB_B
    calib_Ax = DEFAULT_CALIB_AX
    calib_Bx = DEFAULT_CALIB_BX
    calib_Ay = DEFAULT_CALIB_AY
    calib_By = DEFAULT_CALIB_BY

    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r") as f:
                data = json.load(f)
                f_x = data.get("f_x")
                f_y = data.get("f_y", f_x)
                calib_A = data.get("calib_A", DEFAULT_CALIB_A)
                calib_B = data.get("calib_B", DEFAULT_CALIB_B)
                calib_Ax = data.get("calib_Ax", DEFAULT_CALIB_AX)
                calib_Bx = data.get("calib_Bx", DEFAULT_CALIB_BX)
                calib_Ay = data.get("calib_Ay", DEFAULT_CALIB_AY)
                calib_By = data.get("calib_By", DEFAULT_CALIB_BY)
        except Exception:
            pass

    if not f_x or f_x <= 0:
        fov_x_rad = math.radians(default_fov_deg)
        f_x = (width / 2.0) / math.tan(fov_x_rad / 2.0)
        f_y = f_x
        # Automatically generate default calibration.json file if missing
        save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)

    return (
        float(f_x),
        float(f_y),
        float(calib_A),
        float(calib_B),
        float(calib_Ax),
        float(calib_Bx),
        float(calib_Ay),
        float(calib_By),
    )


def save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By):
    """Save 3D calibration parameters (X, Y, Z) to JSON calibration file."""
    with open(CALIB_FILE, "w") as f:
        json.dump({
            "f_x": f_x,
            "f_y": f_y,
            "calib_A": calib_A,
            "calib_B": calib_B,
            "calib_Ax": calib_Ax,
            "calib_Bx": calib_Bx,
            "calib_Ay": calib_Ay,
            "calib_By": calib_By,
        }, f, indent=2)


def main():
    # Initialize lightweight YOLOv8-nano object detector
    model = YOLO("yolov8n.pt")

    # Initialize webcam stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        sys.exit(1)

    # Read initial frame to get camera resolution
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        cap.release()
        sys.exit(1)

    height, width = frame.shape[:2]
    c_x = width / 2.0
    c_y = height / 2.0

    # Load 3D calibration parameters
    (
        f_x,
        f_y,
        calib_A,
        calib_B,
        calib_Ax,
        calib_Bx,
        calib_Ay,
        calib_By,
    ) = load_calibration(width, DEFAULT_FOV_DEG)

    smooth_x, smooth_y, smooth_z = None, None, None
    last_z_raw = None
    last_x_raw = None
    last_y_raw = None

    # State for 2-point interactive Z calibration
    near_point = None  # Tuple of (raw_z, real_z)
    far_point = None   # Tuple of (raw_z, real_z)

    win_name = "3D Coordinate Estimation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict non-person objects (COCO classes 1-79) to filter out humans
        results = model.predict(source=frame, verbose=False, conf=0.20, classes=list(range(1, 80)))
        boxes = [b for b in results[0].boxes if int(b.cls[0]) != 0]

        if len(boxes) > 0:
            # Target primary object (largest non-person bounding box in frame)
            best_box = max(
                boxes,
                key=lambda b: (
                    (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
                ),
            )
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

            box_width_pixels = max(1, x2 - x1)

            # Center pixel coordinates (u, v)
            center_u = (x1 + x2) / 2.0
            center_v = (y1 + y2) / 2.0

            # 3D Spatial Coordinates Calculation
            # 1. Raw Depth Z from pinhole geometry
            Z_raw = (REAL_WIDTH_CM * f_x) / box_width_pixels
            last_z_raw = Z_raw

            # 2. Calibrated Depth Z (Z = calib_A * Z_raw + calib_B)
            Z = max(0.1, calib_A * Z_raw + calib_B)

            # 3. Raw X and Y from pinhole projection
            X_raw = ((center_u - c_x) * Z) / f_x
            Y_raw = ((center_v - c_y) * Z) / f_y
            last_x_raw = X_raw
            last_y_raw = Y_raw

            # 4. Calibrated X and Y (X = Ax * X_raw + Bx, Y = Ay * Y_raw + By)
            X = calib_Ax * X_raw + calib_Bx
            Y = calib_Ay * Y_raw + calib_By

            # Exponential Moving Average (EMA) smoothing to eliminate coordinate jitter
            if smooth_x is None:
                smooth_x, smooth_y, smooth_z = X, Y, Z
            else:
                smooth_x = ALPHA * X + (1.0 - ALPHA) * smooth_x
                smooth_y = ALPHA * Y + (1.0 - ALPHA) * smooth_y
                smooth_z = ALPHA * Z + (1.0 - ALPHA) * smooth_z

            # Strictly formatted coordinate string
            coord_str = f"X: {smooth_x:+.1f} cm | Y: {smooth_y:+.1f} cm | Z: {smooth_z:.1f} cm"

            # Terminal output strictly restricted to coordinates
            print(coord_str, flush=True)

            # Draw bounding box outline
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw center point dot
            cv2.circle(frame, (int(center_u), int(center_v)), 4, (0, 0, 255), -1)

            # Draw formatted text directly above the box
            text_x = x1
            text_y = max(y1 - 10, 20)

            # Background rectangle for legibility
            (t_w, t_h), _ = cv2.getTextSize(
                coord_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (text_x, text_y - t_h - 4),
                (text_x + t_w, text_y + 4),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                coord_str,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Zero Center Calibration for X and Y
        elif key in (ord('0'), ord('z')) and last_x_raw is not None and last_y_raw is not None:
            calib_Bx = -calib_Ax * last_x_raw
            calib_By = -calib_Ay * last_y_raw
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB ZERO SET] X and Y zeroed at camera center. Bx={calib_Bx:.2f} cm, By={calib_By:.2f} cm.\n", flush=True)

        # X & Y Offset Nudge Keys (a/d for X, w/s for Y)
        elif key == ord('a'):  # Nudge X left (-0.5 cm)
            calib_Bx -= 0.5
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB X OFFSET] calib_Bx set to {calib_Bx:.2f} cm\n", flush=True)
        elif key == ord('d'):  # Nudge X right (+0.5 cm)
            calib_Bx += 0.5
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB X OFFSET] calib_Bx set to {calib_Bx:.2f} cm\n", flush=True)
        elif key == ord('w'):  # Nudge Y up (-0.5 cm)
            calib_By -= 0.5
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB Y OFFSET] calib_By set to {calib_By:.2f} cm\n", flush=True)
        elif key == ord('s'):  # Nudge Y down (+0.5 cm)
            calib_By += 0.5
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB Y OFFSET] calib_By set to {calib_By:.2f} cm\n", flush=True)

        # X & Y Scale Multiplier Keys (j/l for X scale, i/k for Y scale)
        elif key == ord('j'):  # Decrease X gain
            calib_Ax -= 0.05
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB X GAIN] calib_Ax set to {calib_Ax:.4f}\n", flush=True)
        elif key == ord('l'):  # Increase X gain
            calib_Ax += 0.05
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB X GAIN] calib_Ax set to {calib_Ax:.4f}\n", flush=True)
        elif key == ord('i'):  # Increase Y gain
            calib_Ay += 0.05
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB Y GAIN] calib_Ay set to {calib_Ay:.4f}\n", flush=True)
        elif key == ord('k'):  # Decrease Y gain
            calib_Ay -= 0.05
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB Y GAIN] calib_Ay set to {calib_Ay:.4f}\n", flush=True)

        # Interactive 2-Point Z Calibration Keys
        elif key == ord('1') and last_z_raw is not None:
            real_near = 3.0
            near_point = (last_z_raw, real_near)
            print(f"\n[CALIB NEAR SET] Raw Z={last_z_raw:.1f} cm mapped to Real Z={real_near} cm.", flush=True)
            if far_point is not None:
                (r_near, z_near), (r_far, z_far) = near_point, far_point
                if abs(r_far - r_near) > 0.01:
                    calib_A = (z_far - z_near) / (r_far - r_near)
                    calib_B = z_near - calib_A * r_near
                    save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
                    smooth_x, smooth_y, smooth_z = None, None, None
                    print(f"[CALIB UPDATE] Computed Affine Model Z = {calib_A:.4f} * Z_raw + {calib_B:.2f}.\n", flush=True)

        elif key == ord('2') and last_z_raw is not None:
            real_far = 40.0
            far_point = (last_z_raw, real_far)
            print(f"\n[CALIB FAR SET] Raw Z={last_z_raw:.1f} cm mapped to Real Z={real_far} cm.", flush=True)
            if near_point is not None:
                (r_near, z_near), (r_far, z_far) = near_point, far_point
                if abs(r_far - r_near) > 0.01:
                    calib_A = (z_far - z_near) / (r_far - r_near)
                    calib_B = z_near - calib_A * r_near
                    save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
                    smooth_x, smooth_y, smooth_z = None, None, None
                    print(f"[CALIB UPDATE] Computed Affine Model Z = {calib_A:.4f} * Z_raw + {calib_B:.2f}.\n", flush=True)

        elif key in (ord('+'), ord('=')):
            calib_A += 0.05
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB TUNING] Slope gain calib_A set to {calib_A:.4f}\n", flush=True)

        elif key in (ord('-'), ord('_')):
            calib_A -= 0.05
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB TUNING] Slope gain calib_A set to {calib_A:.4f}\n", flush=True)

        elif key in (ord(']'), ord('}')):
            calib_B += 1.0
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB TUNING] Offset bias calib_B set to {calib_B:.2f}\n", flush=True)

        elif key in (ord('['), ord('{')):
            calib_B -= 1.0
            save_calibration(f_x, f_y, calib_A, calib_B, calib_Ax, calib_Bx, calib_Ay, calib_By)
            smooth_x, smooth_y, smooth_z = None, None, None
            print(f"\n[CALIB TUNING] Offset bias calib_B set to {calib_B:.2f}\n", flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
