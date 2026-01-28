import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

# ===== ROS 2 ADDITIONS =====
import rclpy
from rclpy.node import Node
from mycobot_msgs.msg import DetectedObject
from scipy.spatial.transform import Rotation as R
# ==========================

# =============================
# CONFIG
# =============================
CONF_THRESH = 0.75

CLASS_COLOR_MAP = {
    "RED CUBE": "red",
    "GREEN CUBE": "green",
    "YELLOW CUBE": "yellow",
    "CYAN CUBE": "cyan"
}

# =============================
# Z ANCHOR CONFIG
# =============================
CUBE_HEIGHT = 0.04
Z_ONE_CUBE = 0.36  # True robot base Z for one cube on table

# Calibration file path
CALIB_FILE = "/home/tejas/z_scale_calibration.json"

# =============================
# CAMERA INTRINSICS
# =============================
camera_matrix = np.array([
    [756.78187368, 0., 327.92680366],
    [0., 751.59057675, 224.30265234],
    [0., 0., 1.]
], dtype=np.float32)

dist_coeffs = np.array(
    [[0.31941327, -2.04882295, -0.00547559, 0.00747853, 3.39222302]],
    dtype=np.float32
)

# =============================
# CAMERA → ROBOT BASE
# =============================
R_base_cam = np.array([
    [ 0,  1,  0],
    [ 1,  0,  0],
    [ 0,  0, -1]
], dtype=np.float32)

t_base_cam = np.array([0.170, 0.0, 0.39], dtype=np.float32)

T_base_cam = np.eye(4, dtype=np.float32)
T_base_cam[:3, :3] = R_base_cam
T_base_cam[:3, 3]  = t_base_cam

# =============================
# OBJECT KEYPOINTS
# =============================
s = 0.02
object_points = np.array([
    [-s, -s, 0.0],
    [ s, -s, 0.0],
    [ s,  s, 0.0],
    [-s,  s, 0.0],
    [ 0,  0, 0.0]
], dtype=np.float32)

# =============================
# AXES (CAMERA FRAME)
# =============================
axis = np.float32([
    [0.1, 0, 0],
    [0, 0.1, 0],
    [0, 0, -0.1]
])

# =============================
# ROS 2 NODE
# =============================
class VisionWorkspaceNode(Node):
    def __init__(self):
        super().__init__("vision_workspace_node")

        # ROS publisher
        self.pub = self.create_publisher(
            DetectedObject,
            "/detected_objects",
            10
        )

        # Load YOLO model
        self.model = YOLO(
            "/home/tejas/YOLO/runs/pose/train/weights/best.pt"
        )

        # Camera (USB camera)
        self.cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("Camera could not be opened")
            return

        # Z calibration variables
        self.z_pnp_ref = None
        self.z_scale_factor = 1.0
        self.calibrated = False

        # Try to load existing calibration
        if not self.load_calibration():
            self.get_logger().info("No calibration found, starting calibration mode...")
            self.manual_calibrate()

        # Timer (only start if calibrated)
        if self.calibrated:
            self.timer = self.create_timer(0.03, self.timer_callback)
            self.get_logger().info("Vision workspace node started")
        else:
            self.get_logger().warn("Calibration incomplete, node not started")

    # =============================
    # CALIBRATION - LOAD/SAVE
    # =============================
    def save_calibration(self):
        """Save calibration to file"""
        try:
            with open(CALIB_FILE, 'w') as f:
                json.dump({
                    "z_pnp_ref": float(self.z_pnp_ref),
                    "z_scale_factor": float(self.z_scale_factor)
                }, f, indent=2)
            self.get_logger().info(f"✓ Calibration saved to {CALIB_FILE}")
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration: {e}")

    def load_calibration(self):
        """Load calibration from file"""
        if os.path.exists(CALIB_FILE):
            try:
                with open(CALIB_FILE, 'r') as f:
                    data = json.load(f)
                    self.z_pnp_ref = data["z_pnp_ref"]
                    self.z_scale_factor = data["z_scale_factor"]
                    self.calibrated = True
                    self.get_logger().info(
                        f"✓ Loaded calibration: Z_ref={self.z_pnp_ref:.3f}m, "
                        f"Scale={self.z_scale_factor:.4f}"
                    )
                    return True
            except Exception as e:
                self.get_logger().error(f"Failed to load calibration: {e}")
                return False
        return False

    # =============================
    # MANUAL CALIBRATION MODE
    # =============================
    def manual_calibrate(self):
        """Interactive calibration mode"""
        self.get_logger().info("=" * 50)
        self.get_logger().info("         Z-SCALE CALIBRATION MODE")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Instructions:")
        self.get_logger().info("  1. Place ONE(anyone) cube in the workspace at pickup position")
        self.get_logger().info("  2. Make sure cube is clearly visible")
        self.get_logger().info("  3. Click on the calibration window and press 'c' to capture (need 5 samples of the cube in any different position)")
        self.get_logger().info("  4. Press 'r' to reset samples")
        self.get_logger().info("  5. Press 'q' to quit without calibrating")
        self.get_logger().info("=" * 50)

        calibration_samples = []
        window_name = "Z-Scale Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.model(frame, conf=CONF_THRESH, verbose=False)
            r = results[0]

            z_pnp_raw = None
            conf_val = 0.0

            if r.boxes is not None and len(r.boxes) > 0:
                # Get first detection
                keypoints = r.keypoints.xy.cpu().numpy()[0].astype(np.float32)
                conf_val = r.boxes.conf.cpu().numpy()[0]
                cls_id = int(r.boxes.cls.cpu().numpy()[0])
                label = self.model.names.get(cls_id, str(cls_id))

                success, rvec, tvec = cv2.solvePnP(
                    object_points, keypoints, camera_matrix, dist_coeffs
                )

                if success:
                    z_pnp_raw = float(tvec[2][0])
                    
                    # Draw info
                    cv2.putText(
                        frame,
                        f"Detected: {label} (conf: {conf_val:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Raw PnP Z: {z_pnp_raw:.4f} m",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )
                    
                    if len(calibration_samples) > 0:
                        avg_z = np.mean(calibration_samples)
                        cv2.putText(
                            frame,
                            f"Samples: {len(calibration_samples)}/5, Avg: {avg_z:.4f} m",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 200, 0),
                            2
                        )
                        
                    # Draw detection box
                    boxes = r.boxes.xyxy.cpu().numpy()
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.putText(
                    frame,
                    "No cube detected - place cube in view",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Instructions overlay - make it prominent
            cv2.rectangle(frame, (5, frame.shape[0] - 50), 
                         (frame.shape[1] - 5, frame.shape[0] - 5), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "CLICK WINDOW FIRST! Press: 'c'=capture | 'r'=reset | 'q'=quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(100) & 0xFF  # Increased wait time for better key detection

            if key == ord('c') and z_pnp_raw is not None:
                calibration_samples.append(z_pnp_raw)
                self.get_logger().info(f"✓ Sample {len(calibration_samples)}/5: Z={z_pnp_raw:.4f}m")
                
                if len(calibration_samples) >= 5:
                    # Use average of samples
                    self.z_pnp_ref = np.mean(calibration_samples)
                    self.z_scale_factor = Z_ONE_CUBE / self.z_pnp_ref
                    self.calibrated = True
                    
                    self.get_logger().info("=" * 50)
                    self.get_logger().info("✓ CALIBRATION COMPLETE!")
                    self.get_logger().info(f"  Raw PnP Z:     {self.z_pnp_ref:.4f} m")
                    self.get_logger().info(f"  True Z:        {Z_ONE_CUBE:.4f} m")
                    self.get_logger().info(f"  Scale Factor:  {self.z_scale_factor:.4f}")
                    self.get_logger().info(f"  Error:         {(self.z_pnp_ref - Z_ONE_CUBE)*1000:.1f} mm")
                    self.get_logger().info("=" * 50)
                    
                    self.save_calibration()
                    break
            elif key == ord('c') and z_pnp_raw is None:
                self.get_logger().warn("Cannot capture - no cube detected!")
                    
            elif key == ord('r'):
                calibration_samples.clear()
                self.get_logger().info("Calibration samples reset (0/5)")
                
            elif key == ord('q'):
                self.get_logger().warn("Calibration cancelled by user")
                break

        cv2.destroyWindow(window_name)

    # =============================
    # MAIN TIMER CALLBACK
    # =============================
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = self.process_frame(frame)
        
        # Add calibration status overlay
        status_text = f"Calibrated: Scale={self.z_scale_factor:.4f} | Press 'r' to recalibrate"
        cv2.putText(
            frame,
            status_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        
        cv2.imshow("Live Cube Detection + Pose", frame)

        key = cv2.waitKey(100) & 0xFF  # Increased wait time
        if key == ord('q'):
            self.get_logger().info("Shutting down...")
            self.shutdown()
        elif key == ord('r'):
            self.get_logger().info("Entering recalibration mode...")
            cv2.destroyAllWindows()
            self.calibrated = False
            self.manual_calibrate()
            if not self.calibrated:
                self.get_logger().warn("Recalibration cancelled, shutting down...")
                self.shutdown()

    # =============================
    # PROCESS FRAME
    # =============================
    def process_frame(self, frame):
        if not self.calibrated:
            return frame

        results = self.model(frame, conf=CONF_THRESH, verbose=False)
        r = results[0]

        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return frame

        boxes = r.boxes.xyxy.cpu().numpy()
        keypoints_all = r.keypoints.xy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for box, image_points, cls_id, conf in zip(
                boxes, keypoints_all, cls_ids, confs):

            image_points = image_points.astype(np.float32)
            x1, y1, x2, y2 = box.astype(int)

            # Label + box
            label = self.model.names.get(cls_id, str(cls_id))
            color = CLASS_COLOR_MAP.get(label, "unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # Draw keypoints
            for j, (x, y) in enumerate(image_points):
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    5,
                    (255, 0, 0) if j == 4 else (0, 0, 255),
                    -1
                )

            # =============================
            # SOLVE PNP
            # =============================
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs
            )
            if not success:
                continue

            # Draw pose axes
            imgpts, _ = cv2.projectPoints(
                axis, rvec, tvec,
                camera_matrix, dist_coeffs
            )
            center = tuple(image_points[4].astype(int))
            frame = cv2.line(frame, center,
                           tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)
            frame = cv2.line(frame, center,
                           tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
            frame = cv2.line(frame, center,
                           tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)

            # =============================
            # Z CORRECTION & LEVEL DETECTION
            # =============================
            z_pnp_raw = float(tvec[2][0])
            
            # Apply scale correction
            z_pnp_corrected = z_pnp_raw * self.z_scale_factor
            
            # Calculate level based on corrected Z
            delta_z = z_pnp_corrected - Z_ONE_CUBE
            level = round(-delta_z / CUBE_HEIGHT)  # Negative: closer = higher stack
            level = max(0, level)  # No negative levels
            
            # Calculate absolute Z in robot base frame
            Z_absolute = Z_ONE_CUBE - level * CUBE_HEIGHT

            # =============================
            # CAMERA → ROBOT BASE TRANSFORM
            # =============================
            R_cam_obj, _ = cv2.Rodrigues(rvec)

            T_cam_obj = np.eye(4, dtype=np.float32)
            T_cam_obj[:3, :3] = R_cam_obj
            T_cam_obj[:3, 3] = [tvec[0][0], tvec[1][0], Z_absolute]

            T_base_obj = T_base_cam @ T_cam_obj
            object_pos_base = T_base_obj[:3, 3]

            # =============================
            # VISUALIZATION
            # =============================
            cv2.putText(
                frame,
                f"Raw Z:{z_pnp_raw:.3f} Corrected:{z_pnp_corrected:.3f} Level:{level}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Base X:{object_pos_base[0]:.3f} "
                f"Y:{object_pos_base[1]:.3f} "
                f"Z:{object_pos_base[2]:.3f}",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                2
            )

            # =============================
            # ROS PUBLISH
            # =============================
            self.publish_detected_object(object_pos_base, color)

        return frame

    # =============================
    # ROS MESSAGE PUBLISH
    # =============================
    def publish_detected_object(self, pos, color):
        msg = DetectedObject()
        msg.x = float(pos[0])
        msg.y = float(pos[1])
        msg.z = float(pos[2])
        msg.color = color
        self.pub.publish(msg)

    # =============================
    # CLEAN SHUTDOWN
    # =============================
    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()

# =============================
# MAIN
# =============================
def main():
    rclpy.init()
    node = VisionWorkspaceNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()