import cv2
import numpy as np
from ultralytics import YOLO

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
Z_ONE_CUBE = 0.36

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
# R_base_cam = np.array([
#     [1,  0,  0],
#     [0, -1,  0],
#     [0,  0, -1]
# ], dtype=np.float32)
R_base_cam = np.array([
    [ 0,  1,  0],
    [1,  0,  0],
    [ 0,  0,  -1]
], dtype=np.float32)


#t_base_cam = np.array([0.0, -0.165, 0.39], dtype=np.float32)
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

        # Z reference
        self.z_pnp_ref = None
        self.z_offset = None
        self.current_level = 0


        # Timer
        self.timer = self.create_timer(0.03, self.timer_callback)

        self.get_logger().info("Vision workspace node started")

    # =============================
    # MAIN TIMER CALLBACK
    # =============================
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = self.process_frame(frame)
        cv2.imshow("Live Cube Detection + Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.shutdown()

    # =============================
    # PROCESS FRAME
    # =============================
    def process_frame(self, frame):

        results = self.model(frame, conf=CONF_THRESH, verbose=False)
        r = results[0]

        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            # self.z_pnp_ref = None
            # self.z_offset = None
            # self.current_level = 0
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,255),
                2
            )

            # Draw keypoints
            for j, (x, y) in enumerate(image_points):
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    5,
                    (255,0,0) if j == 4 else (0,0,255),
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
                             tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3)
            frame = cv2.line(frame, center,
                             tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)
            frame = cv2.line(frame, center,
                             tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3)

            # =============================
            # Z UPDATE
            # =============================
            z_pnp = float(tvec[2][0])
            #print(z_pnp)
            if self.z_pnp_ref is None:
                self.z_pnp_ref = z_pnp

            delta_z = z_pnp - self.z_pnp_ref
            Z_absolute = Z_ONE_CUBE + delta_z
            level = round((Z_ONE_CUBE - Z_absolute) / CUBE_HEIGHT)
            Z_absolute = Z_ONE_CUBE - level * CUBE_HEIGHT

            #z_pnp = float(tvec[2][0])
            #self.z_pnp_ref = None
            # =============================
            # INITIAL CALIBRATION (ONE CUBE)
            # =============================
            # if self.z_pnp_ref is None and conf > 0.85:
            #     self.z_pnp_ref = z_pnp
            #     self.z_offset = Z_ONE_CUBE - z_pnp
            #     self.current_level = 0
            #     Z_absolute = Z_ONE_CUBE
            # else:
            #     if self.z_pnp_ref is None:
            #         continue

            #     # Track PnP delta relative to first cube
            #     delta_pnp = z_pnp - self.z_pnp_ref
            #     if delta_pnp > 0 :
            #         self.z_pnp_ref = None
            #     print(f"Delta PnP: {delta_pnp:.4f} m")

            #     # Convert delta into cube levels
            #     level = int(round(abs(delta_pnp) / CUBE_HEIGHT))
            #     # Clamp level (no negative stacking)
            #     level = max(level, 0)
            #     print(f"Detected Level: {level}")

            #     # Only update if level changed
            #     if level != self.current_level:
            #         self.current_level = level

            #     Z_absolute = Z_ONE_CUBE - self.current_level * CUBE_HEIGHT

        

            # =============================
            # CAMERA → ROBOT BASE
            # =============================
            R_cam_obj, _ = cv2.Rodrigues(rvec)

            T_cam_obj = np.eye(4, dtype=np.float32)
            T_cam_obj[:3, :3] = R_cam_obj
            T_cam_obj[:3, 3] = [tvec[0][0], tvec[1][0], Z_absolute]

            T_base_obj = T_base_cam @ T_cam_obj
            object_pos_base = T_base_obj[:3, 3]

            # =============================
            # VISUALIZATION (RESTORED)
            # =============================
            cv2.putText(
                frame,
                f"Cam X:{tvec[0][0]:.2f} "
                f"Cam Y:{tvec[1][0]:.2f} "
                f"Cam Z:{Z_absolute:.3f} m",
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

            # Terminal log
            # self.get_logger().info(
            #     f"[Robot Base] X={object_pos_base[0]:.3f}, "
            #     f"Y={object_pos_base[1]:.3f}, "
            #     f"Z={object_pos_base[2]:.3f}"
            # )

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
