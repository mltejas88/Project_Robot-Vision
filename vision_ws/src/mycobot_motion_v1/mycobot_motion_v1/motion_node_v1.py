import time
import threading
import sys
import termios
import tty

import rclpy
from rclpy.node import Node

from mycobot_msgs.msg import DetectedObject
from pymycobot.mycobot import MyCobot
import RPi.GPIO as GPIO

# ============================================================
# ----------------------- CONSTANTS --------------------------
# ============================================================

MOVE_SPEED = 50

# -------- TOOL GEOMETRY (TCP) --------
NOZZLE_LENGTH_MM = 68.0
HOVER_MARGIN = 60.0

# -------- MOTION TIMING --------
MOTION_SLEEP = 3.0
VACUUM_START_DELAY = 0.1
VACUUM_DWELL = 0.3

# -------- IDLE BEHAVIOR --------
NO_DETECTION_TIMEOUT = 0.2
IDLE_CHECK_RATE = 1.0

# -------- DEBUG SAFETY --------
DEBUG_STOP_ENABLE = False

# ----------------------- HOME -------------------------------
HOME_COORDS = [51.3, -63.3, 412.67, -91.75, -0.63, -89.98]
HOME_SPEED = 40
HOME_DELAY = 4.0
DROP_Z_MM = 50.0
# ----------------------- WORKSPACE (mm) ---------------------
WORKSPACE_X_MIN = 75.0
WORKSPACE_X_MAX = 225.0
WORKSPACE_Y_MIN = -75.0
WORKSPACE_Y_MAX = 75.0

# ----------------------- BINS -------------------------------
BIN_COORDS = {
    "red":    [132.2, -136.9],
    "yellow": [238.8, -124.1],
    "green":  [115.8, 177.3],
    "blue":   [-6.9, 173.2],
    "cyan":   [-6.9, 173.2],
}

# ============================================================
# ------------------ KEYBOARD LISTENER -----------------------
# ============================================================

def keyboard_listener(stop_flag):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            ch = sys.stdin.read(1)
            if ch.lower() == 's':
                stop_flag["stop"] = True
                print("\n[SAFETY] STOP requested by user")
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ============================================================
# ----------------------- MOTION NODE ------------------------
# ============================================================

class MotionNode(Node):

    def __init__(self):
        super().__init__("motion_node")

        # ---------------- GPIO SETUP ----------------
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(20, GPIO.OUT)   # Vacuum
        GPIO.setup(21, GPIO.OUT)   # Release

        GPIO.output(20, 1)
        GPIO.output(21, 1)

        self.vacuum_on = False

        # ---------------- Subscriber ----------------
        self.sub = self.create_subscription(
            DetectedObject,
            "/detected_objects",
            self.object_callback,
            10
        )

        # ---------------- Robot ----------------
        self.mc = MyCobot("/dev/serial0", 1000000)
        time.sleep(0.2)

        # ---------------- State ----------------
        self.busy = False
        self.at_home = True
        self.target = None
        self.stop_flag = {"stop": False}
        self.last_detection_time = time.time()

        # ---------------- Idle timer ----------------
        self.idle_timer = self.create_timer(
            IDLE_CHECK_RATE,
            self.idle_check_callback
        )

        # ---------------- Safety Keyboard ----------------
        if DEBUG_STOP_ENABLE:
            threading.Thread(
                target=keyboard_listener,
                args=(self.stop_flag,),
                daemon=True
            ).start()

        # Initial HOME
        self.go_home()
        self.get_logger().info("Motion node ready")

    # ========================================================
    # ---------------- WORKSPACE CHECK ------------------------
    # ========================================================

    def is_inside_workspace(self, msg):
        x_mm = msg.x * 1000.0
        y_mm = msg.y * 1000.0

        return (
            WORKSPACE_X_MIN <= x_mm <= WORKSPACE_X_MAX and
            WORKSPACE_Y_MIN <= y_mm <= WORKSPACE_Y_MAX
        )

    # ========================================================
    # ---------------- PUMP CONTROL ---------------------------
    # ========================================================

    def pump_on(self):
        GPIO.output(20, 0)   # release OFF
        GPIO.output(21, 0)   # vacuum ON
        self.vacuum_on = True

    def pump_off(self):
        GPIO.output(20, 1)
        time.sleep(0.3)
        GPIO.output(21, 1)
        time.sleep(2)
        #GPIO.output(21, 0)
        self.vacuum_on = False

    # ========================================================
    # ---------------- SAFETY CHECK ---------------------------
    # ========================================================

    def check_stop(self):
        if self.stop_flag["stop"]:
            self.get_logger().error("Motion stopped by user!")
            self.pump_off()
            self.busy = False
            return True
        return False

    # ========================================================
    # ---------------- CALLBACK -------------------------------
    # ========================================================

    def object_callback(self, msg):
        # Always update detection time
        self.last_detection_time = time.time()

        # Ignore detections during motion
        if self.busy or self.stop_flag["stop"]:
            return

        # Only accept detections when robot is at HOME
        if not self.at_home:
            return

        # Workspace validation
        if not self.is_inside_workspace(msg):
            self.get_logger().warn(
                f"[WORKSPACE] Ignored target "
                f"X={msg.x*1000:.1f}, Y={msg.y*1000:.1f}"
            )
            return

        # Valid target
        self.target = msg
        self.at_home = False
        self.execute_pick_and_place()

    # ========================================================
    # ---------------- IDLE CHECK -----------------------------
    # ========================================================

    def idle_check_callback(self):
        if self.stop_flag["stop"]:
            return

        idle_time = time.time() - self.last_detection_time

        if idle_time > NO_DETECTION_TIMEOUT:
            if not self.busy and not self.at_home:
                self.get_logger().info("[IDLE] No detections — going HOME")
                self.target = None
                self.go_home()
                self.at_home = True

    # ========================================================
    # ---------------- PICK & PLACE ---------------------------
    # ========================================================

    def execute_pick_and_place(self):
        self.busy = True

        obj_x = self.target.x * 1000.0
        obj_y = self.target.y * 1000.0
        obj_z = self.target.z * 1000.0
        color = self.target.color

        self.get_logger().info(
            f"[Target mm] X={obj_x:.1f}, Y={obj_y:.1f}, Z={obj_z:.1f}"
        )

        pick_ee_z = obj_z + NOZZLE_LENGTH_MM
        hover_ee_z = pick_ee_z + HOVER_MARGIN

        # Approach
        if self.check_stop(): return
        self.mc.send_coords([obj_x, obj_y, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # Vacuum ON
        self.pump_on()
        time.sleep(VACUUM_START_DELAY)

        # Descend
        if self.check_stop(): return
        self.mc.send_coords([obj_x, obj_y, pick_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)
        time.sleep(VACUUM_DWELL)

        # Lift
        if self.check_stop(): return
        self.mc.send_coords([obj_x, obj_y, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # Bin move
        if color not in BIN_COORDS:
            self.get_logger().warn(f"Unknown color '{color}'")
            self.pump_off()
            self.busy = False
            return

        bx, by = BIN_COORDS[color]

        if self.check_stop(): return
        self.mc.send_coords([bx, by, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # Drop
        if self.check_stop(): return
        self.mc.send_coords([bx, by, DROP_Z_MM, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        self.pump_off()
        time.sleep(1.0)

        # Retreat
        if self.check_stop(): return
        self.mc.send_coords([bx, by, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # HOME
        self.go_home()
        self.busy = False
        self.at_home = True
        self.target = None

    # ========================================================
    # ---------------- HOME -----------------------------------
    # ========================================================

    def go_home(self):
        self.mc.send_coords(HOME_COORDS, HOME_SPEED, 0)
        time.sleep(HOME_DELAY)

# ============================================================
# ----------------------- MAIN -------------------------------
# ============================================================

def main():
    rclpy.init()
    node = MotionNode()
    rclpy.spin(node)

    GPIO.cleanup()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
