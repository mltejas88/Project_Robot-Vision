import time
import numpy as np
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
NO_DETECTION_TIMEOUT = 0.2   # seconds
IDLE_CHECK_RATE = 1.0        # seconds

# -------- DEBUG SAFETY --------
DEBUG_STOP_ENABLE = False

# ----------------------- HOME -------------------------------
HOME_COORDS = [51.3, -63.3, 412.67, -91.75, -0.63, -89.98]
HOME_SPEED = 40
HOME_DELAY = 4.0

# ----------------------- BINS -------------------------------
BIN_COORDS = {
    "red":    [132.2, -136.9],
    "yellow": [238.8, -125.1],
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

        GPIO.output(20, 1)  # vacuum OFF
        GPIO.output(21, 1)  # release OFF

        self.vacuum_on = False

        # ---------------- Subscriber ----------------
        self.sub = self.create_subscription(
            DetectedObject,
            "/detected_objects",
            self.object_callback,
            1
        )

        # ---------------- Robot ----------------
        self.mc = MyCobot("/dev/serial0", 1000000)
        time.sleep(0.2)

        # ---------------- State ----------------
        self.busy = False
        self.target = None
        self.stop_flag = {"stop": False}
        self.last_detection_time = time.time()
        self.at_home = True

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
            self.get_logger().warn("DEBUG STOP ENABLED — press 's' to stop")

        self.go_home()
        self.get_logger().info("Motion node ready")

    # ========================================================
    # ---------------- PUMP CONTROL ---------------------------
    # ========================================================

    def pump_on(self):
        if not self.vacuum_on:
            GPIO.output(20, 0)
            self.vacuum_on = True
            GPIO.output(21, 0)

    def pump_off(self):
        if not self.vacuum_on:
            return

        # Vacuum OFF
        GPIO.output(20, 1)
        time.sleep(0.3)   # IMPORTANT: let vacuum stop fully

        # Blow / release pulse
        GPIO.output(21, 0)
        time.sleep(0.5)
        GPIO.output(21, 1)

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
        self.last_detection_time = time.time()
        self.at_home = False

        if self.busy or self.stop_flag["stop"]:
            return

        self.target = msg
        self.execute_pick_and_place()

    # ========================================================
    # ---------------- IDLE CHECK -----------------------------
    # ========================================================

    def idle_check_callback(self):
        if self.stop_flag["stop"]:
            return

        if (time.time() - self.last_detection_time) > NO_DETECTION_TIMEOUT:
            if not self.busy and not self.at_home:
                self.get_logger().info("[IDLE] No detections — staying at HOME")
                self.go_home()
                self.at_home = True

    # ========================================================
    # ---------------- PICK & PLACE ---------------------------
    # ========================================================

    def execute_pick_and_place(self):
        self.busy = True
        self.at_home = False

        obj_x = self.target.x * 1000.0
        obj_y = self.target.y * 1000.0
        obj_z = self.target.z * 1000.0
        color = self.target.color

        self.get_logger().info(
            f"[Target mm] X={obj_x:.1f}, Y={obj_y:.1f}, Z={obj_z:.1f}"
        )

        pick_ee_z  = obj_z + NOZZLE_LENGTH_MM
        hover_ee_z = pick_ee_z + HOVER_MARGIN

        # ---------------- APPROACH ----------------
        if self.check_stop(): return
        self.mc.send_coords([obj_x, obj_y, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # ---------------- VACUUM ON ----------------
        self.pump_on()
        time.sleep(VACUUM_START_DELAY)

        # ---------------- DESCEND ----------------
        if self.check_stop(): return
        self.mc.send_coords([obj_x, obj_y, pick_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        time.sleep(VACUUM_DWELL)

        # ---------------- LIFT ----------------
        if self.check_stop(): return
        self.mc.send_coords([obj_x, obj_y, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # ---------------- BIN MOVE ----------------
        if color not in BIN_COORDS:
            self.get_logger().warn(f"Unknown color '{color}', skipping")
            self.pump_off()
            self.busy = False
            return

        bx, by = BIN_COORDS[color]

        if self.check_stop(): return
        self.mc.send_coords([bx, by, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # ---------------- DROP ----------------
        if self.check_stop(): return
        self.mc.send_coords([bx, by, pick_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        self.pump_off()
        time.sleep(1.0)

        # ---------------- RETREAT ----------------
        if self.check_stop(): return
        self.mc.send_coords([bx, by, hover_ee_z, 180, 0, 0], MOVE_SPEED)
        time.sleep(MOTION_SLEEP)

        # ---------------- HOME ----------------
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
