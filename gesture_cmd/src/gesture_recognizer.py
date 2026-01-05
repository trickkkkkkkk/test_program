
# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
from collections import deque

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError("Please install mediapipe: pip install mediapipe") from e


class MediaPipeGestureRecognizer:
    """
    参考 gesture_demo.txt 的核心算法：
    - MediaPipe Hands 两手识别
    - 手掌心 + MCP + 指尖的多点光流，鲁棒中位数
    - 竖向/横向滑动：手宽归一化速度 + EMA + 角度门控 + 一致性
    - 下滑强化：路径积分窗 & 底边 margin bias
    - 张开手掌：静止 + 四指张开 + 稳定计数 + 冷却
    返回:
    detection_result: {
      'hand_present': bool, 'num_hands': int,
      'gesture': str | None,  # 'open_palm'/'swipe_up'/'swipe_down'/'swipe_left'/'swipe_right'
      'cmd': str | None,      # 'toggle'/'seek_forward'/'seek_back'/'vol_up'/'vol_down'
      'primary_center': (x,y) | None,
      'fps': float
    }
    """
    def __init__(self):
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils
        self.drawer_style = mp.solutions.drawing_styles

        # 光流与门控状态
        self.prev_gray = None
        self.prev_points = None  # (N,1,2)
        self.flow_window_dx = deque(maxlen=4)
        self.flow_window_dy = deque(maxlen=4)

        # 阈值与EMA
        self.ema_alpha = 0.25
        self.dy_ema = 0.0
        self.dx_ema = 0.0
        self.vel_thresh_norm_vertical = 1.20
        self.vel_thresh_norm_horizontal = 1.20
        self._dy_gate_high = False
        self._dx_gate_high = False
        self.last_frame_ms = 0

        # 角度门
        self.vertical_angle_gate_ratio_up = 2.2
        self.vertical_angle_gate_ratio_down = 1.8
        self.horizontal_angle_gate_ratio = 2.2

        # 下滑偏置
        self.down_bias = 0.90

        # 下滑路径积分窗
        self.dy_hist_norm = deque(maxlen=24)
        self.dy_hist_t = deque(maxlen=24)
        self.dy_path_window_ms = 250
        self.down_path_thresh = 1.80

        # 张开手掌
        self.open_palm_min_spread_ratio = 1.22
        self.open_palm_max_spread_ratio = 2.01
        self.open_palm_ms = 220
        self.open_palm_cooldown_ms = 300
        self._open_palm_stable_cnt = 0
        self._last_motion_cmd_ms = 0
        self._last_spread = 0.0

        # 轨迹 & 主手选择
        self.tracks = {}
        self.next_track_id = 1
        self.primary_track_id = None
        self.primary_lock_ms = 700
        self.last_primary_set_ms = 0
        self.primary_last_center = None
        self.num_hands = 0

        # 节流
        self.last_cmd_ms = 0
        self.cmd_throttle_ms = 180

        # FPS
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    # ---------- Utils ----------
    @staticmethod
    def _is_finger_up(pts, tip, pip, delta=10):
        return pts[tip][1] < pts[pip][1] - delta

    @staticmethod
    def _palm_center(pts):
        xs = [pts[0][0], pts[5][0], pts[17][0]]
        ys = [pts[0][1], pts[5][1], pts[17][1]]
        return int(np.mean(xs)), int(np.mean(ys))

    @staticmethod
    def _hand_width(pts):
        return abs(pts[17][0] - pts[5][0]) + 1e-6

    def _palm_spread(self, pts, cx, cy):
        tips = [8, 12, 16, 20]
        dists = [(((pts[t][0] - cx) ** 2 + (pts[t][1] - cy) ** 2) ** 0.5) for t in tips]
        avg = float(np.mean(dists)) if dists else 0.0
        spread = avg / self._hand_width(pts)
        self._last_spread = spread
        return spread

    def _update_prev(self, gray, points):
        self.prev_gray = gray.copy()
        self.prev_points = np.array([[p] for p in points], dtype=np.float32)

    @staticmethod
    def _robust_median(values):
        if not values:
            return 0.0
        arr = np.array(values, dtype=np.float32)
        med = np.median(arr)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = max(1e-6, q3 - q1)
        keep = arr[np.abs(arr - med) <= 1.5 * iqr]
        return float(np.median(keep)) if keep.size else float(med)

    def _hand_flow(self, gray, pts, cxcy):
        # 锚点：掌心 + MCP(0,5,9,13,17) + 指尖(8,12,16,20)
        anchor_idxs = [0, 5, 9, 13, 17, 8, 12, 16, 20]
        anchors = [cxcy] + [pts[i] for i in anchor_idxs]
        if self.prev_gray is None or self.prev_points is None:
            self._update_prev(gray, anchors)
            return 0.0, 0.0

        new_points, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        dxs, dys = [], []
        if new_points is not None and st is not None:
            for i in range(len(self.prev_points)):
                if st[i][0] == 1:
                    old = self.prev_points[i][0]
                    new = new_points[i][0]
                    dxs.append(float(new[0] - old[0]))
                    dys.append(float(new[1] - old[1]))
        self._update_prev(gray, anchors)

        dx = self._robust_median(dxs)
        dy = self._robust_median(dys)
        self.flow_window_dx.append(dx)
        self.flow_window_dy.append(dy)
        return dx, dy

    @staticmethod
    def _consistent_sign(values, min_count=3):
        pos = sum(1 for v in values if v > 0)
        neg = sum(1 for v in values if v < 0)
        return (pos >= min_count) or (neg >= min_count)

    def _throttle(self):
        now = int(time.time() * 1000)
        if now - self.last_cmd_ms >= self.cmd_throttle_ms:
            self.last_cmd_ms = now
            return True
        return False

    # ---------- 对外接口 ----------
    def process_frame(self, frame):
        """
        输入 BGR frame，输出 detection_result 与 cmd（需要时）。
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        # FPS 更新
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.start_time = time.time()
            self.frame_count = 0

        res = None
        try:
            res = self.hands.process(rgb)
        except Exception:
            res = None

        hands_pts2d = []
        if res and res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                hands_pts2d.append([(int(p.x * w), int(p.y * h)) for p in lm.landmark])

        now_ms = int(time.time() * 1000)
        self.num_hands = len(hands_pts2d)
        centers = [self._palm_center(pts) for pts in hands_pts2d]

        # 简化：把当前画面中面积最大的手作为主手
        primary_idx = None
        best_area = -1
        for det_idx, pts in enumerate(hands_pts2d):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            area = max(1, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            if area > best_area:
                best_area = area
                primary_idx = det_idx

        gesture, cmd = None, None
        primary_center = None

        if primary_idx is not None:
            pts = hands_pts2d[primary_idx]
            cx, cy = self._palm_center(pts)
            primary_center = (cx, cy)

            # 光流 & 速度
            dx_med, dy_med = 0.0, 0.0
            dx_flow, dy_flow = self._hand_flow(gray, pts, (cx, cy))
            dx_med = float(np.median(self.flow_window_dx)) if self.flow_window_dx else 0.0
            dy_med = float(np.median(self.flow_window_dy)) if self.flow_window_dy else 0.0

            dt_ms = max(16, now_ms - self.last_frame_ms) if self.last_frame_ms else 33
            self.last_frame_ms = now_ms

            hand_w = self._hand_width(pts)
            dy_speed = (dy_med * 1000.0) / dt_ms
            dx_speed = (dx_med * 1000.0) / dt_ms
            dy_norm = dy_speed / (hand_w + 1e-6)
            dx_norm = dx_speed / (hand_w + 1e-6)
            self.dy_ema = (1 - self.ema_alpha) * self.dy_ema + self.ema_alpha * dy_norm
            self.dx_ema = (1 - self.ema_alpha) * self.dx_ema + self.ema_alpha * dx_norm

            # 下滑路径积分
            self.dy_hist_norm.append(dy_norm)
            self.dy_hist_t.append(now_ms)
            down_path_sum = 0.0
            for v, t in zip(reversed(self.dy_hist_norm), reversed(self.dy_hist_t)):
                if (now_ms - t) > self.dy_path_window_ms:
                    break
                if v > 0:
                    down_path_sum += v

            # 滑动角度门控 + 阈值
            is_down = dy_med > 0
            v_gate = self.vertical_angle_gate_ratio_down if is_down else self.vertical_angle_gate_ratio_up
            margin_px = max(0.0, h - cy)
            margin_scale = 1.0
            if margin_px < 120.0:
                margin_scale = 0.85 + 0.15 * (margin_px / 120.0)

            v_thr_base = self.vel_thresh_norm_vertical * (self.down_bias if is_down else 1.0)
            v_thr = v_thr_base * (margin_scale if is_down else 1.0)
            is_vertical = abs(dy_med) > v_gate * abs(dx_med)
            vertical_consistent = self._consistent_sign(self.flow_window_dy, 4)

            # 滑动触发（竖向）
            speed_pass = (abs(self.dy_ema) > v_thr)
            path_pass = (is_down and down_path_sum > self.down_path_thresh)

            # hysteresis reset
            if abs(self.dy_ema) <= self.vel_thresh_norm_vertical * 0.8:
                self._dy_gate_high = False
            if abs(self.dx_ema) <= self.vel_thresh_norm_horizontal * 0.8:
                self._dx_gate_high = False

            four_up = sum([
                1 if self._is_finger_up(pts, 8, 6) else 0,
                1 if self._is_finger_up(pts, 12, 10) else 0,
                1 if self._is_finger_up(pts, 16, 14) else 0,
                1 if self._is_finger_up(pts, 20, 18) else 0
            ])

            if four_up >= 1 and is_vertical and (speed_pass or path_pass) and not self._dy_gate_high and vertical_consistent:
                self._dy_gate_high = True
                self._last_motion_cmd_ms = now_ms
                gesture = "swipe_up" if self.dy_ema < 0 else "swipe_down"
                cmd = "vol_up" if self.dy_ema < 0 else "vol_down"

            # 横向滑动（seek）
            is_horizontal = abs(dx_med) > self.horizontal_angle_gate_ratio * abs(dy_med)
            if four_up >= 1 and is_horizontal and abs(self.dx_ema) > self.vel_thresh_norm_horizontal and not self._dx_gate_high \
               and self._consistent_sign(self.flow_window_dx, 4):
                self._dx_gate_high = True
                self._last_motion_cmd_ms = now_ms
                gesture = "swipe_right" if self.dx_ema > 0 else "swipe_left"
                cmd = "seek_forward" if self.dx_ema > 0 else "seek_back"

            # 张开手掌（静止 + spread）
            cooling = (now_ms - self._last_motion_cmd_ms) < self.open_palm_cooldown_ms
            flow_static_px = max(6, int(max(w, h) * 0.010))  # 与 demo 的静止门控等价
            is_static = (abs(dx_med) < flow_static_px) and (abs(dy_med) < flow_static_px)
            spread = self._palm_spread(pts, cx, cy)
            spread_ok = (four_up >= 4) and (self.open_palm_min_spread_ratio <= spread <= self.open_palm_max_spread_ratio)

            if (not cooling) and is_static and spread_ok:
                self._open_palm_stable_cnt += 1
                if (self._open_palm_stable_cnt * 33) >= self.open_palm_ms:
                    self._open_palm_stable_cnt = 0
                    gesture = "open_palm"
                    cmd = "toggle"
            else:
                self._open_palm_stable_cnt = 0

        detection_result = {
            'hand_present': self.num_hands > 0,
            'num_hands': self.num_hands,
            'gesture': gesture,
            'cmd': cmd,
            'primary_center': primary_center,
            'fps': self.fps
        }
        return detection_result

    def draw_landmarks(self, frame, hands_landmarks):
        try:
            if hands_landmarks and hands_landmarks.multi_hand_landmarks:
                for lm in hands_landmarks.multi_hand_landmarks:
                    self.drawer.draw_landmarks(
                        frame, lm, self.mp_hands.HAND_CONNECTIONS,
                        self.drawer_style.get_default_hand_landmarks_style(),
                        self.drawer_style.get_default_hand_connections_style()
                    )
        except Exception:
            pass
