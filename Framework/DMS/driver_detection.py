import math
import time
import cv2
import mediapipe as mp
import numpy as np
import copy
from PIL import Image
import clip
import torch
from collections import Counter

class Face_mesh():
    def __init__(self):
        super(Face_mesh, self).__init__()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.start_time = time.time()
        self.device = torch.device('cuda:0')
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3,
                                               refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.frame = np.zeros([640, 480], dtype='uint8')
        self.annotated_image = np.zeros([640, 480], dtype='uint8')
        self.left_eye_image = np.zeros([640, 480], dtype='uint8')
        self.right_eye_image = np.zeros([640, 480], dtype='uint8')
        self.mouth_image = np.zeros([640, 480], dtype='uint8')
        self.yawn_time = [-1]
        self.yawn_duration = 0
        self.yawn_interval = [1]
        self.yawn_frequency = []
        self.start_blink_time = [-1]
        self.blink_duration = 0
        self.blink_interval = [1]
        self.blink_frequency = []

        self.angle_yaw = [0]
        self.angle_pitch = [0]
        self.angle_roll = [0]
        self.gaze_yaw = [0]
        self.gaze_pitch = [0]
        self.gaze_roll = [0]
        self.eye_close = False
        self.eye_open = True
        self.yawning = False

        self.right_iris = [474, 475, 476, 477, 474]
        self.left_iris = [469, 470, 471, 472, 469]
        self.right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
        self.left_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
        self.right_eyebrow = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 336]
        self.left_eyebrow = [55, 65, 52, 53, 46, 70, 63, 105, 66, 107, 55]
        self.contours_face = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                              234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356]
        self.mouth_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
        self.mouth_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        self.history_mouth_pts = []
        self.history_eye_pts = []

        self.teacher_model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.classes = ["drinking", "answering the phone", "texting with phone", "drowsiness", "normal driving"]
        self.classes_expression = ["Happy", "Angry", "Neutral"]
        self.text = clip.tokenize(
            ["A man is drinking somthing",
             "A man is answering the phone",
             "A man is staring at his phone",
             "A man is dozing off",
             "A man is sitting upright, staring ahead"
             ]).to(self.device)
        self.text_expression = clip.tokenize(
            ["A man with happy expression",
             "A man with angry expression",
             "A man with neutral expression",
             ]).to(self.device)
        self.results = []
        self.exp_results = []
        self.abnormal_behaviour = ''
        self.expression = ''
        self.behaviour_score = 0
        self.expression_score = 0


    def head_pose_calculation(self, landmarks, landmark, points, frame_shape):
        '''
            yaw is calculate according to landmark point 10 and 164
           pitch is calculate according to landmark point 132 and 361
           roll is calculate according to point 94 and the middle point of 132 and 361
        '''

        coord_len = 0.1
        head_pose_angle_num = 20
        p10 = landmarks[10]
        p4 = landmarks[94]
        p152 = landmarks[164]
        p132 = landmarks[132]
        p361 = landmarks[361]
        pm = [(p132.x + p361.x)/2, (p132.y + p361.y)/2, (p132.z + p361.z)/2]
        coord = [p4.x, p4.y, p4.z]
        annotated_image = np.zeros(frame_shape, dtype='uint8')
        h, w = frame_shape[:-1]

        contour_pts = []
        for c in self.contours_face:
            contour_pts.append(points[c])
        contour_pts = np.array(contour_pts)
        cv2.polylines(annotated_image, [contour_pts], False, (255, 255, 255), thickness=2)

        self.mp_drawing.draw_landmarks(image=annotated_image, landmark_list=landmark,
                                       connections=self.mp_face_mesh.FACEMESH_TESSELATION,  ### 绘制网格
                                       landmark_drawing_spec=None,
                                       connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # coord = pm
        disyaw = np.sqrt(np.square(p10.x - p152.x) + np.square(p10.y - p152.y) + np.square(p10.z - p152.z))
        dispitch = np.sqrt(np.square(p132.x - p361.x) + np.square(p132.y - p361.y) + np.square(p132.z - p361.z))
        disroll = np.sqrt(np.square(pm[0] - p4.x) + np.square(pm[1] - p4.y) + np.square(pm[2] - p4.z))

        vecyaw = [(p152.x - p10.x)/disyaw, (p152.y - p10.y)/disyaw, (p152.z - p10.z)/disyaw]
        vecpitch = [(p361.x - p132.x)/dispitch, (p361.y - p132.y)/dispitch, (p361.z - p132.z)/dispitch]
        vecroll = [(p4.x - pm[0])/disroll, (p4.y - pm[1])/disroll, (p4.z - pm[2])/disroll]

        yaw = [coord[0] + vecyaw[0] * coord_len, coord[1] + vecyaw[1] * coord_len, coord[2] + vecyaw[2] * coord_len]
        pitch = [coord[0] + vecpitch[0] * coord_len, coord[1] + vecpitch[1] * coord_len, coord[2] + vecpitch[2] * coord_len]
        roll = [coord[0] + vecroll[0] * coord_len, coord[1] + vecroll[1] * coord_len, coord[2] + vecroll[2] * coord_len]

        cv2.arrowedLine(annotated_image, (int(coord[0] * w), int(coord[1] * h)), (int(yaw[0] * w), int(yaw[1] * h)),
                        (255, 0, 0), 3)   # the blue axis
        cv2.arrowedLine(annotated_image, (int(coord[0] * w), int(coord[1] * h)), (int(pitch[0] * w), int(pitch[1] * h)),
                        (0, 255, 0), 3)   # the green axis
        cv2.arrowedLine(annotated_image, (int(coord[0] * w), int(coord[1] * h)), (int(roll[0] * w), int(roll[1] * h)),
                        (0, 0, 255), 3)   # the red axis

        angle_yaw = (roll[0] * w - coord[0] * w) * 1.6
        angle_pitch = (roll[1] * h - coord[1] * h) * 2
        angle_roll = math.atan((pitch[1] * h - coord[1] * h) / (pitch[0] * w - coord[0] * w)) * 90

        self.angle_yaw.append(angle_yaw)
        self.angle_pitch.append(angle_pitch)
        self.angle_roll.append(angle_roll)

        if len(self.angle_yaw) > head_pose_angle_num:
            self.angle_yaw.pop(0)
        if len(self.angle_pitch) > head_pose_angle_num:
            self.angle_pitch.pop(0)
        if len(self.angle_roll) > head_pose_angle_num:
            self.angle_roll.pop(0)

        return vecroll, annotated_image


    def gaze_direction_calculation(self, vector, landmarks, frame_shape):
        iris_r = -0.018
        arrow_len = 0.1
        gaze_angle_num = 20
        h, w = frame_shape[:-1]

        ave_left_x, ave_left_y, ave_left_z = 0, 0, 0
        for k in self.left_eye:
            eye_point = landmarks[k]
            ave_left_x = ave_left_x + eye_point.x
            ave_left_y = ave_left_y + eye_point.y
            ave_left_z = ave_left_z + eye_point.z
        ave_left_x = ave_left_x / len(self.left_eye)
        ave_left_y = ave_left_y / len(self.left_eye)
        ave_left_z = ave_left_z / len(self.left_eye)
        left_eye_center = [ave_left_x + vector[0]*iris_r, ave_left_y + vector[1]*iris_r, ave_left_z + vector[2]*iris_r]

        ave_right_x, ave_right_y, ave_right_z = 0, 0, 0
        for k in self.right_eye:
            eye_point = landmarks[k]
            ave_right_x = ave_right_x + eye_point.x
            ave_right_y = ave_right_y + eye_point.y
            ave_right_z = ave_right_z + eye_point.z
        ave_right_x = ave_right_x / len(self.right_eye)
        ave_right_y = ave_right_y / len(self.right_eye)
        ave_right_z = ave_right_z / len(self.right_eye)
        right_eye_center = [ave_right_x + vector[0] * iris_r, ave_right_y + vector[1] * iris_r,
                           ave_right_z + vector[2] * iris_r]

        iris_left_x, iris_left_y, iris_left_z = 0, 0, 0
        for k in self.left_iris:
            iris_point = landmarks[k]
            iris_left_x = iris_left_x + iris_point.x
            iris_left_y = iris_left_y + iris_point.y
            iris_left_z = iris_left_z + iris_point.z
        iris_left_p = [iris_left_x / len(self.left_iris), iris_left_y / len(self.left_iris), iris_left_z / len(self.left_iris)]

        iris_right_x, iris_right_y, iris_right_z = 0, 0, 0
        for k in self.right_iris:
            iris_point = landmarks[k]
            iris_right_x = iris_right_x + iris_point.x
            iris_right_y = iris_right_y + iris_point.y
            iris_right_z = iris_right_z + iris_point.z
        iris_right_p = [iris_right_x / len(self.right_iris), iris_right_y / len(self.right_iris), iris_right_z / len(self.right_iris)]

        left_dis = np.sqrt(np.square(iris_left_p[0] - left_eye_center[0]) + np.square(iris_left_p[1] - left_eye_center[1])
                           + np.square(iris_left_p[2] - left_eye_center[2]))
        left_vector = [(iris_left_p[0] - left_eye_center[0])/left_dis, (iris_left_p[1] - left_eye_center[1])/left_dis,
                       (iris_left_p[2] - left_eye_center[2])/left_dis]

        right_dis = np.sqrt(np.square(iris_right_p[0] - right_eye_center[0]) + np.square(iris_right_p[1] - right_eye_center[1])
                           + np.square(iris_right_p[2] - right_eye_center[2]))
        right_vector = [(iris_right_p[0] - right_eye_center[0]) / right_dis,
                       (iris_right_p[1] - right_eye_center[1]) / right_dis,
                       (iris_right_p[2] - right_eye_center[2]) / right_dis]
        vector = [(left_vector[0] + right_vector[0])/2, (left_vector[1] + right_vector[1])/2, (left_vector[2] + right_vector[2])/2, ]

        left_end = [left_eye_center[0] + vector[0] * arrow_len, left_eye_center[1] + vector[1] * arrow_len,
                    left_eye_center[2] + vector[2] * arrow_len]
        right_end = [right_eye_center[0] + vector[0] * arrow_len,
                     right_eye_center[1] + vector[1] * arrow_len,
                     right_eye_center[2] + vector[2] * arrow_len]

        v_w = vector[0] * w
        v_h = vector[1] * h
        v_l = vector[2] * 500
        gaze_yaw = -math.atan(v_w / v_l) * 72
        gaze_pitch = -math.atan(v_h / v_l) * 72
        self.gaze_yaw.append(gaze_yaw)
        self.gaze_pitch.append(gaze_pitch)

        if len(self.gaze_yaw) > gaze_angle_num:
            self.gaze_yaw.pop(0)
        if len(self.gaze_pitch) > gaze_angle_num:
            self.gaze_pitch.pop(0)

        return left_eye_center, left_end, right_eye_center, right_end


    def draw_eyes(self, eye_points, iris_points, w, h):
        img_h = 60
        img_w = 120
        r = 1.6
        cen_x, cen_y = 0, 0
        for i in range(16):
            p = eye_points[i]
            cen_x = cen_x + p[0]
            cen_y = cen_y + p[1]
        cen_x = cen_x / 16
        cen_y = cen_y / 16
        o_x = int(cen_x - 30)
        o_y = int(cen_y - 15)

        image = np.zeros((img_h, img_w, 3), dtype='uint8')

        x, y = [], []
        for i in range(17):
            eye_points[i][0] = eye_points[i][0] - o_x
            eye_points[i][1] = eye_points[i][1] - o_y
            x.append(eye_points[i][0])
            y.append(eye_points[i][1])

        for i in range(5):
            iris_points[i][0] = iris_points[i][0] - o_x
            iris_points[i][1] = iris_points[i][1] - o_y

        cv2.ellipse(image, (int(img_w / 3), int(img_h / 2)), (int(w), int(w / 2)), 0, 0, 360, (0, 255, 0), 1)
        cv2.ellipse(image, (int(img_w / 3), int(img_h / 2)), (int(w) + 2, int(w / 2) + 2), 0, 0, 360, (0, 255, 0), 1)
        cv2.ellipse(image, (int(img_w/3), int(img_h/2)), (int(w), int(h*r)), 0, 0, 360, (255, 255, 255), -1)

        return image


    def show_eyes(self, points):
        record_time = 4
        blink_rate = 0.16
        left_eye_pts = []
        left_iris = []
        right_eye_pts = []
        right_iris = []

        for i in self.left_eye:
            left_eye_pts.append(points[i])
        for i in self.left_iris:
            left_iris.append(points[i])
        for i in self.right_eye:
            right_eye_pts.append(points[i])
        for i in self.right_iris:
            right_iris.append(points[i])

        eye_pts = left_eye_pts + right_eye_pts + left_iris + right_iris
        delta_eye = 0
        while len(self.history_eye_pts) < 3:
            self.history_eye_pts.append(eye_pts)
        pts = self.history_eye_pts[-1]
        for k in range(len(eye_pts)):
            p1 = pts[k]
            p2 = eye_pts[k]
            delta_eye = delta_eye + np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))
        delta_eye = delta_eye / len(eye_pts)
        if delta_eye > 1.0:
            self.history_eye_pts.pop(0)
            self.history_eye_pts.append(eye_pts)
        eye_points = copy.deepcopy(self.history_eye_pts[-1])

        leye_left = eye_points[0]
        leye_right = eye_points[8]
        leye_top = eye_points[4]
        leye_bottom = eye_points[12]
        reye_left = eye_points[17]
        reye_right = eye_points[25]
        reye_top = eye_points[29]
        reye_bottom = eye_points[21]
        leye_w = np.sqrt(np.square(leye_left[0]-leye_right[0]) + np.square(leye_left[1] - leye_right[1]))
        leye_h = np.sqrt(np.square(leye_bottom[0] - leye_top[0]) + np.square(leye_bottom[1] - leye_top[1]))
        reye_w = np.sqrt(np.square(reye_left[0] - reye_right[0]) + np.square(reye_left[1] - reye_right[1]))
        reye_h = np.sqrt(np.square(reye_bottom[0] - reye_top[0]) + np.square(reye_bottom[1] - reye_top[1]))

        if leye_h/leye_w < blink_rate or reye_h/reye_w < blink_rate:
            interval = time.time() - self.start_blink_time[-1]
            self.eye_close = True
            self.eye_open = False
            if interval > 1:  # 0.5
                self.start_blink_time = []
                self.start_blink_time.append(time.time())
                self.blink_frequency.append(time.time())
            else:
                self.start_blink_time.append(time.time())
                self.blink_duration = (time.time() - self.start_blink_time[0]) * 10 # 100
        else:
            self.eye_open = True
            if time.time() - self.start_blink_time[-1] > 1:
                self.blink_duration = 0
                self.eye_close = False

        blink_interval = time.time() - self.start_blink_time[-1]
        self.blink_interval.append(blink_interval)
        if len(self.blink_interval) > 50:
            self.blink_interval.pop(0)
        if len(self.blink_frequency) > 0:
            inter = time.time() - self.blink_frequency[0]
            if inter > record_time:
                self.blink_frequency.pop(0)

        left_land_pts = np.array(eye_points[:17])
        right_land_pts = np.array(eye_points[17:34])
        left_iris = np.array(eye_points[34:39])
        right_iris = np.array(eye_points[39:])

        left_eye_img = self.draw_eyes(left_land_pts, left_iris, leye_w, leye_h)
        right_eye_img = self.draw_eyes(right_land_pts, right_iris, reye_w, reye_h)

        self.left_eye_image = left_eye_img
        self.right_eye_image = right_eye_img


    def get_order_of_facial_landmark_points(self, frame, landmark):
        h, w = frame.shape[:-1]
        contours = set(self.mp_face_mesh.FACEMESH_CONTOURS)
        contours_pp = []
        for x in contours:
            contours_pp.append(x[0])
            contours_pp.append(x[1])
        contours_set = set(contours_pp)
        left_eye = set(self.left_eye)
        left_eyebrow = set(self.left_eyebrow)
        right_eye = set(self.right_eye)
        right_eyebrow = set(self.right_eyebrow)
        face_oval = set(self.contours_face)
        mouth = contours_set - left_eyebrow - left_eye - right_eyebrow - right_eye - face_oval

        font = cv2.FONT_HERSHEY_SIMPLEX
        for con in mouth:
            if con < 50:
                contours_point = landmark[con]
                cv2.circle(frame, (int(contours_point.x * w), int(contours_point.y * h)), 2, (255, 255, 0), -1)
                cv2.putText(frame, str(con), (int(contours_point.x * w) + 2, int(contours_point.y * h) + 2), font,
                            0.2, (0, 255, 0), 1)

        mouth_pts = []
        mouth_pts.append(14)
        connect_p = 14
        for i in range(10):
            for p in contours:
                if p[1] == connect_p:
                    mouth_pts.append(p[0])
                    connect_p = p[0]
                    break


    def show_mouth(self, points):
        record_time = 8
        yawn_rate = 0.6
        img_h = 50
        img_w = 90
        mouth_image = np.zeros((img_h, img_w, 3), dtype='uint8')
        mouth_inner_pts = []
        mouth_outer_pts = []
        for i in self.mouth_inner:
            mouth_inner_pts.append([points[i][0], points[i][1]])
        for i in self.mouth_outer:
            mouth_outer_pts.append([points[i][0], points[i][1]])

        mouth_pts = mouth_inner_pts + mouth_outer_pts
        delta_mouth = 0
        while len(self.history_mouth_pts) < 3:
            self.history_mouth_pts.append(mouth_pts)
        pts = self.history_mouth_pts[-1]
        for k in range(len(mouth_pts)):
            p1 = pts[k]
            p2 = mouth_pts[k]
            delta_mouth = delta_mouth + np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))
        delta_mouth = delta_mouth / len(mouth_pts)
        if delta_mouth > 1.0:
            self.history_mouth_pts.pop(0)
            self.history_mouth_pts.append(mouth_pts)

        mouth_points = copy.deepcopy(self.history_mouth_pts[-1])
        m_top = mouth_points[5]
        m_bottom = mouth_points[15]
        m_left = mouth_points[0]
        m_right = mouth_points[10]
        mouth_h = np.sqrt(np.square(m_top[0] - m_bottom[0]) + np.square(m_top[1] - m_bottom[1]))
        mouth_w = np.sqrt(np.square(m_right[0] - m_left[0]) + np.square(m_right[1] - m_left[1]))

        if mouth_h/mouth_w > yawn_rate:
            self.yawning = True
            interval = time.time() - self.yawn_time[-1]
            self.yawn_interval.append(interval)
            if interval > 0.2:
                self.yawn_time = []
                self.yawn_time.append(time.time())
                self.yawn_frequency.append(time.time())
            else:
                self.yawn_time.append(time.time())
                self.yawn_duration = (time.time() - self.yawn_time[0]) * 100
        else:
            self.yawn_interval.append(1)
            if time.time() - self.yawn_time[-1] > 1:
                self.yawning = False
                self.yawn_duration = 0

        if len(self.yawn_interval) > 50:
            self.yawn_interval.pop(0)
        if len(self.yawn_frequency) > 0:
            inter = time.time() - self.yawn_frequency[0]
            if inter > record_time:
                self.yawn_frequency.pop(0)

        cen_x = 0
        cen_y = 0
        for p in mouth_points:
            cen_x = cen_x + p[0]
            cen_y = cen_y + p[1]
        cen_x = cen_x / len(mouth_points)
        cen_y = cen_y / len(mouth_points)
        o_x = int(cen_x - 35)
        o_y = int(cen_y - 25)

        for n in range(len(mouth_points)):
            mouth_points[n][0] = mouth_points[n][0] - o_x
            mouth_points[n][1] = mouth_points[n][1] - o_y

        mouth_outer_pts = np.array(mouth_points[21:])
        mouth_inner_pts = np.array(mouth_points[:21])
        cv2.polylines(mouth_image, [mouth_outer_pts], False, (255, 255, 255), thickness=1)
        cv2.polylines(mouth_image, [mouth_inner_pts], False, (255, 255, 255), thickness=1)

        self.mouth_image = mouth_image


    def abnormal_behaviour_detection(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        logits_per_image, logits_per_text = self.teacher_model(image, self.text)
        probs = logits_per_image.detach().softmax(dim=-1).cpu().numpy()
        pred = np.argmax(probs[0])
        if probs[0][pred] < 0.6:
            pred = 4
        cls = self.classes[pred]
        self.results.append(cls)
        if len(self.results) > 20:
            self.results.pop(0)
        collect = Counter(self.results)
        most_num = collect.most_common(1)
        self.abnormal_behaviour = most_num[0][0]
        self.behaviour_score = most_num[0][1] / 20 * 100


    def expression_recognition(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        logits_per_image, logits_per_text = self.teacher_model(image, self.text_expression)
        probs = logits_per_image.detach().softmax(dim=-1).cpu().numpy()
        pred = np.argmax(probs[0])
        if probs[0][pred] < 0.56:
            pred = 2
        cls = self.classes_expression[pred]
        self.exp_results.append(cls)
        if len(self.exp_results) > 20:
            self.exp_results.pop(0)
        collect = Counter(self.exp_results)
        most_num = collect.most_common(1)
        self.expression = most_num[0][0]
        self.expression_score = most_num[0][1]/20 * 100


    def get_3D_face_mesh(self, frame):
        output = []
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_landmarks = results.multi_face_landmarks
        frame_shape = frame.shape
        h, w = frame_shape[:-1]

        if face_landmarks:
            land_pts = []
            landmark = face_landmarks[0]     ## There are total of 468 facial landmark points

            for i in range(478):
                point = landmark.landmark[i]
                x = int(point.x * w)
                y = int(point.y * h)
                land_pts.append((x, y))

            # show the head pose coordinate
            vecroll, annotated_image = self.head_pose_calculation(landmark.landmark, landmark, land_pts, frame_shape)

            # show the gaze direction
            left_eye_center, left_end, right_eye_center, right_end = self.gaze_direction_calculation(vecroll, landmark.landmark, frame_shape)

            # show eyes
            self.show_eyes(land_pts)

            # show mouth
            self.show_mouth(land_pts)

            self.abnormal_behaviour_detection(frame)

            self.expression_recognition(frame)

            if self.eye_open:
                cv2.arrowedLine(frame, (int(left_eye_center[0] * w), int(left_eye_center[1] * h)),
                                (int(left_end[0] * w), int(left_end[1] * h)),
                                (0, 255, 0), 2)
                cv2.arrowedLine(frame, (int(right_eye_center[0] * w), int(right_eye_center[1] * h)),
                                (int(right_end[0] * w), int(right_end[1] * h)), (0, 0, 255), 2)

            ann_h = 300
            ann_w = 360
            land_pts = np.array(land_pts)
            rx, ry, rw, rh = cv2.boundingRect(land_pts)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)
            st_h = max(0, int(ry - 0.4 * rh))
            st_w = max(0, int(rx - 0.5 * rw))
            annotated_image = annotated_image[st_h:int(ry + 1.4 * rh), st_w:int(rx + 1.5 * rw)]
            annotated_image = cv2.resize(annotated_image, (268, 300))
            a_h, a_w = annotated_image.shape[:-1]
            top = buttom = int((ann_h - a_h)/2)
            left = right = int((ann_w - a_w)/2)
            annotated_image = cv2.copyMakeBorder(annotated_image, top, buttom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
            self.frame = frame
            self.annotated_image = annotated_image

        output.append(self.frame)
        output.append(self.annotated_image)
        output.append(self.left_eye_image)
        output.append(self.right_eye_image)
        output.append(self.mouth_image)

        return output

