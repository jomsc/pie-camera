import cv2
import numpy as np
import torch
import sys
import os

from ultralytics import YOLO

crop_factor = 0.46
number_of_rays = 66


def color_detect(self, frame_hsv):
    red_lower = np.array([0, 50, 75])     # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
    red_upper = np.array([10, 220, 100])    # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE

    red_lower2 = np.array([120, 50, 25])
    red_upper2 = np.array([180, 220, 230])

    green_lower = np.array([30, 70, 30])  # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
    green_upper = np.array([85, 255, 255])  # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
    
    mask_r1 = cv2.inRange(frame_hsv, red_lower, red_upper)
    mask_r2 = cv2.inRange(frame_hsv, red_lower2, red_upper2)
    mask_r = mask_r1
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)

    mask_g = cv2.inRange(frame_hsv, green_lower, green_upper)


    kernel = np.ones((3, 3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, kernel)
    
    

    stack_r = np.column_stack(np.where(mask_r > 0))
    avg_r = 0
    # If no white pixels, return None
    if len(stack_r) == 0:
        avg_r = -1
    else:
        avg_r = np.mean(stack_r[:, 1])/self.width # POSITION MOYENNE DE LA COULEUR ROUGE

    stack_g = np.column_stack(np.where(mask_g > 0))
    avg_g = 0
    # If no white pixels, return None
    if len(stack_g) == 0:
        avg_g = -1
    else:
        avg_g = np.mean(stack_g[:, 1])/self.width # POSITION MOYENNE DE LA COULEUR 

    count_r = np.count_nonzero(mask_r)/(self.width*self.height) # POURCENTAGE DE LA COULEUR ROUGE
    count_g = np.count_nonzero(mask_g)/(self.width*self.height) # POURCENTAGE DE LA COULEUR VERTE

    return avg_r, avg_g, count_r, count_g, mask_r, mask_g

def car_detect(model, img, conf_thres=0.5):
    car_label = []
    results = model(img)
    for r in results:
        boxes = r.boxes
        for box, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = box
            label = f'{model.names[int(cls_id)]} {conf:.2f}'
            #print(f"Detected {model.names[int(cls_id)]} with confidence {conf:.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            if 'car' in label and conf>conf_thres:
                car_label.append([int(x1), int(y1), int(x2), int(y2), label])

    return car_label

def sample_label(mask_r, mask_g, v, sample_radius, labels):
    color = 0 # 0 none, 1 red, 2 green, 3 car, 4 obstacle

    avg_r, avg_g = 0, 0
    for i in range(-sample_radius, sample_radius+1):
        for j in range(-sample_radius, sample_radius+1):
            avg_r += mask_r[j+v[1], i+v[0]]
            avg_g += mask_g[j+v[1], i+v[0]]
    
    avg_r = avg_r / (sample_radius*sample_radius)
    avg_g = avg_g / (sample_radius*sample_radius)


    if avg_r >= 127:
        color = 1
    elif avg_g >= 127:
        color = 2
    
    for l in labels:
        if v[0] >= l[0] and v[0] <= l[2]:
            color = 3
        
    if color == 0:
        color = 4
            
    return color

def lidar_sampling(self, frame_disp, car_label, mask_r, mask_g):
    margin = 5
    points = np.linspace(0+margin, self.width-margin-1, number_of_rays, dtype=int)
    height_of_color = 10
    sample_radius = 5

    ray_pos = [(v, height_of_color) for v in points]
    ray_labels = [sample_label(mask_r, mask_g, v, sample_radius, car_label) for v in ray_pos]
    for v in ray_pos:
        a = (int(v[0]-sample_radius), int(v[1]-sample_radius))
        b = (int(v[0]+sample_radius), int(v[1]+sample_radius))
        color = (0, 0, 0)
        i = ray_pos.index(v)
        if ray_labels[i]==1:
            color = (0, 0, 255)
        if ray_labels[i]==2:
            color = (0, 255, 0)
        if ray_labels[i]==3:
            color = (255, 0, 0)
        

        cv2.rectangle(frame_disp, a, b, color, 1)

    return ray_pos, ray_labels, frame_disp


class Camera:
    def __init__(self, width=576, height=325):
        self.width = width
        self.height = height
        self.fromfile = False
        
        weights_path = 'yolov5s.pt'
        self.model = YOLO(weights_path, task='detect')
        pass
    
    def read_frame(self):
        path = input("file name ?")
        img = cv2.imread('photos/2/'+path)
        #img = cv2.imread('photos/2/13.jpg')
        frame = cv2.flip(img, 0)
        frame = cv2.resize(frame, (self.width, self.height))
        return frame
    
    
    def process_stream(self, img):
        # ----------------------------------------------------------------------
        # GET IMAGE
        # ----------------------------------------------------------------------
        if self.fromfile == False:
            frame = img
        else:
            frame = self.read_frame()
        
        crop_height = int(crop_factor*self.height)
        frame = frame[crop_height:, :, :]
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # ----------------------------------------------------------------------
        # COLOR DETECTION
        # ----------------------------------------------------------------------
        avg_r, avg_g, count_r, count_g, mask_r, mask_g = color_detect(self=self, frame_hsv=frame_hsv)

        # ----------------------------------------------------------------------
        # CAR DETECTION 
        # ----------------------------------------------------------------------
        car_label = car_detect(self.model, frame)

        # ----------------------------------------------------------------------
        # LIDAR POINT SAMPLING
        # ----------------------------------------------------------------------
        frame_disp = np.copy(frame)
        ray_pos, ray_labels, frame_disp = lidar_sampling(self, frame_disp, car_label, mask_r, mask_g)

        # ----------------------------------------------------------------------
        # FRAME TO DISPLAY
        # ----------------------------------------------------------------------

        frame_disp[:,:,2][mask_r>0] += 100
        frame_disp[:,:,1][mask_g>0] += 100

        for v in car_label:
            cv2.rectangle(frame_disp, (v[0], v[1]), (v[2], v[3]), (255, 0, 0), 2)
            cv2.putText(frame_disp, v[4], (v[2] + 10, v[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)



        return ray_pos, ray_labels, frame_disp #, count_r, count_g, avg_r, avg_g