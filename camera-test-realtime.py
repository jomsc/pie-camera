import cv2
import os
import numpy as np
from picamera2 import Picamera2

crop_height = 220
number_of_rays = 50

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
        if v[0] >= l[0] and v[0] <= l[1]:
            color = 3
        
    if color == 0:
        color = 4
            
    return color

class Camera:
    def __init__(self, width=854, height=480):
        self.picam2 = Picamera2()
        
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height)},
            lores={"size": (width, height)}
        )
        self.picam2.configure(config)
        
        self.picam2.start()
    
    def read_frame(self):
        img = self.picam2.capture_array()
        frame_d = cv2.flip(img, 0)
        frame = cv2.flip(frame_d, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        return frame
    
    def process_stream(self):
        frame = self.read_frame()
        frame = frame[crop_height:, :, :]
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # COLOR DETECTION
        red_lower = np.array([0, 50, 75])     # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
        red_upper = np.array([10, 220, 100])    # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE

        red_lower2 = np.array([120, 50, 25])
        red_upper2 = np.array([180, 220, 230])

        green_lower = np.array([30, 70, 30])  # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
        green_upper = np.array([85, 255, 255])  # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
     
        mask_r1 = cv2.inRange(frame_hsv, red_lower, red_upper)
        mask_r2 = cv2.inRange(frame_hsv, red_lower2, red_upper2)
        
        mask_r = mask_r2
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)
        
        mask_g = cv2.inRange(frame_hsv, green_lower, green_upper)

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
            
        frame_disp = np.copy(frame)
        frame_disp[:,:,2][mask_r>0] = 180
        frame_disp[:,:,1][mask_g>0] = 180

        
        # EDGE DETECTION 
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (5,5), 0) 
        #sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        canny_img = cv2.Canny(img_blur, 100, 200)

        # CONTOUR (CAR) DETECTION 
        # Apply morphological closing to connect edges
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)

        # Find connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        # Minimum area to consider (adjust based on your scenario)
        min_area = 500

        car_label = []
        # Loop through all detected components (skip background 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                car_label.append((x, x+w))
                cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ----------------------------------------------------------------------
        # LIDAR POINT SAMPLING
        # ----------------------------------------------------------------------
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
        

        count_r = np.count_nonzero(mask_r)/(self.width*self.height) # POURCENTAGE DE LA COULEUR ROUGE
        count_g = np.count_nonzero(mask_g)/(self.width*self.height) # POURCENTAGE DE LA COULEUR VERTE

        return frame_disp, car_label

                              
               
        
        
# EXEMPLE D'IMPLEMENTATION DANS LE MAIN 

def main():
    camera = Camera()

    while True:
        frame_disp, car_label = camera.process_stream()

        cv2.imshow('Frame', frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
