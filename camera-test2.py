import cv2
import os
import numpy as np
from picamera2 import Picamera2

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
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        return frame
    
    def process_stream(self):
        frame = self.read_frame()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)


        # COLOR DETECTION
        red_lower = np.array([0, 180, 180])     # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
        red_upper = np.array([10, 220, 230])    # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE

        green_lower = np.array([35, 70, 30])  # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
        green_upper = np.array([85, 255, 255])  # EN HSV, A AJUSTER POUR QUE LE MUR SOIT DETECTE
     
        mask_r = cv2.inRange(frame_hsv, red_lower, red_upper)
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
            avg_g = np.mean(stack_g[:, 1])/self.width # POSITION MOYENNE DE LA COULEUR VERTE

        
        # EDGE DETECTION 
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_resize = cv2.resize(img_gray, (960, 540))[250:, :]
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_resize, (5,5), 0) 
        #sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        canny_img = cv2.Canny(img_blur, 100, 200)

        # CONTOUR (CAR) DETECTION 
        # Apply morphological closing to connect edges
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)

        # Find connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        # Minimum area to consider (adjust based on your scenario)
        min_area = 700

        # Loop through all detected components (skip background 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(img_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        
        
        # DISPLAY
        mask_rS = cv2.resize(mask_r, (960,540))
        mask_gS = cv2.resize(mask_g, (960,540))
        frame_S = cv2.resize(frame, (960, 540))

        cv2.imshow('Frame', frame_S)
        cv2.imshow('R', mask_rS)
        cv2.imshow('G', mask_gS)
        cv2.imshow('Edges', canny_img)
        cv2.imshow('Contours', img_resize)

        cv2.waitKey()
        cv2.destroyAllWindows()
        self.picam2.stop()
        

        count_r = np.count_nonzero(mask_r)/(self.width*self.height) # POURCENTAGE DE LA COULEUR ROUGE
        count_g = np.count_nonzero(mask_g)/(self.width*self.height) # POURCENTAGE DE LA COULEUR VERTE

        return avg_r, avg_g, count_r, count_g

                              
               
        
        
# EXEMPLE D'IMPLEMENTATION DANS LE MAIN 

def main():
    camera = Camera()
    avg_r, avg_g, count_r, count_g = camera.process_stream()
    print(f"avg_r: {avg_r}, avg_g: {avg_g}, count_r: {count_r}, count_g: {count_g}")
    
   

if __name__ == '__main__':
    main()
