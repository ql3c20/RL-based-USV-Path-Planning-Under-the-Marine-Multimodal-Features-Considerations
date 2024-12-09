import cv2
import numpy as np

# Cannny Edge detection + Contour extraction (which can be replaced by other image processing methods)
image = cv2.imread('Image processing module\Geographic Data\testmap.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


gray_circles = []
radii = []  

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(approx) > 6:  # Assume that the circular contour fits the polygon with a number of sides greater than 6
        area = cv2.contourArea(contour)
        if area > 100:  
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Color threshold
            mask = cv2.inRange(image, (100, 100, 100), (150, 150, 150))  
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            if cv2.mean(masked_image)[0] > 100:  
                gray_circles.append((center, radius))
                radii.append(radius)  


center_change = []
for (center, radius) in gray_circles:
    cv2.circle(image, center, radius, (0, 255, 0), 2)
    cv2.circle(image, center, 3, (0, 0, 255), -1)
    center = (center[0], 600 - center[1])
    center_change.append(center)

# Detected Gray Circle Positions
print("Modified Gray Circle Positions:", center_change)
print("Radii:", radii)

# Detected Gray Circles
cv2.imshow('Detected Gray Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
output_path = 'Image processing module\Geographic Data\testmap_detect.jpg'
cv2.imwrite(output_path, image)