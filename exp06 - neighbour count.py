import cv2
import numpy as np

# Read image as grayscale and convert to binary
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

h, w = binary.shape

count_4 = np.zeros((h, w), dtype=int)       # Arrays to store neighbour counts
count_8 = np.zeros((h, w), dtype=int)


d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]     #4-neighbour directions

d8 = [(-1, 0), (1, 0), (0, -1), (0, 1),   
      (-1, -1), (-1, 1), (1, -1), (1, 1)]   # 8-neighbour directions

# Check every pixel
for r in range(h):
    for c in range(w):
        current_color = binary[r, c]
        
        for dr, dc in d4:      
            nr = r + dr         #calculating new neighbrs
            nc = c + dc

            if 0 <= nr < h and 0 <= nc < w:            #for controling edge cases...checking if new neighbrs within the img 
                if binary[nr, nc] == current_color:    #checking if same clr and increment
                    count_4[r, c] += 1

        for dr, dc in d8:      #now for 8 neighbrs
            nr = r + dr
            nc = c + dc

            if 0 <= nr < h and 0 <= nc < w:
                if binary[nr, nc] == current_color:
                    count_8[r, c] += 1

print("4-neighbour counts:")
print(count_4)

print("\n8-neighbour counts:")
print(count_8)
