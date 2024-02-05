import cv2

# Load the image
img = cv2.imread('images/shelter-prey.png', cv2.IMREAD_UNCHANGED)

# Print original size
print('Original Dimensions : ', img.shape)

# Specify the desired size
width = 300
height = 300
dim = (width, height)

# Resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# Print new size
print('Resized Dimensions : ', resized.shape)

# Save the result
cv2.imwrite('images/shelter-prey.png', resized)