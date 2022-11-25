import numpy as np
import cv2
import matplotlib.pyplot as plt

# laplaceMechanism(img, length, width, epsilon)
# epsilon
#   The privacy budget of the laplace Mechanism.
#   The smaller the value is, the better privacy protection.
def laplaceMechanism(img, x, y, epsilon=1):
    # Flatten the img
    img = img.reshape(x * y)
    # Generate lap+ noise mask
    dp_noise = np.random.laplace(0, 1.0/epsilon, (x * y)).round() % 256 # round # mod 256
    img_out = (img + dp_noise) % 256 # round
    # Reshape to origin form
    img_out = img_out.reshape(y, x)
    return img_out

# Read img
img_gray = cv2.imread('img/cat.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('img/cat.jpg')

# Check data form & size
print(type(img_gray))
(y, x) = img_gray.shape
print(x, y)

# Output the origin img
plt.imshow(img_gray, cmap='gray')
plt.show()

# epsilon 1 -> 0.1 -> 0.05 -> 0.02 -> 0.01
img_out = laplaceMechanism(img_gray, x, y, 0.05)

# Output the img encrypted by dp
plt.imshow(img_out, cmap='gray')
plt.show()