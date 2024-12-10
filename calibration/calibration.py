import cv2
import numpy as np
import glob
import yaml

# read parameters of chessboard
with open('configs/intrinsic.yaml', 'r') as f:
    calibration_config = yaml.load(f, Loader=yaml.FullLoader)

pattern_cols = calibration_config['pattern_cols']  
pattern_rows = calibration_config['pattern_rows']  
square_size_mm = calibration_config['square_size_mm']   # The actual size of each square

# Create world coordinates for a chessboard pattern
object_points = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
object_points[:, :2] = np.indices((pattern_cols, pattern_rows)).T.reshape(-1, 2)
object_points *= square_size_mm  # Convert units to millimeters

# Store all corners and corresponding physical coordinates in the image
all_object_points = []
all_image_points = []

# load images
image_files = glob.glob('assets/images/*.jpg')  

# Traverse each image for calibration
for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (pattern_cols, pattern_rows), None)

    if ret:
        all_object_points.append(object_points)
        all_image_points.append(corners)

        cv2.drawChessboardCorners(img, (pattern_cols, pattern_rows), corners, ret)
        cv2.imshow('Chessboard corners', cv2.resize(img, (800, 400)))
        cv2.waitKey(500)  
    else:
        print("No chessboard corners found in", image_file)

cv2.destroyAllWindows()

# Camera calibration: Calculate the internal parameter matrix K and distortion coefficient
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    all_object_points, all_image_points, gray.shape[::-1], None, None)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Write the calibration results into the same YAML file
calibration_data = {
    'camera_matrix': mtx.tolist(),
    'distortion_coefficients': dist.tolist(),
    'pattern_cols': pattern_cols,
    'pattern_rows': pattern_rows,
    'square_size_mm': square_size_mm
}

with open('configs/intrinsic.yaml', 'w') as f:
    yaml.dump(calibration_data, f, default_flow_style=False)
