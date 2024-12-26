import cv2
import os
import yaml
import numpy as np

# 从YAML文件加载相机参数
def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    camera_matrix = np.array(params['camera_matrix'])
    distortion_coefficients = np.array(params['distortion_coefficients'])
    return camera_matrix, distortion_coefficients

camera_matrix, distortion_coefficient = load_camera_params('configs/intrinsic.yaml')

def function(file):
    if not os.path.exists(file):
        print(f"Error: The file at path '{file}' does not exist.")
        return None  
    image = cv2.imread(file)
   
    image = cv2.undistort(image, camera_matrix, distortion_coefficient)

    output_folder = "assets/test/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  

    existing_files = os.listdir(output_folder)
    
    png_files = [f for f in existing_files if f.endswith('.png')]

    if png_files:
        numbers = [int(f.split('.')[0]) for f in png_files]
        new_index = max(numbers) + 1
    else:
        new_index = 1  

    new_file_name = f"{new_index}.png"
    output_path = os.path.join(output_folder, new_file_name)

    cv2.imwrite(output_path, image)
    print(f"Image saved as {output_path}")

    return image


if __name__ == '__main__':
    file = "assets/datasetvideos/1.png"
    
    image=function(file)

    while True:
        cv2.imshow("choosePoint", image)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()
