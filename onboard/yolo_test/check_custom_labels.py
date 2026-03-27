import cv2
import os
import pdb


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

# Define the base directories for images and labels
base_image_dir = '/home/[USER]/'
base_label_dir = '/home/[USER]/'

# Define the subdirectories for each dataset
subdirectories = ['test', 'train', 'val']

# Define the function to draw bounding boxes
def draw_bounding_boxes(image_path, label_path):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Read the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    # Draw the bounding boxes
    for line in lines:
        # YOLO format: class x_center y_center width height (normalized values)
        values = line.strip().split()
        class_id = int(values[0])
        x_center = float(values[1]) * width
        y_center = float(values[2]) * height
        box_width = float(values[3]) * width
        box_height = float(values[4]) * height
        
        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey()

    
# Iterate through the subdirectories
if __name__ == '__main__':
    img_post_fix = '.png'
    for subdir in subdirectories:
        image_dir = os.path.join(base_image_dir, subdir)
        label_dir = os.path.join(base_label_dir, subdir)
        
        for filename in os.listdir(image_dir):
            if filename.endswith(img_post_fix):
                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, filename.replace(img_post_fix, '.txt'))
                print(image_path)
                #print(label_path)
                #pdb.set_trace()
                if os.path.exists(label_path):
                    draw_bounding_boxes(image_path, label_path)
