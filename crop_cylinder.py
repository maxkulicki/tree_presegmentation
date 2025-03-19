import laspy
import numpy as np
import os

def crop_point_cloud_cylinder(input_path, center_x, center_y, radius, output_path):
    # Read the input point cloud
    las = laspy.read(input_path)
    
    # Extract the x, y coordinates of the points
    x = las.x
    y = las.y
    
    # Compute the distance of each point from the center (center_x, center_y)
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create a mask to filter points within the given radius
    mask = distances <= radius
    
    # Apply the mask to filter points
    cropped_las = laspy.create(point_format=las.point_format, file_version=las.header.version)
    
    for dimension in las.point_format.dimension_names:
        data = las[dimension][mask]
        setattr(cropped_las, dimension, data)
    
    # Update the header statistics
    cropped_las.update_header()

    cropped_las.header.offsets = las.header.offsets
    cropped_las.header.scales = las.header.scales
    
    # Write the cropped point cloud to the output path
    cropped_las.write(output_path)
    
    print(f"Cropped point cloud saved to {output_path}")

def process_directory(directory, center_x, center_y, radius):
    # Create the "cropped" subdirectory if it doesn't exist
    cropped_dir = os.path.join(directory, "cropped")
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)

    # Process each .laz file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".laz"):
            input_path = os.path.join(directory, filename)
            output_filename = os.path.splitext(filename)[0] + "_cropped.laz"
            output_path = os.path.join(cropped_dir, output_filename)
            crop_point_cloud_cylinder(input_path, center_x, center_y, radius, output_path)
