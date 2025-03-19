import laspy
import numpy as np
from scipy.spatial.transform import Rotation

import laspy
import numpy as np
from scipy.spatial.transform import Rotation

def crop_point_cloud_tilted_cylinder(input_path, center_x, center_y, center_z, pca_x, pca_y, pca_z, radius, output_path):
    # Read the input point cloud
    las = laspy.read(input_path)
    
    # Extract the coordinates of the points
    points = np.vstack((las.x, las.y, las.z)).T
    
    # Calculate the direction vector of the cylinder axis
    axis = np.array([pca_x, pca_y, pca_z])
    axis /= np.linalg.norm(axis)
    
    # Create a rotation matrix to align the cylinder with the z-axis
    rotation_matrix = Rotation.align_vectors([[0, 0, 1]], [axis])[0].as_matrix()
    
    # Translate points to the cylinder's base center
    translated_points = points - np.array([center_x, center_y, center_z])
    
    # Rotate points to align the cylinder with the z-axis
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    
    # Compute the distance of each point from the cylinder axis
    distances = np.sqrt(rotated_points[:, 0]**2 + rotated_points[:, 1]**2)
    
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
