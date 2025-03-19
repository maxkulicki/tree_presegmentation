import os
import glob
from tqdm import tqdm
import laspy
import numpy as np
import dendromatics as dm
from crop_tilted import crop_point_cloud_tilted_cylinder
from scipy.spatial.transform import Rotation

def crop_point_cloud_tilted_cylinder(las_data, points, center_x, center_y, center_z, pca_x, pca_y, pca_z, radius, output_path):
    """
    Crop a point cloud cylinder using pre-loaded points.
    
    Args:
        las_data: Original laspy object containing all point attributes
        points: numpy array of shape (n, 3) containing x,y,z coordinates
        ...
    """
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
    cropped_las = laspy.create(point_format=las_data.point_format, file_version=las_data.header.version)
    
    for dimension in las_data.point_format.dimension_names:
        data = las_data[dimension][mask]
        setattr(cropped_las, dimension, data)
    
    # Update the header statistics
    cropped_las.update_header()
    cropped_las.header.offsets = las_data.header.offsets
    cropped_las.header.scales = las_data.header.scales
    
    # Write the cropped point cloud to the output path
    cropped_las.write(output_path)

def create_grid(bounds, tile_size=30, overlap=5):
    """Create a grid of overlapping tiles from point cloud bounds."""
    min_x, max_x = bounds[0], bounds[1]
    min_y, max_y = bounds[2], bounds[3]
    
    tiles = []
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            # Create tile bounds with overlap
            tile_bounds = (
                x - overlap,
                x + tile_size + overlap,
                y - overlap,
                y + tile_size + overlap
            )
            tiles.append(tile_bounds)
            y += tile_size
        x += tile_size
    return tiles

def process_tile(las_data, points, bounds, trees_dir, base_name, n_trees, save_crops=True, save_center_file=None):
    """Process a single tile of points."""
    # Filter points within tile bounds
    mask = (
        (points[:, 0] >= bounds[0]) & (points[:, 0] <= bounds[1]) &
        (points[:, 1] >= bounds[2]) & (points[:, 1] <= bounds[3])
    )
    tile_points = points[mask]
    
    if len(tile_points) < 1000:  # Skip tiles with too few points
        return n_trees

    # Clean ground and generate DTM
    clean_points = dm.clean_ground(tile_points)
    dtm = dm.generate_dtm(clean_points)
    z0_values = dm.normalize_heights(tile_points, dtm)
    coords = np.append(tile_points, np.expand_dims(z0_values, axis=1), 1)
    
    # Extract stripe
    mask = (coords[:, 3] > 0.7) & (coords[:, 3] < 3.5)
    stripe = coords[mask, 0:4]
    
    if len(stripe) < 1000:  # Skip if not enough points in stripe
        return n_trees
    
    # Perform verticality clustering
    try:
        clust_stripe = dm.verticality_clustering(stripe, n_iter=2)
    except Exception as e:
        print(f"Failed to cluster stripe: {str(e)}")
        return n_trees
    
    # Individualize trees
    assigned_cloud, tree_vector, _ = dm.individualize_trees(
        coords, clust_stripe, 0.02, 0.02, 0.7, 3.5,
        0.7, 1.5, 1000, 15, 25, 0.3, 5, 0, 1, 2, tree_id_field=-1
    )
    
    for i, tree in enumerate(tree_vector):

        # Check if tree is within tile bounds
        tx, ty = tree[4], tree[5]  # tree center coordinates
        if (bounds[0] <= tx <= bounds[1] and 
            bounds[2] <= ty <= bounds[3]):
            
            n_trees = n_trees + 1

            #save centers
            if save_center_file:
                save_center_file.write(f"{base_name}_{n_trees},{tree[4]},{tree[5]},{tree[6]}\n")

            #save crops
            if save_crops:
                print(f"  - Processing tree {n_trees}")
                center_x, center_y, center_z = tree[4:7]
                pca_x, pca_y, pca_z = tree[1:4]
                output_path = os.path.join(trees_dir, f"{base_name}_tree_{n_trees}.laz")
                crop_point_cloud_tilted_cylinder(las_data, points, center_x, center_y, center_z, 
                                        pca_x, pca_y, pca_z, 5, output_path)
                print(f"    Tree point cloud saved to: {output_path}")

    return n_trees

def process_point_cloud(input_file, output_dir, tile_size=10, overlap=0.5, save_crops=True, save_center_file=None):
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    scan_dir = os.path.join(output_dir, base_name)
    trees_dir = os.path.join(scan_dir, "trees")
    
    if save_crops:
        # Create output directories    
        # Check if this file has already been processed
        if os.path.exists(trees_dir) and len(os.listdir(trees_dir)) > 0:
            print(f"Skipping {base_name} as it has already been processed.")
            return len(os.listdir(trees_dir))
        
        os.makedirs(trees_dir, exist_ok=True)

    if save_center_file:
        save_center_file = open(save_center_file, "w")  
    
    print(f"\nProcessing file: {base_name}")

    print("Step 1: Reading point cloud and creating grid")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Get point cloud bounds
    bounds = (
        np.min(points[:, 0]), np.max(points[:, 0]),
        np.min(points[:, 1]), np.max(points[:, 1])
    )

    print(f"Point cloud bounds: {bounds}")
    
    # Create grid of tiles
    tiles = create_grid(bounds, tile_size, overlap)
    print(f"Created {len(tiles)} tiles for processing")
    
    # Process each tile
    n_trees = 0
    for i, tile_bounds in enumerate(tqdm(tiles, desc="Processing tiles")):
        n_trees = process_tile(las, points, tile_bounds, trees_dir, base_name, n_trees, save_crops=save_crops, save_center_file=save_center_file)
    
    # Process unique trees
    print(f"Finished processing {base_name}. {n_trees} trees extracted and saved.")
    return n_trees

def main():
    input_dir = 'data/'  # Replace with your input directory
    output_dir = input_dir
    
    # Get all .laz files in the input directory
    laz_files = glob.glob(os.path.join(input_dir, '*.laz'))
    
    print(f"Found {len(laz_files)} .laz files to process")
    
    # Process each file with progress bar
    for laz_file in tqdm(laz_files, desc="Processing point clouds"):
        process_point_cloud(laz_file, output_dir, save_crops=False, save_center_file="tree_centers.csv")

    print("All point clouds processed successfully!")

if __name__ == "__main__":
    main()
