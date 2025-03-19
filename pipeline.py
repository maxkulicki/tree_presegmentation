import os
import glob
from tqdm import tqdm
import laspy
import numpy as np
import dendromatics as dm
from crop_cylinder import crop_point_cloud_cylinder
from crop_tilted import crop_point_cloud_tilted_cylinder
import zipfile
import shutil

def zip_and_clean_single(input_file, scan_dir):
    """Zip a single processed folder and clean up original files."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    folder_path = scan_dir
    zip_file_path = f"{folder_path}.zip"

    try:
        # Create a zip file for the folder
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.path.dirname(folder_path))
                    zipf.write(file_path, arcname)
        print(f"Zipped: {base_name}")

        # Remove the folder after successful zipping
        shutil.rmtree(folder_path)
        print(f"Removed processed folder: {folder_path}")

        # Remove the original .laz file
        if os.path.exists(input_file):
            os.remove(input_file)
            print(f"Removed original .laz file: {input_file}")
        
    except Exception as e:
        print(f"Error during zip and cleanup of '{base_name}': {e}")

def process_point_cloud(input_file, output_dir, initial_crop=True, crop_center=None):
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create output directories
    scan_dir = os.path.join(output_dir, base_name)
    trees_dir = os.path.join(scan_dir, "trees")
    
    # Check if this file has already been processed
    if os.path.exists(trees_dir) and len(os.listdir(trees_dir)) > 0:
        print(f"Skipping {base_name} as it has already been processed.")
        return len(os.listdir(trees_dir))
    
    print(f"\nProcessing file: {base_name}")
    os.makedirs(trees_dir, exist_ok=True)
    
    if initial_crop:
        print("Step 1: Cropping the point cloud")
        # Use crop_center if provided, otherwise default to (0,0)
        center_x = crop_center[0] if crop_center is not None else 0
        center_y = crop_center[1] if crop_center is not None else 0
        
        cropped_file = os.path.join(scan_dir, f"{base_name}_cropped.laz")
        crop_point_cloud_cylinder(input_file, center_x, center_y, 15.0, cropped_file)
        print(f"Cropped point cloud saved to: {cropped_file}")
        point_cloud_file = cropped_file
    else:
        point_cloud_file = input_file
    
    print("Step 2: Processing the point cloud")
    entr = laspy.read(point_cloud_file)
    coords = np.vstack((entr.x, entr.y, entr.z)).transpose()

    print("  - Cleaning ground points")
    clean_points = dm.clean_ground(coords)
    print("  - Generating DTM")
    dtm = dm.generate_dtm(clean_points)
    print("  - Normalizing heights")
    z0_values = dm.normalize_heights(coords, dtm)
    coords = np.append(coords, np.expand_dims(z0_values, axis=1), 1)

    print("  - Extracting stripe")
    lower_limit, upper_limit = 0.7, 3.5
    stripe = coords[(coords[:, 3] > lower_limit) & (coords[:, 3] < upper_limit), 0:4]
    print("  - Performing verticality clustering")
    clust_stripe = dm.verticality_clustering(stripe, n_iter=2)

    print("  - Individualizing trees")
    assigned_cloud, tree_vector, _ = dm.individualize_trees(
        coords, clust_stripe, 0.02, 0.02, lower_limit, upper_limit,
        0.7, 1.5, 1000, 15, 25, 0.3, 5, 0, 1, 2, tree_id_field=-1
    )
    print(f"  - {len(tree_vector)} potential trees identified")

    print("Step 3: Processing individual trees")
    print("  - Computing sections")

    n_trees = 0
    for i, tree in enumerate(tree_vector):
        n_trees += 1
        print(f"  - Processing tree {n_trees}")
        center_x, center_y, center_z = tree[4:7]
        pca_x, pca_y, pca_z = tree[1:4]
        output_path = os.path.join(trees_dir, f"{base_name}_tree_{n_trees}.laz")
        crop_point_cloud_tilted_cylinder(point_cloud_file, center_x, center_y, center_z, 
                                            pca_x, pca_y, pca_z, 5, output_path)
        print(f"    Tree point cloud saved to: {output_path}")

    print(f"Finished processing {base_name}. {n_trees} trees extracted and saved.")
    
    # Add zipping and cleaning step
    print("Step 4: Zipping results and cleaning up")
    zip_and_clean_single(input_file, scan_dir)
    
    return n_trees

def main():
    input_dir = 'input'  # Replace with your input directory
    output_dir = 'output'  # Replace with your output directory
    
    # Get all .laz files in the input directory
    laz_files = glob.glob(os.path.join(input_dir, '*.laz'))
    
    print(f"Found {len(laz_files)} .laz files to process")
    
    # Process each file with progress bar
    for laz_file in tqdm(laz_files, desc="Processing point clouds"):
        n_trees = process_point_cloud(laz_file, output_dir, initial_crop=True)
        tqdm.write(f"Processed {os.path.basename(laz_file)}: {n_trees} trees extracted")

    print("All point clouds processed successfully!")

if __name__ == "__main__":
    main()