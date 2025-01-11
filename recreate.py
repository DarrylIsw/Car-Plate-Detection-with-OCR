import os
import shutil

def copy_n_files(src_dir, dest_dir, n):
    """
    Copies the first `n` files from the images and labels subfolders 
    in the `src_dir` to the corresponding structure in `dest_dir`.

    Args:
        src_dir (str): Source directory containing images and labels subfolders.
        dest_dir (str): Destination directory to copy the files to.
        n (int): Number of files to copy.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    for subfolder in ['images', 'labels']:
        src_subfolder = os.path.join(src_dir, subfolder)
        dest_subfolder = os.path.join(dest_dir, subfolder)

        # Ensure the source subfolder exists
        if not os.path.exists(src_subfolder):
            print(f"Source subfolder '{src_subfolder}' does not exist. Skipping...")
            continue

        # Create the destination subfolder
        os.makedirs(dest_subfolder, exist_ok=True)

        # Get a sorted list of files in the source subfolder
        files = sorted(os.listdir(src_subfolder))

        # Copy the first `n` files
        for file in files[:n]:
            src_file = os.path.join(src_subfolder, file)
            dest_file = os.path.join(dest_subfolder, file)
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")


def main():
    # Directories
    source_dir = "yolov5\data\plates-1"  # Replace with the path to your main source folder
    destination_dir = "yolov5\data\plates-3"  # Replace with the path to your main destination folder
    n = 450  # Number of files to copy from each subfolder

    # Subdirectories to process
    for split in ['train', 'valid', 'test']:
        src_split_dir = os.path.join(source_dir, split)
        dest_split_dir = os.path.join(destination_dir, split)

        # Copy images and labels for the current split
        copy_n_files(src_split_dir, dest_split_dir, n)

if __name__ == "__main__":
    main()
