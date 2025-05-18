import os
import shutil

def organize_by_first_char(folder_path):
    try:
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath):
                if filename:
                    first_char = filename[0]
                    target_folder = os.path.join(folder_path, first_char)

                    os.makedirs(target_folder, exist_ok=True)

                    destination_path = os.path.join(target_folder, filename)

                    try:
                        shutil.move(filepath, destination_path)
                        print(f"Moved '{filename}' to folder '{first_char}'")
                    except Exception as e:
                        print(f"Error moving '{filename}': {e}")
                        
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    folder = input("Enter the path to the folder you want to organize: ")
    organize_by_first_char(folder)
    print("\nFile organization complete.")
