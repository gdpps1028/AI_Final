import os
import cnradical

radical = cnradical.Radical(cnradical.RunOption.Radical)

def append_radical(folder_path):
    try:
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            if os.path.isdir(item_path):
                if item_name:
                    first_char = item_name[0]
                    append_text = "_" + radical.trans_ch(first_char)
                    new_name = item_name + append_text
                    new_path = os.path.join(folder_path, new_name)

                    try:
                        os.rename(item_path, new_path)
                        print(f"Renamed folder from {item_path} to {new_path}")
                    except OSError as e:
                        print(f"Error renaming folder '{item_path}': {e}")

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    folder = input("Enter the path to the folder you want to process: ")
    append_radical(folder)
    print("\nRenaming of immediate subfolders complete.")