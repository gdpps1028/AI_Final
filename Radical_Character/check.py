import csv

def compare_csv_columns(file_path, col1_name, col2_name):
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if col1_name not in reader.fieldnames or col2_name not in reader.fieldnames:
                print(f"Error: Column '{col1_name}' or '{col2_name}' not found in the CSV file.")
                return -1

            same_count = 0
            count = 0
            for row in reader:
                if row[col1_name] == row[col2_name]:
                    same_count += 1
                count += 1
            return count, same_count
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1

csv_file = 'CNN.csv'
column1 = 'id'
column2 = 'char'

count, same_count = compare_csv_columns(csv_file, column1, column2)

if count != -1:
    print(f"Total entries\t: {count}, \nCorrect entries\t: {same_count},\nAccuracy\t: {same_count/count*100}%")
