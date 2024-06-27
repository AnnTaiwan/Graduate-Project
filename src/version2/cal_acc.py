'''
Used for calculating the accuracy of the result from DPU inference.
Need output.txt, csv file(recording the correct labels)
'''
import re
import pandas as pd

def parse_line(line):
    """Extracts the values from a line using regex."""
    # Regex pattern to capture numbers with optional exponent part
    match = re.search(r'0: ([\d.-]+(?:[eE][+\-]?\d+)?)\s*1: ([\d.-]+(?:[eE][+\-]?\d+)?)', line)
    
    # Extract the image name
    name_match = re.search(r'name:([^ ]+)', line)
    if name_match:
        image_name = name_match.group(1)
    else:
        return None, None, None  # Skip if no name match is found

    if match:
        try:
            value_0 = float(match.group(1))
            value_1 = float(match.group(2))
            return image_name, value_0, value_1
        except ValueError as e:
            print(f"ValueError: {e} for line: {line}")
            return None, None, None
    return None, None, None

def read_and_evaluate(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    for i in range(4, len(lines), 6):  # Start from line 4 and step by 6 to get 4, 10, 16, etc.
        if i >= len(lines):
            break  # Ensure there's a line to process

        # Parse the necessary line
        line = lines[i]
        name, raw_0, raw_1 = parse_line(line)
        if raw_0 is not None and raw_1 is not None:
            # Determine the prediction based on the values
            prediction = 0 if raw_0 > raw_1 else 1

            results.append({
                'Line Number': i + 1,
                'Image_Name': name,
                'Prediction': prediction,
                'Value 0': raw_0,
                'Value 1': raw_1
            })

    return results

if __name__ == "__main__":
    file_path = r"C:\Users\User\Desktop\JUNE27th\output_eval100_5.txt"  # Update with the correct path if necessary
    results = read_and_evaluate(file_path)
    
    df = pd.read_csv("eval_info.csv")  # Load the CSV file with labels
    count = 0
    real = 0
    fake = 0
    for result in results:
        image_id = result['Image_Name'][5:-4]  # Extract the image ID from the filename
        label = df[df["filename"] == image_id]["target"].values[0]  # Fetch the label from CSV
        if label:
            fake += 1
        else:
            real += 1
        if label == result['Prediction']:
            count += 1
        else:
            print(f"Mismatch for image {image_id}: Label {label}, Prediction {result['Prediction']}")
            print(f"Value 0: {result['Value 0']}, Value 1: {result['Value 1']}")
    
    if len(results) > 0:
        accuracy = count / len(results)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    else:
        print("No results to evaluate.")

    print(f"Real: {real}, Fake: {fake}")
