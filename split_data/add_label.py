import os
import jsonlines
import pandas as pd
import argparse
import tempfile
import shutil

def add_missing_labels(file_path, test_data_path):
    test_data = pd.read_csv(test_data_path)
    
    label_map = {}
    for _, row in test_data.iterrows():
        label_map[(row['drug_name'], row['disease_name'])] = row['relation']
    
    existing_data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            existing_data.append(line)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
    temp_file_path = temp_file.name
    temp_file.close()
    
    with jsonlines.open(temp_file_path, 'w') as writer:
        updated_count = 0
        for item in existing_data:
            key = (item['drug_name'], item['disease_name'])
            
            if 'label' not in item and key in label_map:
                item['label'] = label_map[key]
                updated_count += 1
            
            writer.write(item)
    
    shutil.move(temp_file_path, file_path)
    
    print(f"Updated {updated_count} entries with missing labels")
    print(f"Original file has been updated: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add missing labels to JSONL file")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONL file")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data CSV")
    
    args = parser.parse_args()
    
    add_missing_labels(args.file_path, args.test_data_path)


