#python script to scrapt logs and find mismatches between #train embeddings used, and #of actual train samples
import os
import re
log_dir = './logs/lejepa_full_benchmark/'

for log_file in os.listdir(log_dir):
    if not log_file.endswith('.log'):
        continue
    with open(os.path.join(log_dir, log_file), 'r') as f:
        log_content = f.read()
        if "Extracting embeddings" not in log_content:
            continue
        #pattern is like "Using 13746 train and 2426 val embeddings"
        match = re.search(r'Using (\d+) train and (\d+) val embeddings', log_content)
        if match:
            num_train_embeddings = int(match.group(1))
            num_val_embeddings = int(match.group(2))
        else:
            print(f"No embedding info found in {log_file}")
            continue
        total_train_samples = num_train_embeddings + num_val_embeddings
        # print(f"{log_file}: Train Embeddings = {num_train_embeddings}, Val Embeddings = {num_val_embeddings}, Total Samples = {total_samples}")

        #now find actual number of train samples from dataset
        """
        Pattern is: 
        ======================== Task Overview ========================
                   Task: Five Fingers MI                    
 Dataset  Train Samples  Test Samples       Train Class Distribution   Test Class Distribution
Kaya2018          16172          1899 [2965, 3190, 3324, 3201, 3492] [343, 371, 392, 377, 416]
        
        """
        match_samples = re.search(r'Task Overview ========================\s+Task: .+\s+Dataset\s+Train Samples\s+Test Samples\s+Train Class Distribution\s+Test Class Distribution\s+.+\s+(\d+)\s+(\d+)\s+\[.*\]\s+\[.*\]', log_content, re.DOTALL)
        if match_samples:
            actual_train_samples = int(match_samples.group(1))
            actual_test_samples = int(match_samples.group(2))
        else:
            # print(f"No sample info found in {log_file}")
            continue

        if total_train_samples < actual_train_samples:
            print(f"Mismatch in {log_file}: Train Embeddings = {total_train_samples}, Actual Train Samples = {actual_train_samples}")

        if total_train_samples > actual_train_samples:
            print(f"Mismatch in {log_file}: Train Embeddings = {total_train_samples}, Actual Train Samples = {actual_train_samples}")