import mne
import os

# Get the base data path
mne_data_path = mne.get_config('MNE_DATA')
if not mne_data_path:
    mne_data_path = os.path.join(os.path.expanduser("~"), "mne_data")

print(f"MNE is looking in: {mne_data_path}")

# Check specifically for Liu zip files
print("Checking for zip files in that directory...")
for root, dirs, files in os.walk(mne_data_path):
    for file in files:
        if "liu" in file.lower() and file.endswith(".zip"):
            print(f"FOUND ZIP: {os.path.join(root, file)}")