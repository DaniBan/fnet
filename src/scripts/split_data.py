import os.path
import pathlib
import pandas as pd
import shutil

PATH_TO_DATA = "../../data/full"
DEST_PATH_TRAIN = "../../data/train"
DEST_PATH_TEST = "../../data/test"

# split images
paths = list(pathlib.Path(PATH_TO_DATA).glob("images/*.jpg"))
train_split = 0.7
train_size = int(0.7 * len(paths))

train_paths = paths[:train_size]
test_paths = paths[train_size:]

print(f"{len(train_paths)}  {len(test_paths)}")

# split labels
df = pd.read_csv(os.path.join(PATH_TO_DATA, "labels", "labels.csv"))
train_labels_df = df.loc[:train_size-1]
test_labels_df = df.loc[train_size:]

print(train_labels_df.shape[0])
print(test_labels_df.shape[0])

# copy train data to train location
for image_path in train_paths:
    shutil.copy(image_path, os.path.join(DEST_PATH_TRAIN, "images"))
train_labels_df.to_csv(os.path.join(DEST_PATH_TRAIN, 'labels/labels.csv'), index=False)

# copy test data to test location
for image_path in test_paths:
    shutil.copy(image_path, os.path.join(DEST_PATH_TEST, "images"))
test_labels_df.to_csv(os.path.join(DEST_PATH_TEST, 'labels/labels.csv'), index=False)
