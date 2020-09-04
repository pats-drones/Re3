import pandas as pd
import numpy as np
import xml.etree.cElementTree as ET
from tqdm import tqdm
import os, glob, cv2, re
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.15  # because we have so many videos we just go for a 90/10 split which can be recommended for deep learning datasets
doimages = True   # also convert images

output_path = "/home/pats/Documents/datasets/completed_dataset"
annotations_dir ="/home/pats/Documents/annotation/OpenLabeling/main/output/PASCAL_VOC"
moth_path = "/home/pats/Documents/datasets/contains_moth.csv"

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def get_xml_object_data(obj):
    class_name = obj.find('name').text
    class_index = 1
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.find('xmin').text))-int(1696/2)
    xmax = int(float(bndbox.find('xmax').text))-int(1696/2)
    ymin = int(float(bndbox.find('ymin').text))
    ymax = int(float(bndbox.find('ymax').text))
    return [class_name, class_index, xmin, ymin, xmax, ymax]

def best_matching_box(bboxes, last_bbox):
    return bboxes[np.argmin([np.linalg.norm(np.array(bbox[-4:])-np.array(last_bbox[:4])) for bbox in bboxes])]

df = pd.read_csv(moth_path, sep=",", names=["path", "contains_moth", "contains_drone", "autolabel"])
df_contain_moth = df[(df["contains_moth"] == 1)]
paths = df_contain_moth["path"].tolist()
train_paths, test_paths = train_test_split(paths, test_size=TEST_SIZE, random_state=42)

def get_grower_name(vid_dir: str):
    return vid_dir.split(os.sep)[5]

def get_img_num(im_path):
    return int(img_path.split("_")[-1].split(".")[0])


for paths_, mode in zip((train_paths, test_paths), ("train", "val")):
    print("\n")
    print(f"Converting images and labels for {mode} set.\n")
    open(os.path.join(output_path, mode, 'image_names.txt'), 'w').close() #clear content in file
    bboxes = []
    vid_id = -1
    frame_id = 0

    img_num_1 = -9001
    for path in tqdm(paths_):
    # for path in paths_:
        
        vid_dir = "/".join(path.split("/")[:8])

        #find out what the path till image is
        grower_name = get_grower_name(vid_dir)
        if grower_name != "batist" and grower_name != "holstein": #new video formats
            if grower_name == "koppert":
                images = natural_sort(glob.glob(os.path.join(vid_dir, "*/cut_video_mkv/*jpg")))
            else:
                images = natural_sort(glob.glob(os.path.join(vid_dir, "cut_video_mkv/*jpg")))
        else: #old video formats
            if grower_name == "holstein":
                ext = "_mkv"
            else:
                ext = "_mp4"
            img_dir = path.split("/")[-1].replace("log", "insect").replace(".csv", ext)
            images = natural_sort(glob.glob(os.path.join(vid_dir, img_dir, "*jpg")))
        
        for img_path in images:
            img_splits = img_path.split("/")
        
            # annotation file
            annotation_name = f"g{img_splits[5]}_d{img_splits[6]}_{img_splits[-1].split('.')[0]}.xml"
            annotation_path = os.path.join(annotations_dir, annotation_name)
            tree = ET.parse(annotation_path)
            annotation = tree.getroot()
            all_labels = [get_xml_object_data(obj)for obj in annotation.findall('object')]
            
            if len(all_labels) >= 1:  ## only write when annotation is found

                img_num_2 = get_img_num(img_path)
                if img_num_1 != img_num_2 -1: #give new vid id to frame skips
                    vid_id += 1
                img_num_1 = img_num_2

                if len(all_labels) > 1:
                    class_name, class_index, xmin, ymin, xmax, ymax = best_matching_box(all_labels, bbox)
                else:
                    class_name, class_index, xmin, ymin, xmax, ymax = all_labels[0]
            
      
                bbox = [xmin, ymin, xmax, ymax, vid_id, 1, frame_id, class_index ,0] #TODO: the 1 is the track_id needs to be updated to be able to handle multitracking
                bboxes.append(bbox)

                #modify image files
                if doimages:
                    img = cv2.imread(img_path)
                    split_img = img[:,:int(1696/2),:]
                    new_name = f"{img_splits[5]}_{img_splits[6]}_{img_splits[-1]}"
                    cv2.imwrite(os.path.join(output_path, mode, new_name), split_img)
                with open(os.path.join(output_path, mode, 'image_names.txt'), 'a') as w:
                    w.write(os.path.join(output_path, mode, new_name) + "\n")
            
                frame_id += 1
   
    
    bboxes = np.array(bboxes)
    if mode == "train":
        set_size = bboxes.shape[0]
    print(f"Dataset contains {bboxes.shape[0]} frames")
    np.save(os.path.join(output_path, mode, "labels.npy"), bboxes)

print(f"Total frames annotated: {set_size + bboxes.shape[0]}")