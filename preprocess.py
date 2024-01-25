import numpy as np
import json
from PIL import Image
import os
import shutil


# 
# what is the file in data tree like? e.g.
# data
# -- cutting
# ---- ims
# ------ 0 # inside which pngs name like 00000x.png
# ------ 1 
# ---- depth 
# ------ 0
# ------ 1 
# ---- gt_masks # the mask for depth where the masked area is invalid depth
# ------ 0 # inside which pngs name like 00000x.png
# ------ 1 
# ---- masks # the mask for rgb where the masked area is the tool to be removed
# ------ 0 
# ------ 1 
# ---- init_pt_cld.npz the initialized point cloud 
# ---- train_meta.json 


# 1. read pose_bounds.npy

def load_pose_bounds(path):
    poses_arr = np.load(path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    poses = poses.transpose((2,0,1))
    pose =poses[0]
    c2w = np.vstack([pose[:, :4], np.array([[0,0,0,1]])])
    w2c = np.linalg.inv(c2w)
    h, w, f = int(pose[0, 4]), int(pose[1, 4]), pose[2, 4]
    K = np.array([[f, 0, (w-1)*0.5], [0, f, (h-1)*0.5], [0, 0, 1]])

    return h,w,f,K,w2c

# 2. rename file name to correct format

def endonerf_to_dygs_format(endonerf_path, frame_nums):
    # write in endonerf_to_dygs_format method, 
    # create two folders with name 0 and 1 in subfolders of the data/pulling and copy 
    # the images inside 0 and 1 and rename them by removing all the characters leaving only number
    
    # Create folders 0 and 1 in subfolders of data/pulling
    # rename  folder_path = "data/pulling/images" to folder_path = "data/pulling/ims"
    # os.rename("data/pulling/images", "data/pulling/ims")

    for subdir in ["images", "depth", "gt_masks", "masks"]:
        subdir_path = os.path.join(endonerf_path, subdir)
        
        os.makedirs(os.path.join(subdir_path, "0"), exist_ok=True)
        os.makedirs(os.path.join(subdir_path, "1"), exist_ok=True)
        # get the number of png images in the images folder, 
        
        # Copy and rename images inside folders 0 and 1
        for i in range(frame_nums):
            if subdir == "gt_masks" or subdir == "masks":
                src_path = os.path.join(subdir_path, "frame-{:06d}.mask.png".format(i))
            elif subdir == "images":
                src_path = os.path.join(subdir_path, "frame-{:06d}.color.png".format(i))
            elif subdir == "depth":
                src_path = os.path.join(subdir_path, "frame-{:06d}.depth.png".format(i))
                
            dst_path_0 = os.path.join(subdir_path, "0", "{:06d}.png".format(i))
            dst_path_1 = os.path.join(subdir_path, "1", "{:06d}.png".format(i))
            shutil.copy(src_path, dst_path_0)
            shutil.copy(src_path, dst_path_1)

# 3. generate the json file

def gen_train_meta_json(h, w, f, K, w2c, json_path, frame_nums):

    ks = []
    w2cs = []
    fns = []
    cam_ids = []

    for i in range(frame_nums):
        ks.append([K.tolist(),K.tolist()])
        w2cs.append([w2c.tolist(),w2c.tolist()])
        fns.append(["0/{:06d}.png".format(i), "1/{:06d}.png".format(i)])
        cam_ids.append([0,1])

    data = {}
    data["h"] = h
    data["w"] = w
    data["k"] = ks
    data["w2c"] = w2cs
    data["fn"] = fns
    data["cam_id"] = cam_ids

    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)


    # create white image with the same size as the original image
    # im = Image.new("RGB", (w, h), (255, 255, 255))
    # # create 100 images in the folder data/cut/seg/0
    # for i in range(100):
    #     im.save("data/cut/seg/0/{:06d}.png".format(i))

# todo

# 4. gen initialized point cloud
def gen_init_pt_cld_npz():
    pass

# 5. run fgt to generate the masked part

# 6. use unimatch to generate stereo 


if __name__ == "__main__":

    task = "pulling"

    data_path = "data/" + task
    npy_path = data_path + "/poses_bounds.npy"
    json_path = data_path + "/train_meta.json"

    frame_nums = len([name for name in os.listdir(os.path.join(data_path, "images")) if name.endswith(".png")])
    print("frame_nums: ", frame_nums)

    h,w,f,K,w2c = load_pose_bounds(npy_path)
    
    gen_train_meta_json(h,w,f,K,w2c,json_path,frame_nums)
    
    endonerf_to_dygs_format(data_path, frame_nums)

    print([h,w,f,K,w2c])
    