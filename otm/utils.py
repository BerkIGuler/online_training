import os
import shutil
import random
import itertools
def folder_cleaner(folder_path):
    # loop over all the files in the folder and delete them
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {file_path} - {e}")


def folder_cleaner_coco(folder_path):
    ### to clean a folder with coco dataset type
    folder_cleaner(folder_path)
    folder_cleaner(folder_path+"/images")
    folder_cleaner(folder_path+"/labels")

def file_mover_coco(source_dir, dest_dir):
    files_images = os.listdir(source_dir+"/images")
    files_labels = os.listdir(source_dir+"/labels")

    # Move each file to the destination directory
    for f in files_images:
        src_path = os.path.join(source_dir+"/images", f)
        dest_path = os.path.join(dest_dir+"/images")
        shutil.move(src_path, dest_path)
        #print(src_path)
        #print(dest_path)
    for f in files_labels:
        src_path = os.path.join(source_dir+"/labels", f)
        dest_path = os.path.join(dest_dir+"/labels")
        shutil.move(src_path, dest_path)

def file_sampler_coco(source_dir, dest_dir, file_nb):
    all_images = os.listdir(source_dir+"/images")
    all_labels = os.listdir(source_dir+"/labels")
    random_array = [random.randint(0, len(all_labels)) for i in range(file_nb)]

    #sampled_images = all_images[random_array]
    #sampled_labels= all_labels[random_array]
    sampled_images = list(itertools.compress(all_images, random_array))
    sampled_labels = list(itertools.compress(all_labels, random_array))

    for f in sampled_images:
        src_path = os.path.join(source_dir+"/images", f)
        dest_path = os.path.join(dest_dir+"/images")
        shutil.copy(src_path, dest_path)
    for f in sampled_labels:
        src_path = os.path.join(source_dir+"/labels", f)
        dest_path = os.path.join(dest_dir+"/labels")
        shutil.copy(src_path, dest_path)

def file_copier_coco(source_dir, dest_dir):
    files_images = os.listdir(source_dir+"/images")
    files_labels = os.listdir(source_dir+"/labels")

    # Move each file to the destination directory
    for f in files_images:
        src_path = os.path.join(source_dir+"/images", f)
        dest_path = os.path.join(dest_dir+"/images")
        shutil.copy(src_path, dest_path)
        #print(src_path)
        #print(dest_path)
    for f in files_labels:
        src_path = os.path.join(source_dir+"/labels", f)
        dest_path = os.path.join(dest_dir+"/labels")
        shutil.copy(src_path, dest_path)


#file_copier_coco("C:/Users/efeml/OneDrive/Desktop/deneme1" , "C:/Users/efeml/OneDrive/Desktop/deneme2")
#folder_cleaner("C:/Users/efeml/OneDrive/Desktop/deneme2/images")
#folder_cleaner("C:/Users/efeml/OneDrive/Desktop/deneme2/labels")
#folder_cleaner("C:/Users/efeml/OneDrive/Desktop/deneme2")
#file_sampler_coco("C:/Users/efeml/OneDrive/Desktop/deneme1" , "C:/Users/efeml/OneDrive/Desktop/deneme2",3)