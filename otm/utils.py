import os
import shutil
import random
import itertools
import logging

logger = logging.getLogger(__name__)


def folder_cleaner(folder_path):
    # helper function to clean a folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting file: {file_path} - {e}")


def folder_cleaner_coco(folder_path):
    # cleans a folder with coco dataset type
    folder_cleaner(folder_path)
    folder_cleaner(folder_path+"/images")
    folder_cleaner(folder_path+"/labels")


def file_mover_coco(source_dir, dest_dir):
    # moves all files from a coco style dataset folder to another directory
    files_images = os.listdir(source_dir + "/images")
    files_labels = os.listdir(source_dir + "/labels")

    for f in files_images:
        src_path = os.path.join(source_dir + "/images", f)
        dest_path = os.path.join(dest_dir + "/images")
        try:
            shutil.move(src_path, dest_path)
        except shutil.Error:
            continue

    for f in files_labels:
        src_path = os.path.join(source_dir + "/labels", f)
        dest_path = os.path.join(dest_dir + "/labels")
        try:
            shutil.move(src_path, dest_path)
        except shutil.Error:
            continue


def file_sampler_coco(source_dir, dest_dir, file_nb):
    # samples n images from a coco style dataset folder, copies to another directory
    all_images = os.listdir(source_dir+"/images")
    all_labels = os.listdir(source_dir+"/labels")
    random_array = [random.randint(0, len(all_labels)) for _ in range(file_nb)]

    sampled_images = list(itertools.compress(all_images, random_array))
    sampled_labels = [filename.replace(".jpg", ".txt").replace(".jpeg", ".txt") for filename in sampled_images]

    for f in sampled_images:
        src_path = os.path.join(source_dir+"/images", f)
        dest_path = os.path.join(dest_dir+"/images")
        shutil.copy(src_path, dest_path)
    for f in sampled_labels:
        src_path = os.path.join(source_dir+"/labels", f)
        dest_path = os.path.join(dest_dir+"/labels")
        shutil.copy(src_path, dest_path)


def file_copier_coco(source_dir, dest_dir):
    # copies labels and images from a coco style dataset folder to another folder
    files_images = os.listdir(source_dir+"/images")
    files_labels = os.listdir(source_dir+"/labels")

    # Move each file to the destination directory
    for f in files_images:
        src_path = os.path.join(source_dir+"/images", f)
        dest_path = os.path.join(dest_dir+"/images")
        shutil.copy(src_path, dest_path)
    for f in files_labels:
        src_path = os.path.join(source_dir+"/labels", f)
        dest_path = os.path.join(dest_dir+"/labels")
        shutil.copy(src_path, dest_path)
