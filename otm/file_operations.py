import os
import zipfile
import shutil


def zip_to_ims(zip_name, target_dir_name, temp_dir_name="temp_unzip_dir"):
    """
    moves the labels and images to target_dir
    assumes zip has one folder with images and labels folders
    assumes target_dir has images and labels folders
    """
    cwd = os.getcwd()
    abs_temp_dir_path = os.path.join(cwd, temp_dir_name)
    abs_zip_path = os.path.join(cwd, zip_name)
    abs_target_dir_path = os.path.join(cwd, target_dir_name)

    # unzip into temp dir
    unzipper = Unzipper(abs_zip_path, abs_temp_dir_path)
    unzipper.unzip()

    temp_to_ims(abs_temp_dir_path, abs_target_dir_path)
    os.remove(abs_zip_path)


def temp_to_ims(temp_path, target_folder_path):
    assert len(os.listdir(temp_path)) == 1, "there must be one folder inside zip"

    inside_zip = os.path.join(temp_path, os.listdir(temp_path)[0])
    assert set(os.listdir(inside_zip)) == {"images", "labels"}, \
        "the folder inside zip must contain labels and images folders"

    # copy to incoming folder
    for file in os.listdir(inside_zip + "/images"):
        img_path = os.path.join(inside_zip + "/images/" + file)
        label_path = os.path.join(inside_zip + "/labels/" + file.replace("jpg", "txt"))
        shutil.move(img_path, target_folder_path + '/images')
        shutil.move(label_path, target_folder_path + '/labels')

    # rm temp folder
    shutil.rmtree(temp_path, ignore_errors=True)  # rm


class Unzipper:
    """Decompresses a zip file"""
    def __init__(self, zip_file, output_folder):
        self.zip_file = zip_file
        self.output_folder = output_folder

    def unzip(self):
        if not zipfile.is_zipfile(self.zip_file):
            raise TypeError(f"{self.zip_file} is not a valid ZIP file.")

        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            try:
                zip_ref.extractall(self.output_folder)
                print(f"Model file is extracted to: {self.output_folder}")
            except Exception as e:
                print(f"Error occurred {e}")


if __name__ == "__main__":
    zip_to_ims("received.zip", "incoming_folder", temp_dir_name="temp_unzip_dir")