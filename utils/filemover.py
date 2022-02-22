import shutil
import sys
from os import listdir
from os.path import join, isfile, exists
def copy_files_by_names(part_name, dest_name, file_type):
    basepath = "C:/Users/kiani/Documents/Studium/Bachelorarbeit Daten/Parkett, new ROI, 5 spots/"
    destpath = "C:/Users/kiani/gait-recognition/data/"
    files = [f for f in listdir(basepath) if isfile(join(basepath, f)) and part_name in f and f.endswith(file_type)]
    for file in files:
        filename = file.replace(file_type, '')
        print(filename)
        number = filename.replace(part_name, '')
        print(number)
        destfile = dest_name + '-' + number + file_type
        if not exists(join(destpath, destfile)):
            shutil.copy(join(basepath, file), join(destpath, destfile))
            print("copied file.")
        else:
            print("file already exists: ", join(destpath, destfile), "! Please delete it manually")
        
    print(files)
if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) <= 2:
        raise ValueError("please provide 2 command line arguments. arg1 must be the participant name, arg2 the destination name ('munichX')")
    elif not (isinstance(sys.argv[1], str) and isinstance(sys.argv[2], str)):
        raise TypeError("the arguments must be of type String!")
    else:
        copy_files_by_names(sys.argv[1], sys.argv[2], ".dat")
        copy_files_by_names(sys.argv[1], sys.argv[2], ".ravi")
            