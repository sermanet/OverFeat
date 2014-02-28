import os
import os.path as path

def findOverfeatDir(file_dir):
    try:
        path_env = os.environ['OVERFEAT_NETDIR']
        if path_env != "":
            return path.abspath(path_env)
    except KeyError:
        pass
    if (file_dir[-12:] == "bin/linux_32") or (file_dir[-12:] == "bin/linux_64") or (file_dir[-9:] == "bin/macos"):
        return path.abspath(path.join(file_dir, "../.."))
    if (file_dir[-15:] == "bin/linux_32/cuda") or (file_dir[-15:] == "bin/linux_64/cuda") or (file_dir[-14:] == "bin/macos/cuda"):
        return path.abspath(path.join(file_dir, "../../.."))
    elif file_dir[-3:] == "src":
        return path.abspath(path.join(file_dir, ".."))
    else:
        idx = max(file_dir.rfind("OverFeat"), file_dir.rfind("overfeat"))
        if idx >= 0:
            return path.abspath(file_dir[:(idx+8)])
        else:
            return file_dir
    
