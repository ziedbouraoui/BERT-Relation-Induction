import string
import os

def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.
 
Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.
 
"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename


def init_logging_path(task_name,file_name):
    dir_log  = os.path.join(os.getcwd(),f"log/{task_name}/{file_name}/")  
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        #os.makedirs(dir_log)
        with open(dir_log , 'w'):
             os.utime(dir_log , None)   
    if not os.path.exists(dir_log):
        os.makedirs(dir_log )
        dir_log  += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log , 'w'):
             os.utime(dir_log , None)   
    return dir_log 