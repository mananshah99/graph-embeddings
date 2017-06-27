import mmap
import os

def count_lines(file_path):  
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def push_notify(message):
    m_str = '"message=' + str(message)
    cmd = 'curl -s --form-string "token=aph11coe778qdkgm431sksmru3r38b" --form-string "user=ufsfrfpyq5oo633jetx71ky9tngabe" --form-string ' + m_str + '" https://api.pushover.net/1/messages.json > /dev/null'
    os.system(cmd)
