"""
This module contains code to partition csv files.
Due to hardware constraint, only 50,000 records can be read into memory at once.

Process flow:
    1. read csv file, determine the number of rows
    2. create a folder in the directory
    3. partition table and save in the directory
"""