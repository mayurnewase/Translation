"""
Load/Download data

input -> filename
output -> [all lines]
"""

import os
import subprocess

def LoadData(filename):
	check = os.path.isfile(filename)
	if(not check):
		process = os.popen("wget http://www.manythings.org/anki/fra-eng.zip").read()
		process = os.popen("unzip fra-eng.zip").read()
		filename = "fra.txt"

	file = open(filename, mode = "rt", encoding = "utf-8")
	data = file.read()
	file.close()

	lines = data.split("\n")
	#data = [s.split("\t") for s in lines]
	return lines
