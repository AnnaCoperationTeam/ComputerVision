import os
import time

path = "./img/img_ex/"

os.chdir(path)
i = 1
for roots, dirs , files in os.walk(os.getcwd()):
	os.chdir(roots)
	for _f in files:
		print(_f + "...Deleted...")

		if _f.endswith(".jpg"):
			os.remove(_f)
			i += 1
print("...Completed...")