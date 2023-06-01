import sys
with open(sys.argv[2], "w") as output_file:
    for i, j in zip(list(range(1,int(sys.argv[1].split()[0]),1)), list(range(int(sys.argv[1].split()[0]),-1, -1))):
        output_file.write(str(i) + " " + str(j) + "\n")