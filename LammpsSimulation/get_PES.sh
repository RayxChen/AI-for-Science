sed -n '127, 50127p' log.lammps | awk -F ' ' '{print $3}' > PES_50K.txt
