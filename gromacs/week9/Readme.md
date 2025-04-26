RCSB 
gmx

gmx pdb2gmx -f 3I40_nowater.pdb -o 3I40.gro -water spce -ignh
1

gmx editconf -f 3I40.gro -o 3I40_box.gro -c -d 1.0 -bt dodecahedron

gmx solvate -cp 3I40_box.gro -cs spc216.gro -p topol.top -o 3I40_box_solvate.gro

gmx grompp -f ions.mdp -c 3I40_box_solvate.gro -p topol.top -o ions.tpr

gmx genion -s ions.tpr -o 3I40_box_solvate_ion.gro -p topol.top -pname NA -nname CL -neutral
13

gmx grompp -f minim.mdp -c 3I40_box_solvate_ion.gro -p topol.top -o em.tpr

gmx mdrun -deffnm em

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr

<!-- gmx mdrun -v -deffnm nvt -ntomp 4 -ntmpi 1 -nb gpu –update gpu -->
gmx mdrun -v -deffnm nvt -ntomp 4 -ntmpi 1 -nb gpu

gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -r nvt.gro -p topol.top -o npt.tpr

gmx mdrun -v -deffnm npt -ntomp 4 -ntmpi 1 -nb gpu

gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_10.tpr

gmx mdrun -v -deffnm md_0_2 -ntomp 4 -ntmpi 1 -nb gpu -s md_0_10.tpr

gmx trjconv -s md_0_10.tpr -f md_0_2.xtc -center -pbc mol -ur compact -o md_0_2_center.xtc
1
0

gmx trjconv -s md_0_10.tpr -f md_0_2_center.xtc -fit rot+trans -o md_0_2_center_fit.xtc
4
0

gmx trjconv -s md_0_10.tpr -f md_0_2_center.xtc -dump 0 -o md_0_2_center_start.pdb
0

先拉進md_0_2_center_start.pdb
再拉進md_0_2_center_fit.xtc
然後在下面打兩次smooth，並且remove掉離子

gmx rms -s md_0_10.tpr -f md_0_2_center_fit.xtc -o rmsd.xvg -tu ns
4
4