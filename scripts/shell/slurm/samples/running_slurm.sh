sbatch <your .sh files>

<<<<<<< HEAD
sstat --jobs=451567 --format=jobid,cputime,maxrss,ntasks

salloc -w n003 -p gpu

salloc --nodes=1 --ntasks=4 --mem=64G --time=4:00:00 --partition=work
=======
sstat --jobs=451567 --format=jobid,cputime,maxrss,ntasks
>>>>>>> afa18c04fdf209d6b14f1c2be166149dd6dfc6a7
