sbatch <your .sh files>

sstat --jobs=451567 --format=jobid,cputime,maxrss,ntasks

salloc -w n003 -p gpu

salloc --nodes=1 --ntasks=4 --mem=64G --time=4:00:00 --partition=work