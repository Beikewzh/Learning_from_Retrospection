module load singularity/3.7.1

export SINGULARITY_CACHEDIR=/home/mila/p/pingsheng.li/scratch/.singularity_cache
export SINGULARITY_TMPDIR=/home/mila/p/pingsheng.li/scratch/.singularity_tmp
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

cd /home/mila/p/pingsheng.li/scratch/Learning_from_Retrospection

singularity shell --nv --cleanenv --bind /home/mila/p/pingsheng.li/scratch/Learning_from_Retrospection:/workspace --pwd /workspace easyr1.sif
