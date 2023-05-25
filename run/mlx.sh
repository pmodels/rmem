#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --time=0:10:00
#SBATCH --ntasks=4
#SBATCH --nodes=2
#SBATCH --account=p200067
#SBATCH --qos=short

echo "loading modules"
#-------------------------------------------------------------------------------
module load env/staging/2022.1
module load GCC
module load UCX
module load Automake Autoconf
module load libtool
module load Perl
module list

#-------------------------------------------------------------------------------
# get a unique tag
TAG=`date '+%Y-%m-%d-%H%M'`-`uuidgen -t | head -c 4`
# get the dir list
DBS_DIR=${HOME}/lib-OFI-1.18.0
HOME_DIR=${HOME}/rmem
SCRATCH_DIR=/project/scratch/p200067/tgillis/benchme_${TAG}_${SLURM_JOBID}

echo "--------------------------------------------------"
echo "running in ${SCRATCH_DIR}"
echo "--------------------------------------------------"

#-------------------------------------------------------------------------------
mkdir -p ${SCRATCH_DIR}
cd ${SCRATCH_DIR}
cp -r ${HOME_DIR}/m4 .
cp -r ${HOME_DIR}/src .
cp -r ${HOME_DIR}/Makefile.am .
cp -r ${HOME_DIR}/configure.ac .
cp -r ${HOME_DIR}/autogen.sh .

#-------------------------------------------------------------------------------
./autogen.sh
CC=gcc CXX=g++ ./configure --enable-perf \
    --with-ofi=${DBS_DIR} \
    --with-pmi=${DBS_DIR} 

make clean
make -j 8

#-------------------------------------------------------------------------------
FI_PROVIDER="psm3" ${DBS_DIR}/bin/mpiexec -ppn 1 -l --bind-to core \
    -n 1 perf record --call-graph dwarf -o perf.0 ./rmem : \
    -n 1 perf record --call-graph dwarf -o perf.1 ./rmem


for id in 0 1
do
    perf script -i perf.${id} > out.${id}.perf
    ${HOME}/FlameGraph/stackcollapse-perf.pl out.${id}.perf > out.${id}.folded
    ${HOME}/FlameGraph/flamegraph.pl out.${id}.folded > ${id}.svg
done

