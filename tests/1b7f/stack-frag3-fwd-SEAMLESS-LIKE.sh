# This version is for the MBI cluster (using ramscratch to store the results)

# It uses Seamless to launch the job, but don't count on it to provide caching for you.

set -u -e

deploymentdir=$1
output=1b7f-frag3-fwd
seamless-run -vvv -y --conda alaric --dry --write-remote-job "$deploymentdir" \
--metavar outdir=/ramscratch/$output -I code/stack.py.DEPS.txt \
--prologue "rm -rf \$outdir" \
""" python3 -u code/stack.py --sequence UG --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --second \
  --angle 24 --dihedral -45 45 \
  --output \$outdir/files \
  && seamless-checksum-index /ramscratch/$output/files && \
  mkdir /ramscratch/$output/bufferdir
  seamless-upload /ramscratch/$output/files --hardlink --destination /ramscratch/$output/bufferdir
  cat /ramscratch/$output/files.INDEX
"""
