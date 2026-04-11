slice=10

min=491
max=600
for i in $(seq $min $slice $max); do
  ii=$((i+slice-1))
  if (( ii > max )); then
    ii=$max
  fi
  echo $i $ii
  bash score-SEAMLESS-invocation-mbi.sh fwd $i $ii
  sleep 1
done
exit

min=861
max=953
for i in $(seq $min $slice $max); do
  ii=$((i+slice-1))
  if (( ii > max )); then
    ii=$max
  fi
  echo $i $ii
  bash score-SEAMLESS-invocation-mbi.sh bwd $i $ii
  sleep 1
done
