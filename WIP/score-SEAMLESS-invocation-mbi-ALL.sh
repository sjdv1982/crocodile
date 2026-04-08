slice=10

max=227
for i in $(seq 1 $slice $max); do
  ii=$((i+slice-1))
  if (( ii > max )); then
    ii=$max
  fi
  echo $i $ii
  bash score-SEAMLESS-invocation-mbi.sh bwd $i $ii
done

max=201
for i in $(seq 1 $slice $max); do
  ii=$((i+slice-1))
  if (( ii > max )); then
    ii=$max
  fi
  echo $i $ii
  bash score-SEAMLESS-invocation-mbi.sh fwd $i $ii
done
