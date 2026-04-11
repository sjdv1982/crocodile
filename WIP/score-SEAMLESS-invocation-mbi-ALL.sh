slice=10

mode=bwd
min=601
max=667
#mode=fwd
#min=251
#max=588
for i in $(seq $min $slice $max); do
  ii=$((i+slice-1))
  if (( ii > max )); then
    ii=$max
  fi
  echo $mode $i $ii
  bash score-SEAMLESS-invocation-mbi.sh $mode $i $ii
  sleep 0.5
done
