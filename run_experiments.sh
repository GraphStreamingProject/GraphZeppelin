#! /bin/bash
iterations=$1

start1=`date +%s`
for ((i=0; i < $iterations; i++))
do
	echo "experiment $((i+1))"
	./experiment
done
end1=`date +%s`
echo "standard $((end1-start1)) seconds"

start2=`date +%s`
for ((i=0; i < $iterations; i++))
do
	echo "Toku_experiment $((i+1))"
	./toku_experiment
done
end2=`date +%s`
echo "toku $((end2-start2)) seconds"

thr=1
for ((j=0; j < 5; j++))
do
	start=`date +%s`
	for ((i=0; i < $iterations; i++))
	do
		echo "Toku_Multi_experiment $((i+1)) threads=$((thr))"
		echo "threads=$((thr))" > graph_worker.conf
		./toku_multi_experiment
	done
	end=`date +%s`
	echo "toku with $((thr)) threads $((end-start))"
	thr=$((thr*2))
done