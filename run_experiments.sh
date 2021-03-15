#! /bin/bash
iterations=$1

start=`date +%s`
for ((i=0; i < $iterations; i++))
do
	echo "experiment $((i+1))"
	./experiment
done
end=`date +%s`
echo $((end-start))

start=`date +%s`
for ((i=0; i < $iterations; i++))
do
	echo "Toku_experiment $((i+1))"
	./toku_experiment
done
end=`date +%s`
echo $((end-start))