#! /bin/bash
iterations=$1

total=1
for ((j=0; j < 5; j++))
do
	for ((size=1; size <= total; size*=2))
	do
		groups=$((total/size))

		start=`date +%s`
		for ((i=0; i < $iterations; i++))
		do
			echo "experiment $((i+1)) groups=$((groups)) of size=$((size))"
			echo "num_groups=$((groups))" > graph_worker.conf
			echo "group_size=$((size))" >> graph_worker.conf
			./experiment > /dev/null
		done
		end=`date +%s`
		echo "Number groups $((groups)), with $((size)) workers each: $((end-start)) seconds"
	done
	total=$((total*2))
done
