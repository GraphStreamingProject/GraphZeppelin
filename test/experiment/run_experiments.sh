#! /bin/bash

# The specific test we want to run
executable=$1

# for every input file specified on the command line
for input in "${@:2}"
do
	echo ""
	echo "input file: $input"
	
	mkdir test_`basename $input`_data
	cp $input test/res/current_test.stream
	total=1
	for ((j=0; j < 5; j++))
	do
		for ((size=1; size <= total; size*=2))
		do
			groups=$((total/size))

			start=`date +%s`
			
			echo "path_prefix=./BUFFTREEDATA/" > streaming.conf
			echo "num_groups=$((groups))"      >> streaming.conf
			echo "group_size=$((size))"        >> streaming.conf
			
			"$executable"
			
			mv runtime_data.txt test_`basename $input`_data/g$((groups))_s$((size))_data.txt
			
			end=`date +%s`
		done
		total=$((total*2))
	done
done

# delete buffer tree data if it exists
rm BUFFTREEDATA/*


