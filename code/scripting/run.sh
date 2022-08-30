echo Evaluating on kron_15 stream with 10^3 batch size... 

/home/ahmed/terrace_CC/terrace/terrace_CC /home/experiment_inputs/streams/kron_15_unique_half_stream.txt 1000 /home/ahmed/terrace_CC/logs/time_15_3.txt &
pid=$!
top -b -Em -p $pid > ../logs/mem_15_3.txt &
top_pid=$!
wait $pid
kill $top_pid

echo Evaluating on kron_15 stream with 10^6 batch size... 

/home/ahmed/terrace_CC/terrace/terrace_CC /home/experiment_inputs/streams/kron_15_unique_half_stream.txt 1000000 /home/ahmed/terrace_CC/logs/time_15_6.txt &
pid=$!
top -b -Em -p $pid > ../logs/mem_15_6.txt &
top_pid=$!
wait $pid
kill $top_pid

echo Computing aspen representation size of kron_16...

/home/ahmed/ligra/utils/SNAPtoAdj -s /home/experiment_inputs/graphs/kron_16_unique_half.txt /home/ahmed/16_Adj.txt
/home/ahmed/aspen/code/memory_footprint -f /home/ahmed/16_Adj.txt > /home/ahmed/16_footprint.txt

echo Computing aspen representation size of kron_17...

/home/ahmed/ligra/utils/SNAPtoAdj -s /home/experiment_inputs/graphs/kron_17_unique_half.txt /home/ahmed/17_Adj.txt
/home/ahmed/aspen/code/memory_footprint -f /home/ahmed/17_Adj.txt > /home/ahmed/17_footprint.txt

echo Formatting kron_16...

/home/experiment_inputs/tools/prepend_count.sh /home/experiment_inputs/graphs/kron_16_unique_half.txt 65536

echo Formatting kron_17...

/home/experiment_inputs/tools/prepend_count.sh /home/experiment_inputs/graphs/kron_17_unique_half.txt 131072

echo Streamifying kron_16...

/home/ahmed/StreamingGraphAlgo/test/util/streamification/in_memory/streamify_in_mem /home/experiment_inputs/graphs/kron_16_unique_half.txt 0.99 10000 0.99 10000 1082661039 /home/experiment_inputs/streams/kron_16_unique_half_stream.txt

echo Streamifying kron_17...

/home/ahmed/StreamingGraphAlgo/test/util/streamification/in_memory/streamify_in_mem /home/experiment_inputs/graphs/kron_17_unique_half.txt 0.99 10000 0.99 10000 4337883873 /home/experiment_inputs/streams/kron_17_unique_half_stream.txt

