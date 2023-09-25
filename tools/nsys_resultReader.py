import sys
import json

if len(sys.argv) != 2 :
    print("Incorrect Argument!")
    print("Arguments: json_file")
    exit()

fun_num = 0
with open(sys.argv[1]) as file:
    line_content = [json.loads(lines) for lines in file.readlines()]
    total_duration = 0
    for line in line_content:
        for key, value in line.items():
            if key == "Type" and value == 48:
                if line["TraceProcessEvent"]["name"] == "889": # cudaStreamAttachMemAsync
                    start = float(line["TraceProcessEvent"]["startNs"])
                    end = float(line["TraceProcessEvent"]["endNs"])
                    total_duration += (end - start)  
                    fun_num += 1
    print("Total Duration:", total_duration / 1000000000)
    print(fun_num)