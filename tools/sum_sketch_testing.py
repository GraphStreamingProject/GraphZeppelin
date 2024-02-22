import sys
import re

"""
The purpose of this file is to parse the output of sketch_testing.cpp into summary statistics
That is, we can answer questions like "how many data points are 2 stddev above .8"
or "What is the mean of the data"
"""

prob = r"([0-9]*[.])?[0-9]+"
which = r"[0-9]+"

pattern = re.compile("(" + which + "): (" + prob + ") \+- (" + prob + ")")

def parse(filename):
  with open(filename) as file:
    lines = file.readlines()[:4000000]
    stats = []
    for l in lines:
      match = pattern.match(l)
      if match:
        t = (int(match.group(1)), float(match.group(2)), float(match.group(4)))
        stats.append(t)
    return stats

def above(stats, target, sigmas):
  above = 0
  below = 0


  for s in stats:
    if (s[1] - sigmas * s[2] > target):
      above += 1
    else:
      below += 1
      print("BELOW")

  print (above / (above + below))
  

def mean(stats, sigmas):
  summ = 0
  count = 0
  for s in stats:
    count += 1
    summ += s[1] - sigmas * s[2]
  print(summ/count)
  
  
stats = parse(sys.argv[1])

above(stats, 0.76, 0)
#above(stats, 0.78, 1)
#above(stats, 0.78, 2)

#mean(stats, 3)







