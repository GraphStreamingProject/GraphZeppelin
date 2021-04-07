from pyspark import SparkContext
import sketch
import random
sc = SparkContext("local","First App")
value = 0
vect_size = 20
vector = [0] * vect_size

def process_update(x):	
	global value,vector,sample_sketch
	sample_sketch = sketch.Sketch(vect_size,700000)
	index = random.randint(0,vect_size-1)
	delta = random.randint(-5,5)
	print(index)
	print("Before: " + str(vector[index]))
	vector[index] += delta
	print("After: " + str(vector[index]))
	sample_update = sketch.Update()
	sample_update.index = index
	sample_update.delta = delta
	sample_sketch.update(sample_update)
	value += x
	print(vector)
	return sample_sketch

score = 0
random.seed()
num_updates = 100
rdd = sc.parallelize(range(num_updates))
print(rdd.map(process_update).reduce( lambda a,b : sketch.plus(a,b)).collect().query())
#query_result = sample_sketch.query()
#if vector[query_result.index] != 0 and vector[query_result.index] == query_result.delta:
#	score += 1
#print(score)

