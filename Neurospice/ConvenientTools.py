import time

def helpme(ob):
	
	'''
	
	function that pretty prints members of a given object
	
	ob=python object
	
	'''
	
	from inspect import getmembers
	import pprint
	pp = pprint.PrettyPrinter()
	pp.pprint(getmembers(ob))

class timeit():
	
	'''
	
	class to manage the timing of different processes. used by wrapping processes with the instantiation of the class and then followed by the class.timeout() using the following pattern:
	
	timer = timeit('adding 2 and 2 took')
	task = 2+2
	timer.timeout()
	
	>>adding 2 and 2 took 0.0002 minutes to complete
	
	'''
	
	def __init__(self,string):
		self.string = string
		self.start = time.time()
		
	def timeout(self):
		self.total=self.string+' '+str(round((time.time()-self.start)/60.0,2))+' minutes to complete'
		print(self.total)

