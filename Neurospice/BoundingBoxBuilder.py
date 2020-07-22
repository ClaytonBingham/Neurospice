import pickle
import os
import numpy as np

class BuildBoundingBox():
	
	'''
	class that can load the NEURON derived current source locations and find a bounding box that would contain all of them and estimate a sufficient boundary to which all voltages would drop to zero
	
	datadir = path/directory that contains the current source data that is driving the model construction
	padding = scalar that is applied to the dimensions of the bounding box that is fitted to the model coordinates to provide a spatial buffer and minimize boundary effects (default=1.2)
	
	
	Usage:
	import pickle
	import random
	locs = dict(range(100),[[random.random()*random.choice([-10,10]),random.random()*random.choice([-10,10]),random.random()*random.choice([-10,10])] for i in range(100))
	with open('locations_single.txt','w') as f:
		pickle.dump(locs)
	
	import neurospice
	from neurospice.BoundingBoxBuilder import BuildBoundingBox
	bbox = BuildBoundingBox(os.getcwd(),padding=1.2)
	print(bbox.recommended_dimensions,bbox.recommended_shift)
	
	'''
	
	def __init__(self,datadir,padding=1.42): #padding is proportional to box side dimensions
		dfiles = [datadir+"/"+fname for fname in os.listdir(datadir) if 'locations_single' in fname]
		
		with open(dfiles[0],'rb') as f:
			dat = pickle.load(f,encoding='latin1')
		
		key = list(dat.keys())[0]
		x = [dat[key][0][0],dat[key][0][0]]
		y = [dat[key][0][1],dat[key][0][1]]
		z = [dat[key][0][2],dat[key][0][2]]
		
		for dfile in dfiles:
			print(dfile)
			with open(dfile,'rb') as f:
				dat = pickle.load(f,encoding='latin1')
				for cell in dat.keys():
					for point in dat[cell]:
						if point[0]<x[0]:
							x[0] = point[0]
						if point[0] > x[1]:
							x[1] = point[0]
						if point[1] < y[0]:
							y[0] = point[1]
						if point[1] > y[1]:
							y[1] = point[1]
						if point[2] < z[0]:
							z[0] = point[2]
						if point[2] > z[1]:
							z[1] = point[2]
		
		self.x = x
		self.y = y
		self.z = z
		print('Model Boundaries: '+'\nx\n'+str(self.x)+'\ny\n'+str(self.y)+'\nz\n'+str(self.z))
		self.model_center = [np.mean(self.x),np.mean(self.y),np.mean(self.z)]
		self.data_dimensions = [self.x[1]-self.x[0],self.y[1]-self.y[0],self.z[1]-self.z[0]]
		self.recommended_dimensions = [self.data_dimensions[0]*padding,self.data_dimensions[1]*padding,self.data_dimensions[2]*padding]
		self.recommended_shift = [self.model_center[0]-self.recommended_dimensions[0]/2.0,self.model_center[1]-self.recommended_dimensions[1]/2.0,self.model_center[2]-self.recommended_dimensions[2]/2.0]
		print('Recommended Model Dimensions: '+str(self.recommended_dimensions))
		print('Recommended Shift: '+str(self.recommended_shift))
