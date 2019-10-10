import numpy as np
import pickle as pkl

def saveVar(var, path, name):
	saveFile = open(path+name, 'wb')
	pkl.dump(var, saveFile)
	saveFile.close()
	return

def loadVar(path, name):
	loadFile = open(path+name, 'rb')
	var = pkl.load(loadFile)
	loadFile.close()
	return var


def getData(m = 30):
	dataX = np.zeros((m, 21), dtype = np.uint8)
	temp = np.random.rand(m, 21)
	dataX[temp < 0.5] = 1
	for i in range(1, 15):
		index = temp[:, i] < 0.75
		dataX[index, i] = dataX[index, i-1]
		dataX[~index, i] = 1 - dataX[~index, i-1]

	dataY = np.zeros(m, dtype = np.uint8)
	temp = np.zeros_like(dataY)
	index = dataX[:, 0] == 0
	temp[index] = np.sum(dataX[index, 1: 8], axis = 1)
	temp[~index] = np.sum(dataX[~index, 8: 15], axis = 1)
	dataY[temp >= 4] = 1

	return (dataX, dataY)

def prtData(data):
	x, y = data
	for i in range(21):
		print('%2d' %i, end = '')
	print('\ty')
	for i in range(y.size):
		print(x[i], end = '\t')
		print(y[i])


class tNode(object):
	#tree node
	def __init__(self, p, valid, data, idx, prune = 0, threshold = 0):
		#data related
		self.X, self.Y = data
		self.idxX, self.idxY = idx
		self.m, self.k = self.X.shape

		#tree related
		self.p = p
		self.l = None #X[key] == 0
		self.r = None #X[key] == 1

		#key and leaf
		if self.p is None:
			self.keyHistory = set()
		else:
			self.keyHistory = self.p.keyHistory | {self.p.key}

		self.key = None
		self.valid = valid

		self.depth = len(self.keyHistory)
		self.size = np.sum(self.valid.astype(np.uint8))

		self.leaf = False
		self.leafVal = None

		self.prune = prune
		self.threshold = threshold

		#init prob
		self.getProb()
		if not self.leaf:
			self.getCondProb()
		return

	def getProb(self):
		num1 = np.sum(self.idxY[self.valid].astype(np.uint8))

		#pre-prune
		if self.prune == 1:
			if self.depth >= self.threshold:
			#	print('I: prune: depth %d' %self.depth)
				self.leaf = True
				self.leafVal = round(num1 / self.size)
				if num1 == self.size - num1:
					self.leafVal = int(np.random.rand() < 0.5)
				return

		if self.prune == 2:
			if self.size <= self.threshold:
			#	print('I: prune: size %d' %self.size)
				self.leaf = True
				self.leafVal = round(num1 / self.size)
				if num1 == self.size - num1:
					self.leafVal = int(np.random.rand() < 0.5)
				return

		#no data
		if self.size == 0:
		#	print('W: no data')
			self.leaf = True
			self.leafVal = int(np.random.rand() < 0.5)
			return

		#unseparatable data
		if len(self.keyHistory) >= self.k:
		#	print('W: unseparatable data')
			self.leaf = True
			self.leafVal = round(num1 / self.size)
			if num1 == self.size - num1:
				self.leafVal = int(np.random.rand() < 0.5)
			return

		#0 leaf
		if num1 == 0: 
			self.leaf = True
			self.leafVal = 0
			return
		#1 leaf
		if num1 == self.size:
			self.leaf = True
			self.leafVal = 1
			return

		self.prob = num1 / self.size #P(Y = 1)
		return self.prob

	def getCondProb(self):
		self.condProb = np.full((2, self.k), np.nan, dtype = np.float16) #[Y=1 | X[key]]
		self.cond = np.full((2, self.k), np.nan, dtype = np.float16) #[X[key]]
		self.condSize = np.zeros((2, self.k), dtype = np.uint32) #[X[key]]
		self.condNum1 = np.zeros((2, self.k), dtype = np.uint32) #[Y=1 and X[key]]

		for i in range(self.k):
			if i in self.keyHistory:
				continue
			
			self.condSize[1, i] = np.sum(self.idxX[self.valid, i].astype(np.uint8)) #X[key] = 1
			self.condSize[0, i] = np.sum((~self.idxX[self.valid, i]).astype(np.uint8)) #X[key] = 0
			self.condNum1[1, i] = np.sum(self.idxY[self.valid] & self.idxX[self.valid, i].astype(np.uint8))
			self.condNum1[0, i] = np.sum(self.idxY[self.valid] & (~self.idxX[self.valid, i]).astype(np.uint8))

			tempNumR = np.sum(self.idxX[self.valid, i].astype(np.uint8))
			tempNumL = np.sum((~self.idxX[self.valid, i]).astype(np.uint8))

			self.cond[1, i] = tempNumR / self.size #P(X[key] = 1)
			self.cond[0, i] = tempNumL / self.size #P(X[key] = 0)


			self.condProb[0, i] = self.condNum1[0, i] / self.condSize[0, i] #P(Y = 1 | X[key] = 0)
			self.condProb[1, i] = self.condNum1[1, i] / self.condSize[1, i] #P(Y = 1 | X[key] = 1)

		return self.condProb

	def getInfo(self):

		def infoFun(p):
			if isinstance(p, float):
				if p == 0 or p == 1:
					return 0

			q = 1 - p
			info = p * np.log2(p) + q * np.log2(q)

			if isinstance(p, np.ndarray):
				idx0 = (p == 0) | (p == 1)
				info[idx0] = 0
			return info

		info = infoFun(self.prob)
		condInfo = np.sum(self.cond * infoFun(self.condProb), axis = 0)
		infoGain = condInfo - info
		infoGain[np.isnan(infoGain)] = -np.inf
		return infoGain


class decisionTree(object):
	def __init__(self, data, prune = 0, threshold = 0):
		self.X, self.Y = data
		self.idxY = self.Y.astype(np.bool)
		self.idxX = self.X.astype(np.bool)
		self.m, self.k = self.X.shape

		self.prune = prune
		self.threshold = threshold

		return

	def chiTest(self, node, key):
		ex = np.empty((2,2), dtype = np.float16)
		ob = np.empty_like(ex)

		ex[:, 0] = (1-node.prob) * node.cond[:, key] #P(X and Y=0)
		ex[:, 1] = node.prob * node.cond[:, key] #P(X and Y=1)

		ex = ex * node.size

		ob[:, 1] = node.condNum1[:, key] #N(X and Y=1)
		ob[:, 0] = node.condSize[:, key] - node.condNum1[:, key] #N(X and Y=0)

		T = np.sum(np.square(ex - ob) / ex)
		return T

	def getKey(self, node):
		value = node.getInfo()
		index = list(np.argsort(value))
		for key in reversed(index):
			if value[key] <= 0:
			#	print('W: negative Info %f' %value[key])
				node.leaf = True
				node.leafVal = round(node.prob)
				if node.prob == 0.5:
					node.leafVal = int(np.random.rand() < 0.5)
				return None
			else:
				if self.prune == 3:
					res = self.chiTest(node, key)
					if res > self.threshold:
						return key
					else:
					#	print('I: prune (skip): significance %f' %res)
						continue
				else:
					return key

		node.leaf = True
		node.leafVal = round(node.prob)
		if node.prob == 0.5:
			node.leafVal = int(np.random.rand() < 0.5)
		return None

	def branch(self, node):
		if node.leaf:
			return

		key = self.getKey(node)
		if key is None:
			return

		node.key = key
		node.l = tNode(node, node.valid & ~self.idxX[:, key], (self.X, self.Y), (self.idxX, self.idxY), self.prune, self.threshold)
		node.r = tNode(node, node.valid & self.idxX[:, key], (self.X, self.Y), (self.idxX, self.idxY), self.prune, self.threshold)
		self.fringe.append(node.l)
		self.fringe.append(node.r)
		return
	
	def fit(self):
		self.root = tNode(None, np.full(self.m, True, dtype = np.bool), (self.X, self.Y), (self.idxX, self.idxY), self.prune, self.threshold)
		self.fringe = [self.root]

		while self.fringe:
			node = self.fringe.pop()
			self.branch(node)

		res = self.predict(self.X)
		err = self.getErrRate(res, self.Y)

		return err

	def predict(self, X):
		m, k = X.shape
		res = np.empty(m, dtype = np.uint8)
		for i in range(m):
			node = self.root
			while not node.leaf:
				if X[i, node.key] == 1:
					node = node.r
				else:
					node = node.l
			res[i] = node.leafVal
		return res

	def getErrRate(self, res, Y):
		err = res.astype(np.bool) ^ Y.astype(np.bool)
		return np.sum(err.astype(np.uint8)) / Y.size

	def visualize(self):
		fringe = [(self.root, '')]
		while fringe:
			node, path = fringe.pop()
			if not node.leaf:
				fringe.append((node.l, path + 'X[%d] = 0, '%node.key))
				fringe.append((node.r, path + 'X[%d] = 1, '%node.key))
			else:
				path = path + 'Y = %d'%node.leafVal
				print(path)
		return

	def getKeySet(self):
		fringe = [self.root]
		keySet = set()
		while fringe:
			node = fringe.pop()
			if not node.leaf:
				fringe.append(node.l)
				fringe.append(node.r)
			else:
				keySet.update(node.keyHistory)
		return keySet


def Q1(mMin = 10, mMax = 10011, mStep = 200):
	for m in range(mMin, mMax, mStep):
		err = np.empty(10, dtype = np.float16)
		for i in range(10):
			X, Y = getData(m = m)
			dt = decisionTree((X, Y))
			dt.fit()
			X, Y = getData(m = 100000)
			res = dt.predict(X)
			err[i] = dt.getErrRate(res, Y)
		print('m = %d:' %m)
		print('..error = %.3f,' %np.mean(err))
	return

def Q2(mMin = 10, mMax = 10011, mStep = 200):
	num = np.empty(50, dtype = np.uint8)
	for m in range(mMin, mMax, mStep):
		for i in range(50):
			X, Y = getData(m = m)
			dt = decisionTree((X, Y))
			dt.fit()
			keySet = dt.getKeySet()
			num[i] = len(keySet & {15, 16, 17, 18, 19, 20})
		print('m = %d' %m)
		print('..#irrelevant vars = %.1f' %np.mean(num))
	return

def getDataSet():
	trainX, trainY = getData(8000)
	testX, testY = getData(2000)
	saveVar(((trainX, trainY), (testX, testY)), '', 'data.pkl')
	return

def Q3a(tMin = 0, tMax = 21, tStep = 1):
	(trainX, trainY), (testX, testY) = loadVar('', 'data.pkl')
	for threshold in range(tMin, tMax, tStep):
		dt = decisionTree((trainX, trainY), prune = 1, threshold = threshold)
		errT = dt.fit()
		res = dt.predict(testX)
		err = dt.getErrRate(res, testY)
		print('threshold = %d:' %threshold)
		print('..error_train = %.3f, error = %.3f' %(errT, err))
	return

def Q3b(tMin = 8000, tMax = 2, tStep = 2):
	(trainX, trainY), (testX, testY) = loadVar('', 'data.pkl')
	threshold = tMin
	while threshold > tMax:
		dt = decisionTree((trainX, trainY), prune = 2, threshold = threshold)
		errT = dt.fit()
		res = dt.predict(testX)
		err = dt.getErrRate(res, testY)
		print('threshold = %d:' %threshold)
		print('..error_train = %.3f, error = %.3f' %(errT, err))
		threshold = round(threshold / tStep)
	return

def Q3c(tMin = 0, tMax = 10, tStep = 1):
	(trainX, trainY), (testX, testY) = loadVar('', 'data.pkl')
	threshold = [0.455, 0.708, 1.323, 2.072, 2.706, 3.841, 5.024, 6.635, 7.879, 10.828]
	for index in range(tMin, tMax, tStep):
		dt = decisionTree((trainX, trainY), prune = 1, threshold = threshold[index])
		errT = dt.fit()
		res = dt.predict(testX)
		err = dt.getErrRate(res, testY)
		print('threshold = %.3f:' %threshold[index])
		print('..error_train = %.3f, error = %.3f' %(errT, err))
	return

def Q4(threshold = 9, mMin = 10, mMax = 10011, mStep = 200):
	num = np.empty(50, dtype = np.uint8)
	for m in range(mMin, mMax, mStep):
		for i in range(50):
			X, Y = getData(m = m)
			dt = decisionTree((X, Y), prune = 1, threshold = threshold)
			dt.fit()
			keySet = dt.getKeySet()
			num[i] = len(keySet & {15, 16, 17, 18, 19, 20})
		print('m = %d' %m)
		print('..#irrelevant vars = %.1f' %np.mean(num))
	return

def Q5(threshold = 16, mMin = 10, mMax = 10011, mStep = 200):
	num = np.empty(50, dtype = np.uint8)
	for m in range(mMin, mMax, mStep):
		for i in range(50):
			X, Y = getData(m = m)
			dt = decisionTree((X, Y), prune = 1, threshold = threshold)
			dt.fit()
			keySet = dt.getKeySet()
			num[i] = len(keySet & {15, 16, 17, 18, 19, 20})
		print('m = %d' %m)
		print('..#irrelevant vars = %.1f' %np.mean(num))
	return

def Q6(threshold = 10.828, mMin = 10, mMax = 10011, mStep = 200):
	num = np.empty(50, dtype = np.uint8)
	for m in range(mMin, mMax, mStep):
		for i in range(50):
			X, Y = getData(m = m)
			dt = decisionTree((X, Y), prune = 1, threshold = threshold)
			dt.fit()
			keySet = dt.getKeySet()
			num[i] = len(keySet & {15, 16, 17, 18, 19, 20})
		print('m = %d' %m)
		print('..#irrelevant vars = %.1f' %np.mean(num))
	return

if __name__ == '__main__':
	Q4()