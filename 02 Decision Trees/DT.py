import numpy as np
import pickle as pkl

def getData(k = 4, m = 30):
	dataX = np.zeros((m, k), dtype = np.uint8)
	temp = np.random.rand(m, k)
	dataX[temp[:, 0] < 0.5, 0] = 1
	for i in range(1, k):
		index = temp[:, i] < 0.75
		dataX[index, i] = dataX[index, i-1]
		dataX[~index, i] = 1 - dataX[~index, i-1]

	dataY = np.zeros(m, dtype = np.uint8)
	temp = np.zeros(m, dtype = np.float16)
	factor = 0
	for i in range(1, k):
		weight = 0.9**(i+1)
		temp = temp + weight * dataX[:, i]
		factor = factor + weight
	index = (temp / factor) < 0.5
	dataY[index] = 1 - dataX[index, 0]
	dataY[~index] = dataX[~index, 0]

	return (dataX, dataY)

class tNode(object):
	#tree node
	def __init__(self, p, valid, data, idx):
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

		self.leaf = False
		self.leafVal = None

		#init prob
		self.getProb()
		if not self.leaf:
			self.getCondProb()
		return

	def getProb(self):
		size = np.sum(self.valid.astype(np.uint8))
		num1 = np.sum(self.idxY[self.valid].astype(np.uint8))

		if size == 0: #no data
			self.leaf = True
			self.leafVal = int(np.random.rand() < 0.5)
			return

		if len(self.keyHistory) >= self.k: #unseparatable data
			self.leaf = True
			self.leafVal = round(num1 / size)
			return

		if num1 == 0: #0 leaf
			self.leaf = True
			self.leafVal = 0
			return
		if num1 == size: #1 leaf
			self.leaf = True
			self.leafVal = 1
			return

		self.prob = num1 / size #P(Y = 1)
		return self.prob

	def getCondProb(self):
		self.condProb = np.full((2, self.k), np.nan, dtype = np.float16) #[Y=1 | X[key]]
		self.cond = np.full((2, self.k), np.nan, dtype = np.float16) #[X[key]]

		for i in range(self.k):
			if i in self.keyHistory:
				continue
			
			tempSizeR = np.sum(self.idxX[self.valid, i].astype(np.uint8)) #X[key] = 1
			tempSizeL = np.sum((~self.idxX[self.valid, i]).astype(np.uint8)) #X[key] = 0
			tempNumR1 = np.sum(self.idxY[self.valid] & self.idxX[self.valid, i].astype(np.uint8))
			tempNumL1 = np.sum(self.idxY[self.valid] & (~self.idxX[self.valid, i]).astype(np.uint8))

			tempNumR = np.sum(self.idxX[self.valid, i].astype(np.uint8))
			tempNumL = np.sum((~self.idxX[self.valid, i]).astype(np.uint8))

			self.cond[1, i] = tempNumR / (tempNumL + tempNumR) #P(X[key] = 1)
			self.cond[0, i] = tempNumL / (tempNumL + tempNumR) #P(X[key] = 0)


			self.condProb[0, i] = tempNumL1 / tempSizeL #P(Y = 1 | X[key] = 0)
			self.condProb[1, i] = tempNumR1 / tempSizeR #P(Y = 1 | X[key] = 1)

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
		return condInfo - info

	def getGini(self):

		def giniFun(p):
			q = 1 - p
			return np.square(p) + np.square(q)
		
		return np.sum(self.cond * giniFun(self.condProb), axis = 0)

class decisionTree(object):
	def __init__(self, data, gini = False):
		self.X, self.Y = data
		self.idxY = self.Y.astype(np.bool)
		self.idxX = self.X.astype(np.bool)
		self.m, self.k = self.X.shape

		self.gini = gini

		return

	def getKey(self, node):
		if self.gini:
			value = node.getGini()
		else:
			value = node.getInfo()
		key = np.nanargmax(value)

		if value[key] <= 0:
			node.leaf = True
			node.leafVal = round(node.prob)
			return None
		else:
			return key

	def branch(self, node):
		if node.leaf:
			return

		key = self.getKey(node)
		if key is None:
			return

		node.key = key
		node.l = tNode(node, node.valid & ~self.idxX[:, key], (self.X, self.Y), (self.idxX, self.idxY))
		node.r = tNode(node, node.valid & self.idxX[:, key], (self.X, self.Y), (self.idxX, self.idxY))
		self.fringe.append(node.l)
		self.fringe.append(node.r)
		return
	
	def fit(self):
		self.root = tNode(None, np.full(self.m, True, dtype = np.bool), (self.X, self.Y), (self.idxX, self.idxY))
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

def Q3():
	X, Y = getData(k = 4, m = 30)
	print('X = ')
	print(X)
	print('Y = ')
	print(Y)
	dt = decisionTree((X, Y))
	err = dt.fit()
	saveVar(dt, '', 'dt.pkl')
	print('********decision tree********')
	print('error_train = ', end = '')
	print(err)
	dt.visualize()

def Q4():
	X, Y = getData(k = 4, m = 1000000)
	dt = loadVar('', 'dt.pkl')
	res = dt.predict(X)
	err = dt.getErrRate(res, Y)
	print('error = ', end = '')
	print(err)

def Q5():
	for m in range(10, 2001, 50):
		errT = np.empty(10, dtype = np.float16)
		err = np.empty(10, dtype = np.float16)
		for i in range(10):
			X, Y = getData(k = 10, m = m)
			dt = decisionTree((X, Y))
			errT[i] = dt.fit()
			X, Y = getData(k = 10, m = 100000)
			res = dt.predict(X)
			err[i] = dt.getErrRate(res, Y)
		print('m = %d:' %m)
		print('..|error_train - error| = %.3f,' %np.mean(np.abs(errT - err)))

def Q6():
	for m in range(10, 2001, 50):
		errT = np.empty(10, dtype = np.float16)
		err = np.empty(10, dtype = np.float16)
		for i in range(10):
			X, Y = getData(k = 10, m = m)
			dt = decisionTree((X, Y), gini = True)
			errT[i] = dt.fit()
			X, Y = getData(k = 10, m = 100000)
			res = dt.predict(X)
			err[i] = dt.getErrRate(res, Y)
		print('m = %d:' %m)
		print('..|error_train - error| = %.3f,' %np.mean(np.abs(errT - err)))

if __name__ == '__main__':
	Q6()