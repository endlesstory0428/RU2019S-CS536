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

#generate m-line k-dim data, margin = epsilon
def getData(m = 100, k = 20, epsilon = 1):
	Y = (np.random.rand(m) < 0.5).astype(np.int8) * 2 - 1

	X = np.empty((m, k), dtype = np.float16)
	X[:, 0: k-1] = np.random.normal(0, 1, (m, k-1))
	X[:, k-1] = (np.random.exponential(1, m) + epsilon) * Y

	return (X, Y)

#visualize
def prtData(data):
	x, y = data
	m, k = x.shape
	for i in range(k):
		print('%2d' %i, end = '\t')
	print('\ty')
	for i in range(m):
		print('[', end = '')
		for j in range(k):
			print('%.2f' %x[i, j], end = '\t')
		print(']', end = '\t')
		print(y[i])


class perceptron(object):
	def __init__(self, data):
		self.X, self.Y = data
		self.m, self.k = self.X.shape
		self.w = np.zeros(self.k)
		self.b = np.zeros(1)
		return

	#visualize related
	def __str__(self):
		retval = ''
		for i in range(self.k):
			retval = retval + ('%2d\t' %i)
		retval = retval + ' b\n' + repr(self)
		return retval

	def __repr__(self):
		retval = ''
		for i in range(self.k):
			retval = retval + ('%.2f\t' %self.w[i])
		retval = retval + ('%.2f' %self.b)
		return retval

	#f(wx+b), for a data point
	def forward(self, x):
		return np.sign(np.sum(self.w * x) + self.b)

	#f(wX+b), for a dataset
	def predict(self, X):
		return np.sign(np.sum(self.w * X, axis = 1) + self.b)

	#update
	def fit(self, stepLimit = None, visualize = False):
		step = 0
		learningFlag = True
		while learningFlag:
			if stepLimit and step > stepLimit:
				break
			learningFlag = False
			for i in range(self.m):
				if self.forward(self.X[i]) != self.Y[i]:
					step = step + 1
					learningFlag = True
					self.w = self.w + self.Y[i] * self.X[i]
					self.b = self.b + self.Y[i]
					if visualize:
					#	print('step %d:' %step)
						print(repr(self))
					break
		return step

def Q2():
	data = getData(m = 100, k = 20, epsilon = 1)
	P = perceptron(data)
	step = P.fit()
	print('After %d steps learning:' %step)
	print(str(P))
	return

def Q3(eMin = 1, eMax = 101, eSetp = 1, size = 100):
	step = np.empty(size, dtype = np.uint8)
	for dummy in range(eMin, eMax ,eSetp):
		epsilon = dummy / 100.
		for i in range(size):
			data = getData(m = 100, k = 20, epsilon = epsilon)
			P = perceptron(data)
			step[i] = P.fit()
		print('epsilon = %.2f' %epsilon)
		print('..step number = %.2f' %np.mean(step))
	return

def Q4(m = 100, kMin = 2, kMax = 41, kStep = 1, size = 100):
	step = np.empty(size, dtype = np.uint8)
	for k in range(kMin, kMax, kStep):
		for i in range(size):
			data = getData(m = m, k = k, epsilon = 1)
			P = perceptron(data)
			step[i] = P.fit()
		print('k = %d' %k)
		print('..step number = %.2f' %np.mean(step))
	return

#m-line k-dim (hopefully) unseparable dataset
def getNSData(k = 2, m = 100):
	X = np.random.normal(0, 1, (m, k))
	Y = (np.sum(np.square(X), axis = 1) > k).astype(np.int8) * 2 - 1
	return (X, Y)

#sufficient but not necessary condition for unseparable
def checkNS2(data):
	X, Y = data
	pX = X[Y == 1]
	nX = X[Y == -1]
	pXpp = pX[(pX[:, 0] > 0) & (pX[:, 1] > 0)]
	pXnp = pX[(pX[:, 0] < 0) & (pX[:, 1] > 0)]
	pXnn = pX[(pX[:, 0] < 0) & (pX[:, 1] < 0)]
	pXpn = pX[(pX[:, 0] > 0) & (pX[:, 1] < 0)]
	if nX.size and pXpp.size and pXnp.size and pXnn.size and pXpn.size: #all 4 quadrants have neg data and there is pos data
		return True
	return False

def Q5():
	data = getNSData()
	P = perceptron(data)
	P.fit(stepLimit = 8192, visualize = True)
	return

if __name__ == '__main__':
	Q4(m = 100)