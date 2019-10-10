import numpy as np

def getData():
	X = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype = np.float64)
	Y = -(X[:, 0] * X[:, 1])
	return (X, Y)

def qKernel(a, b):
	return np.square(1 + np.sum(a * b))


class SVM(object):
	def __init__(self, data, kernel, hidIndex = 0):
		self.X, self.Y = data
		self.m, self.k = self.X.shape

		self.kernel = kernel
		self.K = self.getK()

		self.rate = 1 / 64
		self.maxPatience = 64

		self.hidIndex = hidIndex
		self.keyIndex = list(set(range(self.m)) - {self.hidIndex})
		print(self.keyIndex)

		self.getFrqVal()

		self.initA()

		self.fit()
		return

	#compute the kernel matrix
	def getK(self):
		K = np.empty((self.m, self.m), dtype = np.float64)
		for i in range(self.m):
			for j in range(i, self.m):
				val = self.kernel(self.X[i], self.X[j])
				K[i, j] = val
				K[j, i] = val
		return K


	#compute some frequent using values
	def getFrqVal(self):
		self.y1Y2 = self.Y[self.hidIndex] * self.Y[self.keyIndex]
		print(self.y1Y2)
		self.nyy = -self.y1Y2 / np.sqrt(np.sum(np.square(self.y1Y2)))
		return

	#init alpha
	def initA(self):
		self.a = np.empty((self.m), dtype = np.float64)
		posIdx = np.where(self.Y > 0)[0]
		negIdx = np.where(self.Y < 0)[0]
		self.a[posIdx] = 1 / posIdx.size
		self.a[negIdx] = 1 / negIdx.size
		return self.a

	#check if over shoot
	def checkA(self, a):
		negIdx = np.where(a[self.keyIndex] <= 0)[0]
		if negIdx.size > 0:
			for i in set(negIdx):
				if i >= self.hidIndex:
					a[i+1] = 1
				else:
					a[i] = 1
		#	print(a[self.keyIndex][negIdx]) #It will not change the real value of a, so do NOT use it to project a
		a[self.hidIndex] = -np.sum(a[self.keyIndex] * self.y1Y2)
		if a[self.hidIndex] <= 0:
			a[self.keyIndex] = a[self.keyIndex] - 2 * np.sum(a[self.keyIndex] * self.nyy) * self.nyy
			# print(a)
			a[self.hidIndex] = -np.sum(a[self.keyIndex] * self.y1Y2)
		return a

	#main GD loop
	def update(self, epsilon):
		patience = self.maxPatience

		for iterCount in range(100000):
			ay = self.a * self.Y

			factor = np.sum(self.K * ay, axis = 1) #sum(K2[k row, i col] * ay[i], axis = 1)

			grad = 1 - factor * self.Y #grad(object function)
			bar = 1 / self.a #grad(bar function)

			stepAll = grad + epsilon * bar
			step = stepAll[self.keyIndex] - self.y1Y2 * stepAll[self.hidIndex] #BP a[self.hidIndex] to a[self.keyIndex]

			a = np.copy(self.a)
			a[self.keyIndex] = self.a[self.keyIndex] + self.rate * step
			a = self.checkA(a)
			prevA = self.a
			self.a = a

			if np.sum(np.square(step)) < 1e-3: #too small changes, halt
				patience = patience / 2
				if patience < 2:
					print('no more patience at iter %d' %iterCount, end = '\n..')
					print(self.a)
					break
			else:
				patience = np.min([patience * 4, self.maxPatience])

		else:
			print('reach max iter', end = '\n..')
			print(self.a)
		return self.a


	def fit(self):
		epsilon = 1
		while epsilon > 1e-16:
			print('epsilon ', end = '')
			print(epsilon, end = ':\n..')
			self.a = self.update(epsilon)
			epsilon = epsilon / 2
		return self.a



if __name__ == '__main__':
	S = SVM(getData(), qKernel, hidIndex = 0)