import numpy as np

def getW():
	i = np.array((range(10), ), dtype = np.int8) + 1
	w = np.power(0.6, i, dtype = np.float64)
	return w

def getData(m = 1000):
	X = np.empty((m, 21), dtype = np.float64)

	X[:, 0] = 1
	X[:, 1: 11] = np.random.normal(0, 1, size = (m, 10))
	X[:, 11: 16] = np.random.normal(0, np.sqrt(0.1), size = (m, 5))
	X[:, 16: 21] = np.random.normal(0, 1, size = (m, 5))

	X[:, 11] = X[:, 11] + X[:, 1] + X[:, 2]
	X[:, 12] = X[:, 12] + X[:, 3] + X[:, 4]
	X[:, 13] = X[:, 13] + X[:, 4] + X[:, 5]
	X[:, 14] = X[:, 14] + 0.1 * X[:, 7]
	X[:, 15] = X[:, 15] + 2 * X[:, 2] - 10

	w = getW()

	Y = 10 + np.sum(X[:, 1: 11] * w, axis = 1) + np.random.normal(0, np.sqrt(0.1), size = m)
	return (X, Y)


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


class classic(object):
	def __init__(self, data):
		self.X, self.Y = data
		self.m, self.k = self.X.shape
		self.fit()
		return

	def fit(self):
		XT = np.transpose(self.X)
		Sigma = np.matmul(XT, self.X)

		self.w = np.matmul(np.matmul(np.linalg.inv(Sigma), XT), self.Y)
		
		return self.w
	
	def prtW(self):
		for i in range(self.k):
			print('%2d' %i, end = '\t')
		print('')
		print('weights:')
		for i in range(self.k):
			print('%.4f' %self.w[i], end = '\t')
		print('')
		return

	def predict(self, data):
		X, Y = data
		m, k = X.shape
		y = np.sum(self.w * X, axis = 1)
		return np.mean(np.square(Y - y))


class ridge(object):
	def __init__(self, data, l, index = None):
		self.X, self.Y = data
		self.m, self.prevk = self.X.shape
		self.l = l

		if index is None:
			self.index = np.full(self.prevk, True, dtype = np.bool)
			self.k = self.prevk
		else:
			self.index = index
			self.k = np.sum(index.astype(np.int8))
		self.X = self.X[:, index].reshape((self.m, self.k))

		self.fit()
		return

	def fit(self):
		XT = np.transpose(self.X)
		Sigma = np.matmul(XT, self.X)

		self.w = np.matmul(np.matmul(np.linalg.inv(Sigma + np.identity(self.k) * self.l), XT), self.Y)
		
		return self.w
	
	def prtW(self):
		w = self.getW()
		for i in range(self.prevk):
			print('%2d' %i, end = '\t')
		print('')
		print('weights:')
		for i in range(self.prevk):
			print('%.4f' %w[i], end = '\t')
		print('')
		return

	def predict(self, data):
		X, Y = data
		m, k = X.shape
		y = np.sum(self.w * X[:, self.index], axis = 1)
		return np.mean(np.square(Y - y))

	def getW(self):
		w = np.zeros(self.prevk, dtype = np.float64)
		w[self.index] = self.w
		return w


class lasso(object):
	def __init__(self, data, l):
		self.X, self.Y = data
		self.m, self.k = self.X.shape
		self.l = l
		self.l05 = l / 2
		self.fit()
		return

	def fit(self):
		self.w = np.zeros(21, dtype = np.float64)
		maxPatience = 64
		patience = maxPatience
		prevErr = float('inf')

		X2 = np.sum(np.square(self.X), axis = 0)
		
		for iterCount in range(self.k * 10000):
			index = iterCount % self.k
			loss = self._totalLoss()

			tempErr = np.mean(np.square(loss))
			if np.abs(prevErr - tempErr) < 1e-16:
				patience = patience / 2
				if patience < 2:
					print('I: no more patience.')
					break
			else:
				patience = np.min([patience * 4, maxPatience])
			prevErr = tempErr

			if index == 0:
				self.w[0] = self.w[0] + np.mean(loss)
			else:
				A = X2[index]
				B051 = np.sum(self.X[:, index] * loss)
				B052 = self.w[index] * X2[index]
				B05 = B051 + B052
				if abs(B05) <= self.l05:
					self.w[index] = 0
				else:
					sign = np.sign(B05)
					self.w[index] = self.w[index] + (B051 - sign * self.l05) / A
		else:
			print('I: reach max iter.')

		return self.w


	def prtW(self):
		for i in range(self.k):
			print('%2d' %i, end = '\t')
		print('')
		print('weights:')
		for i in range(self.k):
			print('%.4f' %self.w[i], end = '\t')
		print('')
		return


	def predict(self, data):
		X, Y = data
		m, k = X.shape
		y = np.sum(self.w * X, axis = 1)
		return np.mean(np.square(Y - y))


	def _totalLoss(self):
		return self.Y - np.sum(self.w * self.X, axis = 1)



def Q1():
	C = classic(getData(m = 1000))
	C.prtW()
	w = np.zeros(21, dtype = np.float64)
	w[1: 11] = getW()
	w[0] = 10
	print('true weights:')
	for i in range(21):
		print('%.4f' %w[i], end = '\t')
	print('\n')
	print('weight: (more details)')
	print(C.w)
	print('\n')
	print('error:')
	print(C.predict(getData(m = 1000000)))
	print('\n')
	print('||^w - w||:')
	print(np.sum(np.square(C.w - w)))
	return


def Q2(lmin = 1e-16, lmax = 100):
	l = lmin
	step = lmin
	count = 0
	while l <= lmax:
		print('lambda:')
		print(l)
		print('')
		R = ridge(getData(m = 1000), l)
		print('weight: (more details)')
		print(R.w)
		print('')
		print('error:')
		print(R.predict(getData(m = 1000000)))
		print('\n')
		count = count + 1
		l = l + step
		if count == 9:
			count = 0
			step = step * 10
	return

def Q3(lmin = 1e-16, lmax = 1e4):
	l = lmin
	step = lmin
	count = 0
	while l <= lmax:
		print('lambda:')
		print(l)
		print('')
		L = lasso(getData(m = 1000), l)
		print('weight: (more details)')
		print(L.w)
		print('')
		print('error:')
		print(L.predict(getData(m = 1000000)))
		print('#non-zero weights:')
		print(20 - np.sum((~(L.w.astype(np.bool))).astype(np.int8)))
		print('\n')
		count = count + 1
		l = l + step
		if count == 9:
			count = 0
			step = step * 10
	return

def Q4(lmin = 4e-1, lmax = 1e2, step = 4e-1):
	l = lmin
	while l <= lmax:
		print('lambda:')
		print(l)
		print('')
		L = lasso(getData(m = 1000), l)
		print('weight: (more details)')
		print(L.w)
		print('')
		print('error:')
		print(L.predict(getData(m = 1000000)))
		print('#non-zero weights:')
		print(20 - np.sum((~(L.w.astype(np.bool))).astype(np.int8)))
		print('\n')
		l = l + step
	return

def Q5(lmin = 1e-16, lmax = 100, index = None):
	l = lmin
	step = lmin
	count = 0
	while l <= lmax:
		print('lambda:')
		print(l)
		print('')
		R = ridge(getData(m = 1000), l, index)
		print('weight: (more details)')
		print(R.w)
		print('')
		print('error:')
		print(R.predict(getData(m = 1000000)))
		print('\n')
		count = count + 1
		l = l + step
		if count == 9:
			count = 0
			step = step * 10
	return


if __name__ == '__main__':
	index = np.full(21, True, dtype = np.bool)
	zeroList = [13, 17, 20]
	index[zeroList] = False
	Q5(index = index)