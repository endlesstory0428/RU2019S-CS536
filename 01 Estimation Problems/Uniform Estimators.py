import numpy as np

def getData(j = 1000, n = 100, L = 10):
	return L * np.random.rand(n, j)

def getLmom(data):
	return 2 * data.mean(axis = 0)

def getLmle(data):
	return data.max(axis = 0)

def getMSE(L, Lhat):
	return np.mean(np.square(Lhat - L))

def main(L = 10, n = 100, j = 1000):
	data = getData(j, n, L)
	Lmom = getLmom(data)
	Lmle = getLmle(data)
	MSEmom = getMSE(L, Lmom)
	MSEmle = getMSE(L, Lmle)
	
	print('Estimated:')
	print('MSE_{MOM} = %.4f' %MSEmom)
	print('MSE_{MLE} = %.4f' %MSEmle)
	print('Theoretical:')
	print('MSE_{MOM} = %.4f' %(L**2 / (3. * n)))
	print('MSE_{MLE} = %.4f' %(L**2 * 2. / ((n + 1) * (n + 2))))
	return

if __name__ == '__main__':
	main()