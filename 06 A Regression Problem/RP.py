import numpy as np

def getData(m = 200, w = 1, b = 5, s2 = 0.1):
	X = 2 * np.random.rand(m) + 100
	Y = w * X + b + np.random.normal(0, np.sqrt(s2), size = m)
	return (X, Y) #DO NOT use np.float16, because the precision does matter!

def recenter(data):
	X, Y = data
	return (X - 101, Y)

#compute w and b
def getLine(data):
	X, Y = data
	mx = np.mean(X)
	my = np.mean(Y)

	dx = X - mx
	dy = Y - my

	w = np.sum(dx * dy) / np.sum(np.square(dx))
	b = my - w * mx
	return (w, b)

#compute t times (w, b), get mean and variance
def test(t = 1000, m = 200, w = 1, b = 5, s2 = 0.1):
	W = np.empty((t, 2), dtype = np.float64)
	B = np.empty((t, 2), dtype = np.float64)
	for i in range(t):
		data = getData(m, w, b, s2)
		W[i, 0], B[i, 0] = getLine(data)
		W[i, 1], B[i, 1] = getLine(recenter(data))

	mw = np.mean(W, axis = 0)
	mb = np.mean(B, axis = 0)
	vw = np.mean(np.square(W - mw), axis = 0)
	vb = np.mean(np.square(B - mb), axis = 0)

	return (mw, mb, vw, vb)

#compute mean and variance of (w, b) in theory
def getTheory(m = 200, w = 1, b = 5, s2 = 0.1):
	mw = (w, w)
	mb = (b, b + 101)

	invVar = 3 #2**2 / 12
	factor = s2 / m * invVar
	vw = (factor, factor)
	vb = (factor * (1 / invVar + 10201), factor * (1 / invVar))
	return (mw, mb, vw, vb)


def getRes(t = 1000, m = 200, w = 1, b = 5, s2 = 0.1):
	pmw, pmb, pvw, pvb = test(t, m, w, b, s2)
	tmw, tmb, tvw, tvb = getTheory(m ,w, b, s2)

	print("E(w):")
	print('practice: %.5f, theory: %.5f' %(pmw[0], tmw[0]))
	print("E(w'):")
	print('practice: %.5f, theory: %.5f' %(pmw[1], tmw[1]))
	print("E(b):")
	print('practice: %.5f, theory: %.5f' %(pmb[0], tmb[0]))
	print("E(b'):")
	print('practice: %.5f, theory: %.5f' %(pmb[1], tmb[1]))
	print("var(w):")
	print('practice: %.5f, theory: %.5f' %(pvw[0], tvw[0]))
	print("var(w'):")
	print('practice: %.5f, theory: %.5f' %(pvw[1], tvw[1]))
	print("var(b):")
	print('practice: %.5f, theory: %.5f' %(pvb[0], tvb[0]))
	print("var(b'):")
	print('practice: %.5f, theory: %.5f' %(pvb[1], tvb[1]))
	return

if __name__ == '__main__':
	getRes()