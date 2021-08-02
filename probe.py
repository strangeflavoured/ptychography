#create N probes in zernike polynomials from 2D input sample

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from zernike import CZern

class Probe:
	#initialise probe object by calculating its zernike coefficients
	def __init__(self, sample, N, tolerance):
		self.len=sample.shape[0]//2
		LEN=sample.shape[0]//2
		#transform sample to dual space
		self.ft_sample=np.fft.fftshift(np.fft.fft2(sample))

		#set values close to zero (within tolerance) to 0
		sample_nonzero=self.ft_sample.copy()
		for ix, iy in np.ndindex(sample_nonzero.shape):
			if np.absolute(sample_nonzero[ix,iy])<tolerance:
				sample_nonzero[ix,iy]=0
		self.nonzero=sample_nonzero

		#find bandwidth limit k_max
		#get array of indices with significantly non-zero value
		k=[]
		nonzero=np.nonzero(self.nonzero)
		for i in range(len(nonzero[0])):
			k.append(nonzero[0][i])
			k.append(nonzero[1][i])
		k=np.array(k)
		#maximum non-zero index, centered
		max_index=np.amax(np.abs(k-LEN))
		self.max_index=max_index

		#submatrix
		minIdx=-max_index+LEN
		maxIdx=max_index+LEN
		probe=self.ft_sample[minIdx:maxIdx, minIdx:maxIdx]
		self.probe=probe
		
		#expand sample using Zernike polynomials with radius k_max
		#up to radial order N
		#from example in https://github.com/jacopoantonello/zernike
		cart=CZern(N)
		L,K=probe.shape
		ddx=np.linspace(-1,1,K)
		ddy=np.linspace(-1,1,L)
		xv,yv=np.meshgrid(ddx,ddy)
		cart.make_cart_grid(xv,yv)
		self.cart=cart

		self.coeffs=self.cart.fit_cart_grid(probe)[0]
		#other returns from fit_cart_grid: residuals, rank, singular values (see numpy lstsq)

	#plot probe vs fit
	def test(self):		
		fig, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(18,5))

		#plot original probe
		ax1=sns.heatmap(np.abs(self.probe),ax=ax1)

		#plot fitted zernike polynomial
		fit=self.cart.eval_grid(self.coeffs, matrix=True)
		fit=np.nan_to_num(fit, nan=0)

		ax2=sns.heatmap(np.abs(fit),ax=ax2)

		#plot difference between original and fit
		diff=self.probe-fit
		ax3=sns.heatmap(np.abs(diff),ax=ax3)

		fig.tight_layout()
		plt.show()

	#return coeffs of polynomials order N
	def coeff_N(self, N):
		begin=int(N*(N-1)/2)
		end=int(N*(N+1)/2+1)
		return self.coeffs[begin:end]


if __name__=="__main__":

	import pickle as pkl

	with open("probe_example.pkl","rb") as file:
		sample=pkl.load(file)
	N=10
	tolerance=1
	probe=Probe(sample["probe_re"]+sample["probe_im"]*1j, N, tolerance)
	probe.test()
	print(probe.coeffs)