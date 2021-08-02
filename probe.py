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
		maxIdx=max_index+LEN+1
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
	def test(self, save=False):		
		fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9))=plt.subplots(3,3,sharey=True,figsize=(15,15))

		#plot original probe
		ax1=sns.heatmap(np.abs(self.probe),ax=ax1)
		ax1.set_title("original probe (amplitude)")

		ax4=sns.heatmap(self.probe.real,ax=ax4)
		ax4.set_title("original probe (real)")

		ax7=sns.heatmap(self.probe.imag,ax=ax7)
		ax7.set_title("original probe (imag)")

		#plot fitted zernike polynomial
		fit=self.cart.eval_grid(self.coeffs, matrix=True)
		fit=np.nan_to_num(fit, nan=0)

		ax2=sns.heatmap(np.abs(fit),ax=ax2)
		ax2.set_title("zernike expansion (amplitude)")

		ax5=sns.heatmap(fit.real,ax=ax5)
		ax5.set_title("zernike expansion (real)")

		ax8=sns.heatmap(fit.imag,ax=ax8)
		ax8.set_title("zernike expansion (imag)")

		#plot difference between original and fit
		diff=self.probe-fit
		ax3=sns.heatmap(np.abs(diff),ax=ax3)
		ax3.set_title("original-zernike (amplitude)")

		ax6=sns.heatmap(diff.real,ax=ax6)
		ax6.set_title("original-zernike (real)")

		ax9=sns.heatmap(diff.imag,ax=ax9)
		ax9.set_title("original-zernike (imag)")

		fig.suptitle("comparison of probe and expansion")
		fig.tight_layout()

		if save:
			plt.savefig(f"probe_test_N_{N}.png")
		else:
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
	tolerance=4e-2
	probe=Probe(sample["probe_aperture"], N, tolerance)
	probe.test("save")
	print(probe.coeffs)