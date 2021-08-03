#create N probes in zernike polynomials from 2D input sample

import numpy as np # 1.21.1
import seaborn as sns # 41.4.0
import matplotlib.pyplot as plt
from matplotlib import cm # 3.4.2
from zernike import CZern # 0.0.31

#callable attributes: sample (original), ft_sample (fourier transformed),
#	probe (cutout of ft_sample), max_index (boardes of probe),
#	nonzero (ft_sample low values set to zero)
#	cart (zernike czern object), coeffs 8order list of zernike coeffs
#methods: init (zernike expansion), plot (probe vs expansion),
#	coeff_N (returns coeffs of order N)
class Probe:
	#initialise probe object by calculating its zernike coefficients
	def __init__(self, sample, N="auto", tolerance="auto"):
		self.sample=sample
		LEN=sample.shape[0]

		#transform sample to dual space
		self.ift_sample=np.fft.fftshift(np.fft.ifft2(sample))

		#auto tolerance
		if tolerance=="auto":
			tolerance=np.amax(self.ift_sample)*.2

		#set values close to zero (within tolerance) to 0
		sample_nonzero=self.ift_sample.copy()
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
		max_index=np.amax(np.abs(k-LEN//2))
		self.max_index=max_index

		#submatrix 
		minIdx=-max_index+LEN//2-LEN%2
		maxIdx=max_index+LEN//2+1+LEN%2
		probe=self.ift_sample[minIdx:maxIdx, minIdx:maxIdx]
		self.probe=probe
		
		#expand sample using Zernike polynomials with radius k_max
		#up to radial order N
		if N=="auto":
			N=4*N
		self.N=N

		#create CZern object for zernike polynomials
		cart=CZern(N)
		L,K=self.probe.shape
		ddx=np.linspace(-1,1,K)
		ddy=np.linspace(-1,1,L)
		xv,yv=np.meshgrid(ddx,ddy)
		cart.make_cart_grid(xv,yv)
		self.cart=cart

		#calculate coeffs using lstsq method
		self.coeffs=self.cart.fit_cart_grid(probe)[0]
		#other returns from fit_cart_grid: residuals, rank,
		# singular values (see numpy lstsq)

		#calculate coefficients explicitly
		#coeffs=[]
		#for n in range(N+1):
		#	for m in range(n+1):
		#		coeffs.append(np.sum(self.probe*self.Zernike(n,m)))
		#self.coeffs=np.array(coeffs)

	#plot probe vs fit
	def plot(self, save=False):
		fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9))=plt.subplots(3,3,
			sharey=True,figsize=(15,15))		

		#fitted zernike polynomial
		fit=self.cart.eval_grid(self.coeffs, matrix=True)
		fit=np.nan_to_num(fit, nan=0)

		#define norms for all subplots
		ALL=np.abs([np.abs(self.probe),np.abs(fit),self.probe.real,
			fit.real,self.probe.imag,fit.imag])
		norm=np.amax(ALL)

		#plot original probe
		ax1=sns.heatmap(np.abs(self.probe),ax=ax1,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax1.set_title("original probe (amplitude)")

		ax4=sns.heatmap(self.probe.real,ax=ax4,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax4.set_title("original probe (real)")

		ax7=sns.heatmap(self.probe.imag,ax=ax7,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax7.set_title("original probe (imag)")

		#plot fit
		ax2=sns.heatmap(np.abs(fit),ax=ax2,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax2.set_title("zernike expansion (amplitude)")

		ax5=sns.heatmap(fit.real,ax=ax5,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax5.set_title("zernike expansion (real)")

		ax8=sns.heatmap(fit.imag,ax=ax8,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax8.set_title("zernike expansion (imag)")

		#plot difference between original and fit
		diff=self.probe-fit
		ax3=sns.heatmap(np.abs(diff),ax=ax3,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax3.set_title("original-zernike (amplitude)")

		ax6=sns.heatmap(diff.real,ax=ax6,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax6.set_title("original-zernike (real)")

		ax9=sns.heatmap(diff.imag,ax=ax9,cmap=cm.seismic,vmin=-norm, vmax=norm)
		ax9.set_title("original-zernike (imag)")

		fig.suptitle("comparison of probe and expansion")
		fig.tight_layout()

		if save:
			plt.savefig(f"probe_test_N_{self.N}.png")
		else:
			plt.show()

	#return coeffs of polynomials order N
	def coeff_N(self, N):
		begin=int(N*(N+1)/2)
		end=int((N+1)*(N+2)/2)
		return self.coeffs[begin:end]

	#needed for explicit calculation
	#returns zernike polynomial n,m in shape of probe
	def Zernike(self,n,m):
		cart=CZern(n)
		L,K=self.probe.shape
		ddx=np.linspace(-1,1,K)
		ddy=np.linspace(-1,1,L)
		xv,yv=np.meshgrid(ddx,ddy)
		cart.make_cart_grid(xv,yv)

		coeffs=np.zeros(int((n+1)*(n+2)/2))
		coeffs[int(n*(n+1)/2)+m]=1
		poly=cart.eval_grid(coeffs, matrix=True)
		
		return np.nan_to_num(poly, nan=0)


if __name__=="__main__":

	import pickle as pkl

	with open("probe_example.pkl","rb") as file:
		sample=pkl.load(file)
	probe_complex=sample["probe_re"]+1j*sample["probe_im"]
	
	N=50
	probe=Probe(probe_complex,N)
	probe.plot("save")	

	#print(probe.coeffs)