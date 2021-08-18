#create N probes in zernike polynomials from 2D input sample
import numpy as np # 1.21.1
from zernike import CZern # 0.0.31

#imports from other files
import visualisation as vis
import tests as t

#callable attributes: sample (input, should be square), ift_sample (after ifft2), 
#	ift_sample_crop (cropped to centre for fit), N (max order)
#	nonzero (low values set to zero), nonzero_crop (cropped to centre for fit)
#	cart (zernike czern object), coeffs (ordered list of zernike coeffs)
#	fit (zernike expansion padded), fit_crop (zernike expansion),
#	ft_fit (padded expansion back-transformed), SHIFT (for shift/unshift method),
#	probe (for adorym: contains probe_mag and probe_phase)
#methods: init (zernike expansion),
#	coeff_N (returns coeffs of order N), zernike (returns (n,m) polynomial
# 	weighted with fitted coefficient),
#	shift/unshift (shift/unshift before expansion)
class Probe:
	#initialise probe object by calculating its zernike coefficients
	def __init__(self, sample, N="default", tolerance="auto", shift=2):
		#shift sample by 2 pixels off centre
		self.sample=sample
		self.SHIFT=shift
		LEN=sample.shape[0]

		#transform sample to dual space
		self.ift_sample=self.to_inverse(sample)

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
		nonzero=np.nonzero(sample_nonzero)
		for i in range(len(nonzero[0])):
			k.append(nonzero[0][i])
			k.append(nonzero[1][i])
		k=np.array(k)

		#maximum non-zero index, centered
		max_index=np.amax(np.abs(k-LEN//2))

		#submatrix 
		minIdx=-max_index+LEN//2-LEN%2
		maxIdx=max_index+LEN//2+1+LEN%2
		self.minIdx=minIdx
		self.maxIdx=maxIdx
		self.ift_sample_crop=self.ift_sample[minIdx:maxIdx, minIdx:maxIdx]
		self.nonzero_crop=sample_nonzero[minIdx:maxIdx, minIdx:maxIdx]
		
		#expand sample using Zernike polynomials with radius k_max
		#up to radial order N
		if N=="auto":
			N=int(self.ift_sample_crop.shape[0]//2)
		elif N=="default" or not N:
			N=8
		self.N=N

		#create CZern object for zernike polynomials
		cart=CZern(N)
		L,K=self.ift_sample_crop.shape
		ddx=np.linspace(-1,1,K)
		ddy=np.linspace(-1,1,L)
		xv,yv=np.meshgrid(ddx,ddy)
		cart.make_cart_grid(xv,yv)
		self.cart=cart

		#calculate coeffs using lstsq method
		self.coeffs=self.cart.fit_cart_grid(self.ift_sample_crop)[0]
		#other returns from fit_cart_grid: residuals, rank,
		# singular values (see numpy lstsq)

		#save fitted zernike polynomial
		self.fit_crop=self.cart.eval_grid(self.coeffs, matrix=True)
		self.fit=self.pad(self.fit_crop)
		self.ft_fit=self.to_real(self.fit)

		#generate probe object for ADORYM
		shape=(len(self.coeffs),self.sample.shape[0],self.sample.shape[1])
		probe_mag=np.zeros(shape)
		probe_phase=np.zeros(shape)
		#for each mode, get padded expansion, fft2 and unshift
		for mode, coeff in enumerate(self.coeffs):
			#transform mode to real space
			mode_pad=self.pad(self.zernike(mode))
			mode_pad=self.to_real(mode_pad)
			#get magnitude and phase
			probe_mag[mode]=np.abs(mode_pad)
			probe_phase[mode]=np.angle(mode_pad)
		self.probe=[probe_mag, probe_phase]

	##### METHODS #####
	#transform input to inverse space
	def to_inverse(self, sample):
		sample=np.fft.fftshift(sample)
		sample=self.shift(sample)
		sample=np.fft.ifft2(sample)
		return np.fft.ifftshift(sample)

	#transform output to real space
	def to_real(self, sample):
		sample=np.fft.ifftshift(sample)
		sample=self.unshift(sample)
		sample=np.fft.fft2(sample)
		return np.fft.fftshift(sample)

	#shift 2d array before ifft2
	def shift(self, array):
		return np.roll(array, (self.SHIFT,self.SHIFT))

	#shift array back before fft2
	def unshift(self, array):
		return np.roll(array, (-self.SHIFT,-self.SHIFT))

	#pad cropped arrays with 0 to input shape
	def pad(self, cropped):
		pad=np.zeros(self.sample.shape, dtype=cropped.dtype)
		maxIdx=self.maxIdx
		minIdx=self.minIdx
		pad[minIdx:maxIdx, minIdx:maxIdx]=np.nan_to_num(cropped, nan=0)
		return pad	

	#return coeffs of polynomials order N
	def coeff_N(self, N):
		begin=int(N*(N+1)/2)
		end=int((N+1)*(N+2)/2)
		return self.coeffs[begin:end]

	#returns zernike polynomial n,m in shape of sample_crop
	#if only n is provided, total coefficient number is assumed
	# n>=0, 0<=m<=n
	def zernike(self,n,m=None):
		#make sure valid n and m are provided		
		if m is None:
			#if only n is provided
			idx=int(n)
		else:
			if n>self.N:
				n=self.N
				print(f"n can't be larger than N, setting n={self.N}")
			if m>n:
				m=n
				print(f"m can't be larger than n, setting m={n}")
			idx=int(n*(n+1)/2)+m
			
		#create coeff array with the corresponding coeff from fit
		coeffs=np.zeros(len(self.coeffs),dtype="complex")
		coeffs[idx]=self.coeffs[idx]

		#make polynomial matrix
		poly=self.cart.eval_grid(coeffs, matrix=True)
		
		return np.nan_to_num(poly, nan=0)



####### RUN #######
if __name__=="__main__":

	import pickle as pkl
	import pandas as pd

	with open("probe_example.pkl","rb") as file:
		sample=pkl.load(file)
	probe_complex=sample["probe_re"]+1j*sample["probe_im"]
	
	#fit and plot expansion for sample
	#probe=Probe(probe_complex)
	#vis.visual_coeff(probe,save=True)
	#vis.plot_crop(probe,save=True)	
	#vis.plot_full(probe,save=True)

	#test accuracy for different numbers of polynomials
	# and different shift
	t.test_fit(probe_complex,10, shiftrange=0,save=True)#slow
	#t.plot_shifts(probe_complex,save=True)