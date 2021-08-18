# Methods to test and visualise accuracy of probe fits
import numpy as np # 1.21.1
import seaborn as sns # 41.4.0
import matplotlib.pyplot as plt # 3.4.2
import pandas as pd # 1.3.1

#imoprt from main
from probe import Probe

#plot accuracy of fit for different orders and shifts
def test_fit(sample,end,tolerance="auto",shiftrange=4,shiftstep=2,save=False):
	deviations_crop={}
	deviations_full={}
	for shift in range(0,shiftrange+1, shiftstep):		
		sdiff_crop=[]
		sdiff_full=[]
		#fit for various N and collect square difference
		for i in range(end+1):
			#fit up to order i
			p=Probe(sample, N=i, tolerance=tolerance)
			#square difference: absolute square of the difference between

			# full fit and sample (within zernike radius)
			square_difference=np.absolute(p.ft_fit-p.sample)**2
			square_difference=np.nan_to_num(square_difference, nan=0)

			#collect square difference and coefficients
			sdiff_full.append(np.sum(square_difference))

			#crop fit and sample
			square_difference=np.absolute(p.fit_crop-p.ift_sample_crop)**2
			square_difference=np.nan_to_num(square_difference, nan=0)

			#collect square difference and coefficients
			sdiff_crop.append(np.sum(square_difference))
			print(i, np.sum(square_difference))
		
		#collect square dev for shift	
		deviations_crop[str(shift)]=sdiff_crop
		deviations_full[str(shift)]=sdiff_full

	#plot square deviation
	with plt.style.context('seaborn'):
		fig,(ax1,ax2)=plt.subplots(2,1)
		for key in deviations_crop.keys():
			ax1.scatter(range(end+1),deviations_crop[key], label=f"shift {key}")
			ax2.scatter(range(end+1),deviations_full[key])

		ax1.set_title("fit only (dual space)")
		ax1.set_ylabel("square deviation")

		ax2.set_title("whole probe (real space)")
		ax2.set_xlabel("number of polynomials")		
		ax2.set_ylabel("square deviation")
		#ax.set_yscale("log")
		ax1.legend()
		if save:
			plt.savefig("test_accuracy.png")			
		else:
			plt.show()
	plt.close()

#plot coefficients for different shifts
def plot_shifts(sample,shiftrange=6,shiftstep=2,save=False):
	data=pd.DataFrame({"coeff":[],"abs":[],"shift":[]})	
	for shift in range(0,shiftrange+1,shiftstep):
		p=Probe(sample, shift=shift)
		d=np.abs(p.coeffs)
		df=pd.DataFrame({"coeff":range(len(d)),"abs":d,"shift":np.repeat(shift, len(d))})
		data=data.append(df,ignore_index=True)
		
	sns.relplot(x="coeff",y="abs",kind="line", hue="shift", data=data)
	if save:
		plt.savefig("coefficients_shift.png")
	else:
		plt.show()
	plt.close()