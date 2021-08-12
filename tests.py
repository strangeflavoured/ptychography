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
		for i in range(end):
			#fit up to order i
			p=Probe(sample, N=i, tolerance=tolerance)
			#square difference: absolute square of the difference between
			# fit and sample (within zernike radius)
			square_difference=np.absolute(p.ft_fit-p.sample)**2
			square_difference=np.nan_to_num(square_difference, nan=0)

			#collect square difference and coefficients
			sdiff.append(np.sum(square_difference))
		deviations[str(shift)]=sdiff

	#plot square deviation
	with plt.style.context('seaborn'):
		fig,ax=plt.subplots(1,1)
		for key in deviations.keys():
			ax.scatter(range(end),deviations[key], label=f"shift {key}")
		ax.set_xlabel("number of polynomials")
		ax.set_ylabel("square deviation")
		ax.legend()
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