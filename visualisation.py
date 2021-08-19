#visualisation methods for probe objects (in the following probe)
import numpy as np # 1.21.1
import seaborn as sns # 41.4.0
import matplotlib.pyplot as plt # 3.4.2
from matplotlib import cm # 3.4.2
import pandas as pd

import config

#plot sample_crop vs fit
def plot_crop(probe, save=False):
	#create arrays
	fit=probe.fit_crop
	sample=probe.ift_sample_crop-0*fit
	diff=sample-fit

	#replace nan with 0
	fit=np.nan_to_num(fit, nan=0)
	sample=np.nan_to_num(sample, nan=0)
	diff=np.nan_to_num(diff, nan=0)

	#separate magnitude and phase
	in_mag=np.abs(sample)
	in_phase=np.angle(sample)
	out_mag=np.abs(fit)
	out_phase=np.angle(fit)
	diff_mag=np.abs(diff)
	diff_phase=np.angle(diff)

	#define magnitude norm	
	norm=np.amax([in_mag, out_mag, diff_mag])

	fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,
		sharey=True,figsize=(15,10))		

	#define norms for all subplots
	ALL=np.abs([np.abs(sample),np.abs(fit),sample.real,
		fit.real,sample.imag,fit.imag])
	norm=np.amax(ALL)

	#plot original probe
	ax1=sns.heatmap(in_mag,ax=ax1,cmap=cm.seismic,vmin=-norm, vmax=norm)
	ax1.set_title("original probe (amplitude)")

	ax4=sns.heatmap(in_phase,ax=ax4,cmap=cm.twilight_shifted,vmin=-np.pi, vmax=np.pi)
	ax4.set_title("original probe (real)")

	#plot fit
	ax2=sns.heatmap(out_mag,ax=ax2,cmap=cm.seismic,vmin=-norm, vmax=norm)
	ax2.set_title("zernike expansion (amplitude)")

	ax5=sns.heatmap(out_phase,ax=ax5,cmap=cm.twilight_shifted,vmin=-np.pi, vmax=np.pi)
	ax5.set_title("zernike expansion (real)")

	#plot difference between original and fit
	ax3=sns.heatmap(diff_mag,ax=ax3,cmap=cm.seismic,vmin=-norm, vmax=norm)
	ax3.set_title("original-zernike (amplitude)")

	ax6=sns.heatmap(diff_phase,ax=ax6,cmap=cm.twilight_shifted,vmin=-np.pi, vmax=np.pi)
	ax6.set_title("original-zernike (real)")

	
	fig.suptitle("comparison of probe and expansion")
	fig.tight_layout()

	if save:
		plt.savefig(f"probe_crop_N_{probe.N}_shift_{probe.SHIFT}.png")
	else:
		plt.show()
	plt.close()

#plot full sample vs fit
def plot_full(probe, save=False):
	in_mag=np.abs(probe.sample)
	in_phase=np.angle(probe.sample)
	out_mag=np.abs(probe.ft_fit)
	out_phase=np.angle(probe.ft_fit)

	#define magnitude norm	
	norm=np.amax([in_mag, out_mag])
	
	fig, ((ax1,ax3),(ax2,ax4))=plt.subplots(2,2,
			sharey=True,figsize=(10,10))
	
	#plot original probe
	#mag
	ax1=sns.heatmap(in_mag,ax=ax1,cmap=cm.seismic,vmin=-norm, vmax=norm)
	ax1.set_title("original probe (amplitude)")
	#phase
	ax2=sns.heatmap(in_phase,ax=ax2,cmap=cm.twilight_shifted,vmin=-np.pi, vmax=np.pi)
	ax2.set_title("original probe (phase)")

	#plot fitted probe
	#mag
	ax3=sns.heatmap(out_mag,ax=ax3,cmap=cm.seismic,vmin=-norm, vmax=norm)
	ax3.set_title("fitted probe (amplitude)")
	#phase
	ax4=sns.heatmap(out_phase,ax=ax4,cmap=cm.twilight_shifted,vmin=-np.pi, vmax=np.pi)
	ax4.set_title("fitted probe (phase)")

	fig.suptitle("comparison of probe and expansion")
	fig.tight_layout()
	
	if save:
		plt.savefig(f"probe_full_N_{probe.N}_shift_{probe.SHIFT}.png")
	else:
		plt.show()
	plt.close()

#visualise coefficients of probe obj
def visual_coeff(probe, save=False):
	coeff=probe.coeffs
	with plt.style.context('seaborn'):
		fig, (ax1,ax2,ax3)=plt.subplots(3,1, sharey=True,sharex=True)
		ax1.scatter(range(len(coeff)),coeff.real)
		ax2.scatter(range(len(coeff)),coeff.imag)
		ax3.scatter(range(len(coeff)),np.abs(coeff))

		ax1.set_ylabel("real part")		
		ax2.set_ylabel("imaginary part")
		ax3.set_xlabel("# of coefficient")
		ax3.set_ylabel("absolute")

		fig.tight_layout()
		if save:
			plt.savefig(f"coefficients_shift_{probe.SHIFT}.png")
		else:
			plt.show()
	plt.close()

#plot accuracy of fits for each different setting (description)
def plot_stats():
	stats=config.statistics
	idx=["N","description","shift","size"]
	vals=stats.drop(idx,axis=1).columns
	stats=stats.melt(id_vars=idx, value_vars=vals)
	#normalise square deviation by window size for comparability
	stats["value"]=stats["value"]/stats["size"]**2
	stats=stats.rename(columns={"variable":"order","value":"square deviation"})
	
	sns.catplot(data=stats, x="order", y="square deviation", hue="description", kind="swarm")
	plt.savefig("stats.png")
	plt.close()