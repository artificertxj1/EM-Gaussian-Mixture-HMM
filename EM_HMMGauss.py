import os
import sys
import numpy as np
from scipy.stats import multivariate_normal
import argparse
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

class HMMGauss():
    def __init__(self, K=3,covType="sep"):
        self.K=K        
        self.covType=covType

    def ForwardBackward(self, emissionMtx, transMtx, lamb):
	T=emissionMtx.shape[0]
	forwardAlpha=np.zeros((T, self.K))
	backwardBeta=np.zeros((T, self.K))
	forwardAlpha[0,:]=np.multiply(lamb, emissionMtx[0,:])
	backwardBeta[-1,:]=np.array([1.0]*self.K)
	ZFactor=np.zeros((T,1))
	ZFactor[0]=np.sum(forwardAlpha[0,:])
	forwardAlpha[0,:]/=ZFactor[0]
	for t in range(1,T):
		prevAlpha=forwardAlpha[t-1,:]
		for j in range(self.K):
			for i in range(self.K):
				forwardAlpha[t,j]+=prevAlpha[i]*transMtx[i,j]
			forwardAlpha[t,j]*=emissionMtx[t,j]
		ZFactor[t]=np.sum(forwardAlpha[t,:])
		forwardAlpha[t,:]/=ZFactor[t]
	
	#backwardBeta[-1,:]/=ZFactor[T-1]
	for t in range(T-2,-1,-1):
		for i in range(self.K):
			for j in range(self.K):
				backwardBeta[t,i]+=transMtx[i,j]*emissionMtx[t+1,j]*backwardBeta[t+1,j]
		backwardBeta[t,:]/=ZFactor[t]
	return forwardAlpha, backwardBeta, ZFactor


    def EM(self,X, thresh=0.01, maxIter=500, reg=0.02, random_state=True):
        T,d=X.shape
        lamb=np.array([1./self.K]*self.K)
        if random_state:
            transMtx=np.random.rand(self.K,self.K)
            for i in range(self.K):
		transMtx[i,:]/=np.sum(transMtx[i,:])
        else:
            transMtx=np.ones((self.K,self.K))/float(self.K)
        #mus=X[np.random.choice(T,self.K,replace=False),:]
	kmeans=KMeans(n_clusters=self.K, random_state=0).fit(X)
	mus=kmeans.cluster_centers_
        sigmas=[np.eye(d)]*self.K
	log_likelihood=[]
	##initialize 
	gamma=np.zeros((T,self.K))
	emissionMtx=np.zeros((T,self.K))
	#R=np.zeros((T,self.K))
	while len(log_likelihood)<maxIter:
		print("Iter number: %d"%(len(log_likelihood)))
		for i in range(self.K):
			emissionMtx[:,i]=multivariate_normal.pdf(X,mus[i],sigmas[i])
		## E step
		forwardAlpha,backwardBeta,ZFactor=self.ForwardBackward(emissionMtx, transMtx, lamb)
		#print(forwardAlpha)
		#print(backwardBeta)
		transProb_All=[]
		for t in range(T):
			normFactor=np.sum(np.multiply(forwardAlpha[t,:],backwardBeta[t,:]))
			gamma[t,:]=np.multiply(forwardAlpha[t,:],backwardBeta[t,:])/normFactor
			if t<T-1:
				transProb_t=np.zeros((self.K, self.K))
				for i in range(self.K):
					for j in range(self.K):
						transProb_t[i,j]=gamma[t,i]*transMtx[i,j]*emissionMtx[t+1,j]*backwardBeta[t+1,j]/ backwardBeta[t,i]
				transProb_All.append(transProb_t)
		##### M step
		lamb=np.array(gamma[0,:])
		transProb=np.zeros((self.K,self.K))
		for tp in transProb_All:
			transProb+=tp
		gammaSum=np.sum(gamma[0:(T-1),:],axis=0)
		for i in range(self.K):
			for j in range(self.K):
				transMtx[i,j]=transProb[i,j]/gammaSum[i]
		## renormalize transMtx
		for i in range(self.K):
	                transMtx[i,:]/=np.sum(transMtx[i,:])

		gammaSum=np.sum(gamma,axis=0)
		for i in range(self.K):
			mus[i]=np.zeros((1,d))
			for t in range(T):
				mus[i]+=gamma[t,i]*X[t,:]
			mus[i]/=gammaSum[i]
			sigmas[i]=np.zeros((d,d))
			for t in range(T):
				sigmas[i]+=gamma[t,i]*np.outer(X[t,:]-mus[i], X[t,:]-mus[i])
			sigmas[i]/=gammaSum[i]
		if self.covType=='sep':
			for i in range(self.K):
				sigmas[i]+=reg*np.eye(d)
		else:
			sigma_tied=sigmas[0]
			for i in range(1,self.K):
				sigma_tied+=sigmas[i]
			sigma_tied/=self.K
			sigma_tied+=reg*np.eye(d)
			sigmas=[sigma_tied]*self.K
		#print(mus)
		## update loglikelihood
		loglike_i=0
		for t in range(T):
			loglike_i-=np.log(1./ZFactor[t])
		log_likelihood.append(loglike_i)
		if len(log_likelihood)>2 and abs(loglike_i-log_likelihood[-2])<thresh:
			break
	
	params=namedtuple('params','mus sigmas log_likelihood')
	self.params=params(mus,sigmas, log_likelihood)
	return self.params			


def plot_Gauss(mu, sigma, ax=None):
	def eigsorted(sigma):
		eigVal, eigVec=np.linalg.eigh(sigma)
		order= eigVal.argsort()[::-1]
		return eigVal[order],eigVec[:,order]
	if not ax:
		ax=plt.gca()
	eigVal,eigVec=eigsorted(sigma)
	theta=np.degrees(np.arctan2(*eigVec[:,0][::-1]))
	width,height=1.5*np.sqrt(abs(eigVal))
	ellip=Ellipse(xy=mu,width=width,height=height,angle=theta,alpha=0.6)
	ax.add_artist(ellip)
	return ellip

def plot_results(X, params):
	mus=params.mus
	sigmas=params.sigmas
	loglikelihood=params.log_likelihood
	#val_loglikelihood=params['val_loglikelihood']
	k=len(mus)
	## plot scatters
	fig=plt.figure(figsize=(20,6))
	fig.add_subplot(121)
	plt.cla()
	plt.plot(X.T[0],X.T[1],'.')
	for i in range(k):
		plot_Gauss(mus[i],sigmas[i])
	## plot X likelihood
	fig.add_subplot(122)
	plt.plot(loglikelihood)
	plt.title('Loglikelihood vs Iter, k=%d'%(k),fontsize=20)
	plt.xlabel('iterations',fontsize=18)
	plt.ylabel('Log Likelihood',fontsize=18)
	#fig.add_subplot(133)
	#plt.plot(val_loglikelihood)
        #plt.title('Loglikelihood vs Iter of dev data, k=%d'%(k),fontsize=20)
        #plt.xlabel('iterations',fontsize=18)
        #plt.ylabel('Log Likelihood',fontsize=18)
	#fig.savefig("k=%d_%s.png"%(k, model))
	plt.show()



def main():
	K=sys.argv[1]
	covType=sys.argv[2]
	X=np.genfromtxt("./points.dat")
	#np.random.shuffle(X)
	HMM=HMMGauss(K=int(K), covType=covType)
	res=HMM.EM(X[:,:], random_state=True)
	plot_results(X,res)
if __name__=='__main__':
	main()		
	
