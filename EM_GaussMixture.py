import numpy as np
import matplotlib.pyplot as plt
import argparse,sys
from matplotlib.patches import Ellipse
#import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--DataFilePath', type=str, help='data filepath')
parser.add_argument('--K', type=int,help='clutster number')
parser.add_argument('--CovModel',type=str, help='separate or tied covariance')
parser.add_argument('--plot',help='make plots or no',action='store_true')
args=parser.parse_args()

class EM_Gauss:
	def __init__(self, k=3, thresh=0.001,maxIter=2000):
		self.k=k
		self.thresh=thresh ## stopping threshold
		self.maxIter=maxIter

	def multivar_gauss(self, mu, sig,X):
		prob = np.linalg.det(sig) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(sig) , (X - mu).T).T ) )
		return prob
	
	def fitting(self,X,valX=None, use_val=False, model='separate'):
		if model not in ['separate','tied']:
			sys.stderr.write('Invalid model indicator. Should be either separate or tied\n')		
			sys.exit()
		
		if model=='separate':
			if use_val:
				self.res=self.EM_sep(X,valX=valX,use_val=True)
			else:
				self.res=self.EM_sep(X)
		elif model=='tied':
			if use_val:
				self.res=self.EM_tie(X,valX=valX,use_val=True)
			else:
				self.res=self.EM_tie(X)
		return self.res
	
	def EM_sep(self,X,valX=None,use_val=False):
		N,dim=X.shape
		if use_val:
			valN=valX.shape[0]
			valR=np.zeros((valN,self.k))
		####################
		mus=X[np.random.choice(N,self.k,replace=False)]
		sigmas=[np.eye(dim)]*self.k
		gammas=[1./self.k]*self.k
		R=np.zeros((N,self.k))
		log_likes=[]
		val_log_likes=[]
		while len(log_likes)<self.maxIter:
			for i in range(self.k):
				prob=self.multivar_gauss(mus[i],sigmas[i],X)
				R[:,i]=gammas[i]*prob
				if use_val:
					prob_val=self.multivar_gauss(mus[i],sigmas[i],valX)
					valR[:,i]=gammas[i]*prob_val
			log_like=np.sum(np.log(np.sum(R,axis=1)))
			log_likes.append(log_like)
			R=(R.T/np.sum(R,axis=1)).T
			if use_val:
				val_log_like=np.sum(np.log(np.sum(valR,axis=1)))
				val_log_likes.append(val_log_like)
				valR=(valR.T/np.sum(valR,axis=1)).T
			##############
			num_k=np.sum(R,axis=0)
			## maximization
			for i in range(self.k):
				mus[i]=1./num_k[i]*np.sum(R[:,i]*X.T,axis=1).T
				x_mu=np.matrix(X-mus[i])
				sigmas[i]=np.array(1.0/ num_k[i] *np.dot(np.multiply(x_mu.T, R[:,i]),x_mu))
				gammas[i]=1./N*num_k[i]
			if len(log_likes)>2 and np.abs(log_like-log_likes[-2])<self.thresh:
				break
		res={'mu':mus,'sigma':sigmas,'gamma':gammas,'loglikelihood':log_likes,'val_loglikelihood':val_log_likes}
		return res

	def EM_tie(self,X,valX=None,use_val=False):
    		N,dim=X.shape
        	if use_val:
            		valN=valX.shape[0]
            		valR=np.zeros((valN,self.k))
    		####################
    		mus=X[np.random.choice(N,self.k,replace=False)]
        	sigmas=[np.eye(dim)]*self.k
		sigma=np.sum(sigmas,axis=0)
        	gammas=[1./self.k]*self.k
        	R=np.zeros((N,self.k))
        	log_likes=[]
        	val_log_likes=[]
        	while len(log_likes)<self.maxIter:
            		for i in range(self.k):
                		prob=self.multivar_gauss(mus[i],sigma,X)
                		R[:,i]=gammas[i]*prob
                		if use_val:
                    			prob_val=self.multivar_gauss(mus[i],sigma,valX)
                    			valR[:,i]=gammas[i]*prob_val
            		log_like=np.sum(np.log(np.sum(R,axis=1)))
            		log_likes.append(log_like)
            		R=(R.T/np.sum(R,axis=1)).T
            		if use_val:
                		val_log_like=np.sum(np.log(np.sum(valR,axis=1)))
                		val_log_likes.append(val_log_like)
                		valR=(valR.T/np.sum(valR,axis=1)).T
            		##############
            		num_k=np.sum(R,axis=0)
            		## maximization
            		for i in range(self.k):
                		mus[i]=1./num_k[i]*np.sum(R[:,i]*X.T,axis=1).T
                		x_mu=np.matrix(X-mus[i])
                		sigmas[i]=np.array(1.0/ num_k[i] *np.dot(np.multiply(x_mu.T, R[:,i]),x_mu))
                		gammas[i]=1./N*num_k[i]
			
			sigma=np.sum(sigmas,axis=0)/(self.k)
            		if len(log_likes)>2 and np.abs(log_like-log_likes[-2])<self.thresh:
                		break
		res={'mu':mus,'sigma':[sigma]*self.k,'gamma':gammas,'loglikelihood':log_likes,'val_loglikelihood':val_log_likes}
		return res    			
	
	def pred(self,x):
		p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu , \
                        np.dot(np.linalg.inv(s) , x - mu)))
        	probs = np.array([gamma * p(mu, s) for mu, s, gamma in zip(self.params['mu'], self.params['sigma'], self.params['gamma'])])
	        return probs/np.sum(probs)

## a function to plot the likelihood	
def plot_likelihood(likelihoods,k):
	xAxis=np.arange(len(likelihoods))
	plt.plot(xAxis, likelihoods)
	plt.title('Log likelihood vs. interations, k=%d'%(k))
	plt.xlabel('iteration')
	plt.ylabel('log likelihood')
	plt.show()

def plot_Gauss(mu, sigma, ax=None):
	def eigsorted(sigma):
		eigVal, eigVec=np.linalg.eigh(sigma)
		order= eigVal.argsort()[::-1]
		return eigVal[order],eigVec[:,order]
	if not ax:
		ax=plt.gca()
	eigVal,eigVec=eigsorted(sigma)
	theta=np.degrees(np.arctan2(*eigVec[:,0][::-1]))
	width,height=3.0*np.sqrt(abs(eigVal))
	ellip=Ellipse(xy=mu,width=width,height=height,angle=theta,alpha=0.6)
	ax.add_artist(ellip)
	return ellip

def plot_results(X, params, model):
	mus=params['mu']
	sigmas=params['sigma']
	loglikelihood=params['loglikelihood']
	val_loglikelihood=params['val_loglikelihood']
	k=len(mus)
	## plot scatters
	fig=plt.figure(figsize=(20,6))
	fig.add_subplot(131)
	plt.cla()
	plt.plot(X.T[0],X.T[1],'.')
	for i in range(k):
		plot_Gauss(mus[i],sigmas[i])
	## plot X likelihood
	fig.add_subplot(132)
	plt.plot(loglikelihood)
	plt.title('Loglikelihood vs Iter, k=%d'%(k),fontsize=20)
	plt.xlabel('iterations',fontsize=18)
	plt.ylabel('Log Likelihood',fontsize=18)
	fig.add_subplot(133)
	plt.plot(val_loglikelihood)
        plt.title('Loglikelihood vs Iter of dev data, k=%d'%(k),fontsize=20)
        plt.xlabel('iterations',fontsize=18)
        plt.ylabel('Log Likelihood',fontsize=18)
	#fig.savefig("k=%d_%s.png"%(k, model))
	plt.show()
def main():
	if args.K:
		K=args.K
	else:
		K=3
	if args.plot:
		mkPlot=args.plot
	else:
		mkPlot=False
	if args.CovModel:
		model=args.CovModel
	else:
		model="separate"
	filename=args.DataFilePath
	if not filename:
		print("None data filepath given, exit at error.")
		sys.exit()
	
	
	all_data=np.genfromtxt(filename)
	np.random.shuffle(all_data)
	N=all_data.shape[0]
	idx=int(0.9*N)
	X, val=all_data[:idx,:], all_data[idx:,:]
	#X=all_data
	print(X)
	gaussModel=EM_Gauss(k=K,thresh=0.01,maxIter=2000)		
	res=gaussModel.fitting(X,valX=val,use_val=True,model=model)
	#plot_likelihood(res['loglikelihood'],k=K)
	#plot_likelihood(res['val_loglikelihood'],k=K)
	if mkPlot:
		plot_results(X,res,model)

if __name__=='__main__':
	main()
