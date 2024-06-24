import numpy as np
from sklearn.datasets import load_iris
from grad import numerical_gradient
from utils import layer_dist_plotting

# toy example - two layer neural network (flexing backprops skills be like)

# ------------ loading dataset of your choice!  -------------
X,y=load_iris(return_X_y=True)
# normalization
X=(X-X.mean(axis=0))/X.std(axis=0)
N,D,C=X.shape[0],X.shape[1],3

# ------------ params and hyperparams initialization -----------
H,mb,lr,lam=100,50,5e-4,0
lr_decay=.999
num_iters,print_every=1000,100
nmb=N//mb

X_rand=np.random.normal(loc=0,scale=1e-2, size=(N,D))
y_rand=np.random.randint(C, size=N)

W1,b1=np.random.normal(loc=0, scale=2/np.sqrt(H), size=(D,H)),np.zeros(H)
W2,b2=np.random.normal(loc=0, scale=1/np.sqrt(C), size=(H,C)),np.zeros(C)
gamma,beta=np.ones((H,)),np.zeros((H,))

# ----------------  forward pass as function for gradient check  -----------------
def forward(x,yy,grad=False,activations=False,**kwargs):
    W1,b1,W2,b2=kwargs['W1'],kwargs['b1'],kwargs['W2'],kwargs['b2'],
    mb=x.shape[0]
    aff=x@W1+b1 # NxH
    out=aff * (aff > 0)
    hout=out@W2+b2 # NxC
    # softmax w/ NxC scores
    S=np.exp(hout - hout.max(axis=1, keepdims=True))
    S/=S.sum(axis=1,keepdims=True)
    ll=-np.log(S[np.arange(mb), yy]).sum() / mb
    if activations:
        return [aff,hout]

    if not grad:
        return ll

    # backwards
    S[np.arange(mb),yy]-=1
    dhout=S/mb # notice: NxC
    dW2,db2=out.T@dhout,dhout.sum(axis=0)
    dout=dhout@W2.T
    daff = dout * (aff > 0)
    dW1,db1=x.T@daff,daff.sum(axis=0)
    dx=dout@W1.T # for funsies
    return ll,(dW1,db1,dW2,db2)

# ------------------ numerical gradient check (important!)  -------------
print("Gradient check...")
kwargs = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
num_grads = numerical_gradient(forward, X,y,**kwargs)
_,analytic_grads=forward(X,y,grad=True,activations=False,**kwargs)

for i,k in enumerate(list(kwargs.keys())):
    nrm=np.linalg.norm(num_grads[i]-analytic_grads[i])
    print("\t%s gradient difference norm: %f" % (k, nrm))


# -----------------  plotting distribution of values in activation  --------------
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--plot_activations', action='store_true', help="plot the activation distributions of the layers")
args=parser.parse_args()

# plotting layer activations to see output
if args.plot_activations:
    layer_dist_plotting(forward(X_rand,y_rand,activations=True,**kwargs),show=show_plot)

# ----------------  forward & backward pass! -----------------
for it in range(num_iters):
    acc,loss=0,0
    idx=np.random.choice(N,size=N,replace=False)
    for i in range(nmb):
        # sample mini-batch
        x,yy=X[idx[mb*(i):mb*(i+1)]],y[idx[mb*(i):mb*(i+1)]]

        # affine + relu  w/ minibatch - x,yy (mb := minibatch size)
        aff=x@W1+b1 # NxH

        # batchnorm
        mean,std=aff.mean(0,keepdims=True),aff.std(0,keepdims=True)
        aff=gamma*(aff-mean)/std+beta

        
        # plug and play favorite activation function here
        #out=aff*(aff > 0) # RELU 
        #oo=1/(1+np.exp(-aff)) # SILU - store for grad
        #out=aff*oo # SILU

        out=aff*(aff>0)+(np.exp(aff)-1)*(aff<0) # ELU

        hout=out@W2+b2 # NxC
        # softmax w/ NxC scores
        S=np.exp(hout - hout.max(axis=1, keepdims=True))
        S/=S.sum(axis=1,keepdims=True)
        ll=-np.log(S[np.arange(mb), yy]).sum()/mb + 1/2*((W1**2).sum()+(W2**2).sum())

        # predictions
        preds=S.argmax(axis=1)
        acc+=(preds==yy).mean()
        loss+=ll

        # backwards
        S[np.arange(mb),yy]-=1
        dhout=S/mb # notice: NxC
        dW2,db2=out.T@dhout,dhout.sum(axis=0)
        dout=dhout@W2.T

        #daff = dout*(aff > 0) # SILU grad
        #daff = dout*(oo*(1-oo)*aff+oo) # SILU grad

        daff = dout*((aff>0)+(np.exp(aff))*(aff<0)) # ELU

        # batchnorm backwards
        dbeta=daff.sum(0)
        dgamma=(aff*daff).sum(0)
        daff=gamma*1/std*(daff-1/mb*(aff*dgamma+dbeta))

        dW1,db1=x.T@daff,daff.sum(axis=0)
        dx=dout@W1.T # for funsies get gradient w/ respect to x (deepdream soon??)
        # weight decay - optional
        dW1,dW2=dW1+lam*W1,dW2+lam*W2

        # sgd descent
        W1,b1=W1-lr*dW1, b1-lr*db1
        W2,b2=W2-lr*dW2, b2-lr*db2
        gamma,beta=gamma-lr*dgamma,beta-lr*dbeta

    lr=lr_decay*lr # learning rate decay - optional
    acc,loss=acc/nmb,loss/nmb 
    if it % print_every==0:
        print("Iteration %d: loss {%f}, acc {%f}" % (it,loss,acc))

