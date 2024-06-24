import numpy as np

def numerical_gradient(f, *args,**kwargs):
    h=1e-6
    # assume w1
    grads = []
    for k,v in kwargs.items():
        grad = np.zeros_like(v)
        it=np.nditer(v, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            idx=it.multi_index
            v_p,v_m=v.copy(),v.copy()
            v_p[idx]+=h
            v_m[idx]-=h

            kwargs[k]=v_p
            vf_p=f(*args,**kwargs)
            kwargs[k]=v_m
            vf_m=f(*args,**kwargs)
            gg=(vf_p - vf_m) / (2*h)
            grad[idx]=gg
            it.iternext()
        grads.append(grad)
    return grads
