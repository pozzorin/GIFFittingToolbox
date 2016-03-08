import numpy as np

globaldict = {}
for i in range(55000, 100000):
    print i
    gid = str('%05d' % i)
    try:
        a=np.load("cellparams_"+gid+".npy")
        globaldict[i] = a
    except:
        print "No gid", gid

np.save("GIFparametersHBP", globaldict)


