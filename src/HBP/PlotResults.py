import matplotlib.pyplot as plt
import numpy as np

import cPickle as pickle


results = pickle.load(open( "/Users/christianpozzorini/Desktop/HBP_modelparam.pkl", "r" ))

def appendClamped_smaller(mylist, value, clamp):
    
    if value < clamp :
        mylist.append(clamp)
    else :
        mylist.append(value)


def appendClamped_larger(mylist, value, clamp):
    
    if value > clamp :
        mylist.append(clamp)
    else :
        mylist.append(value)



L_test = []
L_train = []
V_test = []
V_train = []
V_exp = []

DV = []

gamma_a_1 = []


problem_cnt = 0

for r in results :
    
    if r['fit_problem'] == False :
            
        appendClamped_smaller(L_test, r['likelihood_testset'], -10.0)
        appendClamped_smaller(L_train, r['likelihood_trainingset'], -10.0)  
          
        appendClamped_smaller(V_test, r['pct_var_explained_testset'], -1.0)
        appendClamped_smaller(V_train, r['pct_var_explained_trainingset'], -1.0)    
        
        appendClamped_smaller(V_exp, r['pct_var_explained_changeduetoexpassumption'], -100.0)  
             
        appendClamped_larger(DV, r['model']['DV'], 10.0)  
        
        appendClamped_larger(gamma_a_1, r['model']['gamma'][2][0], 100.0)          

    else :
        
        problem_cnt 
        
        
        
plt.figure(facecolor='white', figsize=(14,10))

# plot likelihood
plt.subplot(3,3,1)
plt.hist(L_test, bins=13, range=(-10,10), color='red', histtype='stepfilled')
plt.hist(L_train, bins=13, range=(-10,10), color='black', histtype='step',lw=3)
plt.xlabel('Likelihood (bits/spike)')

# plot pct var explained
plt.subplot(3,3,2)
plt.hist(V_test, bins=13,range=(-1,100), color='red', histtype='stepfilled')
plt.hist(V_train, bins=13, range=(-1,100), color='black', histtype='step',lw=3)
plt.xlabel('Var Explained (%)')

# plot pct var explained drop due to exp 
plt.subplot(3,3,3)
plt.hist(V_exp, bins=25,range=(-100,100), color='red', histtype='stepfilled')
plt.xlabel('Var Explained Change due to Exp(%)')


# plot pct var explained drop due to exp 
plt.subplot(3,3,4)
plt.hist(DV, bins=25,range=(0,10), color='red', histtype='stepfilled')
plt.xlabel('DV (mV)')

# plot pct var explained drop due to exp 
plt.subplot(3,3,5)
plt.hist(gamma_a_1, bins=25,range=(-100,100), color='red', histtype='stepfilled')
plt.xlabel('Gamma a1 (mV)')



plt.show()
    

    