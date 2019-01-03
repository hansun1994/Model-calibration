# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker

#%% zero-coupon rate
def zero_coupon_rate(tau, r0, kappa, theta, sigma, alpha, model):
    if model == 'Vasicek':
        b = (1-np.exp(-kappa*tau))/kappa;
        a = (theta-sigma**2/(2*(kappa**2)))*(b-tau)-((sigma**2)/(4*kappa))*(b**2)
    elif model == 'CIR':
        g = np.sqrt(kappa**2+2*(sigma**2))     
        tmp = 2*kappa*theta/(sigma**2)
        tmp1 = kappa*tau/2
        tmp2 = g*tau/2
        
        a = tmp*np.log(np.exp(tmp1)/(np.cosh(tmp2)+(kappa/g)*np.sinh(tmp2)))
        b = 2/(kappa+g*(np.cosh(g*tau/2)/np.sinh(g*tau/2)))
    elif model == 'Ho-Lee':
        u = 90 - tau
        dt = 0.1
        p5 = np.exp(-r0*90)
        p1 = np.exp(-r0*u)
        slope = (np.log(np.exp(-r0*(u+dt))) - np.log(np.exp(-r0*(u-dt))))/(2*dt)
        a = np.log(p5/p1) - tau*slope - 0.5*(sigma**2)*u*(tau**2)
        b = tau
    elif model == 'Hull-White':
        u = 90 - tau
        p5 = np.exp(-r0*90)
        p1 = np.exp(-r0*u)
        db = (1/alpha)*(1-np.exp(-alpha*tau))
        dt = 0.1
        slope = (np.log(np.exp(-r0*(u+dt))) - np.log(np.exp(-r0*(u-dt))))/(2*dt)
        a = np.log(p5/p1) - db*slope - (alpha**2)*((np.exp(-alpha*90)-np.exp(-alpha*u))**2)*(np.exp(2*alpha*u)-1)/(4*(alpha**3))
        b = db
        
    p = np.exp(a-b*r0)
    return p

def zero_coupon_swaption(tau, r0, kappa, theta, sigma, model):
    if model == 'Vasicek':
        b = (1-np.exp(-kappa*tau))/kappa;
        a = (theta-sigma**2/(2*(kappa**2)))*(b-tau)-((sigma**2)/(4*kappa))*(b**2)
    elif model == 'CIR':
        g = np.sqrt(kappa**2+2*(sigma**2))     
        tmp = 2*kappa*theta/(sigma**2)
        tmp1 = kappa*tau/2
        tmp2 = g*tau/2
        
        a = tmp*np.log(np.exp(tmp1)/(np.cosh(tmp2)+(kappa/g)*np.sinh(tmp2)))
        b = 2/(kappa+g*(np.cosh(g*tau/2)/np.sinh(g*tau/2)))
    
    unit = np.ones(len(r0))
    a_ = a.reshape(len(a),1)*(unit)
    b_ = b.reshape(len(b),1)*(unit)
    
    p = np.exp(a_-b_*r0)
    return p

#%% swaprate and libor rate function
def swap_rate(tau, p, term):
    ttemp = np.asarray([i/2 for i in range(1,61)])
    ptemp = np.interp(ttemp, tau, p)
    dis = np.cumsum(ptemp)
    pterm = np.interp(term, tau, p)
    index = 2*term-1
    s = 200*(1-pterm)/dis[index]
    
    return s

def libor_rate(tau, p, term):
    pterm = np.interp(term, tau, p)
    l = 100*(1/pterm - 1)/term
    
    return l

#%% swaption
def swaption(tau, tn, tN, param, model):
    r0 = param[0]
    kappa = param[1]
    theta = param[2]
    sigma = param[3]
    pt0 = zero_coupon_rate(tau,r0,kappa,theta,sigma,0,model)
    
    dt = 1/12
    delta = 1/2
    multiplier = 20000
    nsims = 10000
    
    n1 = len(tn)
    n2 = len(tN)
    nstep = np.zeros(n1)
    for i in range(n1):
        nstep[i] = int(tn[i]/dt)

    z = np.random.normal(0,1,(int(nstep[n1-1]),nsims))
    r = np.ones([int(nstep[n1-1])+1,nsims])*r0
    
    if model == 'Vasicek':
        for j in range(1,int(nstep[n1-1])+1):
            r[j,:] = r[j-1,:]+kappa*(theta-r[j-1,:])*dt+sigma*np.sqrt(dt)*z[j-1,:]
            r[j,r[j,:]<0] = 0.0000001
            
    elif model == 'CIR':
        for j in range(1,int(nstep[n1-1])+1):
            r[j,:] = r[j-1,:]+kappa*(theta-r[j-1,:])*dt+sigma*np.sqrt(r[j-1,:])*np.sqrt(dt)*z[j-1,:]
            r[j,r[j,:]<0] = 0.0000001
            
    bTn = np.zeros([n1,nsims])
    pTn = []
    
    for i in range(n1):
        bTn[i,:] = np.exp(-dt*sum(r[:int(nstep[i]),:]))
        pTn.append(zero_coupon_swaption(tau,r[int(nstep[i]),:],kappa,theta,sigma,model))
    
    swaption_p = np.zeros([n1,n2])
    pt0_delta = pt0[range(0,len(pt0)+1,int(delta/dt))]
    
    for i1 in range(n1):
        for i2 in range(n2):
            opt_mat = tn[i1]
            swap_mat = tN[i2]
            
            ynNt = (pt0_delta[int(opt_mat/delta)] - pt0_delta[int((opt_mat+swap_mat)/delta)])/(delta*sum(pt0_delta[int(opt_mat/delta)+1:int((opt_mat+swap_mat)/delta)+1]))
            
            pTn_delta = pTn[i1][range(0,len(pTn[i1]),int(delta/dt)),:]
            
            pnN = delta*sum(pTn_delta[1:int(swap_mat/delta),:])
            
            ynNTn = (1-pTn_delta[int(swap_mat/delta),:])/pnN
            
            tmp = (pnN/bTn[i1,:])*((ynNTn-ynNt).clip(min=0))
            
            swaption_p[i1,i2] = np.mean(tmp)*multiplier
            
    swaption_p = pd.DataFrame(swaption_p,index=tn,columns=tN)
            
    return swaption_p

#%% objective function
def objective(params, tau, LIBOR, SWAP, model):
# unpack params
    r0 = params[0]
    kappa = params[1]
    theta = params[2]
    sigma = params[3]
    alpha = params[4]

    p = zero_coupon_rate(tau, r0, kappa, theta, sigma, alpha, model)
    # now that we have zero-coupon bond prices p(t,T)
    # now it is time to calculate MODEL LIBOR rates and SWAP rates
    S = swap_rate(tau, p, np.asarray(SWAP.index))
    L = libor_rate(tau, p, np.asarray(LIBOR.index))

    # the goal is to minimize the distance between model rates and market rates
    rel1 = (S - np.asarray(SWAP)) / np.asarray(SWAP)
    rel2 = (L - np.asarray(LIBOR)) / np.asarray(LIBOR)

    #rel1 = (S-SWAP(:,2))
    #rel2 = (L-LIBOR(:,2))

    #mae = (sum(abs(rel1))+sum(abs(rel2)))
    mae = np.sum(rel1**2) + np.sum(rel2**2)
    
    return mae

#%% calibration
def calibration(fun, param_0, tau, LIBOR, SWAP, model):
     # change tolerance
    opt = {'maxiter':1000, 'maxfev':5e3}
    sol = minimize(objective, param_0, args = (tau, LIBOR, SWAP, model), method='Nelder-Mead', options=opt)
    print(sol.message)
    par = np.array(sol.x)
    print('parameters = ' + str(par))
    r_star = par[0]
    kappa_star = par[1]
    theta_star = par[2]
    sigma_star = par[3]
    alpha_star = par[4]
    param = [r_star,kappa_star,theta_star,sigma_star]
    p = zero_coupon_rate(tau, r_star, kappa_star, theta_star, sigma_star, alpha_star, model)
    L = libor_rate(tau, p, np.asarray(LIBOR.index))
    S = swap_rate(tau, p, np.asarray(SWAP.index))
    return p, L, S, sol.fun, param

#%% data loading and processing
df = pd.read_csv('F:/CU/18 fall/Computational method in finance/project/swapLiborData.csv')

for i in range(df.shape[0]):
    df.loc[i,'Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df.loc[i,'Date'],'D')
    
libor = pd.DataFrame(index=df.iloc[:,0], columns=[i/12 for i in [1,2,3,6,12]])
swap = pd.DataFrame(index=df.iloc[:,0], columns=[2,3,5,7,10,15,30])

for i in range(df.shape[0]):
    libor.iloc[i] = np.asarray(df.iloc[i,1:6])
    swap.iloc[i] = np.asarray(df.iloc[i,6:])


#%% main
tau_v = np.arange(0, 30 + 1/12, 1/12)
tau = np.arange(0, 90 + 1/12, 1/12)

tn = [1,2,5,10]
tN = [2,5,10]

param_v = [0.02,2,0.2,0.1,0.1]
param_0 = [0.25,5,0.2,0.1,0.1]

vasicek = pd.DataFrame(index=libor.index,columns=list(libor.columns)+list(swap.columns)+['error'])
cir = pd.DataFrame(index=libor.index,columns=list(libor.columns)+list(swap.columns)+['error'])
ho_lee = pd.DataFrame(index=libor.index,columns=list(libor.columns)+list(swap.columns)+['error'])
hull_white = pd.DataFrame(index=libor.index,columns=list(libor.columns)+list(swap.columns)+['error'])
swaption_vasicek = {}
swaption_cir = {}

for i in range(df.shape[0]):
    model = 'Vasicek'
    p, l, s, error, param = calibration(objective, param_v, tau_v, libor.iloc[i,:], swap.iloc[i,:], model)
    vasicek.iloc[i,:] = list(l) + list(s) + [error]
    swaption_p = swaption(tau_v,tn,tN,param,model)
    swaption_vasicek[df.iloc[i,0]] = swaption_p
    
    model = 'CIR'
    p, l, s, error, param = calibration(objective, param_v, tau_v, libor.iloc[i,:], swap.iloc[i,:], model)
    cir.iloc[i,:] = list(l) + list(s) + [error]
    swaption_p = swaption(tau_v,tn,tN,param,model)
    swaption_cir[df.iloc[i,0]] = swaption_p
    
    model = 'Ho-Lee'
    p, l, s, error, param = calibration(objective, param_v, tau, libor.iloc[i,:], swap.iloc[i,:], model)
    ho_lee.iloc[i,:] = list(l) + list(s) + [error]
    
    model = 'Hull-White'
    p, l, s, error, param = calibration(objective, param_v, tau, libor.iloc[i,:], swap.iloc[i,:], model)
    hull_white.iloc[i,:] = list(l) + list(s) + [error]
    

#%% error surface while fixing parameters


#%% plot
plt.figure(figsize=(20,42))

plt.subplot(3,1,1)
plt.subplots_adjust(hspace=0.4)
plt.plot(vasicek.index, vasicek.iloc[:,0])
plt.plot(vasicek.index, cir.iloc[:,0])
plt.plot(vasicek.index, ho_lee.iloc[:,0])
plt.plot(vasicek.index, hull_white.iloc[:,0])
plt.plot(vasicek.index, libor.iloc[:,0])
plt.xlabel('Date')
plt.ylabel('L(t,T)')
plt.legend(['Vasicek', 'CIR', 'Ho-Lee', 'Hull-White', 'Actual'])
plt.title('Model results for 1/12 LIBOR on different date ')

plt.subplot(3,1,2)
plt.plot(vasicek.index, vasicek.iloc[:,5])
plt.plot(vasicek.index, cir.iloc[:,5])
plt.plot(vasicek.index, ho_lee.iloc[:,5])
plt.plot(vasicek.index, hull_white.iloc[:,5])
plt.plot(vasicek.index, swap.iloc[:,0])
plt.xlabel('Date')
plt.ylabel('S(t,T)')
plt.legend(['Vasicek', 'CIR', 'Ho-Lee', 'Hull-White', 'Actual'])
plt.title('Model results for 2-year SWAP on different date ')

plt.subplot(3,1,3)
plt.plot(vasicek.index, vasicek['error'])
plt.plot(vasicek.index, cir['error'])
plt.plot(vasicek.index, ho_lee['error'])
plt.plot(vasicek.index, hull_white['error'])
plt.xlabel('Date')
plt.ylabel('SSRE')
plt.legend(['Vasicek', 'CIR', 'Ho-Lee','Hull-White'])
plt.title('Error for different model')

plt.show()

#%% plot
i = df.shape[0]-1
model = 'Vasicek'
p_v, l_v, s_v, error, param = calibration(objective, param_v, tau_v, libor.iloc[i,:], swap.iloc[i,:], model)

model = 'CIR'
p_c, l_c, s_c, error, param = calibration(objective, param_v, tau_v, libor.iloc[i,:], swap.iloc[i,:], model)

model = 'Ho-Lee'
p_ho, l_ho, s_ho, error, param = calibration(objective, param_v, tau, libor.iloc[i,:], swap.iloc[i,:], model)

model = 'Hull-White'
p_hu, l_hu, s_hu, error, param = calibration(objective, param_v, tau, libor.iloc[i,:], swap.iloc[i,:], model)


plt.figure(figsize=(20,42))

plt.subplot(3,1,1)
plt.subplots_adjust(hspace=0.4)
plt.plot(np.arange(0,60.1,1/6), p_v)
plt.plot(np.arange(0,60.1,1/6), p_c)
plt.plot(np.arange(0,60.1,1/6), p_ho[:361])
plt.plot(np.arange(0,60.1,1/6), p_hu[:361])
plt.xlabel('Time to maturity')
plt.ylabel('Zero-coupon prices')
plt.legend(['Vasicek', 'CIR', 'Ho-Lee', 'Hull-White'])

plt.subplot(3,1,2)
plt.plot(libor.columns, libor.iloc[-1,:])
plt.plot(libor.columns, l_v,'o')
plt.plot(libor.columns, l_c,'o')
plt.plot(libor.columns, l_ho,'o')
plt.plot(libor.columns, l_hu,'o')
plt.xlabel('maturity')
plt.ylabel('L(t,T')
plt.legend(['market','Vasicek', 'CIR', 'Ho-Lee', 'Hull-White'])

plt.subplot(3,1,3)
plt.plot(swap.columns, swap.iloc[-1,:])
plt.plot(swap.columns, s_v,'o')
plt.plot(swap.columns, s_c,'o')
plt.plot(swap.columns, s_ho,'o')
plt.plot(swap.columns, s_hu,'o')
plt.xlabel('swap term')
plt.ylabel('S(t,T)')
plt.legend(['market','Vasicek', 'CIR', 'Ho-Lee', 'Hull-White'])

plt.show()