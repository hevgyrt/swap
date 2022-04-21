import pytest
import sys, os
import numpy as np
import matplotlib.pyplot as plt

testdir = os.path.dirname(os.getcwd() + '/')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from wave_dissipation_parametrizations import S_ds
from idealized_spectrums import fully_developed_pm, JONSWAP, donelan

#@pytest.fixture

def test_rogers2012_ex1():
    f = np.linspace(1e-5,5,1000)
    df = f[1]-f[0]
    U10 = 12 #[m/s]
    d=100 #[m] depth

    g = 9.81 # [m/s**2]
    sigma = 2*np.pi*f
    #if deepwater:
    k = sigma**2 / g
    cg = 0.5*(g/sigma)

    cp = np.array([U10 / 0.9, U10 / 3.5])

#    for j,cp in enumerate([U10 / 0.9, U10 / 3.5]):# See Rogers (2012)
    fp = g/(2*np.pi*cp)

    f_tr = 3*fp

    #E_don = donelan(f=f, fp=fp, cp=cp, U10=U10, implemented_type="rogers")
    E_don = [donelan(f=f, fp=fp[i], cp=cp[i], U10=U10, implemented_type="rogers") for i in range(len(cp))]

    idx_tr = [np.argmin(np.abs(f-f_tr[0])),np.argmin(np.abs(f-f_tr[1]))]

    E_don[0][f>f_tr[0]] = E_don[0][idx_tr[0]]*((f_tr[0]/f[f>f_tr[0]])**(5))
    E_don[1][f>f_tr[1]] = E_don[1][idx_tr[1]]*((f_tr[1]/f[f>f_tr[1]])**(5))

    hm0_ex1 = 4*np.sqrt(np.sum(E_don[0])*df)
    hm0_ex2 = 4*np.sqrt(np.sum(E_don[1])*df)

    fig, ax = plt.subplots()
    ax.plot(f/fp[0],E_don[0]*f**4,label=r'$U/c_p$={}, $H_s$={}, $f_p$={}'.format(np.round(U10/cp[0],2),np.round(hm0_ex1,2),np.round(fp[0],2)))
    ax.plot(f/fp[1],E_don[1]*f**4,label=r'$U/c_p$={}, $H_s$={}, $f_p$={}'.format(np.round(U10/cp[1],2),np.round(hm0_ex2,2),np.round(fp[1],2)))

    ax.set_xlim([0,8])
    ax.set_xlabel(r'$f/f_p$')
    ax.set_ylim([-1e-8,5e-3])
    ax.set_ylabel(r'$E(f) f^4$')
    ax.grid()
    ax.legend()
    plt.show()

    Sds_s = [S_ds(f,E_don[i],d, deepwater=True) for i in range(len(cp))]

    L = [1,2,1,4]
    M = [1,2,4,4]
    E_generic_type = ["var", "thres","thres","thres"]
    a1 = [2e-4, 8.8e-6, 5.7e-5, 5.7e-7]
    a2 = [1.6e-3, 1.1e-4, 3.2e-6, 8e-6]

    f1_idx = 3
    fig, ax = plt.subplots(2,2,sharex=True,figsize=(10,10))

    for i in range(0,len(L)):
        for j,Sds in enumerate(Sds_s):
            Sds_rog, T1, T2 = Sds.rogers_2012(a1[i], a2[i], L[i], M[i], E_generic_type[i], f1_idx, return_T1_T2=True)
            ax[1,j].plot(f/fp[j],T1,'-',label="T1:{}".format(i))
            ax[1,j].plot(f/fp[j],T2,'--', label='T2:{}'.format(i),alpha=0.7)

            ax[0,j].plot(f/fp[j],Sds_rog, label='S_ds {}'.format(i))

    ax[0,0].set_ylim([0,4e-4])
    ax[1,0].set_ylim([0,4e-4])

    ax[0,1].set_ylim([0,2e-4])
    ax[1,1].set_ylim([0,1e-4])

    for aax in ax.flatten():
        aax.set_xlim([0,6])
        aax.legend()
        aax.grid()
    plt.show()
