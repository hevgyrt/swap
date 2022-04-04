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
    ex1=True
    f = np.linspace(1e-5,5,1000)
    df = f[1]-f[0]
    U10 = 12 #[m/s]
    d=100 #[m] depth

    g = 9.81 # [m/s**2]
    sigma = 2*np.pi*f
    #if deepwater:
    k = sigma**2 / g
    cg = 0.5*(g/sigma)

    if ex1:
        cp = U10 / 0.9 # See Rogers (2012)
        fp = g/(2*np.pi*cp)
    else:
        cp = U10 / 3.5 # See Rogers (2012)
        fp = g/(2*np.pi*cp)

    f_tr = 3*fp

    E_don = donelan(f=f, fp=fp, cp=cp, U10=U10, implemented_type="rogers")
    E_don[f>f_tr]*=(f_tr/f[f>f_tr])**(5)

    plt.plot()
    plt.plot(f/fp,E_don*f**4)

    plt.xlim([0,8])
    plt.ylim([-1e-8,5e-3])
    plt.grid()
    plt.show()

    Sds = S_ds(f,E_don,d, deepwater=True)

    L = [1,2,1,4]
    M = [1,2,4,4]
    E_generic_type = ["var", "thres","thres","thres"]
    a1 = [2e-4, 8.8e-6, 5.7e-5, 5.7e-7]
    a2 = [1.6e-3, 1.1e-4, 3.2e-6, 8e-6]

    f1_idx = 3
    fig, ax = plt.subplots(1,2,figsize=(16,5))

    for i in range(0,len(L)):
        Sds_rog, T1, T2 = Sds.rogers_2012(a1[i], a2[i], L[i], M[i], E_generic_type[i], f1_idx, return_T1_T2=True)
        ax[0].plot(f/fp,T1,'-',label="T1:{}".format(i))
        ax[0].plot(f/fp,T2,'--', label='T2:{}'.format(i),alpha=0.7)

        ax[1].plot(f/fp,Sds_rog, label='S_ds {}'.format(i))

    ax[0].set_ylim([0,4e-4])
    ax[1].set_ylim([0,4e-4])

    for aax in ax:
        aax.set_xlim([0,6])
        aax.legend()
        aax.grid()
    plt.show()
