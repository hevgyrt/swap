import numpy as np
import logging


class S_ds():

    def __init__(self,f,E, d, deepwater=True):
        self.f = f
        self.E = E
        self.d = d
        self.g = 9.81 # [m/s**2]
        self.sigma = 2*np.pi*f
        if deepwater:
            self.k = self.sigma**2 / self.g
            self.cg = 0.5*(self.g/self.sigma)
        else:
            logging.error('Intermediate waters are not included yet')

    def komen_1984(self, n=0, C_wc = 2.36e-5):
        """ Standard whitecapping parametrization from Hasselmann (1974)

        Args:
            n: add wave number dependance
            C_wc: white capping coefficient

            NOTE: Standard WAM Cyckle III formulations for n and C_wc are
            implemented as default. For WAM Cyckle IV, it is
                C_wc = 4.10e-5 for WAM Cycle IV
                n    = 0.5 for WAM Cyckle IV
        Returns:
            S_wc in frequency form
        """
        k = self.k
        sigma = self.sigma
        d_sigma = sigma[1]-sigma[0]
        E = self.E
        tp = 2*np.pi
        E_sigma = E/tp


        # tunable coefficients
        s_tilde_pm = np.sqrt(3.02e-3)
        p = 4

        m0 = np.sum(E_sigma)*d_sigma

        sigma_tilde = ( (1/m0) * np.sum(E_sigma/sigma)*d_sigma )**(-1)
        k_tilde = ( (1/m0) * np.sum(E_sigma/np.sqrt(k))*d_sigma )**(-2)
        #print("k_tilde: {}".format(k_tilde))
        #print("sigma_tilde: {}".format(sigma_tilde))

        s_tilde = k_tilde*np.sqrt(m0)

        mu = C_wc*((1-n) + n*(k/k_tilde))*((s_tilde/s_tilde_pm)**p)*(sigma_tilde/k_tilde)
        #mu = C_wc*((s_tilde/s_tilde_pm)**p)*(sigma_tilde/k_tilde)
        S_wc_sigma = -mu*k*E_sigma
        S_wc_f = S_wc_sigma*(2*np.pi) #Jacobian from sigma to frequency

        return S_wc_f


    def westhuyjsen_2007(self):
        """ Westhuyjsen (2007)
        NOTE: Only implemented unidirectional for deepwater

        Args:

        Returns:
            S_wc in frequency form

        """
        f = self.f
        d_f = f[1]-f[0]
        k = self.k
        sigma = self.sigma
        d_sigma = sigma[1]-sigma[0]
        g = self.g
        cg = self.cg
        E = self.E
        tp = 2*np.pi
        E_sigma = E/tp


        # Defining some constants
        C_ds = 5e-5 # see p. 157 in paper
        Br = 1.75e-3 # see p. 157 in paper
        p = 4 #NOTE: depends on wave age (bottom p.155). Optimally computed in eq. (11)

        #kk = k.reshape(k.size,1)

        cg_k = cg*(k**3)
        #ccg_kk =cg_k.reshape(cg_k.size,1)

        # Computing B(k)
        #b_k = ccg_kk*E_sigma
        b_k = cg_k*E_sigma

        # Compute threshold saturation
        b_ratio = b_k/Br
        b_ratio.mask[b_ratio<=1] = True


        # broadcast numpy
        S_wc_sigma = -C_ds*((b_k/Br)**(0.5*p))*np.sqrt(g*k)*(E_sigma)
        S_wc_sigma.mask = b_ratio.mask
        S_wc_f = S_wc_sigma*(2*np.pi) #Jacobian from sigma to frequency

        # Integrated s_wc
        s_wc_integrated = np.sum(S_wc_f,axis=0)*d_f

        return S_wc_f

    def westhuyjsen_2012(self):
        return 0

    def rogers_2012(self, a1, a2, L, M, E_generic_type, f1_idx, return_T1_T2=False, return_delta_f=False):
        """ implemented from Rogers (2012)

        Args:
            a1, a2, L, M: parameters (see Table 1 in Rogers (2012))
            f1_idx: frequency idx where breaking starts
            E_generic: "var" / "thres"

        Returns:
            T2(f,theta) or T2(f) in frequency form

        """
        f = self.f
        df = f[1]-f[0]
        k = self.k
        g = self.g
        cg = self.cg
        E = self.E

        B_nt = 0.035**2 #Babanin et al. (2007)
        A_inverse = np.ones(len(f))
        A = 1/A_inverse

        E_t_f = (A_inverse)*(2*np.pi*B_nt)/(cg*(k**3))

        delta_f = np.ma.masked_array(E-E_t_f)
        delta_f.mask = False
        delta_f.mask[delta_f.data<=0] = True # Waves only breaks if they overshoot the threshold

        if E_generic_type == "var":
            E_generic = E
        elif E_generic_type == "thres":
            E_generic = E_t_f
        else:
            raise Exception('E_generic_type must be either "var" or "thres"')


        T1 = self.T1(a1=a1, A=A, delta_f=delta_f, E_generic=E_generic, L=L)
        T2 = self.T2(a2=a2, A=A, f1_idx=f1_idx, delta_f=delta_f, E_generic=E_generic, M=M)

        S_ds = T1 + T2

        if not return_T1_T2:
            return S_ds
        else:
            if not return_delta_f:
                return S_ds, T1, T2
            else:
                return S_ds, T1, T2, delta_f




    def T1(self,a1, A, delta_f, E_generic, L):
        """ implemented from Rogers (2012)
        NOTE: E_f_theta can also be E_f

        Args:
            a1: parameter (see Table 1 in Rogers (2012))
            A: Inverse spectrum narrowness. Set to 1
            delta_f: difference threshold and spectrum. NOTE: Should be masked
                array since waves below threshold does not break
            E_generic: Normalization/generic spectral density
            L: parameter (see Table 1 in Rogers (2012))

        Returns:
            T1(f,theta) or T1(f) in frequency form

        """
        E_f_theta = self.E


        f = self.f

        T1 = a1 * A * f * (delta_f/E_generic)**L * E_f_theta
        return T1


    def T2(self, a2, A, f1_idx, delta_f, E_generic, M):
        """ implemented from Rogers (2012)
        NOTE: E_f_theta can also be E_f
        NOTE: This is a cummulative effect, thus not local in frequency!

        Args:
            a2: parameter (see Table 1 in Rogers (2012))
            A: Inverse spectrum narrowness. Set to 1
            f1_idx: frequency idx where breaking starts
            delta_f: difference threshold and spectrum. NOTE: Should be masked
                array since waves below threshold does not break
            E_generic: Normalization/generic spectral density
            M: parameter (see Table 1 in Rogers (2012))

        Returns:
            T2(f,theta) or T2(f) in frequency form

        """
        E_f_theta = self.E
        f = self.f
        df = f[1]-f[0]

        integral = np.zeros(len(delta_f))
        for f2 in range(f1_idx,len(delta_f)):
            integral[f2] = np.ma.sum( A[f1_idx:f2] * ((delta_f[f1_idx:f2]/E_generic[f1_idx:f2])**M) )*df

        T2 = np.ma.masked_array(a2 * integral * E_f_theta)
        T2.mask = False
        T2.mask = delta_f.mask

        return T2


    def rapizo_2016(self, a1, a2, a3, L, M, E_generic_type, f1_idx, U, return_T1_T2_T3=False):
        """ Rapizo et al. (2016)
        NOTE: Only implemented unidirectional for deepwater

        Args:

        Returns:
            S_ds in frequency form

        """
        f = self.f
        df = f[1]-f[0]
        k = self.k
        sigma = self.sigma
        g = self.g
        cg = self.cg
        E = self.E


        Sds_rog, T1, T2, delta_f = self.rogers_2012(a1, a2, L, M, E_generic_type, f1_idx, return_T1_T2=True, return_delta_f=True)

        current_effect = -(U*k)/sigma
        current_effect_max = np.nanmax(current_effect)
        T3 = a3*np.max([current_effect_max,0])*(delta_f)

        S_ds =(1+T3)*(T1 + T2)

        if not return_T1_T2_T3:
            return S_ds
        else:
            return S_ds, T1, T2, T3






def main():
    import matplotlib.pyplot as plt
    g = 9.81
    f = np.ma.masked_array(np.linspace(0.00001,1,501))
    df = f[1]-f[0]

    U10 = 12 #[m/s]
    Hm0_pm = (0.24*U10**2)/g
    T_peak = 7 #[s]
    fp = 1/T_peak
    #deepwater:
    #k = (2*np.pi*fp)**2 / g
    cp = g/(2*np.pi*fp)
    print("cp: {}".format(cp))

    E_pm = fully_developed_pm(U10,f)
    E_jw = JONSWAP(f,1/T_peak,U10)
    E_don = donelan(f,fp,cp,U10,implemented_type="rogers")



    m0_jw = np.sum(E_jw)*df
    Hm0_jw = 4*np.sqrt(m0_jw)
    print("Hm0_pm = {}, Hm0_jw = {}\n".format(round(Hm0_pm,3),round(Hm0_jw,3)))

    d = 50
    Sds_jw = S_ds(f,E_jw,d, deepwater=True)
    S_wc_west_jw = Sds_jw.westhuyjsen_2007()
    S_wc_kom_jw = Sds_jw.komen_1984(n=0.5,C_wc = 4.10e-5)
    S_wc_rog_jw = -Sds_jw.rogers_2012(a1=2e-4, a2=1.6e-3, L=1, M=1, E_generic_type='var',
                                                    f1_idx=1)

    # Plotting
    font_size=15
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10,8))

    ax[0].plot(f,E_don,c='tab:blue',label='Donelan')
    ax[0].plot(f,E_jw,c='tab:orange',label='Jonswap')
    ax[0].plot(f, E_pm,c='tab:green',label='Pierson Moskowitz')
    ax[0].set_ylabel(r'$H_{s} [m]$', fontsize=font_size)


    ax[1].plot(f,S_wc_west_jw,c='tab:orange',ls='--',label='westhuyjsen_2007')
    ax[1].plot(f,S_wc_kom_jw,c='tab:orange',ls='-',label='komen_1984')
    ax[1].plot(f,S_wc_rog_jw,c='tab:orange',ls=':',label='Rogers 2012')

    ax[1].set_xlabel('Frequency [Hz]', fontsize=font_size)
    ax[1].set_ylabel(r'$S_{wc} [m^2]$', fontsize=font_size)

    for aax in ax:
        aax.legend()
        aax.grid()

    plt.show()

if __name__=='__main__':
    main()
