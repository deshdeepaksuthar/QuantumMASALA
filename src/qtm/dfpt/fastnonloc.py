# XXX works perfectly fine I am just stupid but only at one crystal
# XXX check the fermi function for metals
# XXX should be q-point independent for ONCV

# the rest of contribution(non local) of the Norm Conserving pseudo potential
# to the dynamical matrix
# Equation B32 except the last part

# gspace is grid in G space
# need D, beta (from projectors)
# beta  = beta(k,v,sigma, s,m) = sum over G ( c_(k+G; v;sigma)  beta*(m) (k+ G) e^(i(k+G_.tau_s)
# k = kpoint
# v is band index
# s is atom
# m is index for beta projectors
# c is the fourier component of wavefunction
# sigma is spin index fo rband
# alpha =
# gamma =

import numpy as np
from qtm.pseudo.nloc import NonlocGenerator
from qtm.gspace.gkspc import GkSpace
def d2vion2(crystal, gspace, qpoint=[0,0,0], wfn=None):

    nat = np.sum([sp.numatoms for sp in crystal.l_atoms])

    # change this to work with all species

    # print("Number of k points", len(wfn))
    numk = len(wfn)
    # print("Number of bands", wfn[0][0].numbnd)
    num_bands = wfn[0][0].numbnd

    # call for each k point
    # wavefn ( list( kpoint, 0, ) , band , gk)

    dynmat = np.zeros((nat*3, nat*3), dtype=complex)
    for i in range(numk):
        kpoint = wfn[i][0].k_cryst
        k_weight =  wfn[i][0].k_weight
        temp_wfn = wfn[i][0].evc_gk.data
        temp = per_k(crystal, gspace, wfn[i][0].gkspc,  kpoint,qpoint, temp_wfn, num_bands)*k_weight
        dynmat += temp
    return dynmat

def per_k(crystal, gspace, gkspace, kpoint, qpoint, wfn, num_bands):
    # boilerplate
    nat = np.sum([sp.numatoms for sp in crystal.l_atoms])
    alat = crystal.reallat.alat
    tpiba = 2*np.pi/alat
    tpiba2 = tpiba**2
    # XXX Multi species
    sp = crystal.l_atoms[0]
    nlpot = NonlocGenerator(sp, gspace)
    # print(kpoint)
    # gkspace = GkSpace(gspace, kpoint)
    gk_tpiba = gkspace.gk_tpiba
    fac = 2*np.pi
    l_atoms=crystal.l_atoms
    coords_cart_all = np.concatenate([sp.r_cryst for sp in l_atoms], axis=1)
    gktau = coords_cart_all.T @ gkspace.gk_tpiba



    dynmat = np.zeros((nat*3, nat*3), dtype=complex)

    # end boilerplate
    vkb, dij, vkb_diag = nlpot.gen_vkb_dij(gkspace)
    # XXX for some reason vkb has oppsite sign in quantum espresso
    vkb = vkb.data
    num_beta = nlpot.numvkb

    # print("vkbshape:", vkb.shape)
    # print("wfn:     ", wfn.shape)
    # print("gk_tpiba:   ", gk_tpiba.shape)
    # print("gktau:   ", gktau.shape)
    # generating gamma Eqn B12
    # print(gk_tpiba.T)

    gamma = np.zeros((num_bands, nat, 3, 3,  num_beta), dtype=complex)
    for band in range(num_bands):
        for atom in range(nat):
            for alpha_dir in range(3):
                for beta_dir in range(3):
                    for k in range(num_beta):
                        gamma[band][atom][alpha_dir][beta_dir][k] -= np.sum(wfn[band]*np.conj(vkb[k + num_beta*atom])*\
                                                gk_tpiba[alpha_dir] * gk_tpiba[beta_dir] * tpiba2* \
                                                np.exp(1j * fac * gktau[atom]))
    # DONE

    # generating the new beta Eqn B 8 // can be replaced with a inner product
    # becp1 in qe
    bbeta = np.zeros((num_bands, nat, num_beta), dtype=complex)
    for band in range(num_bands):
        for atom in range(nat):
            for k in range(num_beta):
                bbeta[band][atom][k] += np.sum(wfn[band]*np.conj(vkb[k + num_beta*atom])* \
                                        np.exp(1j*fac*gktau[atom]))
    # DONE


    # generating alpha Eqn B10
    # number of direction is 3, hence third place is 3
    # something is wrong, things don't come out same by changing coords

    # XXX alpha does not match with eq
    aalpha = np.zeros((num_bands, nat, 3, num_beta), dtype=complex)
    for band in range(num_bands):
        for atom in range(nat):
            for alpha_dir in range(3):
                for k in range(num_beta):
                    aalpha[band][atom][alpha_dir][k] += np.sum(wfn[band] * np.conj(vkb[k + num_beta*atom]) *\
                            1j * gk_tpiba[alpha_dir] * tpiba * \
                            np.exp(1j*fac*gktau[atom]))
    # DONE


    for alpha_dir in range(3):
        for beta_dir in range(3):
            for atom in range(nat):
                for i in range(num_bands):
                    for k1 in range(num_beta):
                        for k2 in range(num_beta):
                            dynmat[atom*3 + alpha_dir][atom*3 + beta_dir] += \
                                                    (dij[k1 + num_beta*atom][k2 + num_beta*atom] ) * \
                                                    ( gamma[i][atom][alpha_dir][beta_dir][k1].conj() * bbeta[i][atom][k2] +\
                                                      bbeta[i][atom][k1].conj() * gamma[i][atom][alpha_dir][beta_dir][k2] +\
                                                      aalpha[i][atom][alpha_dir][k1].conj() * aalpha[i][atom][beta_dir][k2] +\
                                                      aalpha[i][atom][beta_dir][k1].conj() * aalpha[i][atom][alpha_dir][k2] )



    return dynmat
