# to generate the right hand side of equation B25 Dal corso
# that is p, equation B29
# is a hack may not work for other positions
import numpy as np
from qtm.pseudo.loc2 import loc_generate_pot_rhocore
from qtm.pseudo.nloc import NonlocGenerator
from qtm.gspace.gkspc import GkSpace
from qtm.containers import FieldGType, FieldRType, get_FieldG
from qtm.dft.kswfn import *

def dvqpsi_nl(crystal, gspace, kpoints, wfn, mode, qpoint=[0,0,0]):

    nat = np.sum([sp.numatoms for sp in crystal.l_atoms])

    # change this to work with all species

    # print("Number of k points", len(wfn))
    numk = len(wfn)
    # print("Number of bands", wfn[0][0].numbnd)
    num_bands = wfn[0][0].numbnd

    # call for each k point
    # wavefn ( list( kpoint, 0, ) , band , gk)

    dynmat = 0 # placeholder
    for i in range(numk):
        kpoint = wfn[i][0].k_cryst
        k_weight =  wfn[i][0].k_weight
        temp_wfn = wfn[i][0]
        dynmat = per_k(crystal, gspace, wfn[i][0].gkspc,  kpoint,qpoint, temp_wfn, num_bands, k_weight, mode)
    return dynmat

def per_k(crystal, gspace, gkspace, kpoint, qpoint, cwfn, num_bands, k_weight, mode):
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
    fac = 1
    l_atoms=crystal.l_atoms
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    gktau = coords_cart_all.T @ gkspace.gk_cart

    wfn = cwfn.evc_gk.data
    # wfn2 = np.loadtxt("wfn.txt", dtype=complex)
    # cwfn.evc_gk.data[:,gkspace.idxsort] = wfn2
    # wfn = cwfn.evc_gk.data


    # end boilerplate
    vkb, dij, vkb_diag = nlpot.gen_vkb_dij(gkspace)
    # XXX for some reason vkb has oppsite sign in quantum espresso
    vkb = vkb.data
    num_beta = nlpot.numvkb

    bbeta = np.zeros((num_bands, nat, num_beta), dtype=complex)
    for band in range(num_bands):
        for atom in range(nat):
            for k in range(num_beta):
                bbeta[band][atom][k] += np.sum(wfn[band]*np.conj(vkb[k + num_beta*atom]))

    aalpha = np.zeros((num_bands, nat, 3, num_beta), dtype=complex)
    for band in range(num_bands):
        for atom in range(nat):
            for alpha_dir in range(3):
                for k in range(num_beta):
                    aalpha[band][atom][alpha_dir][k] += np.sum(wfn[band] * np.conj(vkb[k + num_beta*atom]) *\
                            1j * gkspace.gk_cart[alpha_dir]  )


    # print("size of wfn", wfn.shape)
    dvloc = get_FieldG(gspace).zeros( dtype=complex)
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    gtau = coords_cart_all.T @ gspace.g_cart

    num_typ=len(l_atoms)
    labels=np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    numg = gspace.size_g
    v_loc=np.zeros((num_typ, numg), dtype=complex)
    for isp in range(num_typ):
        v_loc[isp]=loc_generate_pot_rhocore(l_atoms[isp], gspace)[0].data/np.prod(gspace.grid_shape)
    for atom in range(nat):
        for ig in range(gspace.g_cart.shape[1]):
            dvloc.data[ig] += -1j* \
                                (   gspace.g_cart[0][ig]*mode[atom][0] +\
                                    gspace.g_cart[1][ig]*mode[atom][1] +\
                                    gspace.g_cart[2][ig]*mode[atom][2] ) *\
                                np.exp(-1j*gtau[atom][ig]) * v_loc[labels[atom],ig]

    dvloc_r = dvloc.to_r()
    wfn_r = cwfn.evc_gk.to_r()
    # print(dvloc_r.data)

    dvloc_out = KSWfn(gkspace, k_weight, int(num_bands), is_noncolin=False)
    dvloc_out_r = dvloc_out.evc_gk.to_r()
    # print(wfn_r.data[0])

    for band in range(num_bands):
            dvloc_out_r.data[band] = dvloc_r.data * wfn_r.data[band]
    # np.savetxt("dvlocr.txt", 2*dvloc_out_r.data[0]*np.prod(gspace.grid_shape))

    dvloc_out.evc_gk.data[:,:] = dvloc_out_r.to_g().data

    # non local part

    for band in range(num_bands):
        for atom in range(nat):
            for ikb in range(num_beta):
                for jkb in range(num_beta):
                    for alpha_dir in range(3):
                        dvloc_out.evc_gk.data[band,:] += dij[ikb + num_beta*atom][jkb + num_beta*atom]* \
                                                        aalpha[band, atom, alpha_dir, jkb] * \
                                                        vkb[ikb + atom*num_beta] *mode[atom][alpha_dir]
    # ps1 = np.zeros(num_beta*nat, num_bands)
    for band in range(num_bands):
        for atom in range(nat):
            for alpha_dir in range(3):
                for ikb in range(num_beta):
                    for jkb in range(num_beta):
                        dvloc_out.evc_gk.data[band,:] += (-1j)*dij[ikb + num_beta*atom][jkb + num_beta*atom]* \
                                                        bbeta[band, atom, jkb] * \
                                                        vkb[ikb + atom*num_beta] * \
                                                        gkspace.gk_cart[alpha_dir] *\
                                                        mode[atom][alpha_dir]


    return dvloc_out
