# computing dhxc
# derivative of hartree and xc potential
# this is for just one mode
import numpy as np
import scipy as sc
# from qtm.dfpt.stolen.ksham import KSHam
from qtm.containers.field import *
from qtm.gspace.gkspc import GkSpace
from qtm.dft.kswfn import *
from qtm.pot import hartree, xc
from qtm.pseudo import loc_generate_rhoatomic, loc_generate_pot_rhocore, NonlocGenerator
from qtm.symm.symmetrize_field import SymmFieldMod

def compute(crystal, gspace, rho, l_wfn, dpsi, drho=None, qpoint=[0,0,0]):
    gkspace = l_wfn[0][0].gkspc
    wfn =  l_wfn[0][0]
    eigvals = wfn.evl
    num_bands = len(wfn.evl)
    k_weight = wfn.k_weight
    grho = gspace
    gwfn = gspace
    FieldG_rho: FieldGType = get_FieldG(gspace)
    omega = crystal.reallat.cellvol
    alat = crystal.reallat.alat
    tpiba = 2*np.pi/alat
    tpiba2 = tpiba**2

    if drho is None:
        drho = get_FieldR(gspace).zeros()
    else :
        drho = drho.to_r()/np.prod(gspace.grid_shape)
        # drho.data[:] = 0
    wr = wfn.evc_gk.to_r().data
    dpsir = dpsi.evc_gk.to_r().data
    # factor = 4*gspace.reallat_dv/np.prod(gspace.grid_shape)
    k_weight = 2
    factor = 2*k_weight/omega
    # factor = 4*gspace.reallat_dv/crystal.reallat.cellvol

    # for band in range(num_bands):
    #         drho.data[:] += 4*np.real((wr[band,:].conj() *\
    #                 dpsir[band]))/np.prod(gspace.grid_shape)
    for band in range(num_bands):
            drho.data[:] +=  factor*(wr[band,:].conj() *\
                                dpsir[band])
    # drho.data[:] /= 1.3
    print("drho", np.linalg.norm(drho.data))


    dvscf = get_FieldG(gspace).zeros()
    e2 = 1.0
    drho_g = drho.to_g()

    # XXX
    # symmterizing drho
    symm = SymmFieldMod(crystal, gspace)
    # drho_g = symm.symmetrize(drho_g)

    # XXX
    # check the division again
    dvscf.data[0] = 0
    gq2 =  (gspace.g_cart[0] + qpoint[0])**2
    gq2 += (gspace.g_cart[1] + qpoint[1])**2
    gq2 += (gspace.g_cart[2] + qpoint[2])**2
    dvscf.data[1:] = (4*np.pi*e2*(1/(gq2[1:])) * drho_g.data[1:])
    # dvscf.data[1:] = 4*np.pi*e2*(1/(gq2[1:]))
    # dvscf.data[0] = np.sum(dvscf.data[:])/np.prod(gspace.grid_shape)

    # hartree done
    # getting the densitites
    # rho

    # rho should be in real space
    # had to make a matrix instead of array

    rho_g =  rho
    # rho = rho_g.to_r()/np.prod(gspace.grid_shape)

    FieldG_rho: FieldGType = get_FieldG(gspace)
    v_ion, rho_core = FieldG_rho.zeros(()), FieldG_rho.zeros(1)
    l_nloc = []
    for sp in crystal.l_atoms:
        v_ion_sp, rho_core_sp = loc_generate_pot_rhocore(sp, grho)
        v_ion += v_ion_sp
        rho_core += rho_core_sp
        l_nloc.append(NonlocGenerator(sp, gwfn))
    libxcf = xc.get_libxc_func(crystal)
    dvxcdR, en= xc.compute2(rho , rho_core, *libxcf)

    # XXX
    # temp = dvscf.to_r()/np.prod(gspace.grid_shape)
    temp = dvscf.to_r()
    # temp = get_FieldG(gspace).zeros()
    drho = drho.to_r()/np.prod(gspace.grid_shape)

    # my version
    # temp.data[:] +=  dvxcdR.data.ravel()[:] * drho.data[:]

    # temp.data[:] *= gspace.reallat_dv/np.prod(gspace.grid_shape)


    dvscf = temp.to_g()
    drho = drho.to_g()
    # dvscf = temp.to_g()/np.prod(gspace.grid_shape)
    print(" dvscf", np.linalg.norm(dvscf.data))
    return dvscf, drho
