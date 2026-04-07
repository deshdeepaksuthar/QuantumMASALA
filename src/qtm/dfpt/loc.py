# second derivative of the ionic potential(local part)
# B32 PRB 64 235118
# shape of dynamical matrix (3*nat, 3*nat)
#
# write the equation as well
# done


import numpy as np
from qtm.pseudo.loc2 import loc_generate_pot_rhocore
from qtm.containers import FieldGType, FieldRType, get_FieldG
from qtm.constants import RYDBERG


# k independent part
# last part of B32
# check garbage folder for a faster version
# this seems q independent as well
def d2vion3(crystal, gspace, rho=None):
    """
    TODO : CHECK THE ARGUMENTS OF THE FUNCTION
    """
    alat = crystal.reallat.alat
    nat = np.sum([sp.numatoms for sp in crystal.l_atoms ])
    omega = crystal.reallat.cellvol
    tpiba2 = (2*np.pi/alat)**2
    cryst = crystal
    l_atoms=cryst.l_atoms
    tot_num = np.sum([sp.numatoms for sp in l_atoms])
    num_typ=len(l_atoms)
    labels=np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    # rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)
    rho=rho._data[0, :]/np.prod(gspace.grid_shape)

    dynmat = np.zeros((3*nat, 3*nat), dtype=complex)
    gspc = gspace
    idxsort = gspc.idxsort
    numg = gspc.size_g
    # cart_g = (gspc.g_cart.T[idxsort]).T
    # tpiba_g= (gspc.g_tpiba.T[idxsort]).T
    cart_g = gspc.g_cart
    tpiba_g= gspc.g_tpiba
    gtau = coords_cart_all.T @ cart_g
    omega=gspc.reallat_cellvol
    alat=cryst.reallat.alat

    v_loc=np.zeros((num_typ, numg), dtype=complex)
    for isp in range(num_typ):
        v_loc[isp]=loc_generate_pot_rhocore(l_atoms[isp], gspc)[0].data/np.prod(gspace.grid_shape)
    fac = 1

    for alpha in range(3):
        for beta in range(3):
            for atom in range(nat):
                for g_index in range(0, numg):
                    phase =(fac*(gtau[atom][g_index]))
                    label = labels[atom]
                    dynmat[alpha+ 3*atom][beta + 3*atom] += -(omega*tpiba2) *\
                            (v_loc[label][g_index]) *\
                            (np.real(rho[g_index])*np.cos(phase) - np.imag(rho[g_index])*np.sin(phase)) *\
                            tpiba_g[alpha][g_index] * tpiba_g[beta][g_index]
    return dynmat
