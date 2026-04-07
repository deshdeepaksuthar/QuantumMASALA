import numpy as np
import scipy as sc
from qtm.dft.ksham import KSHam
from qtm.containers.field import *
from qtm.gspace.gkspc import GkSpace
from qtm.dft.kswfn import *
from qtm.pot import hartree, xc
from qtm.pseudo import loc_generate_rhoatomic, loc_generate_pot_rhocore, NonlocGenerator

# start and initialize the dpsi
# we have dvpsi
# for first step, dpsi is 0

# for a single k point only
def solve(crystal, gspace, rho, l_wfn, right):


    gkspace = l_wfn[0][0].gkspc
    wfn =  l_wfn[0][0]
    eigvals = wfn.evl
    num_bands = len(wfn.evl)
    k_weight = wfn.k_weight
    grho = gspace
    gwfn = gspace
    FieldG_rho: FieldGType = get_FieldG(gspace)
    v_ion, rho_core = FieldG_rho.zeros(()), FieldG_rho.zeros(1)
    l_nloc = []
    for sp in crystal.l_atoms:
        v_ion_sp, rho_core_sp = loc_generate_pot_rhocore(sp, grho)
        v_ion += v_ion_sp
        rho_core += rho_core_sp
        l_nloc.append(NonlocGenerator(sp, gwfn))
    v_ion = v_ion.to_r()

    libxc_func = xc.get_libxc_func(crystal)
    # print(libxc_func)
    v_hart, en_hartree = hartree.compute(rho.to_g())
    v_xc, en_xc = xc.compute(rho.to_g(), rho_core, *libxc_func)
    vloc = v_ion + v_hart + v_xc
    vloc /= np.prod(gwfn.grid_shape)
    vloc_g0 = np.sum(vloc, axis=-1)
    v_ion_g0 = np.sum(v_ion) / np.prod(grho.grid_shape)
    vloc_g0 = np.sum(vloc, axis=-1) / np.prod(grho.grid_shape)
    # from qtm.dfpt.stolen.ksham import KSHam
    ksham = KSHam( gkspace, False, vloc, l_nloc)

    # print(eigvals)
    out = KSWfn(gkspace, k_weight,  int(num_bands), is_noncolin=False)
    gkvec = gkspace.gk_cart

    import scipy as sc
    shape = out.evc_gk.data.shape[1]

    # since metals
    P = np.zeros((shape,shape), dtype=complex)

    for band in range(num_bands):
        for i in range(shape):
            for j in range(shape):
                P[i][j] += wfn.evc_gk.data[band][i]*np.conj(wfn.evc_gk.data[band][j])

    dpsi_out  = KSWfn(gkspace, k_weight, int(num_bands), is_noncolin=False)
    for band in range(num_bands):
        def temp_fun(inp):
            psi = KSWfn(gkspace, k_weight,   1, is_noncolin=False)
            hpsi = KSWfn(gkspace, k_weight,  1, is_noncolin=False)
            psi.evc_gk.data[:] = inp
            ksham.h_psi(psi.evc_gk, hpsi.evc_gk)

            hpsi.evc_gk.data[:] -= eigvals[band]*psi.evc_gk.data[:]
            return hpsi.evc_gk.data
        H_prec = np.eye(shape, dtype=complex)*eigvals[band]
        eprec = np.sum(wfn.evc_gk.data[band] *np.conj( wfn.evc_gk.data[band]) *eigvals[band] *2*1.35)
        H_prec = H_prec/np.maximum(1, (np.linalg.norm(gkvec, axis=0)**2)/eprec)

        H = sc.sparse.linalg.LinearOperator((shape, shape), temp_fun)


        # no precursor

        x, k = sc.sparse.linalg.cg(H, P@right.evc_gk.data[band] - right.evc_gk.data[band] , atol=1e-15)
        # x, k = sc.sparse.linalg.cg(H, -P@right.evc_gk.data[band], M=H_prec , atol=1e-20)
        if(k != 0):
            print ("CONNOT CONVErGE THE CONJUgATE GRAIenT")
            return 1
        dpsi_out.evc_gk.data[band,:] = np.copy(x)
    # print("converged cg")

    return dpsi_out
