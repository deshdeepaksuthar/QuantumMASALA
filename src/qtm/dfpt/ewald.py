"""
Reviews of Modern Physics Vol.73 No.2 equation B2 - Baroni
computes the second derivative of the Energy contribution due to ewald summation
(ion-ion interaction)

 should work for all q-points

"""
import scipy.special
import numpy as np
from qtm.lattice import ReciLattice
from qtm.crystal import Crystal
from qtm.gspace import GSpace
# alpha = 2.8


def ew_fc(crystal: Crystal, gspace: GSpace, qpoint, error_thr=1e-7) -> np.ndarray:
    """
    Ewald contribution the force constant matrix for a certain qpoint

    Input:
        crystal
        gspace
        Optional:
        error_threshold: default 1e-7

    Output:
        ndarray(nat*3,nat*3)
    """
    gcryst = gspace.g_cryst[:, :]
    gcart = gspace.g_cart[:, :]
    gg = gspace.g_norm2[:]

    l_atoms = crystal.l_atoms
    reallat = crystal.reallat
    alat = reallat.alat
    vol = reallat.cellvol

    latvec = np.array(reallat.axes_alat)
    recilat = ReciLattice.from_reallat(reallat=reallat)
    valence_all = np.repeat([sp.ppdata.valence for sp in l_atoms], [sp.numatoms for sp in l_atoms])

    # concatenated version of coordinate arrays where ith column represents the coordinate of ith atom.
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    coords_cryst_all = np.concatenate([sp.r_cryst for sp in l_atoms], axis=1)
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])


    # getting the error bounds TODO: write the error formula
    alpha = 2.9

    # XXX get from qtm.constants
    pi = np.pi
    tpi = 2 * pi
    tpiba2 = (tpi / alat)**2
    tpiba = np.sqrt(tpiba2)
    atomlist = l_atoms
    # electronic charge square
    e2 = 1.0
    # need to make the 2-d arrays 1-d in order to stack them
    gkk = gg
    gkx = gcart[0]
    gky = gcart[1]
    gkz = gcart[2]


    gkk = gkk.reshape(len(gkk))
    gkx = gkx.reshape(len(gkx))
    gky = gky.reshape(len(gky))
    gkz = gkz.reshape(len(gkz))
    gkstack = np.vstack((gkx, gky, gkz))

    if (len(atomlist) != 0):
        number_of_atoms = tot_atom
        Ewald = np.zeros((3, 3, number_of_atoms, number_of_atoms), dtype=complex)

        xyzlist = coords_cart_all.T    # position list of  atoms in real space

        gcutm = gspace.ecut
        atomic_charge = np.zeros(number_of_atoms)
        atomic_charge = valence_all

        '''choosing alpha, routine copied from qe
        note: alpha is the eta from RMP 73, 2, 2001'''

        alpha = 2.9
        upperbound = 1
        # XXX a new check introduced so that alpha does go to zero
        while upperbound >= 1.0e-9 and alpha >0.1:
        # while upperbound >= 1.0e-12 :
####### empirical formula for error in sum over G
            alpha -= 0.1
            upperbound = 2.0 * (atomic_charge.sum())**2 * np.sqrt(2.0 * alpha / tpi)\
                * scipy.special.erfc(np.sqrt(tpiba2 * gcutm / 4.0 / alpha))
        print("alpha used for ewald sum is", alpha)

########Compute G-space sums here, first term
        Ewald1 = np.zeros((3, 3, number_of_atoms, number_of_atoms), dtype=complex)
        gtq2 = ((gkx + qpoint[0])**2 + (gky + qpoint[1])**2 + (gkz + qpoint[2])**2)
#        gtq2 = np.sort(gtq2)
        # XXX another hard coded value
        gflag = abs(gtq2) > 1.0e-8

        facq = np.zeros(len(gtq2), dtype=complex)
        #print(qpoint)
        facq[gflag] = -e2 * 4.0 * pi / vol *\
            np.exp(-gtq2[gflag] / alpha / 4.0) / gtq2[gflag]
        for at_a in range(number_of_atoms):
            tau_a = xyzlist[at_a]
            for at_b in range(number_of_atoms):
                tau_b = xyzlist[at_b]
                argq = ((gkx + qpoint[0]) * (tau_a[0] - tau_b[0]) +
                        (gky + qpoint[1]) * (tau_a[1] - tau_b[1]) +
                        (gkz + qpoint[2]) * (tau_a[2] - tau_b[2]))
                facg = facq * atomic_charge[at_a] * \
                    atomic_charge[at_b] * np.exp(1j * argq)
                for alph_dirn in range(3):
                    for beta_dirn in range(3):
                        Ewald1[alph_dirn,
                               beta_dirn,
                               at_a, at_b] = (facg * ((gkstack[alph_dirn]
                                                       + qpoint[alph_dirn]) *
                                                      (gkstack[beta_dirn]
                                                       + qpoint[beta_dirn]))).sum()
        # Compute the second term
        # note that all our g-space terms have tpiba already accounted for

        Ewald2 = np.zeros((3, 3, number_of_atoms, number_of_atoms), dtype=complex)
        gt2 = gkk
        # XXX do not use hard coded value for the tolerance
        gflag2 = abs(gt2) > 1.0e-8
        fac = np.zeros(len(gt2), dtype=complex)
        fac[gflag2] = -e2 * 4.0 * pi / vol * \
            np.exp(- gt2[gflag2] / alpha / 4.0) / gt2[gflag2]
        for at_a in range(number_of_atoms):
            tau_a = xyzlist[at_a]
            fnat = 0  + 0j
            for at_b in range(number_of_atoms):
                tau_b = xyzlist[at_b]
                arg = (gkx * (tau_a[0] - tau_b[0]) +
                       gky * (tau_a[1] - tau_b[1]) +
                       gkz * (tau_a[2] - tau_b[2]))
                facg = fac * atomic_charge[at_a] * atomic_charge[at_b] * np.cos(arg)
                fnat = fnat + facg
            for alph_dirn in range(3):
                for beta_dirn in range(3):
                    Ewald2[alph_dirn,
                           beta_dirn,
                           at_a, at_a] = (fnat * (gkstack[alph_dirn] *
                                                  gkstack[beta_dirn])).sum()

        # XXX change this as well, should not be hard coded
        mxr = 100
        # Compute the real space part
        Ewald3 = np.zeros((3, 3, number_of_atoms, number_of_atoms), dtype=complex)
        rmax = 5.0 / np.sqrt(alpha) / alat
        # C = latvec *alat
        C = latvec
        CI = np.linalg.inv(C).T
        xyzlist = coords_cryst_all.T
        for at_a in range(number_of_atoms):
            for at_b in range(number_of_atoms):
                dtau = (xyzlist[at_a] - xyzlist[at_b])
                ''' now generate the Rs to sum over'''
                ds = np.dot(dtau, CI.T)
                ds = ds - np.round(ds)
                dtau_0 = np.dot(ds, C)
                #
                # estimate the max values of needed integer indices
                nm1 = int(np.linalg.norm(CI[:, 0]) * rmax) + 2
                nm2 = int(np.linalg.norm(CI[:, 1]) * rmax) + 2
                nm3 = int(np.linalg.norm(CI[:, 2]) * rmax) + 2
                r = np.zeros((3, mxr), dtype=float)
                t = np.zeros(3, dtype=float)
                r2 = np.zeros(mxr, dtype=float)
                nrm = 0
                for i in range(-nm1 -1, nm1 ):
                    for j in range(-nm2 -1, nm2 ):
                        for k in range(-nm3 -1, nm3 ):
                            tt = 0.0
                            for pol in range(3):
                                t[pol] = i * C[pol, 0] + j * \
                                    C[pol, 1] + k * C[pol, 2] - dtau_0[pol]
                                tt += t[pol] ** 2
                            if tt <= rmax**2 and abs(tt) > 1.0e-10:
                                nrm += 1
                                if nrm > mxr:
                                    sys.exit("error in l 130, ewald.py")
                                r[:, nrm] = t
                                r2[nrm] = tt
                # truncate and sort the r vectors now ( horrible hacks! TOOD: cleanup!)
                mask2 = np.ones(mxr, dtype=bool)
                delarr = np.arange(nrm + 1, mxr)
                mask2[nrm + 1:] = False
                if not (r2[nrm + 1:] == 0).all():
                    print("The R-space array is longer than expected! (ewald)")
                r = np.delete(r, delarr, axis=1)
                r2 = r2[mask2]
                indsor = r2.argsort()
                r = r[:, indsor]
                r2 = r2[indsor]
                zeros = np.where(r2 == 0)
                r2 = np.delete(r2, zeros[0])
                r = np.delete(r, zeros[0], axis=1)
                rr = np.sqrt(r2) * alat
                dtau = dtau * alat
                ar = np.sqrt(alpha) * rr
                qrg = np.dot(qpoint, ((r * alat).T + dtau.T).T)
                d2f = (3.0 * scipy.special.erfc(ar) + np.sqrt(8.0 / tpi) * ar *
                       (3.0 + 2.0 * ar**2) * np.exp(-ar**2)) / rr**5
                df = (-scipy.special.erfc(ar) - np.sqrt(8.0 / tpi) * ar
                      * np.exp(-ar**2)) / rr**3
                for alph_dirn in range(3):
                    for beta_dirn in range(3):
                        Ewald3[alph_dirn,
                               beta_dirn,
                               at_a, at_b] += (e2 * atomic_charge[at_a] *
                                               atomic_charge[at_b] * np.exp((1j) * qrg) *
                                               (d2f * alat * r[alph_dirn]
                                                * alat * r[beta_dirn])).sum()
                        Ewald3[alph_dirn,
                               beta_dirn,
                               at_a, at_a] -= (e2 * atomic_charge[at_a] *
                                               atomic_charge[at_b] *
                                               (d2f * alat *
                                                r[alph_dirn])
                                               * alat * r[beta_dirn]).sum()
                    Ewald3[alph_dirn,
                           alph_dirn,
                           at_a, at_b] += (e2 * atomic_charge[at_a] *
                                           atomic_charge[at_b] * np.exp((1j) * qrg) * df).sum()
                    Ewald3[alph_dirn,
                           alph_dirn,
                           at_a, at_a] -= (e2 * atomic_charge[at_a] * atomic_charge[at_b] * df).sum()

    Ewald = -(Ewald1 - Ewald2 + Ewald3)
    temp = np.zeros((3*number_of_atoms, 3*number_of_atoms), dtype=complex)
    for i1 in range(3):
        for i2 in range(3):
            for j1 in range(number_of_atoms):
                for j2 in range(number_of_atoms):
                    temp[j1*3 + i1][j2*3 + i2] = Ewald[i1][i2][j1][j2]

    return temp
