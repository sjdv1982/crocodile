    for proto in all_proto:
        p = candpool[proto]
        prototype = prototypes[proto]
        p_target_conformer = p["target_conformer"]
        p_source_conformer = p["source_conformer"]
        p_source_mat = p["source_mat"]
        p_target_mat = p["target_mat"]
        all_tconfs = np.unique(p_target_conformer)
        proto_tconf_align = {}
        for tconf in all_tconfs:
            mat = superimpose(lib.coordinates[tconf], prototype)[0]
            proto_tconf_align[tconf] = Rotation.from_matrix(mat)

        cand_tconf_align = np.array(
            [proto_tconf_align[conf] for conf in p_target_conformer]
        )

        all_sconfs = np.unique(p_source_conformer)
        proto_sconf_align = {}
        for sconf in all_sconfs:
            proto_sconf_align[sconf] = proto_align[proto, sconf]

        cand_sconf_align = np.array(
            [proto_sconf_align[conf] for conf in p_source_conformer]
        )
        print(
            proto,
            len(p_source_conformer),
            cand_sconf_align.shape,
            cand_tconf_align.shape,
        )
        source_rotvec_align = np.array(
            [
                (r * proto_sconf_align[conf]).as_rotvec()
                for r, conf in zip(
                    Rotation.from_matrix(p_source_mat), p_source_conformer
                )
            ]
        )
        target_rotvec_align = np.array(
            [
                (r * proto_tconf_align[conf]).as_rotvec()
                for r, conf in zip(
                    Rotation.from_matrix(p_target_mat), p_target_conformer
                )
            ]
        )
        print(source_rotvec_align.shape, target_rotvec_align)
        """
            rr = csource_rotaconformers_align * cluster.inv()
            ax = rr.as_rotvec()
            ang = np.linalg.norm(ax, axis=1)
            ang = np.maximum(ang, 0.0001)
            ax /= ang[:, None]
            fac = (np.cos(ang) - 1) ** 2 + np.sin(ang) ** 2
            cross = (scalevec * scalevec) * (1 - ax * ax)
            curr_rmsd = np.sqrt(fac * cross.sum(axis=-1))
            rmsd[n, :] = curr_rmsd
        """
