
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>

typedef double Matrix[3][3];
typedef double Vector[3];
typedef uint Cluster[4];

typedef struct HierarchyStruct
{
    const double *data;
    unsigned int shape[1];
} HierarchyStruct;

typedef struct RandomRotationsStruct
{
    const double *data;
    unsigned int shape[3];
} RandomRotationsStruct;

typedef struct ScalevecStruct
{
    const double *data;
    unsigned int shape[1];
} ScalevecStruct;

typedef struct ResultStruct
{
    double *data;
    unsigned int shape[3];
} ResultStruct;

extern "C" int transform(const HierarchyStruct *hierarchy, const RandomRotationsStruct *random_rotations, const ScalevecStruct *scalevec, ResultStruct *result);

///////////////////

inline int sign(int x)
{
    return (x > 0) - (x < 0);
}

inline double norm(const Vector &v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline void get_axis_angle(const Matrix &mat, Vector &axis, double &angle)
{
    //From https://github.com/rock-learning/pytransform3d/blob/master/pytransform3d/rotations.py"""
    double trace = mat[0][0] + mat[1][1] + mat[2][2];
    angle = acos((trace - 1.0) / 2.0);

    if (fabs(angle) < 1e-12)
    { // mat == identity
        axis[0] = 1;
        axis[1] = 0;
        axis[2] = 0;
        angle = 0;
    }
    else
    {
        /*
        # We can usually determine the rotation axis by inverting Rodrigues'
        # formula. Subtracting opposing off-diagonal elements gives us
        # 2 * sin(angle) * e,
        # where e is the normalized rotation axis.
        */
        Vector axis_unnormalized = {
            mat[2][1] - mat[1][2],
            mat[0][2] - mat[2][0],
            mat[1][0] - mat[0][1]
        };

        if (fabs(angle - M_PI) < 1e-4)
        { // trace(mat) close to -1
            /*
            # The threshold is a result from this discussion:
            # https://github.com/rock-learning/pytransform3d/issues/43
            # The standard formula becomes numerically unstable, however,
            # Rodrigues' formula reduces to mat = I + 2 (ee^T - I), with the
            # rotation axis e, that is, ee^T = 0.5 * (mat + I) and we can find the
            # squared values of the rotation axis on the diagonal of this matrix.
            # We can still use the original formula to reconstruct the signs of
            # the rotation axis correctly.
            */
            axis[0] = sqrt(0.5 * (mat[0][0] + 1.0)) * sign(axis_unnormalized[0]);
            axis[1] = sqrt(0.5 * (mat[1][1] + 1.0)) * sign(axis_unnormalized[1]);
            axis[2] = sqrt(0.5 * (mat[2][2] + 1.0)) * sign(axis_unnormalized[2]);
        }
        else
        {            
            axis[0] = axis_unnormalized[0];
            axis[1] = axis_unnormalized[1];
            axis[2] = axis_unnormalized[2];            
            /*
            # The norm of axis_unnormalized is 2.0 * np.sin(angle), that is, we
            # could normalize with a[:3] = a[:3] / (2.0 * np.sin(angle)),
            # but the following is much more precise for angles close to 0 or pi:
            */

            double axnorm = norm(axis);
            axis[0] /= axnorm;
            axis[1] /= axnorm;
            axis[2] /= axnorm;
        }
    }
}

inline void dif_matrix(const Matrix &mat1, const Matrix &mat2, Matrix &result)
{
    //mat1.dot(mat2.T)
    result[0][0] = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[0][1] + mat1[0][2] * mat2[0][2];
    result[0][1] = mat1[0][0] * mat2[1][0] + mat1[0][1] * mat2[1][1] + mat1[0][2] * mat2[1][2];
    result[0][2] = mat1[0][0] * mat2[2][0] + mat1[0][1] * mat2[2][1] + mat1[0][2] * mat2[2][2];

    result[1][0] = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[0][1] + mat1[1][2] * mat2[0][2];
    result[1][1] = mat1[1][0] * mat2[1][0] + mat1[1][1] * mat2[1][1] + mat1[1][2] * mat2[1][2];
    result[1][2] = mat1[1][0] * mat2[2][0] + mat1[1][1] * mat2[2][1] + mat1[1][2] * mat2[2][2];

    result[2][0] = mat1[2][0] * mat2[0][0] + mat1[2][1] * mat2[0][1] + mat1[2][2] * mat2[0][2];
    result[2][1] = mat1[2][0] * mat2[1][0] + mat1[2][1] * mat2[1][1] + mat1[2][2] * mat2[1][2];
    result[2][2] = mat1[2][0] * mat2[2][0] + mat1[2][1] * mat2[2][1] + mat1[2][2] * mat2[2][2];
}

inline double vecsum(const Vector &v)
{
    return v[0] + v[1] + v[2];
}

inline double get_msd(const Matrix &dmat, const Vector &scalevec)
{
    double ang;
    Vector ax;
    get_axis_angle(dmat, ax, ang);
    double x = cos(ang) - 1;
    double y = sin(ang);
    double fac = x * x + y * y;

    double cross_sum =
        scalevec[0] * scalevec[0] * (1 - ax[0] * ax[0]) +
        scalevec[1] * scalevec[1] * (1 - ax[1] * ax[1]) +
        scalevec[2] * scalevec[2] * (1 - ax[2] * ax[2]);

    double msd = fac * cross_sum;
    return msd;
}

bool is_close_to_cluster(
    const Matrix &mat, uint matnr, const Matrix *random_mats,
    const Cluster *clusters, const uint *indices,
    uint curr_cluster,
    const Vector &scalevec,
    float *cum_thresholds, uint nthresholds,
    int hierarchy_pos)
{
    const Cluster &c = clusters[curr_cluster];
    uint c_matindex = c[0];
    uint c_offset = c[1];
    uint c_size = c[2];
    if (curr_cluster == 0)
    {
        assert(hierarchy_pos == -1);
    }
    else
    {
        assert(hierarchy_pos >= 0);
        const Matrix &c_mat = random_mats[c_matindex];
        Matrix dmat;
        dif_matrix(mat, c_mat, dmat);
        double msd = get_msd(dmat, scalevec);
        //printf("MSD1 %d %.6f %.3f\n", hierarchy_pos, msd, cum_thresholds[hierarchy_pos]);
        if (msd > cum_thresholds[hierarchy_pos])
            return false;
    }
    if (hierarchy_pos == nthresholds - 2)
    {
        for (uint childnr = 0; childnr < c_size; childnr++)
        {
            uint child = indices[c_offset + childnr];
            const Matrix &c_mat = random_mats[child];
            Matrix dmat;
            dif_matrix(mat, c_mat, dmat);
            double msd = get_msd(dmat, scalevec);
            //printf("MSD2 %d %.6f %.3f %d %d\n", nthresholds - 1, msd, cum_thresholds[-1], childnr, child);
            if (msd < cum_thresholds[nthresholds-1])
                return true;
        }
        return false;
    }

    for (uint childnr = 0; childnr < c_size; childnr++)
    {
        uint ind = indices[c_offset + childnr];
        bool result = is_close_to_cluster(
            mat, matnr, random_mats,
            clusters, indices,
            ind,
            scalevec, cum_thresholds,
            nthresholds,
            hierarchy_pos + 1);
        if (result)
            return true;
    }
    return false;
}

void add_to_cluster(
    Cluster &clus,
    uint &index, uint *indices, uint &nindices)
{
    uint &c_matindex = clus[0];
    uint &c_offset = clus[1];
    uint &c_size = clus[2];
    uint &c_maxsize = clus[3];
    if (c_size == c_maxsize)
    {
        uint new_maxsize = 2 * c_maxsize;
        memcpy(indices + nindices, indices + c_offset, c_size * sizeof(uint));
        c_offset = nindices;
        c_maxsize = new_maxsize;
        nindices += new_maxsize;
    }
    indices[c_offset + c_size] = index;
    c_size++;
}

void new_cluster(
    uint matindex,
    Cluster *clusters, uint &nclusters,
    uint *indices, uint &nindices)
{
    uint ini_size = 10;
    Cluster &clus = clusters[nclusters];
    clus[0] = matindex;
    clus[1] = nindices;
    clus[2] = 0;
    clus[3] = ini_size;
    nclusters++;
    nindices += ini_size;
}

void new_clusters(
    uint matindex,
    Cluster *clusters, uint &nclusters,
    uint *indices, uint &nindices,
    uint nest)
{
    new_cluster(matindex, clusters, nclusters, indices, nindices);
    Cluster &clus = clusters[nclusters - 1];
    if (nest == 1)
    {
        //printf("ADD   %d %d %d %d\n", 1, nclusters-1, matindex, clusters[nclusters-1][2]+1);
        add_to_cluster(clus, matindex, indices, nindices);
        return;
    }
    //printf("PRE   %d %d %d\n", nest, nclusters-1, nclusters);
    add_to_cluster(clus, nclusters, indices, nindices);
    new_clusters(matindex, clusters, nclusters, indices, nindices, nest - 1);
}

bool insert_in_clustering(
    uint matnr, const Matrix &mat, const Matrix *random_mats,
    Cluster *clusters, uint &nclusters,
    uint *indices, uint &nindices,
    uint curr_cluster,
    const Vector &scalevec,
    float *thresholds, uint nthresholds,
    int hierarchy_pos)
{
    const Cluster &c = clusters[curr_cluster];
    uint c_matindex = c[0];
    uint c_offset = c[1];
    uint c_size = c[2];
    if (curr_cluster == 0)
    {
        assert(hierarchy_pos == -1);
    }
    else
    {
        assert(hierarchy_pos >= 0);
        const Matrix &c_mat = random_mats[c_matindex];
        Matrix dmat;
        dif_matrix(mat, c_mat, dmat);
        double msd = get_msd(dmat, scalevec);
        //printf("MSD3 %d %.6f %.3f %d %d\n", hierarchy_pos, msd, thresholds[hierarchy_pos], curr_cluster, c_matindex);
        if (msd > thresholds[hierarchy_pos])
            return false;
    }
    bool found = false;
    if (hierarchy_pos == nthresholds - 2)
    {
        //printf("ADD2  %d %d %d\n", curr_cluster, matnr, clusters[curr_cluster][2]+1);
        add_to_cluster(clusters[curr_cluster], matnr, indices, nindices);
    }
    else
    {
        for (uint childnr = 0; childnr < c_size; childnr++)
        {
            uint ind = indices[c_offset + childnr];
            bool result = insert_in_clustering(
                matnr, mat,
                random_mats,
                clusters, nclusters,
                indices, nindices,
                ind,
                scalevec,
                thresholds, nthresholds,
                hierarchy_pos + 1);
            if (result)
                return true;
        }
        //printf("PRE2  %d %d\n", curr_cluster, nclusters);
        add_to_cluster(clusters[curr_cluster], nclusters, indices, nindices);
        new_clusters(
            matnr,
            clusters, nclusters,
            indices, nindices,
            nthresholds - 2 - hierarchy_pos);
    }
    return true;
}

int build_rotamers(
    const Matrix *random_mats,
    uint nrandom_mats,
    const Vector &scalevec,
    const double *hierarchy,
    uint nhierarchy,
    Matrix *result)
{
    uint max_clusters = uint(1e07);
    uint max_indices = uint(1e08);
    Cluster *clusters = (Cluster *)malloc(max_clusters * sizeof(Cluster));
    uint nclusters = 0;
    uint *indices = (uint *)malloc(max_indices * sizeof(uint));
    uint nindices = 0;
    new_cluster(0, clusters, nclusters, indices, nindices);
    float thresholds[nhierarchy];
    float cum_thresholds[nhierarchy];
    float cumsum = 0;
    int nresult = 0;
    for (int n = 0; n < nhierarchy; n++)
    {
        int nrev = nhierarchy - n - 1;
        float h = hierarchy[n] * hierarchy[n];
        thresholds[nrev] = h;
        cumsum += h;
        cum_thresholds[nrev] = cumsum;
    }
    
    for (uint matnr = 0; matnr < nrandom_mats; matnr++){
        //printf("CONF  %d\n", matnr);
        const Matrix &mat = random_mats[matnr];
        bool close = is_close_to_cluster(
            mat, matnr, random_mats,
            clusters, indices,
            0, scalevec,
            cum_thresholds, nhierarchy,
            -1
        );
        //printf("CLOSE %d %d\n", matnr, close);
        //printf("STAT %d %d\n", nclusters, nindices);
        if (!close){
            memcpy(result + nresult, mat, sizeof(mat));
            nresult++;
            insert_in_clustering (
                matnr, mat, random_mats,
                clusters, nclusters,
                indices, nindices,
                0, scalevec,
                thresholds, nhierarchy,
                -1
            );
        }
        //print_clust(0, clusters, indices, 0, hierarchy)
        //print()
    }
    //print_clust(0, clusters, indices, 0, hierarchy)
    free(clusters);
    free(indices);
    return nresult;
}

extern "C" int transform(
    const HierarchyStruct *hierarchy, const RandomRotationsStruct *random_rotations,
    const ScalevecStruct *scalevec, ResultStruct *result)
{
    int nrotamers = build_rotamers(  
        (const Matrix *) random_rotations->data, 
        random_rotations->shape[0],
        *((const Vector *) scalevec->data),
        hierarchy->data, 
        hierarchy->shape[0],
        (Matrix *) result->data
    );

    result->shape[0] = nrotamers;
    return 0;
}
