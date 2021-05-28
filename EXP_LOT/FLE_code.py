import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wot
import anndata
import sys

sys.path.append("../")
import LinSinkhorn
import os

from sklearn.decomposition import PCA


def filter_adata(adata, obs_filter=None, var_filter=None):
    if obs_filter is not None:
        if os.path.exists(obs_filter):
            adata = adata[
                adata.obs.index.isin(wot.io.read_sets(obs_filter).obs.index)
            ].copy()
        else:
            obs_filter = obs_filter.split(",")
            if (
                len(obs_filter) == 1 and obs_filter[0] in adata.obs
            ):  # boolean field in obs
                adata = adata[adata.obs[obs_filter] == True].copy()
            else:  # list of ids
                adata = adata[adata.obs.index.isin(obs_filter)].copy()
    if var_filter is not None:
        if os.path.exists(var_filter):
            adata = adata[
                :, adata.var.index.isin(wot.io.read_sets(var_filter).obs.index)
            ].copy()
        else:
            var_filter = var_filter.split(",")
            if (
                len(var_filter) == 1 and var_filter[0] in adata.var
            ):  # boolean field in var
                adata = adata[:, adata.var[var_filter[0]]].copy()
            else:  # list of ids
                adata = adata[:, adata.var.index.isin(var_filter)].copy()
    return adata


def read_dataset(
    path, obs=None, var=None, obs_filter=None, var_filter=None, **keywords
):
    """
    Read h5ad, loom, mtx, 10X h5, and csv formatted files
    Parameters
    ----------
    path: str
        File name of data file.
    obs: {str, pd.DataFrame}
        Path to obs data file or a data frame
    var: {str, pd.DataFrame}
        Path to var data file or a data frame
    obs_filter {str, pd.DataFrame}
        File with one id per line, name of a boolean field in obs, or a list of ids
    var_filter: {str, pd.DataFrame}
        File with one id per line, name of a boolean field in obs, or a list of ids
    Returns
    -------
    Annotated data matrix.
    """

    _, ext = os.path.splitext(str(path).lower())
    if ext == ".txt":
        df = pd.read_csv(path, engine="python", header=0, sep=None, index_col=0)
        adata = anndata.AnnData(
            X=df.values,
            obs=pd.DataFrame(index=df.index),
            var=pd.DataFrame(index=df.columns),
        )
    elif ext == ".h5ad":
        adata = anndata.read(path)
    elif ext == ".loom":
        adata = anndata.read_loom(path)
    elif ext == ".mtx":
        adata = anndata.read_mtx(path)
    elif ext == ".zarr":
        adata = anndata.read_zarr(path)
    else:
        raise ValueError("Unknown file format: {}".format(ext))

    def get_df(meta):
        if not isinstance(meta, pd.DataFrame):
            tmp_path = None
            if meta.startswith("gs://"):
                tmp_path = download_gs_url(meta)
                meta = tmp_path
            meta = pd.read_csv(meta, sep=None, index_col="id", engine="python")
            if tmp_path is not None:
                os.remove(tmp_path)
        return meta

    if obs is not None:
        if not isinstance(obs, list) and not isinstance(obs, tuple):
            obs = [obs]
        for item in obs:
            adata.obs = adata.obs.join(get_df(item))
    if var is not None:
        if not isinstance(var, list) and not isinstance(var, tuple):
            var = [var]
        for item in var:
            adata.var = adata.var.join(get_df(item))

    return filter_adata(adata, obs_filter=obs_filter, var_filter=var_filter)


def read_sets(path, feature_ids=None, as_dict=False):
    path = str(path)
    hash_index = path.rfind("#")
    set_names = None
    if hash_index != -1:
        set_names = path[hash_index + 1 :].split(",")
        path = path[0:hash_index]
    ext = get_filename_and_extension(path)[1]
    if ext == "gmt":
        gs = read_gmt(path, feature_ids)
    elif ext == "gmx":
        gs = read_gmx(path, feature_ids)
    elif ext == "txt" or ext == "grp":
        gs = read_grp(path, feature_ids)
    else:
        raise ValueError('Unknown file format "{}"'.format(ext))
    if set_names is not None:
        gs_filter = gs.var.index.isin(set_names)
        gs = gs[:, gs_filter]
    if as_dict:
        return wot.io.convert_binary_dataset_to_dict(gs)
    return gs


def get_filename_and_extension(name):
    name = os.path.basename(name)
    dot_index = name.rfind(".")
    ext = ""
    basename = name
    if dot_index != -1:
        ext = name[dot_index + 1 :].lower()
        basename = name[0:dot_index]
        if ext == "txt":  # check for .gmt.txt e.g.
            dot_index2 = basename.rfind(".")
            if dot_index2 != -1:
                ext2 = basename[dot_index2 + 1 :].lower()
                if ext2 in set(["gmt", "grp", "gmx"]):
                    basename = basename[0:dot_index2]
                    return basename, ext2
    return basename, ext


def read_gmt(path, feature_ids=None):
    with open(path) as fp:
        row_id_lc_to_index = {}
        row_id_lc_to_row_id = {}
        if feature_ids is not None:
            for i in range(len(feature_ids)):
                fid = feature_ids[i].lower()
                row_id_lc_to_index[fid] = i
                row_id_lc_to_row_id[fid] = feature_ids[i]

        members_array = []
        set_descriptions = []
        set_names = []
        for line in fp:
            if line == "" or line[0] == "#":
                continue
            tokens = line.split("\t")
            if len(tokens) < 3:
                continue
            set_names.append(tokens[0].strip())
            description = tokens[1].strip()
            if "BLANK" == description:
                description = ""
            set_descriptions.append(description)
            ids = tokens[2:]
            ids_in_set = []
            members_array.append(ids_in_set)
            for i in range(len(ids)):
                value = ids[i].strip()
                if value != "":
                    value_lc = value.lower()
                    row_index = row_id_lc_to_index.get(value_lc)
                    if feature_ids is None:
                        if row_index is None:
                            row_id_lc_to_row_id[value_lc] = value
                            row_index = len(row_id_lc_to_index)
                            row_id_lc_to_index[value_lc] = row_index

                    if row_index is not None:
                        ids_in_set.append(value)

        if feature_ids is None:
            feature_ids = np.empty(len(row_id_lc_to_index), dtype="object")
            for rid_lc in row_id_lc_to_index:
                feature_ids[row_id_lc_to_index[rid_lc]] = row_id_lc_to_row_id[rid_lc]

        x = np.zeros(shape=(len(feature_ids), len(set_names)), dtype=np.int8)
        for j in range(len(members_array)):
            ids = members_array[j]
            for id in ids:
                row_index = row_id_lc_to_index.get(id.lower())
                x[row_index, j] = 1

        obs = pd.DataFrame(index=feature_ids)
        var = pd.DataFrame(data={"description": set_descriptions}, index=set_names)
        return anndata.AnnData(X=x, obs=obs, var=var)


def read_gmx(path, feature_ids=None):
    with open(path) as fp:
        set_ids = fp.readline().split("\t")
        descriptions = fp.readline().split("\t")
        nsets = len(set_ids)
        for i in range(len(set_ids)):
            set_ids[i] = set_ids[i].rstrip()

        row_id_lc_to_index = {}
        row_id_lc_to_row_id = {}
        x = None
        array_of_arrays = None
        if feature_ids is not None:
            for i in range(len(feature_ids)):
                fid = feature_ids[i].lower()
                row_id_lc_to_index[fid] = i
                row_id_lc_to_row_id[fid] = feature_ids[i]
            x = np.zeros(shape=(len(feature_ids), nsets), dtype=np.int8)
        else:
            array_of_arrays = []
        for line in fp:
            tokens = line.split("\t")
            for j in range(nsets):
                value = tokens[j].strip()
                if value != "":
                    value_lc = value.lower()
                    row_index = row_id_lc_to_index.get(value_lc)
                    if feature_ids is None:
                        if row_index is None:
                            row_id_lc_to_row_id[value_lc] = value
                            row_index = len(row_id_lc_to_index)
                            row_id_lc_to_index[value_lc] = row_index
                            array_of_arrays.append(
                                np.zeros(shape=(nsets,), dtype=np.int8)
                            )
                        array_of_arrays[row_index][j] = 1
                    elif row_index is not None:
                        x[row_index, j] = 1
        if feature_ids is None:
            feature_ids = np.empty(len(row_id_lc_to_index), dtype="object")
            for rid_lc in row_id_lc_to_index:
                feature_ids[row_id_lc_to_index[rid_lc]] = row_id_lc_to_row_id[rid_lc]

        if array_of_arrays is not None:
            x = np.array(array_of_arrays)
        obs = pd.DataFrame(index=feature_ids)
        var = pd.DataFrame(data={"description": descriptions}, index=set_ids)
        return anndata.AnnData(x, obs=obs, var=var)


def read_grp(path, feature_ids=None):
    with open(path) as fp:
        row_id_lc_to_index = {}
        row_id_lc_to_row_id = {}
        if feature_ids is not None:
            for i in range(len(feature_ids)):
                fid = feature_ids[i].lower()
                row_id_lc_to_index[fid] = i
                row_id_lc_to_row_id[fid] = feature_ids[i]

        ids_in_set = set()
        for line in fp:
            if line == "" or line[0] == "#" or line[0] == ">":
                continue
            value = line.strip()
            if value != "":
                value_lc = value.lower()
                row_index = row_id_lc_to_index.get(value_lc)
                if feature_ids is None:
                    if row_index is None:
                        row_id_lc_to_row_id[value_lc] = value
                        row_index = len(row_id_lc_to_index)
                        row_id_lc_to_index[value_lc] = row_index

                if row_index is not None:
                    ids_in_set.add(value)

        if feature_ids is None:
            feature_ids = np.empty(len(row_id_lc_to_index), dtype="object")
            for rid_lc in row_id_lc_to_index:
                feature_ids[row_id_lc_to_index[rid_lc]] = row_id_lc_to_row_id[rid_lc]

        x = np.zeros(shape=(len(feature_ids), 1), dtype=np.int8)
        for id in ids_in_set:
            row_index = row_id_lc_to_index.get(id.lower())
            x[row_index, 0] = 1

        obs = pd.DataFrame(index=feature_ids)
        var = pd.DataFrame(
            index=[wot.io.get_filename_and_extension(os.path.basename(path))[0]]
        )
        return anndata.AnnData(X=x, obs=obs, var=var)


path = "Experiment_FLE/"
FLE_COORDS_PATH = path + "data/fle_coords.txt"
FULL_DS_PATH = path + "data/ExprMatrix.h5ad"
VAR_DS_PATH = path + "data/ExprMatrix.var.genes.h5ad"
CELL_DAYS_PATH = path + "data/cell_days.txt"
GENE_SETS_PATH = path + "data/gene_sets.gmx"
GENE_SET_SCORES_PATH = path + "data/gene_set_scores.csv"
CELL_SETS_PATH = path + "data/cell_sets.gmt"
CELL_GROWTH_PATH = path + "data/growth_gs_init.txt"
SERUM_CELL_IDS_PATH = path + "data/serum_cell_ids.txt"

coord_df = pd.read_csv(FLE_COORDS_PATH, index_col="id", sep="\t")
days_df = pd.read_csv(CELL_DAYS_PATH, index_col="id", sep="\t")
ids_serum_df = pd.read_csv(SERUM_CELL_IDS_PATH, sep="\t", header=None)

# adata_1 = read_dataset(FULL_DS_PATH, obs=[days_df,coord_df])
# adata_1.shape
# unique_days = adata_1.obs['day'].unique()
# unique_days = unique_days[np.isnan(unique_days) == False]
#
# adata_1.obs['x']
# adata_1.obs['day']
# # plot visualization coordinates
# figure = plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.tight_layout()
# plt.scatter(adata_1.obs['x'], adata_1.obs['y'],c= adata_1.obs['day'],
#                s=4, marker=',', edgecolors='none', alpha=0.8)
# cb = plt.colorbar()
# cb.ax.set_title('Day')
#
# adata_1
# adata_1.var.index.values


# load data
adata = read_dataset(
    VAR_DS_PATH, obs=[CELL_DAYS_PATH, CELL_GROWTH_PATH], obs_filter=SERUM_CELL_IDS_PATH
)
adata.shape


###  Get the data and apply PCA ####
unique_days = adata.obs["day"].unique()
unique_days = unique_days[np.isnan(unique_days) == False]

cell_ids_serum = {}
for day in unique_days:
    list_day = adata[adata.obs["day"] == day].obs.index.values.tolist()
    cell_ids_serum[day] = list_day


n_components = 30  # 5
PCA30_SERUM_DAY = {}
SERUM_DAY = {}
CELL_GROWTH_RATE = {}
for ind_day, day in enumerate(unique_days[:-1]):
    tmp_1 = adata[adata.obs["day"] == unique_days[ind_day]].to_df().values
    tmp_2 = adata[adata.obs["day"] == unique_days[ind_day + 1]].to_df().values
    tmp_3 = (
        adata[adata.obs["day"] == unique_days[ind_day]].obs["cell_growth_rate"].values
    )
    X = PCA(n_components=n_components).fit_transform(
        np.concatenate([tmp_1, tmp_2], axis=0)
    )
    PCA30_SERUM_DAY[day] = X[: np.shape(tmp_1)[0], :], X[np.shape(tmp_1)[0] :, :]
    SERUM_DAY[day] = tmp_1
    CELL_GROWTH_RATE[day] = tmp_3


cell_sets = read_sets(CELL_SETS_PATH)
cell_sets = filter_adata(cell_sets, obs_filter=SERUM_CELL_IDS_PATH)
name = cell_sets.var.index[0]
cell_set = cell_sets[:, name]
IPS_all = cell_set[cell_set.X > 0].obs.join(coord_df).join(days_df)
IPS_18 = IPS_all[IPS_all["day"] == unique_days[-1]]
list_IPS_18 = list(IPS_18.index)

serum_18 = adata[adata.obs["day"] == unique_days[-1]].to_df()
index_IPS_18 = []
for k in range(len(list_IPS_18)):
    index_IPS_18.append(serum_18.index.get_loc(list_IPS_18[k]))


CELL_DISTRIBUTION_IPSC_DAY18 = np.zeros(serum_18.shape[0])
CELL_DISTRIBUTION_IPSC_DAY18[index_IPS_18] = 1
CELL_DISTRIBUTION_IPSC_DAY18 = CELL_DISTRIBUTION_IPSC_DAY18 / np.sum(
    CELL_DISTRIBUTION_IPSC_DAY18
)


### Compute the Optimal transport ###
Square_Euclidean_cost = lambda X, Y: LinSinkhorn.Square_Euclidean_Distance(X, Y)
Square_Euclidean_factorized_cost = lambda X, Y: LinSinkhorn.factorized_square_Euclidean(
    X, Y
)

cost = Square_Euclidean_cost
cost_factorized = Square_Euclidean_factorized_cost


rank_LR = 500
reg = 0
gamma_init = "arbitrary"
gamma_0 = 1 / 5
method = "Dykstra"
alpha = 0.00001
Init_trivial = True
Init_random = False
seed_init = 10
reg_init = 1e-1
max_iter_LR = 100
delta_LR = 1e-3


max_iter_Sin = 100
delta_Sin = 1e-3
epsilon = 5


dict_results_Sin = {}
dict_results_LOT_PCA = {}
# dict_results_LOT = {}
start = 0
for i, day in enumerate(unique_days[start:]):
    if day == 18.0:
        continue
    print(
        "\r"
        + f"Executing Sinkhorn and LOT between the days {day} and {unique_days[start + i+1]} / 18",
        end="",
    )
    # Computes the marginals
    delta_days = unique_days[start + i + 1] - day
    # The original data has been transformed with PCA as described in [1]
    a = np.power(CELL_GROWTH_RATE[day], delta_days)
    a = a / np.sum(a)

    n = PCA30_SERUM_DAY[day][1].shape[0]
    b = np.ones(n) / n

    ### Sinkhorn with PCA ###
    C = cost(PCA30_SERUM_DAY[day][0], PCA30_SERUM_DAY[day][1])
    res_Sin, acc_Sin, times_Sin, P_Sin = LinSinkhorn.Sinkhorn_LSE(
        C, epsilon, a, b, max_iter=max_iter_Sin, delta=delta_Sin, lam=0, time_out=100
    )
    dict_results_Sin[day] = P_Sin

    ### LOT methods ###
    ### LOT with PCA ###
    res_LR, acc_LR_PCA, times_LR, Q, R, g = LinSinkhorn.Lin_LOT_MD(
        PCA30_SERUM_DAY[day][0],
        PCA30_SERUM_DAY[day][1],
        a,
        b,
        rank_LR,
        reg,
        alpha,
        cost,
        cost_factorized,
        Init_trivial=Init_trivial,
        Init_random=Init_random,
        seed_init=seed_init,
        reg_init=reg_init,
        gamma_init=gamma_init,
        gamma_0=gamma_0,
        method=method,
        max_iter=max_iter_LR,
        delta=delta_LR,
        max_iter_IBP=10000,
        delta_IBP=1e-3,
        lam_IBP=0,
        time_out=100,
    )
    P_LR_PCA = np.dot(Q / g, R.T)
    dict_results_LOT_PCA[day] = P_LR_PCA

    ### LOT without PCA ###
    # res_LR, acc_LR, times_LR, Q, R, g = LinSinkhorn.Lin_LOT_MD(SERUM_DAY[day],SERUM_DAY[unique_days[start+i+1]],a,b,rank_LR,reg,alpha,cost,cost_factorized,Init_trivial=Init_trivial,Init_random=Init_random,seed_init=seed_init,reg_init=reg_init,gamma_init=gamma_init,gamma_0=gamma_0,method=method,max_iter=max_iter_LR,delta=delta_LR,max_iter_IBP=10000,delta_IBP=1e-3,lam_IBP=0,time_out=10)
    # P_LR = np.dot(Q/g,R.T)
    # dict_results_LOT[day] = P_LR


cell_distribution_ipsc_Sin = {}
cell_distribution_ipsc_LOT_PCA = {}
reverse_days = unique_days[::-1]
for i, day in enumerate(reverse_days):
    print("\r" + f"Infering ancestor cells at day {day}", end="")
    if day == 0.0:
        continue
    if day == 18.0:
        cell_distribution_ipsc_Sin[day] = CELL_DISTRIBUTION_IPSC_DAY18
        cell_distribution_ipsc_LOT_PCA[day] = CELL_DISTRIBUTION_IPSC_DAY18

    # Calculates cells' ancestors
    P_Sin = dict_results_Sin[reverse_days[i + 1]]
    cell_distribution_Sin = np.dot(P_Sin, cell_distribution_ipsc_Sin[day])
    cell_distribution_ipsc_Sin[reverse_days[i + 1]] = cell_distribution_Sin / np.sum(
        cell_distribution_Sin
    )

    P_LOT_PCA = dict_results_LOT_PCA[reverse_days[i + 1]]
    cell_distribution_LOT_PCA = np.dot(P_LOT_PCA, cell_distribution_ipsc_LOT_PCA[day])
    cell_distribution_ipsc_LOT_PCA[
        reverse_days[i + 1]
    ] = cell_distribution_LOT_PCA / np.sum(cell_distribution_LOT_PCA)


nbins = 500
xrange = coord_df["x"].min(), coord_df["x"].max()
yrange = coord_df["y"].min(), coord_df["y"].max()
coord_df["x"] = np.floor(
    np.interp(coord_df["x"], [xrange[0], xrange[1]], [0, nbins - 1])
).astype(int)
coord_df["y"] = np.floor(
    np.interp(coord_df["y"], [yrange[0], yrange[1]], [0, nbins - 1])
).astype(int)


coord_ancestors_ipsc = dict()
for day in unique_days:
    cell_ids = np.array(cell_ids_serum[day])
    coord_ancestors_ipsc[day] = coord_df[coord_df.index.isin(cell_ids)][
        ["x", "y"]
    ].values


alpha_bins = [1, 0.5, 0.0]
binned_cell_distribution_ipsc_Sin = {}
binned_cell_distribution_ipsc_LOT_PCA = {}
for day in unique_days:

    tmp = cell_distribution_ipsc_Sin[day].copy()
    tmp[tmp >= 1e-2] = alpha_bins[0]
    tmp[np.logical_and(1e-2 > tmp, tmp >= 5e-4)] = alpha_bins[1]
    tmp[5e-4 > tmp] = alpha_bins[2]
    binned_cell_distribution_ipsc_Sin[day] = tmp

    tmp = cell_distribution_ipsc_LOT_PCA[day].copy()
    tmp[tmp >= 1e-2] = alpha_bins[0]
    tmp[np.logical_and(1e-2 > tmp, tmp >= 5e-4)] = alpha_bins[1]
    tmp[5e-4 > tmp] = alpha_bins[2]
    binned_cell_distribution_ipsc_LOT_PCA[day] = tmp

import csv

#
w = csv.writer(open("binned_LOT_500.csv", "w"))
for key, val in binned_cell_distribution_ipsc_LOT_PCA.items():
    w.writerow([key, val])

w = csv.writer(open("binned_Sinkhorn_500.csv", "w"))
for key, val in binned_cell_distribution_ipsc_Sin.items():
    w.writerow([key, val])


path = "Experiment_FLE/"

# binned_cell_distribution_ipsc_Sin = pd.read_csv(path+'binned_LOT.csv',header=None)
# binned_cell_distribution_ipsc_Sin = pd.read_csv(path+'binned_Sinkhorn.csv',header=None)


cm = plt.get_cmap("jet")
cNorm = mpl.colors.Normalize(vmin=0, vmax=len(unique_days))
scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)

fig = plt.figure(figsize=(13, 10))
plt.title(f"Sinkhorn, $\epsilon$={epsilon}", fontsize=34)
plt.plot(
    coord_df["x"],
    coord_df["y"],
    marker=".",
    color="grey",
    ls="",
    markersize=0.3,
    alpha=0.07,
)
for i, day in enumerate(unique_days):
    colorVal = scalarMap.to_rgba(i)
    for b in alpha_bins:
        ind_alpha = np.where(binned_cell_distribution_ipsc_Sin[day] == b)[0]
        colorVal = np.array(colorVal)
        colorVal[3] = b
        plt.plot(
            coord_ancestors_ipsc[day][ind_alpha, 0],
            coord_ancestors_ipsc[day][ind_alpha, 1],
            marker=".",
            color=colorVal,
            ls="",
            markersize=1,
        )
plt.xlabel("FLE1", fontsize=34)
plt.ylabel("FLE2", fontsize=34)
ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=1)
cbar = mpl.colorbar.ColorbarBase(
    ax, cmap=cm, norm=mpl.colors.Normalize(vmin=0, vmax=18)
)
cbar.remove()
fig.savefig("plot_iPSC_Sinkhorn.png", bbox_inches="tight")
plt.show()


fig = plt.figure(figsize=(13, 10))
plt.title(f"LOT, $\gamma$={1/epsilon}", fontsize=34)
plt.plot(
    coord_df["x"],
    coord_df["y"],
    marker=".",
    color="grey",
    ls="",
    markersize=0.3,
    alpha=0.07,
)
for i, day in enumerate(unique_days):
    colorVal = scalarMap.to_rgba(i)
    for b in alpha_bins:
        ind_alpha = np.where(binned_cell_distribution_ipsc_LOT_PCA[day] == b)[0]
        colorVal = np.array(colorVal)
        colorVal[3] = b
        plt.plot(
            coord_ancestors_ipsc[day][ind_alpha, 0],
            coord_ancestors_ipsc[day][ind_alpha, 1],
            marker=".",
            color=colorVal,
            ls="",
            markersize=1,
        )
plt.xlabel("FLE1", fontsize=34)
plt.ylabel("FLE2", fontsize=34)
ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=1)
cbar = mpl.colorbar.ColorbarBase(
    ax, cmap=cm, norm=mpl.colors.Normalize(vmin=0, vmax=18)
)

fig.savefig("plot_iPSC_LOT.png", bbox_inches="tight")
plt.show()


# import matplotlib as mpl
# from matplotlib import pyplot as plt
# import numpy as np
# from ott.core import sinkhorn
# from ott.geometry import pointcloud
#
#
# # Defines the optimal transport regularisation parameter
# epsilon = 5
#
# dict_results = {}
# for i, day in enumerate(unique_days):
#   if day == 18.:
#     continue
#   print('\r' + f'Executing Sinkhorn between the days {day} and {unique_days[i+1]} / 18', end='')
#   # Computes the marginals
#   delta_days = unique_days[i+1] - day
#   n = PCA30_SERUM_DAY[day][1].shape[0]
#   # The original data has been transformed with PCA as described in [1]
#   a = np.power(CELL_GROWTH_RATE[day], delta_days) / np.mean(
#       np.power(CELL_GROWTH_RATE[day], delta_days)) / n
#   b = np.ones(n) / n
#
#   # # Applies optimal transport
#   geom = pointcloud.PointCloud(
#       PCA30_SERUM_DAY[day][0], PCA30_SERUM_DAY[day][1], epsilon=epsilon)
#   out = sinkhorn.sinkhorn(geom, a, b, tau_a=1/(1+epsilon), tau_b=1)
#
#   # Saves the geometry and the potentials to calculate the ancestors at a later time
#   dict_results[day] = [geom, out.f, out.g]
#
#
#
# cell_distribution_ipsc = {}
# reverse_days = unique_days[::-1]
# for i, day in enumerate(reverse_days):
#   print('\r' + f'Infering ancestor cells at day {day}', end='')
#   if day == 0.:
#     continue
#   if day == 18.:
#     cell_distribution_ipsc[day] = CELL_DISTRIBUTION_IPSC_DAY18
#   # Calculates cells' ancestors
#   geom = dict_results[reverse_days[i+1]][0]
#   f = dict_results[reverse_days[i+1]][1]
#   g = dict_results[reverse_days[i+1]][2]
#   cell_distribution = geom.apply_transport_from_potentials(
#       f, g, cell_distribution_ipsc[day], axis=1)
#
#   cell_distribution_ipsc[reverse_days[i+1]] = cell_distribution / np.sum(
#       cell_distribution)
#
# cell_distribution_ipsc[unique_days[2]]
#
#
#
# alpha_bins = [1, 0.5, 0.]
# binned_cell_distribution_ipsc = {}
# for day in unique_days:
#   tmp = cell_distribution_ipsc[day].copy()
#   tmp = np.array(tmp)
#   tmp[tmp >= 1e-2] = alpha_bins[0]
#   tmp[np.logical_and(1e-2 > tmp, tmp >= 5e-4)] = alpha_bins[1]
#   tmp[5e-4 > tmp] = alpha_bins[2]
#   binned_cell_distribution_ipsc[day] = tmp
#
#
#
# cm = plt.get_cmap('jet')
# cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(unique_days))
# scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
#
# fig = plt.figure(figsize=(13, 10))
# plt.title(f'Cell type: iPSC, medium: serum, $\epsilon$={epsilon}', fontsize=24)
# plt.plot(coord_df['x'], coord_df['y'], marker='.', color='grey', ls='',
#          markersize=0.3, alpha=0.07)
# for i, day in enumerate(unique_days):
#   colorVal = scalarMap.to_rgba(i)
#   for b in alpha_bins:
#     ind_alpha = np.where(binned_cell_distribution_ipsc[day] == b)[0]
#     colorVal = np.array(colorVal)
#     colorVal[3] = b
#     plt.plot(coord_ancestors_ipsc[day][ind_alpha, 0],
#              coord_ancestors_ipsc[day][ind_alpha, 1],
#            marker='.', color=colorVal, ls='', markersize=1)
# plt.xlabel('FLE1', fontsize=24)
# plt.ylabel('FLE2', fontsize=24)
# ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=1)
# cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm,
#                        norm=mpl.colors.Normalize(vmin=0, vmax=18))
# plt.show()
list = [
    9.5,
    5,
    9,
    15.5,
    6,
    8,
    16,
    6.5,
    12,
    12,
    14,
    15.5,
    9.5,
    6,
    7,
    9,
    8.5,
    6,
    10,
    18,
    8.5,
]
array_list = np.array(list)
np.mean(array_list)
