from collections import defaultdict
from itertools import combinations

import igraph as ig
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logsumexp

REQUIRED_COLUMNS = ["repo", "user_id", "year", "office", "organization_role"]
ROLE_MAP = {
    "FACULTY": "Faculty",
    "GRADUATE": "Graduate",
    "UNDRGRAD": "Undergraduate",
    "PROSTAFF": "Professional staff",
    "SCIENTST": "Scientist",
    "POSTDOC": "Postdoc",
    "RETIRED": "Retired",
    "SELFEMPL": "Self-employed",
    "OTHER": "Other",
    "NONE": "Unknown",
}
ROLE_ORDER = [
    "Graduate", "Postdoc", "Undergraduate", "Scientist", "Faculty",
    "Other", "Professional staff", "Retired", "Self-employed", "Unknown",
]
OFFICE_ORDER = [
    "Basic Energy Sciences",
    "High Energy Physics",
    "NERSC Directors Reserve",
    "Biological and Environmental Research",
    "Advanced Scientific Computing Research",
    "Fusion Energy Sciences",
    "Nuclear Physics",
    "ASCR Leadership Computing Challenge",
]
OFFICE_LABELS = {
    "Basic Energy Sciences": "BES",
    "High Energy Physics": "HEP",
    "NERSC Directors Reserve": "NERSC DDR",
    "NERSC Director's Reserve": "NERSC DDR",
    "Biological and Environmental Research": "BER",
    "Advanced Scientific Computing Research": "ASCR",
    "Fusion Energy Sciences": "FES",
    "Nuclear Physics": "NP",
    "ASCR Leadership Computing Challenge": "ASCR LCC",
    "Other (<1%)": "Other (<1%)",
    "Unknown": "Unknown",
}


def load_memberships(path):
    data = pd.read_csv(path, dtype={"repo": "string", "user_id": "string"})
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = data[REQUIRED_COLUMNS].copy()
    for column in ["repo", "user_id", "office", "organization_role"]:
        data[column] = data[column].astype("string").str.strip()
    data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)
    return data.dropna(subset=["repo", "user_id", "year"]).drop_duplicates().reset_index(drop=True)


def collaboration_edges(data):
    pair_repo_sum = defaultdict(float)
    pair_repo_years = defaultdict(int)
    for (repo, _year), group in data.groupby(["repo", "year"], sort=False):
        users = sorted(group["user_id"].dropna().unique())
        if len(users) < 2:
            continue
        contribution = 1.0 / (len(users) - 1)
        for source, target in combinations(users, 2):
            key = (source, target, repo)
            pair_repo_sum[key] += contribution
            pair_repo_years[key] += 1
    rsa = defaultdict(float)
    for (source, target, repo), total in pair_repo_sum.items():
        rsa[(source, target)] += total / pair_repo_years[(source, target, repo)]
    return pd.DataFrame(
        [(source, target, 1.0, weight) for (source, target), weight in sorted(rsa.items())],
        columns=["source", "target", "weight_unweighted", "weight_rsa"],
    )


def build_user_graph(data):
    users = sorted(data["user_id"].dropna().unique())
    index = {user: i for i, user in enumerate(users)}
    edges = collaboration_edges(data)
    graph = ig.Graph(
        n=len(users),
        edges=[(index[s], index[t]) for s, t in edges[["source", "target"]].itertuples(index=False)],
        directed=False,
    )
    graph.vs["name"] = users
    if graph.ecount():
        graph.es["weight_unweighted"] = edges["weight_unweighted"].tolist()
        graph.es["weight_rsa"] = edges["weight_rsa"].tolist()
    return graph, edges


def node_scores(graph):
    return pd.DataFrame({
        "user_id": graph.vs["name"],
        "degree": graph.degree(),
        "rsa_strength": graph.strength(weights="weight_rsa") if graph.ecount() else np.zeros(graph.vcount()),
    })


def latest_roles(data, users=None):
    work = data[["user_id", "repo", "year", "organization_role"]].copy()
    work["role_code"] = work["organization_role"].astype("string").str.strip().str.upper()
    work["role"] = work["role_code"].map(ROLE_MAP)
    valid = work[work["role"].notna() & work["role"].ne("Unknown")].drop_duplicates(
        ["user_id", "year", "repo", "role_code"]
    )
    counts = valid.groupby(["user_id", "year", "role_code", "role"], as_index=False).size()
    if len(counts):
        latest_year = counts.groupby("user_id")["year"].transform("max")
        counts = counts[counts["year"].eq(latest_year)]
        maximum = counts.groupby("user_id")["size"].transform("max")
        winners = counts[counts["size"].eq(maximum)]
        resolved = winners.groupby("user_id").agg(
            role_code=("role_code", lambda x: x.iloc[0] if x.nunique() == 1 else "NONE"),
            role=("role", lambda x: x.iloc[0] if x.nunique() == 1 else "Unknown"),
        ).reset_index()
    else:
        resolved = pd.DataFrame(columns=["user_id", "role_code", "role"])
    user_list = sorted(data["user_id"].dropna().unique()) if users is None else list(users)
    result = pd.DataFrame({"user_id": user_list}).merge(resolved, on="user_id", how="left")
    result["role_code"] = result["role_code"].fillna("NONE")
    result["role"] = result["role"].fillna("Unknown")
    return result


def exact_top(data, score_column, n=100):
    if len(data) < n:
        raise ValueError(f"Only {len(data)} eligible users; top {n} requested.")
    return data.sort_values([score_column, "user_id"], ascending=[False, True], kind="mergesort").head(n).reset_index(drop=True)


def integer_log_bins(values, count=24):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    minimum, maximum = int(np.floor(values.min())), int(np.ceil(values.max()))
    upper = maximum + 1
    if upper - minimum < count:
        count = upper - minimum
    edges = np.rint(np.geomspace(minimum, upper, count + 1)).astype(int)
    edges[0], edges[-1] = minimum, upper
    for i in range(1, count):
        edges[i] = int(np.clip(edges[i], minimum + i, upper - (count - i)))
    for i in range(1, count):
        edges[i] = max(edges[i], edges[i - 1] + 1)
    for i in range(count - 1, 0, -1):
        edges[i] = min(edges[i], edges[i + 1] - 1)
    labels = [f"{a:,}" if a == b - 1 else f"{a:,}–{b - 1:,}" for a, b in zip(edges[:-1], edges[1:])]
    return edges.astype(float), labels


def _format_rsa(value):
    value = float(value)
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def continuous_log_bins(values, count=24):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    edges = np.geomspace(values.min(), np.nextafter(values.max(), np.inf), count + 1)
    labels = [f"{_format_rsa(a)}–{_format_rsa(b)}" for a, b in zip(edges[:-1], edges[1:])]
    return edges, labels


def assign_bins(data, score_column, edges, labels):
    result = data.copy()
    result["bin_order"] = pd.cut(
        result[score_column], bins=edges, labels=False, right=False, include_lowest=True
    ).astype(int)
    result["bin_label"] = pd.Categorical(
        result["bin_order"].map(dict(enumerate(labels))),
        categories=labels,
        ordered=True,
    )
    return result


def category_composition(
    binned_users, assignments, category_column, weight_column, category_order
):
    if isinstance(binned_users["bin_label"].dtype, pd.CategoricalDtype):
        labels = list(binned_users["bin_label"].cat.categories)
    else:
        labels = (
            binned_users.sort_values("bin_order")
            .drop_duplicates("bin_order")["bin_label"]
            .tolist()
        )

    orders = range(len(labels))
    bin_counts = (
        binned_users.groupby("bin_order")["user_id"]
        .nunique()
        .reindex(orders, fill_value=0)
        .rename("n_unique_users")
        .reset_index()
    )
    bin_counts["bin_label"] = bin_counts["bin_order"].map(dict(enumerate(labels)))

    merged = binned_users[["user_id", "bin_order"]].merge(
        assignments, on="user_id", how="left"
    )
    index = pd.MultiIndex.from_product(
        [orders, category_order], names=["bin_order", category_column]
    )

    composition = (
        merged.groupby(["bin_order", category_column], dropna=False)[weight_column]
        .sum()
        .reindex(index, fill_value=0.0)
        .rename(weight_column)
        .reset_index()
        .merge(bin_counts, on="bin_order", how="left")
    )
    composition["percentage"] = np.where(
        composition["n_unique_users"] > 0,
        100.0 * composition[weight_column] / composition["n_unique_users"],
        0.0,
    )

    return bin_counts, composition


def role_assignments(data, users):
    roles = latest_roles(data, users)
    roles["weight"] = 1.0
    return roles[["user_id", "role", "weight"]]


def office_assignments(data, users, threshold=1.0):
    assignments = data[data["user_id"].isin(users)][["user_id", "office"]].dropna().drop_duplicates()
    missing = sorted(set(users) - set(assignments["user_id"]))
    if missing:
        assignments = pd.concat([assignments, pd.DataFrame({"user_id": missing, "office": "Unknown"})], ignore_index=True)
    assignments["weight"] = 1.0 / assignments.groupby("user_id")["office"].transform("nunique")
    shares = 100.0 * assignments.groupby("office")["weight"].sum() / assignments["weight"].sum()
    small = [office for office, share in shares.items() if office != "Unknown" and share < threshold]
    if small:
        assignments.loc[assignments["office"].isin(small), "office"] = f"Other (<{threshold:g}%)"
        assignments = assignments.groupby(["user_id", "office"], as_index=False)["weight"].sum()
    return assignments


def fit_cutoff_power_law(positive_degrees):
    values = np.asarray(positive_degrees, dtype=int)
    maximum = int(values.max())
    counts = np.bincount(values, minlength=maximum + 1)[1:]
    support = np.arange(1, maximum + 1, dtype=float)
    observed = counts > 0
    indices = np.flatnonzero(observed)
    log_probability = np.log(counts[observed] / values.size)
    weights = np.sqrt(counts[observed])
    log_support = np.log(support)

    def residuals(log_parameters):
        tau, z_c = np.exp(log_parameters)
        log_weights = -tau * log_support - support / z_c
        model = log_weights - logsumexp(log_weights)
        return weights * (model[indices] - log_probability)

    bounds = (np.log([0.01, 0.5]), np.log([10.0, 1_000_000.0]))
    starts = [(0.5, 25), (0.5, 100), (0.75, 500), (1, 50), (1, 500), (1.5, 100), (2, 1000)]
    results = [least_squares(residuals, np.log(start), bounds=bounds, max_nfev=10000) for start in starts]
    best = min(results, key=lambda result: result.cost)
    tau, z_c = np.exp(best.x)
    dof = max(len(best.fun) - 2, 1)
    covariance_log = np.linalg.pinv(best.jac.T @ best.jac) * np.sum(best.fun ** 2) / dof
    errors = np.sqrt(np.maximum(np.diag(covariance_log), 0.0))
    return {
        "tau": float(tau),
        "z_c": float(z_c),
        "tau_se": float(tau * errors[0]),
        "z_c_se": float(z_c * errors[1]),
        "covariance_log": covariance_log,
    }


def cutoff_pmf(plot_support, normalization_support, tau, z_c):
    normalization_log_weights = -tau * np.log(normalization_support) - normalization_support / z_c
    normalizer = logsumexp(normalization_log_weights)
    return np.exp(-tau * np.log(plot_support) - plot_support / z_c - normalizer)


def cutoff_band(plot_support, normalization_support, tau, z_c, tau_se, z_c_se, correlation=0.0, draws=1500, seed=2026):
    covariance = np.array([
        [(tau_se / tau) ** 2, correlation * (tau_se / tau) * (z_c_se / z_c)],
        [correlation * (tau_se / tau) * (z_c_se / z_c), (z_c_se / z_c) ** 2],
    ])
    samples = np.exp(np.random.default_rng(seed).multivariate_normal(np.log([tau, z_c]), covariance, size=draws))
    probabilities = np.vstack([
        cutoff_pmf(plot_support, normalization_support, sample_tau, sample_z_c)
        for sample_tau, sample_z_c in samples
    ])
    return np.quantile(probabilities, 0.025, axis=0), np.quantile(probabilities, 0.975, axis=0)


def network_metrics(data, label):
    graph, _ = build_user_graph(data)
    memberships = data[["user_id", "repo"]].drop_duplicates()
    components = graph.connected_components()
    sizes = sorted(components.sizes(), reverse=True)
    giant = components.giant() if sizes else graph
    if giant.vcount() > 1:
        mean_distance = giant.average_path_length(directed=False)
        maximum_distance = giant.diameter(directed=False)
    else:
        mean_distance, maximum_distance = np.nan, 0
    return {
        "network": label,
        "total_repos": int(data["repo"].nunique()),
        "total_users": int(data["user_id"].nunique()),
        "mean_repos_per_user": memberships.groupby("user_id")["repo"].nunique().mean(),
        "mean_users_per_repo": memberships.groupby("repo")["user_id"].nunique().mean(),
        "mean_collaborators_per_user": float(np.mean(graph.degree())) if graph.vcount() else np.nan,
        "mean_distance_giant_component": mean_distance,
        "maximum_distance_giant_component": maximum_distance,
        "global_clustering_coefficient": graph.transitivity_undirected(mode="zero") if graph.vcount() else np.nan,
        "giant_component_size": sizes[0] if sizes else 0,
        "second_largest_component_size": sizes[1] if len(sizes) > 1 else 0,
    }
