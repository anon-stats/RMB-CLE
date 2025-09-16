#%%
import matplotlib.pyplot as plt
from generator import GenerateDataset
import matplotlib.lines as mlines
import itertools

def plot_data(
    regression,
    num_dims,
    num_tasks,
    num_instances,
    num_clusters,
):
    gen_data = GenerateDataset()
    df = gen_data(
        regression=regression,
        num_dims=num_dims,
        num_tasks=num_tasks,
        num_instances=num_instances,
        num_clusters=num_clusters,
    )

    tasks_per_cluster = num_tasks // num_clusters
    cluster_task_map = {}
    task_counter = 0
    for c in range(num_clusters):
        cluster_task_map[c] = list(range(task_counter, task_counter + tasks_per_cluster))
        task_counter += tasks_per_cluster

    cmap = plt.cm.tab10
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p'])

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Plot tasks
    task_markers = {}
    for cluster_idx, tasks in cluster_task_map.items():
        color = cmap(cluster_idx % 10)
        for t in tasks:
            task_points = df[df["Task"] == t][["Feature 0", "Target"]].values
            marker = next(markers)
            task_markers[t] = marker
            ax.scatter(
                task_points[:, 0],
                task_points[:, 1],
                label=f"Task {t}",
                color=color,
                marker=marker,
                alpha=0.7,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Custom legend: show task markers and cluster connection lines
    legend_handles = []
    for cluster_idx, tasks in cluster_task_map.items():
        color = cmap(cluster_idx % 10)
        for t in tasks:
            handle = mlines.Line2D([], [], color=color, marker=task_markers[t],
                                   linestyle='None', markersize=8, label=f'Task {t}')
            legend_handles.append(handle)
        cluster_line = mlines.Line2D([], [], color=color, linestyle='-', linewidth=2)
        legend_handles.append(cluster_line)

    ax.legend(handles=legend_handles, ncol=1, frameon=True, loc='center right', bbox_to_anchor=(-0.15, 0.5))
    plt.savefig("multi_task_clusters.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_data(
        regression=True,
        num_dims=1,
        num_tasks=8,
        num_instances=100,
        num_clusters=4,
    )
#%%
