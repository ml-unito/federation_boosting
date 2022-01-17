from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from copy import deepcopy
import pandas as pd

class RanksCounter:
    """
    Stores ranks for each algorithm in a dictionary and provides methods to print them
    """

    def __init__(self, console):
        self.ranks = {"FedAlgorithms.adaboost": [], "FedAlgorithms.distsamme": [
        ], "FedAlgorithms.samme": [], "FedAlgorithms.preweaksamme": []}
        self.ranks_by_skw = {
            "Noniidness.uniform": deepcopy(self.ranks),
            "Noniidness.num_examples_skw": deepcopy(self.ranks),
            "Noniidness.lbl_skw": deepcopy(self.ranks),
            "Noniidness.dirichlet_lbl_skw": deepcopy(self.ranks),
            "Noniidness.pathological_skw": deepcopy(self.ranks),
            "Noniidness.covariate_shift": deepcopy(self.ranks)
        }
        self.console = console
        self.rank_tables = []

    def hm(self, model):
        """
        Hilights model name if it contains "FedAlgorithms.adaboost"
        """
        if model == "FedAlgorithms.adaboost":
            return "[bold]FedAlgorithms.adaboost.F1[/]"
        else:
            return model

    def add_rank(self, group):
        """
        Updates dictionaries with ranks induced by the given group
        """
        data: pd.DataFrame = group.sort_values(
            by=['F1'], inplace=False, ascending=False)
        skw = data.head(1).Skw.item()

        for rank, row in enumerate(data.itertuples()):
            self.ranks[row.Model].append(rank + 1)
            self.ranks_by_skw[skw][row.Model].append(rank + 1)

    def avg_rank(self, ranks):
        """
        Returns average rank of the given ranks
        """
        return sum(ranks) / len(ranks)

    def print_global_rank(self):
        """
        Prints global ranks (a.k.a. by-algorithms ranks)
        """
        table = Table(title="[bold red]Ranks by Algorithms[/]")
        table.add_column("Algorithm", justify="left", style="red")
        table.add_column("Rank", justify="right", style="bold")

        global_ranks = [[method, self.avg_rank(
            self.ranks[method])] for method in self.ranks.keys()]
        global_ranks.sort(key=lambda x: x[1])

        for model, rank in global_ranks:
            table.add_row(self.hm(model), f"{rank:.3f}")

        self.console.print(table)

    def print_skw_ranks(self):
        """
        Prints ranks by skw (a.k.a. by-skewness ranks)
        """

        tables = []
        for skw in self.ranks_by_skw.keys():
            table = Table(title=f"[bold cyan]{skw}[/]")
            table.add_column("Algorithm", justify="left", style="cyan")
            table.add_column("Rank", justify="right", style="bold")

            skw_ranks = [[method, self.avg_rank(
                self.ranks_by_skw[skw][method])] for method in self.ranks_by_skw[skw].keys()]
            skw_ranks.sort(key=lambda x: x[1])

            for method, rank in skw_ranks:
                table.add_row(self.hm(method), f"{rank:.3f}")

            tables.append(table)

        self.console.print(
            Panel(Columns(tables), title="[bold cyan]By Skewness[/]", border_style="cyan"))

    def add_rank_table(self, group):
        """
        Prints ranks tables for the given group (used by --by-all option)
        """
        data: pd.DataFrame = group.sort_values(
            by=['F1'], inplace=False, ascending=False)

        ds_skw = data.head(1)["Dataset"].item() + " " + \
            data.head(1)["Skw"].item()
        table = Table(title=f"[bold magenta]{ds_skw}[/]")
        table.add_column("#", justify="left", style="bold")
        table.add_column("Model", justify="left", style="magenta")
        table.add_column("F1", justify="right", style="green")

        for i, row in enumerate(data.itertuples()):
            table.add_row(str(i+1), self.hm(row.Model), f"{row.F1:.3f}")

        self.rank_tables.append(table)

    def print_rank_tables(self):
        self.console.print(Panel(Columns(
            self.rank_tables), title="[bold magenta]By Skewness[/]", border_style="magenta"))
