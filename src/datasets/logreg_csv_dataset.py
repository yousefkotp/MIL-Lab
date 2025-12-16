from src.datasets.linear_csv_dataset import LinearCSVDataset


class LogRegCSVDataset(LinearCSVDataset):
    """Dedicated dataset alias for logistic regression probes to avoid impacting linear probe behavior."""

    pass
