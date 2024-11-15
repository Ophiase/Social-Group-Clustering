from enum import Enum

class DimensionalityReductionMethod(Enum):
    PCA = "PCA"
    TSNE = "t-SNE"

    @staticmethod
    def from_string(method: str) -> "DimensionalityReductionMethod":
        try:
            return DimensionalityReductionMethod(method)
        except ValueError:
            raise ValueError(f"Invalid method '{method}'. Use one of: {[e.value for e in DimensionalityReductionMethod]}")
