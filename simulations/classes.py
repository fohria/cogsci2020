from dataclasses import dataclass


@dataclass
class Parameters:
    """
        convenience object to hold parameter values
        examples:
            str(Parameters) for plot titles
            eval(str) to convert back to object
                -- yes eval is considered unsafe :)
    """

    alpha: float
    beta: float
    gamma: float
    task: str
    trials: int
