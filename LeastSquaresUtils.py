import numpy as np

def solve_MMSE(outputs, inputs, coefficients=None):
    """
    Solves the following optimization
    |o_1|   |c_1,1   c_1,2   ...   c_1,m|   |i_1|
    |o_2|   |c_1,1   c_1,2   ...   c_1,m|   |i_2|
    |o_3| = |c_1,1   c_1,2   ...   c_1,m| * |i_3|
    |...|   | ...     ...    ...    ... |   |...|
    |o_n|   |c_n,1   c_n,2   ...   c_n,m|   |i_m|
    """
    if coefficients is None:
        coefficients = float('nan') * np.ones((len(outputs), len(inputs)))

    for row in range(len(coefficients)):
        data = coefficients[row]
        # nan check
        freeIndicies = np.where(data != data)[0]
        fixedIndicies = np.where(data == data)[0]

        if len(fixedIndicies) > 0:
            x = inputs[freeIndicies].T
            adjustment = (inputs[fixedIndicies].T * data[fixedIndicies]).sum(axis=1)
            adjustedOutputs = outputs[row] - adjustment
        else:
            adjustedOutputs = outputs
            x = inputs

        xTxi = np.linalg.inv(x.T @ x)

        solvedCoefs = xTxi @ x.T @ adjustedOutputs

        solvedCoefIdx = 0
        for idx in freeIndicies:
            data[idx] = solvedCoefs[solvedCoefIdx]
            solvedCoefIdx += 1
    return coefficients


def calculate_MSE(outputs, inputs, coefficients):
    return ((outputs - coefficients @ inputs) ** 2).sum(1)
