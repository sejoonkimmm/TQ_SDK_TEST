from flask import Flask, request, jsonify
from ttopt import TTOpt
from ttopt import ttopt_init
import numpy as np

app = Flask(__name__)

np.random.seed(42)

d = 100                     # Number of function dimensions:
rank = 4                    # Maximum TT-rank while cross-like iterations
def f(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)

# Flask route for the optimization
@app.route('/optimize', methods=['POST'])
def optimize():
    # Extract the JSON data from the request
    data = request.json
    print(data)
    
    # Args:
    #     f (function): the function of interest. Its argument X should represent
    #         several spatial points for calculation (is 2D numpy array of the
    #         shape [samples, d]) if "is_vect" flag is True, and it is one
    #         spatial point for calculation (is 1D numpy array of the shape [d])
    #         in the case if "is_vect" flag is False. For the case of the tensor
    #         approximation (if "is_func" flag is False), the argument X relates
    #         to the one or many (depending on the value of the flag "is_vect")
    #         multi-indices of the corresponding array/tensor. Function should
    #         return the values in the requested points (is 1D numpy array of the
    #         shape [samples] of float or only one float value depending on the
    #         value of "is_vect" flag). If "with_opt" flag is True, then function
    #         should also return the second argument (is 1D numpy array of the
    #         shape [samples] of any or just one value depending on the "is_vect"
    #         flag) which is the auxiliary quantity corresponding to the
    #         requested points (it is used for debugging and in specific parallel
    #         calculations; the value of this auxiliary quantity related to the
    #         "argmin / argmax" point will be passed to "callback" function).
    #     d (int): number of function dimensions.
    #     a (float or list of len d of float): grid lower bounds for every
    #         dimension. If a number is given, then this value will be used for
    #         each dimension.
    #     b (float or list of len d of float): grid upper bounds for every
    #         dimension. If a number is given, then this value will be used for
    #         each dimension.
    #     n (int or list of len d of int): number of grid points for every
    #         dimension. If a number is given, then this value will be used for
    #         each dimension. If this parameter is not specified, then instead of
    #         it the values for both "p" and "q" should be set.
    #     p (int): the grid size factor (if is given, then there will be n=p^q
    #         points for each dimension). This parameter can be specified instead
    #         of "n". If this parameter is specified, then the parameter "q" must
    #         also be specified, and in this case the QTT-based approach will be
    #         used.
    #     q (int): the grid size factor (if is given, then there will be n=p^q
    #         points for each dimension). This parameter can be specified instead
    #         of "n". If this parameter is specified, then the parameter "p" must
    #         also be specified, and in this case the QTT-based approach will be
    #         used.
    #     evals (int or float): the number of requests to the target function
    #         that will be made.
    #     name (str): optional display name for the function of interest. It is
    #         the empty string by default.
    #     callback (function): optional function that will be called after each
    #         optimization step (in Func.comp_opt) with related info (it is used
    #         for debugging and in specific parallel calculations).
    #     x_opt_real (list of len d): optional real value of x-minimum or maximum
    #         (x). If this value is specified, then it will be used to display the
    #         current approximation error within the algorithm iterations (this
    #         is convenient for debugging and testing/research).
    #     y_opt_real (float): optional real value of y-optima (y=f(x)). If
    #         this value is specified, then it will be used to display the
    #         current approximation error within the algorithm iterations (this
    #         is convenient for debugging and testing/research).
    #     is_func (bool): if flag is True, then we optimize the function (the
    #         arguments of f correspond to continuous spatial points), otherwise
    #         we approximate the tensor (the arguments of f correspond to
    #         discrete multidimensional tensor multi-indices). It is True by
    #         default.
    #     is_vect (bool): if flag is True, then function should accept 2D
    #         numpy array of the shape [samples, d] (batch of points or indices)
    #         and return 1D numpy array of the shape [samples]. Otherwise, the
    #         function should accept 1D numpy array (one multidimensional point)
    #         and return the float value. It is True by default.
    #     with_cache (bool): if flag is True, then all requested values are
    #         stored and retrieved from the storage upon repeated requests.
    #         Note that this leads to faster computation if one point is
    #         computed for a long time. On the other hand, this can in some
    #         cases slow down the process, due to the additional time spent
    #         on checking the storage and using unvectorized slow loops in
    #         python. It is False by default.
    #     with_log (bool): if flag is True, then text messages will be
    #         displayed during the optimizer query process. It is False by
    #         default.
    #     with_opt (bool): if flag is True, then function of interest returns
    #         opts related to output y (scalar or vector) as second argument
    #         (it will be also saved and passed to "callback" function). It is
    #         False by default.
    #     with_full_info (bool): if flag is True, then the full information will
    #         be saved, including multi-indices of requested points (it is used
    #         by animation function) and best found multi-indices and points.
    #         Note that the inclusion of this flag can significantly slow down
    #         the process of the algorithm. It is False by default.
    #     with_wrn (bool): if flag is True, then warning messages will be
    #         presented (in the current version, it can only be messages about
    #         early convergence when using the cache). It is True by default.

    # We initialize the TTOpt class instance with the correct parameters:
    tto = TTOpt(
        f=f,                    # Function for minimization with data.X as parameter
        d=d,                    # Number of function dimensions
        a=-10.,                 # Grid lower bound (number or list of len d)
        b=+10.,                 # Grid upper bound (number or list of len d)
        p=2,                    # The grid size factor (there will n=p^q points)
        q=12,                   # The grid size factor (there will n=p^q points)
        evals=1.E+5,            # Number of function evaluations
        name='Alpine',          # Function name for log (this is optional)
        x_opt_real=np.ones(d),  # Real value of x-minima (x; this is for test)
        y_opt_real=0.,          # Real value of y-minima (y=f(x); this is for test)
        with_log=True)

    # And now we launching the minimizer:
    tto.optimize(rank)

    # We log the final state:
    print('-' * 70 + '\n' + tto.info() +'\n\n')
    
    # Run the minimization
    result = tto.info()
    
    # Return the result as a JSON response
    return jsonify({"minimum_value": result})


if __name__ == '__main__':
    # Run the Flask app on a specific port (e.g., 5000)
    app.run(host='0.0.0.0', port=5000)
