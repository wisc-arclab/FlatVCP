# FlatVCP: Optimal Control for Kinematic Bicycle Model
This repository contains source code for the paper Optimal Control for Kinematic Bicycle Model with Continuous-time Safety Guarantees: A Sequential Second-order Cone Programming Approach.
The repository contains both Python3 and MATLAB code.

If you use this code, we would appreciate it if you cited as follows:
```
@article{freire2022optimal,
  title={Optimal Control for Kinematic Bicycle Model with Continuous-time Safety Guarantees: A
Sequential Second-order Cone Programming Approach},
  author={Freire, Victor and Xu, Xiangru},
  journal={arXiv preprint arXiv:2204.08980},
  year={2022}
}
```

## Python3
### Dependencies
The repository is structured as a Python3 package. Thus, you can run the following command to
install the dependencies.
```
pip3 install -r requirements.txt
```
For [cvxpy](https://www.cvxpy.org/), we recommend using [MOSEK](https://www.mosek.com/) noting
that you will need a license (free academic license available). We also tested
[ECOS](https://web.stanford.edu/~boyd/papers/ecos.html) with satisfactory performance.

### Example
For a quick-start, inspect the file `examples/fig_s.py` and run it with:
```
python3 examples/fig_s.py
```

The output should be a trajectory file `traj_figS.csv` with the generated state-space
trajectory and the following plot of the x-y trajectory.

<img
src="https://github.com/ARC-Lab-Research-Group/FlatVCP/img/bk_python_example.png"
width="600" alt="Bicycle Python Example">

Note that this example uses [matplotlib](https://matplotlib.org/) for visualization.


## MATLAB
### Dependencies
The MATLAB implementation requires a working installation of
[YALMIP](https://yalmip.github.io/) and we recommend using [MOSEK](https://www.mosek.com/).
noting that you will need a license (free academic license available). However, any
second-order cone programming solver should also work.

Also required is the
[B-splines](https://www.mathworks.com/matlabcentral/fileexchange/27374-b-splines) Add-On by
[Levente Hunyadi](https://www.mathworks.com/matlabcentral/profile/authors/1879353).

You might also find the following MATHWORKS toolboxes useful/necessary:
* Optimization Toolbox
* Symbolic Math Toolbox
* Control Systems Toolbox

### Example

