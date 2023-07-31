# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:06:16 2021

@author: dcasasor
"""


from PharmaPy.Connections import interpolate_inputs
import numpy as np


def analyze_controls(di):

    controls = {}

    for key, val in di.items():
        if isinstance(val, dict) and key != 'kwargs':
            if 'fun' not in val:
                raise KeyError("'%s' dictionary must have a 'fun' field")
            elif not callable(val['fun']):
                raise TypeError(
                    "Object passed to the 'fun' field must be a "
                    "callable with signature fun(time, *args, **kwargs)")

            out = val
        else:
            if not callable(val):
                raise TypeError(
                    "Object passed to the 'fun' field must be a "
                    "callable with signature fun(time, *args, **kwargs)")

            out = {'fun': val}

        if 'args' not in out:
            out['args'] = ()

        if 'kwargs' not in out:
            out['kwargs'] = {}

        controls[key] = out

    return controls


class DynamicInput:
    def __init__(self):
        self.controls = {}
        self.args_control = {}

        # Attributes assigned from UO instance
        self.parent_instance = None

    def add_variable(self, variable_name=None, function=None, data=None,
                     args_control=None):

        if function is not None:
            self.controls[variable_name] = function
        elif data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("Passed data must be a numpy array")
            elif data.ndim != 2:
                raise ValueError("Passed data must be two dimensional")

            self.controls[variable_name] = data

        else:
            raise ValueError("Please provide a callable using the 'function' "
                             "argument or data using the 'data' argument.")

        if args_control is None:
            args_control = ()

        self.args_control[variable_name] = args_control

    def evaluate_inputs(self, time):
        controls_out = {}

        for key, obj in self.controls.items():
            if callable(obj):
                args = self.args_control[key]
                controls_out[key] = obj(time, *args)

            else:
                t_inlet, y_inlet = obj.T
                controls_out[key] = interpolate_inputs(time, t_inlet, y_inlet)

        return controls_out
