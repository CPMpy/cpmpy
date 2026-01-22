"""
Parser for the MPS format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_mps   
    write_mps

========================
List of helper functions
========================

.. autosummary::
    :nosignatures:

    _parse_mps
    _load_mps
"""

from __future__ import annotations

import os
import cpmpy as cp
import numpy as np
from io import StringIO
from typing import Any, List, Optional, TextIO, Tuple, Union
from enum import Enum

from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.decompose_global import decompose_in_tree, decompose_objective
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_objective
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.linearize import linearize_constraint, only_positive_bv, only_positive_bv_wsum
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from cpmpy.transformations.safening import no_partial_functions, safen_objective


class ConstraintType(Enum):
    EQUAL = "E"             # ==
    GREATER_THAN = "G"      # >
    LESS_THAN = "L"         # <
    NON_CONSTRAINING = "N"  # objective

class VariableType(Enum):
    INTEGER = "I"
    CONTINUOUS = "C"      # not supported
    FLOATING_POINT = "F"  # not supported
    BINARY = "B"
    FREE = "F"            # not supported 
    CONSTANT = "X"        # only integers (for now float constants not supported, even in objective function)

def _get_constraint_type(constraint_type: str) -> ConstraintType:
    """
    Gets the constraint type from a string.

    Arguments:
        constraint_type (str): The constraint type string.

    Returns:
        ConstraintType: The constraint type.
    """
    if constraint_type == "E":
        return ConstraintType.EQUAL
    elif constraint_type == "G":
        return ConstraintType.GREATER_THAN
    elif constraint_type == "L":
        return ConstraintType.LESS_THAN
    elif constraint_type == "N":
        return ConstraintType.NON_CONSTRAINING
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}")

class MPS:

    def __init__(self, assume_interger_variables:bool=True):
        """
        Initializes the MPS object.

        Arguments:
            assume_interger_variables (bool): Whether to assume integer variables. Default is True. 
                                            If True, floating point variables will be converted to integer variables.
                                            If False, floating point variables will be kept as floating point variables 
                                            and an exception will be raised (cpmpy does not support floating point decision variables)
        """
        self.ASSUME_INTEGER_VARIABLES = assume_interger_variables
        self._metadata = dict()                # metadata on the MPS instance
        self._row_map = dict()                 # maps constraint names to types of constraint (ConstraintType)
        self.objective = None                            # name of the expression which represents the objective
        self.minimize = True                             # direction of optimisation
        self._A_matrix = {}                              # A matrix (variable x constraint)    
        self._rhs_map = dict()                 # right hand side of the expressions, maps expression name to its rhs
        self._lb_map = dict()                  # lower bounds of the variables, maps variable name to its lb
        self._ub_map = dict()                  # upper bounds of the variables, maps variable name to its ub
        self._type_map = dict()                # for each variable name, stores the type of variable it represents (VariableType)
        self._intorg = False                             # state management for the INTORG marker (in COLUMNS section)

    @property
    def metadata(self) -> dict:
        """
        Returns the metadata of the MPS instance.

        Returns:
            dict: The metadata of the MPS instance.
        """
        return self._metadata

    def _get_bounds(self, variable_name:str) -> Tuple:
        lb = self._lb_map.get(variable_name, 0) 
        if variable_name not in self._ub_map:
            raise ValueError(f"Upper bound not found for variable: {variable_name}. CPMpy does not support unbounded variables.")
        ub = self._ub_map[variable_name]
        return lb, ub


    def to_cpmpy(self, model_constants:bool=False, filter_zeros:bool=True) -> cp.Model:
        """
        Converts the MPS instance to a CPMpy model.

        Returns:
            cp.Model: The CPMpy model.
        """

        _var_map = dict()

        def _get_variable(variable_name: str):
            if variable_name not in _var_map:

                type = self._type_map.get(variable_name, VariableType.FREE)
                if type == VariableType.INTEGER:
                    _var_map[variable_name] = cp.intvar(name=variable_name, lb=self._get_bounds(variable_name)[0], ub=self._get_bounds(variable_name)[1])
                elif type == VariableType.FLOATING_POINT:
                    if self.ASSUME_INTEGER_VARIABLES:
                        _var_map[variable_name] = cp.intvar(name=variable_name, lb=int(self._get_bounds(variable_name)[0]), ub=int(self._get_bounds(variable_name)[1]))
                    else:
                        raise ValueError(f"Floating point variables are not supported: {variable_name}")
                elif type == VariableType.BINARY:
                    _var_map[variable_name] = cp.boolvar(name=variable_name)
                elif type == VariableType.CONSTANT:
                    if model_constants:
                        _var_map[variable_name] = cp.intvar(name=variable_name, lb=self._get_bounds(variable_name)[0], ub=self._get_bounds(variable_name)[0])
                    else:
                        _var_map[variable_name] = self._get_bounds(variable_name)[0]
                else:
                    raise ValueError(f"Invalid variable type: {type} for variable: {variable_name}")

            return _var_map[variable_name]
        
        def _get_variables(variable_names: list[str]):
            return np.array([_get_variable(variable_name) for variable_name in variable_names])

        model = cp.Model()

        inverted_A_matrix = self.invert_A_matrix()

        for constraint_name, constraint_type in self._row_map.items():
            print(constraint_name, constraint_type)
            if constraint_type == ConstraintType.NON_CONSTRAINING:
                print(inverted_A_matrix)
                obj_array = np.array(list(inverted_A_matrix[constraint_name].values())) * _get_variables(list(inverted_A_matrix[constraint_name].keys()))
                if filter_zeros:
                    obj_array = [o for o in obj_array if not (isinstance(o, (int, np.integer)) and o == 0)]
                objective = cp.sum(obj_array)
                if self.minimize:
                    model.minimize(objective)
                else:
                    model.maximize(objective)

            else:
                if constraint_type == ConstraintType.EQUAL:
                    lhs = np.array(list(inverted_A_matrix[constraint_name].values())) * _get_variables(list(inverted_A_matrix[constraint_name].keys()))
                    if filter_zeros:
                        lhs = [l for l in lhs if not (isinstance(l, (int, np.integer)) and l == 0)] 
                    model += cp.sum(lhs) == self._rhs_map[constraint_name]
                elif constraint_type == ConstraintType.GREATER_THAN:
                    lhs = np.array(list(inverted_A_matrix[constraint_name].values())) * _get_variables(list(inverted_A_matrix[constraint_name].keys()))
                    if filter_zeros:
                        lhs = [l for l in lhs if not (isinstance(l, (int, np.integer)) and l == 0)]
                    model += cp.sum(lhs) >= self._rhs_map[constraint_name]
                elif constraint_type == ConstraintType.LESS_THAN:
                    lhs = cp.cpm_array(list(inverted_A_matrix[constraint_name].values())) * _get_variables(list(inverted_A_matrix[constraint_name].keys()))
                    if filter_zeros:
                        lhs = [l for l in lhs if not (isinstance(l, int) and l.value != 0)]
                    model += cp.sum(list(lhs)) <= self._rhs_map[constraint_name] 
                else:
                    raise ValueError(f"Invalid constraint type: {constraint_type} for constraint: {constraint_name}")
            
        return model

    @staticmethod
    def _transform(cpm_cons: list[cp.Expression], csemap: dict) -> list[cp.Expression]:
        """
        Transforms a list of CPMpy expressions to a list of linearised expressions, compatible with the MPS format.

        Arguments:
            cpm_cons (list[cp.Expression]): The list of CPMpy expressions to transform.
            csemap (dict): The context-sensitive evaluation map.
        """
        # TODO: for now just straight copy from CPM_gurobi
        cpm_cons = toplevel_list(cpm_cons)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div"})  # linearize expects safe exprs
        supported = {"min", "max", "abs", "alldifferent"} # alldiff has a specialized MIP decomp in linearize
        cpm_cons = decompose_in_tree(cpm_cons, supported, csemap=csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']), csemap=csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=csemap)  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons, csemap=csemap)
        cpm_cons = only_implies(cpm_cons, csemap=csemap)  # anything that can create full reif should go above...
        print(cpm_cons)
        # gurobi does not round towards zero, so no 'div' in supported set: https://github.com/CPMpy/cpmpy/pull/593#issuecomment-2786707188
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "sub"}), csemap=csemap, prune_trivial=False)  # the core of the MIP-linearization
        print(cpm_cons)
        cpm_cons = only_positive_bv(cpm_cons, csemap=csemap)  # after linearization, rewrite ~bv into 1-bv
        return cpm_cons

    @staticmethod
    def _transform_objective(cpm_obj: cp.Expression, csemap: dict) -> cp.Expression:
        """
        Transforms a CPMpy expression to a linearised expression, compatible with the MPS format.
        """

        # transform objective
        obj, safe_cons = safen_objective(cpm_obj)
        obj, decomp_cons = decompose_objective(obj, csemap=csemap)
        obj, flat_cons = flatten_objective(obj, csemap=csemap)
        obj = only_positive_bv_wsum(obj)  # remove negboolviews

        additional_constraints = (safe_cons + decomp_cons + flat_cons)
        return obj, additional_constraints


    @staticmethod
    def from_cpmpy(model: cp.Model) -> MPS:
        """·
        Converts a CPMpy model to an MPS object.

        Arguments:
            model (cp.Model): The CPMpy model to convert.
        """
        cpm_expr = model.constraints
        for c in cpm_expr:
            print(c)
        csemap = dict()
        cpm_cons = MPS._transform(cpm_expr, csemap=csemap)
        for c in cpm_cons:
            print(c)

        mps_obj = MPS()


        # --------------------------------- Objective -------------------------------- #    

        objective, additional_constraints = MPS._transform_objective(model.objective_, csemap=csemap)

        objective_name = ('min' if model.minimize else 'max') + 'obj'
        mps_obj.minimize = model.minimize
        mps_obj.set_constraint_type(objective_name, ConstraintType.NON_CONSTRAINING)
        if objective.name == "wsum":
            weights, variables = objective.args
            for weight, variable in zip(weights, variables):
                mps_obj.update_column(objective_name, variable.name, weight)
        elif objective.name == "sum":
            variables_with_weights = objective.args
            weights, variables = zip[tuple[Any, ...]](*[(a.args[0], a.args[1]) if isinstance(a, cp.Operator) and a.name == "mul" and len(a.args) == 2 else (1, a) for a in variables_with_weights])
            for weight, variable in zip(weights, variables):
                mps_obj.update_column(objective_name, variable.name, weight)
        else:
            raise ValueError(f"Invalid constraint type: {type(objective)}")
        
        # -------------------------------- Constraints ------------------------------- #

        for i, cpm_con in enumerate(cpm_cons + additional_constraints):
            if isinstance(cpm_con, cp.expressions.core.Comparison):
                # Comparison type
                if cpm_con.name == "==":
                    mps_obj.set_constraint_type(f'c{i}', ConstraintType.EQUAL)
                elif cpm_con.name == ">=":
                    mps_obj.set_constraint_type(f'c{i}', ConstraintType.GREATER_THAN)
                elif cpm_con.name == "<=":
                    mps_obj.set_constraint_type(f'c{i}', ConstraintType.LESS_THAN)
                else:
                    raise ValueError(f"Invalid comparison operator: {cpm_con.name}")

                # LHS
                if cpm_con.args[0].name == "wsum":
                    weights, variables = cpm_con.args[0].args
                    for weight, variable in zip(weights, variables):
                        mps_obj.update_column(f'c{i}', variable.name, weight)
                elif cpm_con.args[0].name == "sum":
                    print(cpm_con)
                    variables_with_weights = cpm_con.args[0].args
                    weights, variables = tuple(zip(*[(a.args[0], a.args[1]) if isinstance(a, cp.expressions.core.Operator) and a.name == "mul" and len(a.args) == 2 else (1, a) for a in variables_with_weights]))
                    print("----")
                    print(weights, variables)
                    print(type(weights), type(variables))
                    for weight, variable in zip(weights, variables):
                        mps_obj.update_column(f'c{i}', variable.name, weight)
                else:
                    raise ValueError(f"Invalid constraint type: {type(cpm_con.args[0])}")

                # RHS
                mps_obj.update_rhs(f'c{i}', cpm_con.args[1])

            else:
                raise ValueError(f"Invalid constraint type: {type(cpm_con)}")

        # --------------------------------- Variables -------------------------------- #

        variables = get_variables(cpm_cons)
        for variable in variables:
            lb, ub = variable.get_bounds()
            if lb == ub:
                mps_obj.update_bounds(variable.name, "FX", lb)
            else:
                mps_obj.update_bounds(variable.name, "LI", lb)
                mps_obj.update_bounds(variable.name, "UI", ub)

        

        # ------------------------------------- - ------------------------------------ #
 
        return mps_obj

    @staticmethod
    def _format_space(string:str, space:Optional[int]=None, leading:int=0) -> str:
        if isinstance(string, int):
            string = str(string)
        if space is None:
            space=len(string)
        if len(string) <= space:
            return f"{'':<{leading}}{string:<{space}}"
        else:
            raise ValueError(f"String {string} is longer than {space} characters")

    @staticmethod
    def _format_line(strings, spaces, format:str, leading:int=0) -> str:
        if format == "fixed":
            line = MPS._format_space(strings[0], spaces[0], leading=leading)
            if len(strings) > 1:
                line +=''.join([MPS._format_space(string,space) for (string,space) in zip(strings[1:], spaces[1:])])
            return line
        elif format == "free":
            return MPS._format_space('', leading) + ' '.join(val for pair in zip(strings, spaces) for val in pair)
        else:
            raise ValueError(f"Invalid format: {format}")
      
    @staticmethod
    def _write_name(name, format:str) -> str:
        print(name)
        return MPS._format_line(('NAME', name), (14,None), format=format)

    @staticmethod
    def _write_objective(minimize:bool, format:str) -> str:
        return MPS._format_line(('N', f"{'min' if minimize else 'max'}obj"), (4, None), leading=1, format=format)

    @staticmethod
    def _write_row(row_name:str, constraint_type: ConstraintType, format:str) -> str:
        return MPS._format_line((constraint_type.value, row_name), (3, None), leading=1, format=format)
    
    @staticmethod
    def _write_opening_marker(format:str):
        return MPS._format_line(('MARK0000', 'MARKER', 'INTORG'), (10, 21, None), leading=4, format=format)

    @staticmethod
    def _write_closing_marker(format):
        return MPS._format_line(('MARK0001', 'MARKER', 'INTEND'), (10, 21, None), leading=4, format=format)

    @staticmethod 
    def _write_column(column_name:str, variables_with_coefficients:List[Tuple[str, int]], format:str) -> str:
        for a,b in zip(variables_with_coefficients[::2],variables_with_coefficients[1::2]):
            yield MPS._format_line((column_name, a[0], a[1], b[0], b[1]), (10, 21, 4, 21, None), leading=4, format=format)
        if len(variables_with_coefficients) % 2 != 0:
            yield MPS._format_line((column_name, variables_with_coefficients[-1][0], variables_with_coefficients[-1][1]), (10, 21, None), leading=4, format=format)

    @staticmethod
    def _write_rhs(variables_with_coefficients:List[Tuple[str, int]], format:str) -> str:
        for a,b in zip(variables_with_coefficients[::2],variables_with_coefficients[1::2]):
            yield MPS._format_line(('rhs', a[0], a[1], b[0], b[1]), (10, 21, 4, 21, None), leading=4, format=format)
        if len(variables_with_coefficients) % 2 != 0:
            yield MPS._format_line(('rhs', variables_with_coefficients[-1][0], variables_with_coefficients[-1][1]), (10, 21, None), leading=4, format=format)


    def write_mps(self, file_path: Optional[str] = None, name:Optional[str]=None, format: str = "fixed"):
        mps_string = []


        if format == "fixed":
            # Name
            print(name, type(name))
            name = name if name is not None else str(file_path).split(os.sep)[-1].split(".")[0].capitalize() if file_path is not None else "CPMPYMODEL"
            mps_string.append(self._write_name(name, format=format))
            # Rows
            mps_string.append("ROWS")
            # mps_string.append(self._write_objective(self.minimize, format=format))

            for row_name, constraint_type in self._row_map.items():
                mps_string.append(self._write_row(row_name, constraint_type, format=format))
            # Columns
            mps_string.append("COLUMNS")
            mps_string.append(self._write_opening_marker(format=format))
            for column_name, column_rows in self._A_matrix.items():
                for line in self._write_column(column_name, list(zip(column_rows.keys(), column_rows.values())), format=format):
                    mps_string.append(line)
            mps_string.append(self._write_closing_marker(format=format))
            # RHS
            mps_string.append("RHS")
            for line in self._write_rhs(list(zip(self._rhs_map.keys(), self._rhs_map.values())), format=format):
                mps_string.append(line)
            # Bounds
            mps_string.append("BOUNDS")
            print(self._A_matrix)
            for column_name in self._A_matrix.keys():
                if column_name == 'minobj' or column_name == 'maxobj':
                    continue
                variable_type = self._type_map[column_name]
                if variable_type == VariableType.FLOATING_POINT:
                    if column_name in self._lb_map:
                        mps_string.append(self._format_line(('LO', 'bnd', column_name, self._lb_map.get(column_name, 0)), (3, 10, 21, None), leading=1, format=format))
                    if column_name in self._ub_map:
                        mps_string.append(self._format_line(('UP', 'bnd', column_name, self._ub_map[column_name]), (3, 10, 21, None), leading=1, format=format))
                elif variable_type == VariableType.INTEGER:
                    if column_name in self._lb_map:
                        mps_string.append(self._format_line(('LI', 'bnd', column_name, self._lb_map.get(column_name, 0)), (3, 10, 21, None), leading=1, format=format))
                    if column_name in self._ub_map:
                        mps_string.append(self._format_line(('UI', 'bnd', column_name, self._ub_map[column_name]), (3, 10, 21, None), leading=1, format=format))
                elif variable_type == VariableType.BINARY:
                    mps_string.append(self._format_line(('BV', 'bnd', column_name, self._lb_map.get(column_name, 0)), (3, 10, 21, None), leading=1, format=format))
                elif variable_type == VariableType.CONSTANT:
                    mps_string.append(self._format_line(('FX', 'bnd', column_name, self._lb_map[column_name]), (3, 10, 21, None), leading=1, format=format))
                else:
                    raise ValueError(f"Invalid variable type: {variable_type} for variable: {column_name}")
            # End
            mps_string.append("ENDATA")

        mps_string = "\n".join(mps_string)

        if file_path is not None:
            with open(file_path, "w") as f:
                f.write(mps_string)

        return mps_string

    def set_objective(self, expression_name: str):
        """
        Sets the name of the expression that represents the objective.

        Arguments:
            expression_name (str): The name of the expression that represents the objective.
        """
        self.objective = expression_name

    def set_constraint_type(self, constraint_name: str, constraint_type: ConstraintType):
        """
        Sets the type of a constraint.

        Arguments:
            constraint_name (str): The name of the constraint.
            constraint_type (ConstraintType): The type of the constraint.
        """
        self._row_map[constraint_name] = constraint_type

    def set_marker(self, marker: str):
        """
        Sets the marker for the INTORG/INTEND section.

        Arguments:
            marker (str): The marker to set.
        """
        if "'INTORG'" == marker:
            self._intorg = True
        elif "'INTEND'" == marker:
            self._intorg = False

    def update_column(self, column_name: str, row_name: str, row_coeff: str):
        """
        Updates the A matrix.

        Arguments:
            column_name (str): The name of the column.      constraint name
            row_name (str): The name of the row.            variable name
            row_coeff (str): The coefficient of the row.
        """
        if self._intorg:
            row_coeff = int(row_coeff)
        else:
            if self.ASSUME_INTEGER_VARIABLES:
                row_coeff = int(row_coeff)
            else:
                raise ValueError(f"Floating point variables are not supported: {row_coeff}")
        self._A_matrix[row_name] = self._A_matrix.get(row_name, {}) | {column_name: row_coeff}

    def update_rhs(self, row_name: str, row_coeff: str):
        """
        Updates the right hand side of a constraint.

        Arguments:
            row_name (str): The name of the constraint.
            row_coeff (str): The right hand side of the constraint.
        """
        if self._intorg:
            row_coeff = int(row_coeff)
        else:
            if self.ASSUME_INTEGER_VARIABLES:
                row_coeff = int(row_coeff)
            elif row_coeff != int(row_coeff):
                raise ValueError(f"Floating point variables are not supported: {row_coeff}")
            else:
                row_coeff = int(row_coeff)
        self._rhs_map[row_name] = row_coeff

    def update_bounds(self, row_name: str, type: str, bound_value: str):
        """
        Updates the bounds of a variable.

        Arguments:
            row_name (str): The name of the variable.
            type (str): The type of the bound.
            bound_value (str): The value of the bound.
        """
        if type == "LO":
            self._type_map[row_name] = VariableType.FLOATING_POINT
            if self.ASSUME_INTEGER_VARIABLES:
                self._lb_map[row_name] = int(bound_value)
            else:
                if bound_value != int(bound_value):
                    raise ValueError(f"Floating point bounds are not supported: {bound_value}")
                self._lb_map[row_name] = int(bound_value)
        elif type == "UP":
            self._type_map[row_name] = VariableType.FLOATING_POINT
            if self.ASSUME_INTEGER_VARIABLES:
                self._ub_map[row_name] = int(bound_value)
            else:
                if bound_value != int(bound_value):
                    raise ValueError(f"Floating point bounds are not supported: {bound_value}")
                self._ub_map[row_name] = int(bound_value)
        elif type == "FX":
            self._type_map[row_name] = VariableType.CONSTANT
            if bound_value != int(bound_value):
                if self.ASSUME_INTEGER_VARIABLES:
                    bound_value = int(bound_value)
                else:
                    raise ValueError(f"Floating point bounds are not supported: {bound_value}")        
            self._lb_map[row_name] = int(bound_value)
            self._ub_map[row_name] = int(bound_value)
        elif type == "BV":
            self._type_map[row_name] = VariableType.BINARY
            self._lb_map[row_name] = 0
            self._ub_map[row_name] = 1
        elif type == "LI":
            self._type_map[row_name] = VariableType.INTEGER
            self._lb_map[row_name] = int(bound_value)
        elif type == "UI":
            self._type_map[row_name] = VariableType.INTEGER
            self._ub_map[row_name] = int(bound_value)
        elif type == "SC":
            pass
        elif type == "SI":
            pass
        elif type == "FR":
            pass
        elif type == "MI":
            pass
        elif type == "PL":
            pass
        else:
            raise ValueError(f"Invalid bound type: {type}")

    def invert_A_matrix(self):
        """
        Inverts the A matrix, becoming a (constraint x variable) matrix.

        Returns:
            dict: The inverted A matrix.
        """
        inverted_A_matrix = dict()
        for column_name, column_rows in self._A_matrix.items():
            for row_name, row_coeff in column_rows.items():
                inverted_A_matrix[row_name] = inverted_A_matrix.get(row_name, {}) | {column_name: row_coeff}
        return inverted_A_matrix
        
    @staticmethod
    def _read_line(line, starts:List[int], format:str, required:Optional[List[bool]]=None) -> List:
        if required is not None:
            for i, (s, r) in enumerate(zip(starts, required)):
                if s >= len(line):
                    if r:
                        raise ValueError(f"Required field {i} is missing")
                    else:
                        i -= 1
                        break
            starts = starts[:i+1]
        if format == "fixed":
            res = []
            for a,b in zip(starts[:], starts[1:]):
                res.append(line[a:b].strip())
            res.append(line[starts[-1]:].strip())
            if required is not None:
                res += [None]*(len(required)-len(starts))
            return res
        elif format == "free":
            return line.split() + [None]*(len(required)-len(starts)) if required is not None else line.split()
        else:
            raise ValueError(f"Invalid format: {format}")


def _parse_mps(f: TextIO, format: str = "fixed", **kwargs) -> MPS:
    """
    Parses an MPS string and returns an MPS object.

    Arguments:
        mps (str): The MPS string to parse.
    """

    mps_obj = MPS()  

    lines = f.readlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        print(line)

        if line.startswith("NAME"):
            mps_obj._metadata["name"] = mps_obj._read_line(line, (14,), format=format)[0]
            i += 1
            line = lines[i]
        elif line.startswith("OBJSENSE"): # optional, not part of core specification
            direction = mps_obj._read_line(line, (9,), format=format)
            if direction == "MIN":
                pass # default is minimize
            elif direction == "MAX":
                mps_obj.minimize = False
            else:
                raise ValueError(f"Invalid optimisation direction: {direction}")
            i += 1  
        elif line.startswith("*"): # comment line
            i += 1
        elif line.startswith("ROWS"): # name of constraints
            i += 1
            line = lines[i]
            while i < len(lines) and (line[0] == " " or line[0] == "*"):
                # create mapping of constraint name to constraint type
                constraint_type, constraint_name = mps_obj._read_line(line, (1, 4), format=format)
                constraint_type = _get_constraint_type(constraint_type.lstrip()) # operators can be in column 2 or 3
                print(constraint_name)
                if constraint_type == ConstraintType.NON_CONSTRAINING:
                    mps_obj.set_objective(constraint_name)
                mps_obj.set_constraint_type(constraint_name, constraint_type)
                i += 1
                line = lines[i]
        elif line.startswith("COLUMNS"):
            i += 1
            line = lines[i]
            while i < len(lines) and (line[0] == " " or line[0] == "*"):                
                if len(line) >= 32 and line[14:20] == "MARKER":
                    mps_obj.set_marker(line[24:34])
                else:
                    column_name, row_name, row_coeff, row2_name, row2_coeff = mps_obj._read_line(line, (4, 14, 35, 39, 60), required=(True, True, True, False, False), format=format)
                    mps_obj.update_column(row_name, column_name, row_coeff)
                    if row2_name is not None:
                        mps_obj.update_column(row2_name, column_name, row2_coeff)
                i += 1
                line = lines[i]
        elif line.startswith("RHS"):
            i += 1
            line = lines[i]
            while i < len(lines) and (line[0] == " " or line[0] == "*"):

                row_name, row_coeff, row2_name, row2_coeff = mps_obj._read_line(line, (14, 24, 39, 49), required=(True, True, False, False), format=format)
                mps_obj.update_rhs(row_name, row_coeff)
                if row2_name is not None:
                    mps_obj.update_rhs(row2_name, row2_coeff)
                i += 1
                line = lines[i]
        elif line.startswith("BOUNDS"):
            i += 1
            line = lines[i]
            while i < len(lines) and (line[0] == " " or line[0] == "*"):
                type, _, row_name, bound_value = mps_obj._read_line(line, (1, 3, 14, 35), required=(True, True, True, False), format=format)
                if bound_value is None:
                    bound_value = 0
                print(line)
                print(row_name, type, bound_value)
                mps_obj.update_bounds(row_name, type, bound_value)               
                i += 1
                line = lines[i]
        elif line.startswith("ENDATA"):
            break
        else:
            raise ValueError(f"Invalid line: {line}")
            i += 1

    return mps_obj

def _load_mps(mps_obj: MPS, **kwargs) -> cp.Model:
    """
    Loads an MPS object into a CPMpy model.

    Arguments:
        mps_obj (MPS): The MPS object to load.
    """
    return mps_obj.to_cpmpy(**kwargs)


_std_open = open
def read_mps(mps: Union[str, os.PathLike], open=open, format:str="fixed", **kwargs) -> cp.Model:
    """
    Parser for MPS format. Reads in an instance and returns its matching CPMpy model.

    Arguments: 
        mps (str or os.PathLike):
            - A file path to a MPS file
            - OR a string containing the MPS content directly
        open: (callable):
            If mps is the path to a file, a callable to "open" that file (default=python standard library's 'open').
        format: (str):
            The format of the MPS file. Can be "fixed" or "free". Default is "fixed".
    """

    # If mps is a path to a file -> open file
    if isinstance(mps, (str, os.PathLike)) and os.path.exists(mps):
        if open is not None:
            f = open(mps)
        else:
            f = _std_open(mps, "rt")
    # If mps is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(mps)

 
    mps_obj = _parse_mps(f, format=format, **kwargs)
    model = _load_mps(mps_obj, **kwargs)
    return model
   

def write_mps(model: cp.Model, file_path: Optional[str] = None, format: str = "fixed") -> str:
    """
    Writes a CPMpy model to an MPS string / file.

    Arguments:
        model (cp.Model): The CPMpy model to write.
        file_path (Optional[str]): Optional path to the MPS file to write.

    Returns:
        str: The MPS string.
    """
    mps_obj = MPS.from_cpmpy(model)
    print("======")
    print(mps_obj._A_matrix)
    mps_string = mps_obj.write_mps(file_path, format=format)
    return mps_string



