import cpmpy as cp
from cpmpy.expressions.core import Operator

from xml.etree import ElementTree as ET
import re

from cpmpy.expressions.variables import NDVarArray



"""
    Initial version of a XCSP3-core parser for CPMpy
    
    Currently implemented:
        - variable creation with arrays
        - array of variables with different domains
        - constraint groups
        - constraint blocks
        - constraints:
            - simple arithmetic
            - alldiff
            - gcc
            - minimum
            - maximum

    Features to add for sure:
        - creating variables without arrays
        - wildcarting of %...
        - objective functions
        - rest of constraints
    Nice to haves:
        - 
        - ...?
    NOT to add:
        - set variables
        - ...?
"""

class XCSPParser(cp.Model): # not sure if we should subclass Model

    def __init__(self, fname):
        super().__init__()
        tree = ET.parse(fname)
        # check instance
        instance = tree.getroot()
        self.varpool = dict()

        xml_vars = instance.findall("./variables/")
        self.parse_variables(xml_vars)

        self.abstract_args = None
        for cons in self.parse_constraint(instance.findall("./constraints/")):
            self += cons

    def parse_variables(self, lst_of_defs):
        #
        for xml_el in lst_of_defs:
            if xml_el.tag == "array":  # dispatch to multidim
                self.parse_n_dimvar(xml_el)
            elif xml_el.tag =='var':
                self.parse_singular_var(xml_el)
            else:
                raise ValueError(f"Unknown variable {xml_el}, todo?")

    def parse_singular_var(self,xml_element):
        varname = xml_element.attrib['id']
        description = xml_element.attrib['note']
        lb, ub = self.parse_text_domain(xml_element.text)
        if lb == 0 and ub == 1:
            self.varpool[varname] = cp.boolvar(name=varname,description=description)
        else:
            self.varpool[varname] = cp.intvar(lb, ub, name=varname)

    def parse_n_dimvar(self, xml_element):

        assert xml_element.tag == "array"

        varname = xml_element.attrib['id']
        #TODO we can add the note as a description
        str_shape = xml_element.attrib['size']
        shape = tuple([int(i.strip("[]")) for i in str_shape.split("][")])

        domains = xml_element.findall('domain')
        if domains == []: #domain is same for all vars
            lb, ub = self.parse_text_domain(xml_element.text)
            if lb == 0 and ub == 1:
                self.varpool[varname] = cp.boolvar(shape=shape, name=varname)
            else:
                self.varpool[varname] = cp.intvar(lb, ub, shape=shape, name=varname)
        else: #different domains
            #start by making the vars with arbitrary domain, correct it later
            self.varpool[varname] = cp.intvar(1,1, shape =shape, name=varname)
            for domain in domains:
                vrs, lb, ub = self.parse_domain(domain)
                for var in vrs:
                    cpm_vars = self.get_vars(var)
                    if isinstance(cpm_vars, list):
                        for cpm_var in cpm_vars:
                            cpm_var.lb = lb
                            cpm_var.ub= ub
                    else:
                        cpm_vars.lb = lb
                        cpm_vars.ub = ub

    def get_vars(self, str_var: str):
        """
            Translate string variable to CPMpy var
             -> also supports multi-dimensional vars!
        """
        if re.fullmatch("\%[0-9]", str_var):
            return self.abstract_args[int(str_var.strip("%"))]

        if re.fullmatch("\%...", str_var):
            return self.abstract_args

        if "[" not in str_var:  # simple var, value or interval
            if '..' in str_var:  # interval of values (not vars)
                i, j = str_var.split('..')
                return [x for x in range(int(i), int(j) + 1)]
            else: #simple var/value
                return self.varpool.get(str_var, int(str_var))

        if '(' in str_var: #it's a subexpression used as a var
            return self.parse_intension(str_var)

        # multi-dimensional var, find indices to pass
        split = str_var.index("[")
        name, indices = str_var[:split], str_var[split:]

        np_index = ""
        for idx in re.findall("\[[0-9]*\.*[0-9]*\]", indices):
            idx = idx.strip("[]")
            if idx == "":
                np_index += ":,"
            elif ".." in idx:
                i, j = idx.split("..")
                np_index += f"{int(i)}:{int(j) +1},"
            else:
                np_index += idx + ","

        var = eval(f"self.varpool[name][{np_index}]")

        if isinstance(var, NDVarArray):
            return var.flatten().tolist()
        return var

    def parse_domain(self, domain):
        lb, ub = self.parse_text_domain(domain.text)
        vars = domain.attrib['for']
        return vars.split(' '), lb, ub


    def parse_text_domain(self, txt):

        if ".." in txt:
            # integer range
            lb, ub = txt.split("..")
            return int(lb), int(ub)
        elif txt.strip() == "0 1":
            return 0, 1

        raise ValueError("Unknown domain:", txt)

    # parsers for all constraints
    def parse_constraint(self, xml_cons):

        if isinstance(xml_cons, list):
            return [self.parse_constraint(cons) for cons in xml_cons]

        if xml_cons.tag == "group":
            return self.parse_group(xml_cons)
        else:
            # find parsing function name
            func = eval(f"self.parse_{xml_cons.tag.lower()}")
            return func(xml_cons)

    def parse_group(self, group):
        xml_cons = next(iter(group)) # find template
        constraints = []
        for arglist in group.findall("./args"):
            # find each fill for template and store
            self.abstract_args = []
            for str_var in arglist.text.strip().split(" "):
                cpm_var = self.get_vars(str_var)
                if isinstance(cpm_var, list):
                    self.abstract_args += cpm_var
                else:
                    self.abstract_args.append(cpm_var)
            # call constructor of template with filled in args
            constraints.append(self.parse_constraint(xml_cons))

        return constraints

    def parse_block(self, block):
        cons = []
        for constraint in block:
            cons.append(self.parse_constraint(constraint))
        return cons

    funcmap = {
        # Arithmetic
        "neg": (1, lambda x: -x),
        "abs": (1, lambda x: abs(x)),
        "add": (0, lambda x: cp.sum(x)),
        "sub": (2, lambda x, y: x - y),
        "mul": (2, lambda x, y: x * y),
        "div": (2, lambda x, y: x / y),
        "mod": (2, lambda x, y: x % y),
        "sqr": (1, lambda x: x ** 2),
        "pow": (2, lambda x, y: x ** y),
        "min": (0, lambda x: cp.min(x)),
        "max": (0, lambda x: cp.max(x)),
        "dist": (2, lambda x, y: abs(x - y)),
        # Relational
        "lt": (2, lambda x, y: x < y),
        "le": (2, lambda x, y: x <= y),
        "ge": (2, lambda x, y: x >= y),
        "gt": (2, lambda x, y: x > y),
        "ne": (2, lambda x, y: x != y),
        "eq": (0, lambda x: x[0] == x[1] if len(x) == 2 else cp.AllEqual(x)),
        # Set
        'in': (2, lambda x, y: cp.InDomain(x,y)),
        # Logic
        "not": (1, lambda x: ~x),
        "and": (0, lambda x: cp.all(x)),
        "or": (0, lambda x: cp.any(x)),
        "xor": (0, lambda x: cp.Xor(x)),
        "iff": (0, lambda x: cp.all(a == b for a, b in zip(x[:-1], x[1:]))),
        "imp": (2, lambda x, y: x.implies(y)),
        # control
        "if": (3, lambda b, x, y: cp.IfThenElse(b, x, y))
    }

    def parse_intension(self, xml_cons):
        """
            Parse recursive functions
        """
        if isinstance(xml_cons, ET.Element):
            #TODO: expression can be in <function> block
            txt = xml_cons.text.strip()
        elif isinstance(xml_cons, str):
            txt = xml_cons.strip()
        else:
            raise ValueError(f"Unknown argument {xml_cons}")

        if "(" not in txt:
            # found a leave node
            if "," in txt:
                return [self.get_vars(str_var) for str_var in txt.split(",")]
            return self.get_vars(txt)

        # find function name
        i = txt.index("(")
        fname, str_args = txt[:i], txt[i + 1:-1]
        arity, cpm_op = self.funcmap[fname]

        start, level = 0, 0
        cpm_args = []
        for i, c in enumerate(str_args):
            if c == "(": level += 1
            if c == ")": level -= 1
            if c == "," and level == 0:
                cpm_args += [self.parse_intension(str_args[start:i])]
                start = i + 1

        cpm_args.append(self.parse_intension(str_args[start:]))

        if arity != 0 and isinstance(cpm_args, list):
            return cpm_op(*cpm_args)

        return cpm_op(cpm_args)

    def _cpm_vars_from_attr(self, xml_vars):
        """ helper function to get variables"""
        cpm_vars = []
        for v in xml_vars.text.strip().split(" "):
            cpm_var = self.get_vars(v)
            if isinstance(cpm_var, list):
                cpm_vars += cpm_var
            else:
                cpm_vars.append(cpm_var)
        return cpm_vars

    def parse_extension(self, xml_cons): # table constraint
        cpm_list = self._cpm_vars_from_attr(xml_cons.find("./list"))
        if xml_cons.find("./supports") is not None: #positife table constraint
            suptxt = xml_cons.find("./supports").text.strip()
            if '*' in suptxt: #wildcards
                raise NotImplementedError()
            else:
                tab = self.get_table_values(suptxt)
                return cp.Table(cpm_list,tab)
        elif xml_cons.find("./conflicts") is not None: #negative table constraint
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def parse_sum(self, xml_cons):

        cpm_vars = self._cpm_vars_from_attr(xml_cons.find("./list"))
        condition = xml_cons.find("./condition")
        operator, rhs = condition.text.strip()[1:-1].split(",")
        cpm_rhs = self.get_vars(rhs)
        arity, cpm_op = self.funcmap[operator]

        #if var pattern is just %... we will take all of them, including the rhs (because we only learn here which one it is)
        #so remove it from the lhs. #TODO same for other constraints?
        try:
            cpm_vars.remove(cpm_rhs)
        except ValueError:
            pass

        coeffs = xml_cons.find("./coeffs")
        if coeffs is None:
            lhs = cp.sum(cpm_vars)
        else:
            weights = [int(c) for c in coeffs.text.strip().split(" ")]
            lhs = Operator("wsum", [weights, cpm_vars])

        if arity == 0:
            return cpm_op([lhs, self.get_vars(rhs)])
        return cpm_op(lhs, self.get_vars(rhs))

    def parse_count(self, xml_cons):
        cpm_list = self._cpm_vars_from_attr(xml_cons.find("./list"))
        cpm_values = self._cpm_vars_from_attr(xml_cons.find("./values"))
        assert len(cpm_values) == 1 #only 1 value is supported
        cpm_value = cpm_values[0]
        condition = xml_cons.find("./condition")
        operator, rhs = condition.text.strip()[1:-1].split(",")
        cpm_rhs = self.get_vars(rhs)

        # if var pattern is just %... we will take all of them, including the rhs (because we only learn here which one it is)
        # so remove it from the lhs.
        try:
            cpm_list.remove(cpm_rhs)
        except ValueError:
            pass
        arity, cpm_op = self.funcmap[operator]
        return cpm_op([cp.Count(cpm_list,cpm_value), cpm_rhs])

    def parse_nvalues(self, xml_cons):
        raise NotImplementedError()

    def parse_cardinality(self, xml_cons):
        cpm_vars = self._cpm_vars_from_attr(xml_cons.find("./list"))
        vals = xml_cons.find("./values")
        vals = [int(v) for v in vals.text.strip().split(" ")]
        counts = self._cpm_vars_from_attr(xml_cons.find("./occurs"))

        return cp.GlobalCardinalityCount(cpm_vars, vals, counts)

    def parse_regular(self, xml_cons):
        raise NotImplementedError("Regular is not supported")

    def parse_mdd(self, xml_cons):
        raise NotImplementedError("MDD is not supported")

    def parse_minimum(self, xml_cons):
        cpm_vars = self._cpm_vars_from_attr(xml_cons.find("./list"))
        condition = xml_cons.find("./condition")
        operator, rhs = condition.text.strip()[1:-1].split(",")
        cpm_rhs = self.get_vars(rhs)
        arity, cpm_op = self.funcmap[operator]
        return cpm_op([cp.Minimum(cpm_vars), cpm_rhs])

    def parse_maximum(self, xml_cons):
        cpm_vars = self._cpm_vars_from_attr(xml_cons.find("./list"))
        condition = xml_cons.find("./condition")
        operator, rhs = condition.text.strip()[1:-1].split(",")
        cpm_rhs = self.get_vars(rhs)
        arity, cpm_op = self.funcmap[operator]
        return cpm_op([cp.Maximum(cpm_vars), cpm_rhs])

    def parse_element(self, xml_cons):
        raise NotImplementedError()

    def parse_channel(self, xml_cons):
        lists = xml_cons.findall("./list")
        if len(lists) == 0: #simplified form
            raise NotImplementedError()
        elif len(lists) == 1:
            value = xml_cons.find("./value/")
            if value is None: # just one list
                raise NotImplementedError()
            else: # list and value
                raise NotImplementedError()
        elif len(lists) == 2:
            for list in lists:
                if list.find('./startIndex') is not None:
                    raise NotImplementedError()
            cpm_list1 = self._cpm_vars_from_attr(lists[0])
            cpm_list2 = self._cpm_vars_from_attr(lists[1])[0:len(cpm_list1)] #make them same length, since last part is irrelevant anyway
            return cp.Inverse(cpm_list1,cpm_list2)
        else:
            raise NotImplementedError()

    def parse_alldifferent(self, xml_cons):
        if xml_cons.text != "":
            # simplified version
            cpm_vars = [self.get_vars(v) for v in xml_cons.text.strip().split(" ")]
            return cp.AllDifferent(*cpm_vars)
        else:
            raise NotImplementedError("Extended version of alldiff not implemented")

    def parse_allequal(self, xml_cons):
        raise NotImplementedError()

    def parse_stretch(self, xml_cons):
        raise NotImplementedError("Stretch is not supported")

    def parse_nooverlap(self, xml_cons):
        raise NotImplementedError("No overlap is not supported for now")

    def parse_cumulative(self, xml_cons):
        raise NotImplementedError()

    def parse_ordered(self, xml_cons):
        #TODO when we have a global constraint for this
        raise NotImplementedError()

    def get_table_values(self, suptxt):
        if '..' in suptxt: #split intervals
            if '(' in suptxt: #tuples
                ii = suptxt.index("..")
                nn = suptxt.index(')', ii)
                mm = suptxt.rindex('(', 0, ii)
                splice = suptxt[mm:nn + 1]
                i = splice.index('..')
                n = splice.find(',', i)
                m = splice.rfind(',', 0, i)
                if m == -1:
                    ssplice = splice[m + 2:n]
                else:
                    ssplice = splice[m + 1:n]
                a, b = ssplice.split("..")
                extended = suptxt[:mm] + suptxt[nn + 1:]
                if m == -1: #if interval is first element of tuple we need an extra '('
                    m += 1
                for j in range(int(a), int(b) + 1):
                    extended += splice[:m + 1] + str(j) + splice[n:]
                return self.get_table_values(extended)
            else: #singular values
                i = suptxt.index("..")
                n = suptxt.find(' ', i)
                m = suptxt.rfind(' ', 0, i)

                if n == -1:
                    splice = suptxt[m + 1:]
                else:
                    splice = suptxt[m + 1:n]
                a, b = splice.split("..")

                if m == -1 and n == -1:  # it's the only element
                    extended = ""
                elif m == -1:
                    extended = suptxt[n + 1:]
                elif n == -1:
                    extended = suptxt[:m]
                else:
                    extended = suptxt[:m] + suptxt[n:]
                for j in range(int(a), int(b) + 1):
                    extended += ' '
                    extended += str(j)

                return self.get_table_values(extended)
        elif '(' in suptxt: #multiple options
            tab = []
            for tpl in suptxt[1:-1].split(')('):
                vals = tuple([int(val) for val in tpl.split(',')])
                tab.append(vals)
            return tab
        elif '*' in suptxt:
            raise NotImplementedError()
        else: #just one option
            return [tuple([int(val)]) for val in suptxt.split(" ")]





if __name__ == "__main__":


    import os
    from cpmpy.transformations.get_variables import get_variables
    from cpmpy.transformations.normalize import toplevel_list

    dir = "C:\\Users\\wout\\Downloads\\CSP23"
    fnames = [fname for fname in os.listdir(dir) if fname.endswith(".xml")]
    for fname in sorted(fnames)[114:]:
        print(fname)
        model = XCSPParser(os.path.join(dir,fname))
        print(model)
        if model.solve(time_limit=20):
            print('sat')
        print(model.status())
