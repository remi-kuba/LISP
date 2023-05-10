import sys
import traceback
sys.setrecursionlimit(20_000)


class SchemeError(Exception):
    pass


class SchemeSyntaxError(SchemeError):
    pass


class SchemeNameError(SchemeError):
    pass


class SchemeEvaluationError(SchemeError):
    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    # Break up new lines and remove comments
    source = [e[:(e + ";").find(";")].replace("(", " ( ").replace(")", " ) ")
        for e in source.split("\n") if e[:(e + ";").find(";")]]
    # Split elements in each line
    return [j.strip() for e in source for j in e.split()]


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    tokens = [number_or_symbol(t) if t not in ("(",")") else t for t in tokens]
    def parens_counter(string):
        return sum(c == "(" for c in string) != sum(c == ")" for c in string)
    if tokens[0] != "(" or tokens[-1] != ")" or parens_counter(tokens):
        if len(tokens) != 1 or tokens[0] in ("(",")"):
            raise SchemeSyntaxError
    def parse_expression(index):
        if tokens[index] == "(":
            nested_list = []
            index += 1 
            while index < len(tokens) and tokens[index] != ")":
                sublist,index = parse_expression(index)
                nested_list.append(sublist)
            return nested_list, index + 1
        else:
            return tokens[index],index+1
    return parse_expression(0)[0]


######################
# Built-in Functions #
######################

scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": lambda args: args[0] if len(args) == 1 else (args[0]
        * scheme_builtins["*"](args[1:])),
    "/": lambda args: args[0] if len(args) == 1 else (scheme_builtins["/"](
        args[:-1]) / args[-1]),
    "equal?": lambda args: args[0] == args[1] if len(args) == 2 else (args[0]
        == args[1] and scheme_builtins["equal?"](args[2:])),
    ">": lambda args: args[0] > args[1] if len(args) == 2 else (args[0] > args[1] \
        and scheme_builtins[">"](args[1:])),
    ">=": lambda args: args[0] >= args[1] if len(args) == 2 else (args[0] >= args[1] \
        and scheme_builtins[">="](args[1:])),
    "<": lambda args: args[0] < args[1] if len(args) == 2 else (args[0] < args[1] \
        and scheme_builtins["<"](args[1:])),
    "<=": lambda args: args[0] <= args[1] if len(args) == 2 else (args[0] <= args[1] \
        and scheme_builtins["<="](args[1:])),
    "not": lambda arg: not evaluate(arg[0]) if len(arg) == 1 else "raiseError",
    "list?": lambda arg: True if arg[0] is None else False if not isinstance(arg[0],
        Pair) else arg[0].cdr is None if not isinstance(arg[0].cdr,Pair
        ) else scheme_builtins["list?"]([arg[0].cdr]),
    "length": lambda arg: "raiseError" if not scheme_builtins["list?"](arg
        ) else 0 if arg[0] is None else (arg[0]).length(),
    "list-ref": lambda args: (args[0].car if isinstance(args[0],Pair) and args[1] == 0 
        else "raiseError") if not scheme_builtins["list?"](args[0]) or args[1] >= 
        (args[0]).length() else ((args[0])[args[1]]).car,
    "append": lambda args: None if not args else concat_lists(args),
    "map": lambda args: None if args[1] is None else "raiseError" if not
        scheme_builtins["list?"](args[1]) else evaluate(args[1]).funct_copy(args[0]),
    "filter": lambda args: None if args[1] is None else "raiseError" if not
        scheme_builtins["list?"](args[1]) else filter_func(args),
    "reduce": lambda args: args[2] if args[1] is None else "raiseError" if not
        scheme_builtins["list?"](args[1]) else args[1].reduce(evaluate(args[0]
        ),evaluate(args[2])),
    "begin": lambda args: args[-1]
    }

def filter_func(args):
    return (args[1].copy).filter(scheme_builtins["map"]([evaluate(i) for i in args]))

def concat_lists(lists,new_pair = None):
    lists = [e for e in lists if e is not None]
    for num,e in enumerate(lists):
        if not scheme_builtins["list?"](e):
            raise SchemeEvaluationError("One of the 'pairs' is not a Pair")
        if num == 0: 
            new_pair = e.copy
        else:
            new_pair.append(e.copy)
    return new_pair
        

class Frame:
    """
    Frame attributes are parent frame, and variables located w/in frame
    __getitem__: checks parent frames for the variable, too
    """
    def __init__(self, parent = 0, variables = None):
        self.parent = Frame(None,scheme_builtins) if parent == 0 else parent
        self.variables = {} if variables is None else variables
    def __setitem__(self,key,val):
        self.variables[key] = val
    def __getitem__(self,key):
        if self.parent is not None:
            return self.variables.get(key, self.parent.__getitem__(key))
        return self.variables.get(key,"None")
    def get_enclosed(self,key):
        var = self.variables.get(key,None)
        if var is not None:
            return self
        elif self.parent is not None:
            return self.parent.get_enclosed(key)
        else:
            return None
    

class Func:
    """
    __call__: when function object is called, want to set the variables
    to the arguments sent in -- do this my dict comprehension
    Make a new frame to evaluate the function (like all functions do, with
    its parent frame being the function's enclosing frame)
    Evaluate the function (tree = list of strs that can be evaluated)
    """
    def __init__(self,arg_list,body,f1):
        self.variables = arg_list
        self.func = body
        self.frame = f1
    def __call__(self,arg_list):
        if len(self.variables) != len(arg_list): # Enough inputs
            raise SchemeEvaluationError
        variables = dict(zip(self.variables,arg_list))
        f1 = Frame(self.frame,variables)
        return evaluate(self.func,f1)

    
##############
# Evaluation #
##############


def evaluate(tree, f1 = Frame()):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    return result_and_frame(tree,f1)[0]


def result_and_frame(tree, f1 = None):
    """
    If no frame is given, make a new frame (new frames have scheme builtin as a
    parent frame)
    If the tree is a list: go to that function
    Otherwise, it is either a string or an integer (a recursive call of the function)
    """
    if f1 is None: f1 = Frame()
    if isinstance(tree,list):
        return list_eval(tree,f1)
    return nonlist_eval(tree,f1)

def key_val(tree,f1):
    key = tree[1][0] if isinstance(tree[1],list) else tree[1]
    val = make_lambda(tree[1][1:],tree[2],f1) if isinstance(tree[1],list) \
        else evaluate(tree[2:],f1) if tree[2] == "(" else evaluate(
        tree[2],f1)
    f1[key] = val
    return val, f1

def total_lambda(tree,f1):
    return make_lambda(tree[1],tree[2],f1), f1

def make_lambda(args,body,f1):
    return Func(args,body,f1)

def if_state(tree,f1):
    if evaluate(tree[1],f1):
        return result_and_frame(tree[2],f1)
    return result_and_frame(tree[3], f1)

def and_or(tree,f1):
    return other(tree[1:],tree[0],f1), f1  

def other(conditions,andor,f1):
    boolean = andor == "or"
    for condition in conditions:
        if evaluate(condition,f1) == boolean:
            return boolean
    return not boolean

def con_state(tree,f1):
    return cons([evaluate(e,f1) for e in tree[1:]],f1), f1

def car_cdr(tree,f1):
    if len(tree) != 2 or not isinstance(evaluate(tree[1],f1),Pair):
        raise SchemeEvaluationError(f"Wrong implementation of {tree[0]}")
    return getattr(evaluate(tree[1],f1),tree[0]), f1

def list_state(tree,f1):
    return make_cons(tree[1:], f1), f1

def del_state(tree,f1):
    removed = f1.variables.pop(tree[1],None)
    if removed is None:
        raise SchemeNameError(f"Cannot delete {tree[1]} because it \
        does not exist in frame")
    return removed, f1

def let_state(tree,f1):
    variables = {key:evaluate(val,f1) for key,val in tree[1]}
    subframe = Frame(f1,variables)
    return evaluate(tree[2],subframe), f1

def set_state(tree,f1):
    new_frame, evaluated_exp = f1.get_enclosed(tree[1]), evaluate(tree[2],f1)
    if new_frame is None:
        raise SchemeNameError
    new_frame[tree[1]] = evaluated_exp
    return evaluated_exp, f1

def list_eval(tree,f1):
    """
    If define is the function: set the left-hand variable to right-hand side
    If RHS is a variable: just set it (send in only tree[2]), else must evaluate
    rest of function (e.g. (+ 3 2)), so must send in list (tree[2:])
    Return the RHS evaluated

    If lambda: Make a Func object (args are variable input list, str representing
    what lambda should do, frame in which we're making the function)

    If not: evaluate every argument (first argument should be a function)
    Call tree[0] (args[0]) with arguments args[1:]
    If not a function, will raise a evaluation error
    """
    if not tree: raise SchemeEvaluationError("No input was given")
    special_funct = {"define": key_val, "lambda": total_lambda, "if": if_state,
        "and": and_or, "or": and_or, "cons": con_state, "car": car_cdr, "cdr": car_cdr,
        "list": list_state, "del": del_state, "let": let_state, "set!": set_state}
    if isinstance(tree[0],str) and tree[0] in special_funct:
        return special_funct[tree[0]](tree,f1)
    return return_answer([evaluate(arg,f1) for arg in tree],f1,tree[0])

def return_answer(args, f1, insert):
    try:
        evaluated = args[0](args[1:])
        if evaluated == "raiseError":
            raise SchemeEvaluationError(f"{insert} has too many inputs")
        return evaluated, f1
    except TypeError:
        raise SchemeEvaluationError(f"{insert} is not a real function")

def nonlist_eval(variable,f1):
    """
    If the variable is a string: look for what number the variable represents
    in enclosing and all parent frames
    Else: just return the number
    """
    if isinstance(variable,str):
        # print(f"\n\nVariables: {f1.variables, f1.parent.variables}\n\n")
        # f1: {f1}\nparent: {f1.parent}
        if variable in ("#t", "#f"):
            return variable == "#t", f1
        elif variable == "nil":
            return None, f1
        elif f1[variable] != "None":
            return f1[variable], f1
        raise SchemeNameError(f"{variable} does not exist in the frame")
    return variable,f1


##############
# Lists #
##############

class Pair:
    """
    Linked list class with car being first element and
    cdr being the second element that corresponds to a linked list
    unless it is the last element (then cdr == None)
    """
    def __init__(self,car,cdr,f1):
        self.car = car
        self.cdr = cdr
        self.frame = Frame() if not f1 else f1
    def length(self, node = 1):
        if self.cdr is None: return node
        return self.cdr.length(node + 1)
    def __getitem__(self,node):
        if node == 0: 
            return self
        return self.cdr.__getitem__(node - 1)
    @property
    def copy(self):
        # Deep Copies
        if self.cdr is None:
            return Pair(self.car,self.cdr,self.frame)
        return Pair(self.car,self.cdr.copy,self.frame)
    def funct_copy(self,funct):
        # While deep copying, applies function to each element
        car = evaluate([funct,self.car])
        if self.cdr is None: 
            return Pair(car,self.cdr,self.frame)
        return Pair(car,self.cdr.funct_copy(funct),self.frame)
    def append(self,next_pair):
        # Add Pair to end of another Pair
        end_node = self[self.length() - 1]
        end_node.cdr = next_pair
    def remove(self,node):
        # Remove a node 
        if node == self.length() - 1:
            # Make Pair invalid and check for this in return
            self[node - 1].cdr = None
        elif node != 0:
            self[node - 1].cdr = self[node + 1]
        else: 
            self[0].car, self[0].cdr = self[1].car, self[1].cdr
    def filter(self,truth_table):
        """Pair that is being called should be a copy"""
        # Make a list of the nodes that are invalid 
        # Make sure to do it backwards so removing doesn't change
        # the node number
        ans = [n for n in range(self.length()-1,-1,-1) if not
            truth_table[n].car]
        if len(ans) == self.length():
            return None
        for num in ans:
            self.remove(num)
        return self
    def reduce(self,funct,initial,start = 0):
        """Go through each node and build up number"""
        while True:
            node = self[start].car
            initial = evaluate([funct,initial,node])
            start += 1
            if start == self.length():
                break
        return initial
    def __repr__(self):
        if self.cdr is None:
            return "[" + f"{self.car},{self.cdr}" + "]"
        return "[" + f"{self.car}," + f"{repr(self.cdr)}" + "]"


def cons(a, f1 = None):
    if len(a) != 2:
        raise SchemeEvaluationError(f"{len(a)} arguments given. Should have only 2")
    return Pair(a[0],a[1],f1)

def make_cons(a, f1):
    if not a:
        return None
    elif len(a) == 1:
        return cons([evaluate(a[0],f1),None],f1)
    else:
        return cons([evaluate(a[0],f1),make_cons(a[1:],f1)],f1)


def evaluate_file(file, f1 = None, add_frame = False):
    """Evaluate one file"""
    fil = open(file, "r")
    string = parse(tokenize(fil.read()))
    if add_frame: # Also give the frame
        return result_and_frame(string,f1)
    return evaluate(string,f1)


def evaluate_files(files, f1 = None):
    """Run through multiple files"""
    for file in files:
        ans, f1 = evaluate_file(file,f1,True)
    return ans, f1



def repl(verbose=False, f1 = None):
    """
    Read in a single line of user input, evaluate the expression, and print 
    out the result. Repeat until user inputs "QUIT"
    
    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """
    if f1 is None:
        _, f1 = result_and_frame(["+"])  # make a global frame
    while True:
        input_str = input("in> ")
        if input_str == "quit":
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if verbose:
                print("expression>", expression)
            output, f1 = result_and_frame(expression, f1)
            print("  out>", output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print("Error>", repr(e))

 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("here", sys.argv)
        answer, frame = evaluate_files(sys.argv[1:])
        repl(True,frame)
    else:
        repl(True)
