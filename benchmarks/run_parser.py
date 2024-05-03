from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3

from callbackscpmpy import CallbacksCPMPy



import glob
from os.path import join
problems = glob.glob(join("CSP", '*.xml'))
problems += glob.glob(join("..", "MiniCOP", '*.xml'))
print(problems)
count = 160
ecount = 0
print(problems[count])
for xml in problems[count:]:
    count += 1
    '''if len(sys.argv) < 2:
        sys.argv.append(xml)
    assert len(sys.argv) >= 2'''
    #we are not using commandline
    #parser = ParserXCSP3(os.path.join("./", sys.argv[1]))
    parser = ParserXCSP3(xml)
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    # e.g., callbacks.recognize_unary_primitives = False
    callbacker = CallbackerXCSP3(parser, callbacks)
    try:
        callbacker.load_instance()
    except NotImplementedError as e:
        ecount += 1
        raise e
    except Exception as e:
        print(problems[count-1])
        raise e
    cb = callbacker.cb
    #print(cb.cpm_model)
    #print(cb.cpm_variables)
    #print('________________________________________________________')
    print(count, ecount)
    print('________________________________________________________')
    print('________________________________________________________')

print('not implemented:', ecount, 'out of', count)
# obj = build_dynamic_object(sys.argv[2], sys.argv[3])