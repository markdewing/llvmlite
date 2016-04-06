from __future__ import print_function

from ctypes import CFUNCTYPE, c_int, POINTER, c_double, c_float
import sys
try:
    from time import perf_counter as time
except ImportError:
    from time import time

import numpy as np

try:
    import faulthandler; faulthandler.enable()
except ImportError:
    pass

import math
from pprint import pprint

import llvmlite.ir as ll
import llvmlite.binding as llvm


# Create a function to apply an expression to an input array and write it
# to an output array.

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# Print -Rpass messages
llvm.enable_diagnostic_handler()

# Needs a debug build of LLVM to work
#llvm.enable_debug_output()

feature_dict = llvm.get_host_cpu_features()
#print('features')
#pprint(feature_dict)

simd_lib = "libsleef_sse.so"
if feature_dict.get('avx'):
    simd_lib = "libsleef_avx.so"
if feature_dict.get('avx2'):
    simd_lib = "libsleef_avx2.so"

llvm.load_library_permanently(simd_lib)
#fp_type = ll.FloatType()
#cfp_type = c_float

fp_type = ll.DoubleType()
cfp_type = c_double

t1 = time()

fnty = ll.FunctionType(fp_type, [fp_type.as_pointer(),
                                        ll.IntType(32),
                                 fp_type.as_pointer()])
module = ll.Module()
module.triple = llvm.get_default_triple()
print("Triple = ",module.triple)
print("Default Triple = ",llvm.get_default_triple())
print("CPU = ",llvm.get_host_cpu_name())

func = ll.Function(module, fnty, name="sum")

func.args[0].add_attribute("nocapture")
func.args[0].add_attribute("noalias")
func.args[2].add_attribute("nocapture")
func.args[2].add_attribute("noalias")

bb_entry = func.append_basic_block()
printf_type = ll.FunctionType(ll.IntType(32), [ll.IntType(32).as_pointer()])
printf_fn = ll.Function(module, printf_type, "printf")

bb_loop = func.append_basic_block()
bb_exit = func.append_basic_block()
builder = ll.IRBuilder()
builder.position_at_end(bb_entry)

#builder.call(printf_fn, [ll.Constant(ll.IntType(32).as_pointer(),"hello\n")])

builder.branch(bb_loop)
builder.position_at_end(bb_loop)

index = builder.phi(ll.IntType(32))
index.add_incoming(ll.Constant(index.type, 0), bb_entry)

ptr = builder.gep(func.args[0], [index])
dptr = builder.bitcast(ptr, fp_type.as_pointer())
value = builder.load(dptr,align=32)

fn_type = ll.FunctionType(fp_type, [fp_type])
#exp_fn = ll.Function(module, fn_type, "exp")
exp_fn = ll.Function(module, fn_type, "llvm.sqrt.f64")

#value2 = builder.fmul(value, ll.Constant(fp_type, -2))

evalue = builder.call(exp_fn, [value])
#evalue2 = builder.fadd(evalue, ll.Constant(fp_type, 4.0))
#evalue = value

out_ptr = builder.gep(func.args[2], [index])
out_dptr = builder.bitcast(out_ptr, fp_type.as_pointer())
store = builder.store(evalue, out_dptr, align=32)

#added = builder.fadd(accum, evalue)
#accum.add_incoming(added, bb_loop)

indexp1 = builder.add(index, ll.Constant(index.type, 1))
index.add_incoming(indexp1, bb_loop)

cond = builder.icmp_unsigned('<', indexp1, func.args[1])
#cond = builder.icmp_unsigned('<', indexp1, ll.Constant(index.type, 100))
br =  builder.cbranch(cond, bb_loop, bb_exit)
mref2 = builder.module.add_metadata([ll.MetaDataString(builder.module,"llvm.loop.vectorize"),ll.Constant(ll.IntType(32), 1)])
mref3 = builder.module.add_metadata([ll.MetaDataString(builder.module,"llvm.loop.vectorize.width"),ll.Constant(ll.IntType(32), 4)])
mref4 = builder.module.add_metadata([ll.MetaDataString(builder.module,"llvm.loop.interleave.count"),ll.Constant(ll.IntType(32), 4)])
#mref1 = builder.module.add_metadata([ll.MetaDataString(builder.module,"0"),mref2])
print("mref2 name =",mref2.name)
# name here doesn't matter, will get replaced later
mtmp = ll.MDSelfValue(builder.module, mref2.name)
#mref1 = builder.module.add_metadata([mtmp,mref2,mref3,mref4])
mref1 = builder.module.add_metadata([mtmp,mref4,mref2,mref3])
# Would like to get rid of this next renaming step
mtmp.pname = mref1.name
print("ref1 name = ",mref1.name)
br.set_metadata("llvm.loop",mref1)


#value.set_metadata("llvm.mem.parallel_loop_acces",mref1)
#store.set_metadata("llvm.mem.parallel_loop_acces",mref1)


builder.position_at_end(bb_exit)
builder.ret(ll.Constant(fp_type, 0.0))

strmod = str(module)
print("IR")
print(strmod)


t2 = time()

print("-- generate IR:", t2-t1)

t3 = time()

llmod = llvm.parse_assembly(strmod)

t4 = time()

print("-- parse assembly:", t4-t3)

#print(llmod)
options=dict(cpu=llvm.get_host_cpu_name())
target_machine = llvm.Target.from_default_triple().create_target_machine(**options)
#target_machine = llvm.Target.from_default_triple().create_target_machine()


targ_info =  llvm.create_target_library_info()
pmb = llvm.create_pass_manager_builder()
pmb.opt_level = 2
pmb.loop_vectorize = True

pm = llvm.create_module_pass_manager()
fpm = llvm.create_function_pass_manager(llmod)

llvm.add_target_transform_info(target_machine, pm)

pmb.add_library_info(targ_info)
pmb.populate(pm)
pmb.populate(fpm)




#llvm.create_and_add_target_library_info(fpm)
#llvm.create_and_add_target_library_info(pm)
#targ_info.add_pass(pm)
#print("targ info = ",type(targ_info))
#print("pm ptr = ",fpm._ptr)
#llvm.add_target_library_info(fpm, targ_info)
#print("loop vectorize = ",pmb.loop_vectorize)
#print("loop unroll = ",pmb.disable_unroll_loops)


t5 = time()

for func in llmod.functions:
    fpm.initialize()
    fpm.run(func)
    fpm.finalize()

pm.run(llmod)


t6 = time()

print("-- optimize:", t6-t5)

t7 = time()



#cfp_type = c_float
with llvm.create_mcjit_compiler(llmod, target_machine) as ee:
    #pm.run(llmod)
    ee.enable_jit_events()
    ee.finalize_object()
    cfptr = ee.get_function_address("sum")

    t8 = time()
    print("-- JIT compile:", t8 - t7)

    #print(target_machine.emit_assembly(llmod))

    cfunc = CFUNCTYPE(cfp_type, POINTER(cfp_type), c_int, POINTER(cfp_type))(cfptr)
    A = np.arange(10000, dtype=np.double)
    #A = np.arange(100, dtype=np.float)
    #B = np.arange(100, dtype=np.double)
    #B = np.zeros(100, dtype=np.float)
    B = np.zeros(10000, dtype=np.double)
    print(A[0], B[0], A[1], B[1])
    t8 = time()
    for i in range(10000):
        res = cfunc(A.ctypes.data_as(POINTER(cfp_type)), A.size, B.ctypes.data_as(POINTER(cfp_type)))
    t9 = time()

    print(A[0], B[0], math.exp(A[0]) )
    print(A[1], B[1], math.exp(A[1]) )
    print("-- run time : ",t9-t8)

