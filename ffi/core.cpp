#include "core.h"
#include <stdio.h>

#include "llvm-c/Support.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Debug.h"


extern "C" {


API_EXPORT(void)
LLVMPY_SetDebugFlag()
{
#ifndef NDEBUG
    llvm::DebugFlag = true;
#endif
}

API_EXPORT(const char *)
LLVMPY_CreateString(const char *msg) {
    return strdup(msg);
}

API_EXPORT(const char *)
LLVMPY_CreateByteString(const char *buf, size_t len) {
    char *dest = (char *) malloc(len + 1);
    if (dest != NULL) {
        memcpy(dest, buf, len);
        dest[len] = '\0';
    }
    return dest;
}

API_EXPORT(void)
LLVMPY_DisposeString(const char *msg) {
    free(const_cast<char*>(msg));
}

API_EXPORT(LLVMContextRef)
LLVMPY_GetGlobalContext() {
    return LLVMGetGlobalContext();
}

API_EXPORT(void)
LLVMPY_SetCommandLine(const char *name, const char *option)
{
    const char *argv[] = {name, option};
    LLVMParseCommandLineOptions(2, argv, NULL);
}

void diagnostic_handler(LLVMDiagnosticInfoRef d, void *data)
{
    printf("diagnostic handler: %s\n",LLVMGetDiagInfoDescription(d));
}

API_EXPORT(void)
LLVMPY_EnableDiagnosticHandler()
{
    LLVMContextSetDiagnosticHandler(LLVMGetGlobalContext(), diagnostic_handler, NULL);
}



} // end extern "C"
