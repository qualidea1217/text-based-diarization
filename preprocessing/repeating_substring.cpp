#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <string>

static PyObject* is_repeating(PyObject* self, PyObject* args) {
    const char* input_cstr;
    int min_length;
    int min_repeat;

    if (!PyArg_ParseTuple(args, "sii", &input_cstr, &min_length, &min_repeat)) {
        return NULL;
    }

    std::string input_str(input_cstr);
    int str_length = input_str.length();
    int max_length = str_length / min_repeat;

    for (int length = max_length; length >= min_length; --length) {
        for (int start = 0; start <= str_length - length * min_repeat; ++start) {
            std::string substring = input_str.substr(start, length);
            std::string pattern = substring;
            for (int i = 1; i < min_repeat; ++i) {
                pattern += substring;
            }
            if (input_str.find(pattern) != std::string::npos) {
                Py_RETURN_TRUE;
            }
        }
    }

    Py_RETURN_FALSE;
}

static PyMethodDef repeating_substring_methods[] = {
        {"is_repeating",  is_repeating, METH_VARARGS, "Check if a string contains a repeating substring"},
        {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef repeating_substring_module = {
        PyModuleDef_HEAD_INIT,
        "repeating_substring",   /* name of module */
        NULL,          /* module documentation, may be NULL */
        -1,          /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        repeating_substring_methods
};

PyMODINIT_FUNC
PyInit_repeating_substring(void) {
    return PyModule_Create(&repeating_substring_module);
}
