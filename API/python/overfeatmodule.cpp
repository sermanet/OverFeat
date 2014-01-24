#include <Python.h>
#include <cassert>
#include "overfeat.hpp"
#include <numpy/arrayobject.h>

const int FLOAT32_DTYPE = 11;

THTensor* THFromData(float* data, int d1, int d2 = -1, int d3 = -1) {
  if (d2 == -1) {
    THStorage* storage = THStorage_(newWithData)(data, d1);
    return THTensor_(newWithStorage1d)(storage, 0, d1, 1);
  } else if (d3 == -1) {
    THStorage* storage = THStorage_(newWithData)(data, d1*d2);
    return THTensor_(newWithStorage2d)(storage, 0, d1, d2, d2, 1);
    return NULL;
  } else {
    THStorage* storage = THStorage_(newWithData)(data, d1*d2*d3);
    return THTensor_(newWithStorage3d)(storage, 0, d1, d2*d3, d2, d3, d3, 1);
  }
}

THTensor* THFromContiguousArray(PyArrayObject* array) {
  int ndim = PyArray_NDIM(array);
  assert(PyArray_TYPE(array) = FLOAT32_DTYPE);
  assert(ndim < 4);
  npy_intp* dims = PyArray_DIMS(array);
  float* data = (float*)PyArray_DATA(array);
  if (ndim == 1)
    return THFromData(data, dims[0]);
  else if (ndim == 2)
    return THFromData(data, dims[0], dims[1]);
  else
    return THFromData(data, dims[0], dims[1], dims[2]);
}

static PyObject*
overfeat_init(PyObject* self, PyObject* args) {
  const char* weight_file_path;
  int net_idx;
  if (!PyArg_ParseTuple(args, "si", &weight_file_path, &net_idx))
    return NULL;
  overfeat::init(weight_file_path, net_idx);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
overfeat_free(PyObject* self, PyObject* args) {
  overfeat::free();
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
overfeat_fprop(PyObject* self, PyObject* args) {
  PyArrayObject* input = NULL;
  if (!PyArg_ParseTuple(args, "O", &input))
    return NULL;
  if (PyArray_TYPE(input) != FLOAT32_DTYPE) {
    PyErr_SetString(PyExc_TypeError,
		    "Overfeat: arrays must have type numpy.float32");
    return NULL;
  }
  if (PyArray_NDIM(input) != 3) {
    PyErr_SetString(PyExc_TypeError,
		    "Overfeat: fprop expect a 3d array");
    return NULL;
  }

  PyArrayObject* input_c = PyArray_GETCONTIGUOUS(input);
  THTensor* input_th = THFromContiguousArray(input_c);

  THTensor* output_th = overfeat::fprop(input_th);

  npy_intp sizes[3] = {0,0,0};
  for (int i = 0; i < output_th->nDimension; ++i)
    sizes[i] = output_th->size[i];
  
  PyArrayObject* output = (PyArrayObject*)PyArray_SimpleNewFromData(output_th->nDimension,
								    sizes,
								    NPY_FLOAT,
								    THTensor_(data)(output_th));

  Py_DECREF(input_c);
  
  return PyArray_Return(output);
}

static PyObject*
overfeat_get_n_layers(PyObject* self, PyObject* args) {
  int n_layers = overfeat::get_n_layers();
  return Py_BuildValue("i", n_layers);
}

static PyObject*
overfeat_get_output(PyObject* self, PyObject* args) {
  int i;
  if (!PyArg_ParseTuple(args, "i", &i))
    return NULL;
  THTensor* output_th = overfeat::get_output(i);
  
  npy_intp sizes[3] = {0,0,0};
  for (int i = 0; i < output_th->nDimension; ++i)
    sizes[i] = output_th->size[i];
  
  PyArrayObject* output = (PyArrayObject*)PyArray_SimpleNewFromData(output_th->nDimension,
								    sizes,
								    NPY_FLOAT,
								    THTensor_(data)(output_th));
  
  return PyArray_Return(output);
}

static PyObject*
overfeat_get_class_name(PyObject* self, PyObject* args) {
  int i;
  if (!PyArg_ParseTuple(args, "i", &i))
    return NULL;
  std::string class_name = overfeat::get_class_name(i);
  return Py_BuildValue("s", class_name.c_str());
}

static PyMethodDef OverfeatMethods[] = {
  {"init", overfeat_init, METH_VARARGS, "Initializes overfeat"},
  {"free", overfeat_free, METH_VARARGS, "Releases ressources allocated by overfeat"},
  {"fprop", overfeat_fprop, METH_VARARGS, "Runs the feature extractor"},
  {"get_n_layers", overfeat_get_n_layers, METH_VARARGS, "Return the number of layers in the network"},
  {"get_output", overfeat_get_output, METH_VARARGS, "Returns the output of the i-th layer"},
  {"get_class_name", overfeat_get_class_name, METH_VARARGS, "Returns the name of the i-th class"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initoverfeat(void) {
  (void)Py_InitModule("overfeat", OverfeatMethods);
  import_array();
}
