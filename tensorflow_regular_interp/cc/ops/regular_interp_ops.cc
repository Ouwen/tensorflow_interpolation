#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("RegularInterp")
  .Attr("T: realnumbertype")
  .Attr("ndim: int >= 1")
  .Attr("check_sorted: bool = true")
  .Attr("bounds_error: bool = false")
  .Input("points: ndim * T")
  .Input("values: T")
  .Input("xi: T")
  .Output("zi: T")
  .Output("dz: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    int ndim;
    TF_RETURN_IF_ERROR(c->GetAttr("ndim", &ndim));

    shape_inference::ShapeHandle shape, values_shape, xi_shape, zi_shape, tmp;

    // Get the dimensions of each axis
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &shape));
    for (int i = 1; i < ndim; ++i) {
      TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &tmp));
      TF_RETURN_IF_ERROR(c->Concatenate(shape, tmp, &shape));
    }

    // Make sure that the first ndim axes of values have the right shape
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(ndim), ndim, &values_shape));
    TF_RETURN_IF_ERROR(c->Subshape(values_shape, ndim, &zi_shape));
    TF_RETURN_IF_ERROR(c->Subshape(values_shape, 0, ndim, &values_shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, values_shape, &values_shape));

    // Make sure that the last dimension of xi is ndim
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(ndim+1), 1, &xi_shape));
    shape_inference::DimensionHandle dim = c->Dim(xi_shape, -1);
    TF_RETURN_IF_ERROR(c->WithValue(dim, ndim, &dim));

    // Compute the output shape
    TF_RETURN_IF_ERROR(c->Subshape(xi_shape, 0, -1, &tmp));
    TF_RETURN_IF_ERROR(c->Concatenate(tmp, zi_shape, &tmp));
    c->set_output(0, tmp);

    TF_RETURN_IF_ERROR(c->Concatenate(xi_shape, zi_shape, &tmp));
    c->set_output(1, tmp);

    return Status::OK();
  });
  