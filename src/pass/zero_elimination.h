/*!
 *  Copyright (c) 2018 by Contributors
 * \file zero_elimination.h
 * \brief Transform tensors in such a way as to eliminate summation over zeros.
 */
#ifndef TVM_PASS_ZERO_ELIMINATION_H_
#define TVM_PASS_ZERO_ELIMINATION_H_

#include <tvm/ir.h>
#include <tvm/tensor.h>

namespace tvm {
namespace ir {


/*!
 * \brief Transform the expression into `c ? e : 0`, that is lift the condition of being
 *  possible to be non-zero to the top level.
 */
EXPORT Expr LiftNonzeronessCondition(const Expr& expr);

/*!
 * \brief Perform lifting of conditions of being possible to be non-zero together with
 *  applying some transformations like simplifying the reduction domain. Works only with
 *  this particular tensor's body, i.e. doesn't perform inlining.
 */
EXPORT Tensor OptimizeAndLiftNonzeronessConditions(const Tensor& tensor);

/*!
 * \brief TODO
 */
EXPORT Tensor InlineTailCall(const Tensor& tensor);

/*!
 * \brief Inline tensors which are not reductions.
 *
 * \param inlineable A list of tensors which are allowed to be inlined. If empty, try
 *  to inline all tensors recursively.
 */
EXPORT Tensor InlineNonReductions(const Tensor& tensor, const Array<Tensor>& inlineable);

/*!
 * \brief Clone the reduction by cloning its iteration variables.
 */
// TODO: Probably move this function somewhere
Expr CloneReduction(const Expr& expr);

/*!
 * \brief Get the set of all tensors which are called from within the expression.
 */
// TODO: Probably move this function somewhere and rename it
std::unordered_set<Tensor> Subtensors(const Expr& expr);

// TODO: Probably move this function somewhere
EXPORT std::string PrintTensorName(const Tensor& tensor);

// TODO: Probably move this function somewhere
EXPORT std::string PrintTensorRecursively(const Tensor& tensor);

}  // namespace ir
}  // namespace tvm
#endif  // TVM_PASS_ZERO_ELIMINATION_H_
