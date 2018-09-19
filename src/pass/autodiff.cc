/*!
 *  Copyright (c) 2018 by Contributors
 * \file autodiff.cc
 * \brief Automatic differentiation of IR Expr
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <topi/tags.h>
#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include "../op/op_util.h"


namespace tvm {
namespace ir {

// TODO: Move to some header
Expr CloneReduction(const Expr& expr);

#define NOT_IMPLEMENTED { throw dmlc::Error("Derivative of this op is not implemented"); }

/*! \brief Differentiate an expression wrt a variable or a tensor element */
class JacobianMutator : public IRMutator {
  public:
    /*!
     * \brief Differentiate wrt `input(indices)`.
     * \param input The input tensor.
     * \param indices The indices of the element with respect to which to differentiate.
     */
    explicit JacobianMutator(Tensor input, Array<Expr> indices)
      : input_(input), indices_(indices) {}
    /*!
     * \brief Differentiate wrt the input variable.
     * \param input The input variable.
     */
    explicit JacobianMutator(VarExpr input)
      : input_var_(input) {}

    Expr Mutate_(const Variable* op, const Expr& e) {
      if (input_var_.operator->() && input_var_.get() == op)
        return FloatImm::make(op->type, 1.0);
      else
        return make_zero(op->type);
    }

    Expr Mutate_(const Load* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const Let* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const Call* op, const Expr& e) {
      if (op->call_type == Call::CallType::Halide) {
        if (input_.operator->() && op->func.same_as(input_->op) &&
            op->value_index == input_->value_index) {
          Expr condition = UIntImm::make(Bool(), 1);
          for (size_t i = 0; i < input_.ndim(); ++i) {
            condition = And::make(condition, EQ::make(indices_[i], op->args[i]));
          }
          return Cast::make(op->type, condition);
        }
        else
          return make_zero(op->type);
      }
      else if (op->call_type == Call::CallType::PureIntrinsic) {
        if (op->name == "exp")
          return Mul::make(Mutate(op->args[0]), e);
        else if (op->name == "log")
          return Div::make(Mutate(op->args[0]), op->args[0]);
        else if (op->name == "fabs") {
          auto type = op->args[0].type();
          return Mul::make(Mutate(op->args[0]),
                           Select::make(GE::make(op->args[0], make_zero(type)),
                                        FloatImm::make(type, 1.0), FloatImm::make(type, -1.0)));
        }
        else
          throw dmlc::Error("Derivative of this op is not implemented: " + op->name);
      }
      NOT_IMPLEMENTED
    }

    Expr Mutate_(const Add* op, const Expr& e)  {
      return op->make(Mutate(op->a), Mutate(op->b));
    }

    Expr Mutate_(const Sub* op, const Expr& e)  {
      return op->make(Mutate(op->a), Mutate(op->b));
    }

    Expr Mutate_(const Mul* op, const Expr& e) {
      return Add::make(Mul::make(Mutate(op->a), op->b), Mul::make(op->a, Mutate(op->b)));
    }

    Expr Mutate_(const Div* op, const Expr& e) {
      return Div::make(
          Sub::make(Mul::make(Mutate(op->a), op->b), Mul::make(op->a, Mutate(op->b))),
          Mul::make(op->b, op->b));
    }

    Expr Mutate_(const Mod* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const Min* op, const Expr& e) {
      return Select::make(LE::make(op->a, op->b), Mutate(op->a), Mutate(op->b));
    }

    Expr Mutate_(const Max* op, const Expr& e) {
      return Select::make(GE::make(op->a, op->b), Mutate(op->a), Mutate(op->b));
    }

    Expr Mutate_(const EQ* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const NE* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const LT* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const LE* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const GT* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const GE* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const And* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const Or* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const Reduce*, const Expr& e) {
      // This case is relatively difficult because a reduction expression
      // may use an arbitrary combiner.
      // The resulting reduction expression will return a tuple containing
      // both derivatives and the original results (in exactly this order).

      // We have to clone the reduction axes because otherwise the original expression
      // cannot be used together with the derivative (it will lead to errors during lowering)
      Expr expr_with_new_axes = CloneReduction(e);
      const Reduce* op = expr_with_new_axes.as<Reduce>();

      // New lhs and rhs variables of the new combiner consist of variables
      // representing derivatives followed by the original variables.
      Array<Var> new_lhs;
      for (const auto& var : op->combiner->lhs)
        new_lhs.push_back(var.copy_with_suffix(".der"));
      for (const auto& var : op->combiner->lhs)
        new_lhs.push_back(var);

      Array<Var> new_rhs;
      for (const auto& var : op->combiner->rhs)
        new_rhs.push_back(var.copy_with_suffix(".der"));
      for (const auto& var : op->combiner->rhs)
        new_rhs.push_back(var);

      // The new combiner result also consists of the resulting derivatives
      // followed by the original results.
      Array<Expr> new_result;
      for (const auto& res : op->combiner->result) {
        // Each resulting derivative is computed as a sum of derivatives
        // wrt lhs and rhs multiplied by the derivatives of lhs and rhs
        Expr new_res = make_zero(res.type());
        for (size_t i = 0; i < op->combiner->lhs.size(); ++i) {
          Expr res_di = Derivative(res, op->combiner->lhs[i]);
          // new_lhs[i] is the derivative of lhs[i] (wrt our input tensor)
          new_res = Add::make(new_res, Mul::make(new_lhs[i], res_di));
        }
        for (size_t i = 0; i < op->combiner->rhs.size(); ++i) {
          Expr res_di = Derivative(res, op->combiner->rhs[i]);
          new_res = Add::make(new_res, Mul::make(new_rhs[i], res_di));
        }
        new_result.push_back(new_res);
      }
      for (const auto& res : op->combiner->result)
        new_result.push_back(res);

      // The identity is transformed in a similar way
      Array<Expr> new_identity;
      for (const auto& id : op->combiner->identity_element)
        new_identity.push_back(Mutate(id));
      for (const auto& id : op->combiner->identity_element)
        new_identity.push_back(id);

      Array<Expr> new_source;
      for (const auto& src : op->source)
        new_source.push_back(Mutate(src));
      for (const auto& src : op->source)
        new_source.push_back(src);

      CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
      // Also simplify the resulting combiner (mostly to get rid of unused components)
      return SimplifyCombiner(
          Reduce::make(new_combiner, new_source, op->axis, op->condition, op->value_index));
    }

    Expr Mutate_(const Cast* op, const Expr& e) {
      if (op->type.is_float())
        return Cast::make(op->type, Mutate(op->value));
      else
        return make_zero(op->type);
    }

    Expr Mutate_(const Not* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const Select* op, const Expr& e) {
      return Select::make(op->condition, Mutate(op->true_value), Mutate(op->false_value));
    }

    Expr Mutate_(const Ramp* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const Broadcast* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const IntImm* op, const Expr& e) { return op->make(op->type, 0); }
    Expr Mutate_(const UIntImm* op, const Expr& e) { return op->make(op->type, 0); }
    Expr Mutate_(const FloatImm* op, const Expr& e) { return op->make(op->type, 0); }

    Expr Mutate_(const StringImm* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const Shuffle* op, const Expr& e) NOT_IMPLEMENTED

  private:
    Tensor input_;
    Array<Expr> indices_;
    VarExpr input_var_;
};

// Collect all tensors used by the given tensor
class IRCollectSubtensors : public IRVisitor {
  public:
    void Visit_(const Call* op) {
      if (op->call_type == Call::CallType::Halide)
        if (op->func->derived_from<OperationNode>()) {
          // TODO: node_ is not supposed to be used
          Operation operation(std::static_pointer_cast<OperationNode>(op->func.node_));
          subtensors.insert(operation.output(op->value_index));
        }
      for (auto& e : op->args)
        Visit(e);
    }

    std::unordered_set<Tensor> subtensors;
};

Expr Jacobian(Expr expr, Tensor input, Array<Expr> indices) {
  return JacobianMutator(input, indices).Mutate(expr);
}

Expr Derivative(Expr expr, VarExpr var) {
  return JacobianMutator(var).Mutate(expr);
}

Tensor Jacobian(Tensor output, Tensor input) {
  if (const ComputeOpNode* op = output->op.as<ComputeOpNode>()) {
    //std::cout << "Jacobian of " << output << " = " << op->body << std::endl;
    //std::cout << "wrt " << input << std::endl;

    // We have to clone the iteration axes because otherwise the original expression
    // cannot be used together with the derivative (it will lead to errors during lowering)
    Array<IterVar> new_axis;
    std::unordered_map<const Variable*, Expr> vmap;
    for (IterVar iv : op->axis) {
      IterVar new_v =
        IterVarNode::make(iv->dom, iv->var.copy_with_suffix(".jac.out"),
            iv->iter_type, iv->thread_tag);
      new_axis.push_back(new_v);
      vmap[iv->var.operator->()] = new_v;
    }

    // Generate new itervars for the input
    Array<Expr> input_itervars;
    size_t i = 0;
    for (Expr ext : input->shape) {
      IterVar new_v =
        IterVarNode::make(Range(0, ext), Var("jacobian_i" + std::to_string(i)),
            IterVarType::kDataPar);
      // Append them to new_axis
      new_axis.push_back(new_v);
      // We also need a separate array of these itervars
      input_itervars.push_back(new_v);
      ++i;
    }

    // The differentiation itself happens here
    Expr new_body =
      Jacobian(Substitute(op->body[output->value_index], vmap), input, input_itervars);

    //std::cout << "resulting body = " << new_body << "\n" << std::endl;

    int value_index = 0;
    Array<Expr> new_bodies;

    // If this is a reduction then it may return a tuple and we have
    // to repeat the body several times
    if (const Reduce* red = new_body.as<Reduce>()) {
      value_index = red->value_index;
      for (size_t i = 0; i < red->source.size(); ++i)
        new_bodies.push_back(
            Reduce::make(red->combiner, red->source, red->axis, red->condition, i));
    }
    else {
      new_bodies.push_back(new_body);
    }

    auto new_op =
      ComputeOpNode::make(op->name + ".jacobian", op->tag, op->attrs, new_axis, new_bodies);

    // new_shape = output.shape + input.shape
    Array<Expr> new_shape = output->shape;
    for (const auto& e : input->shape)
      new_shape.push_back(e);

    return TensorNode::make(new_shape, output->dtype, new_op, value_index);
  }
  else
    NOT_IMPLEMENTED;
}


/*!
 * \brief A generalization of matrix multiplication to tensors.
 *
 *  `Res[i_1, ... , j_1, ...] = Sum_{k_1, ...} A[i_1 ..., k_1, ...]*B[k_1, ..., j_1, ...]`
 *  The number of `k` variables is \p ndims_to_reduce.
 *
 * \param A The tensor A
 * \param B The tensor B
 * \param ndims_to_reduce The number of dimensions to reduce over
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor computing the result
 */
tvm::Tensor generalized_matmul(const tvm::Tensor& A,
                               const tvm::Tensor& B,
                               int ndims_to_reduce,
                               std::string name = "tensor",
                               std::string tag = topi::kMatMul) {
  CHECK_GE(A->shape.size(), ndims_to_reduce);
  CHECK_GE(B->shape.size(), ndims_to_reduce);

  Array<tvm::Expr> output_shape(A->shape.begin(), A->shape.end() + (-ndims_to_reduce));
  for (auto it = B->shape.begin() + ndims_to_reduce; it != B->shape.end(); ++it)
    output_shape.push_back(*it);

  Array<tvm::IterVar> iter_vars;
  for (int i = 0; i < ndims_to_reduce; ++i)
    iter_vars.push_back(tvm::reduce_axis(tvm::Range(0, B->shape[i]), "k"));

  auto func =
    [&A, &B, &iter_vars, ndims_to_reduce]
    (const Array<tvm::Var>& input_indices) {
      Array<tvm::Expr> A_indices(
          input_indices.begin(),
          input_indices.begin() + (A->shape.size() - ndims_to_reduce));
      for (auto& v : iter_vars)
        A_indices.push_back(v);

      Array<tvm::Expr> B_indices;
      for (auto& v : iter_vars)
        B_indices.push_back(v);

      auto it = input_indices.begin() + (A->shape.size() - ndims_to_reduce);
      for (; it != input_indices.end(); ++it)
        B_indices.push_back(*it);

      return tvm::sum(A(A_indices)*B(B_indices), iter_vars);
    };

  return tvm::compute(output_shape, func, name, tag);
}

Tensor JacobianRecursive(Tensor output, Tensor input, Tensor head) {
  if (const ComputeOpNode* op = output->op.as<ComputeOpNode>()) {
    IRCollectSubtensors subtensors;
    subtensors.Visit(op->body[output->value_index]);

    Tensor res;

    // We have to compute jacobians/gradients wrt all the subtensors, multiply them
    // by jacobians of subtensor wrt the input, and sum the results
    for (auto& subtensor : subtensors.subtensors) {
      // Placeholders have zero derivative wrt input (unless it is input)
      if (subtensor->op.as<PlaceholderOpNode>() && subtensor != input)
        continue;

      // jacobian/gradient wrt the subtensor
      Tensor jac_output_subtensor = Jacobian(output, subtensor);
      Tensor part = generalized_matmul(head, jac_output_subtensor, output->shape.size());
      part = FuseTensors(part, {jac_output_subtensor});

      // if the subtensor is not the input, compute its jacobian wrt input recursively
      if (subtensor != input)
        part = JacobianRecursive(subtensor, input, part);

      // Add this part to the result
      if (res.operator->())
        res = topi::add(res, part);
      else
        res = part;
    }

    if (res.operator->())
      return res;
    else {
      // If the output doesn't use input and non-placeholder tensors then the result is
      // just a zero tensor of the correct shape
      Array<tvm::Expr> result_shape(
          head->shape.begin(),
          head->shape.end() + (-output->shape.size()));
      for (auto e : input->shape)
        result_shape.push_back(e);
      return topi::full(result_shape, output->dtype, make_zero(output->dtype));
    }
  }
  else
    NOT_IMPLEMENTED;
}

}  // namespace ir
}  // namespace tvm
