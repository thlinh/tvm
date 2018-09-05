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

#define NOT_IMPLEMENTED { throw dmlc::Error("Derivative of this op is not implemented"); }

class JacobianMutator : public IRMutator {
  public:
    explicit JacobianMutator(Tensor input, Array<Expr> indices)
      : input_(input), indices_(indices) {}

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
        // TODO
        NOT_IMPLEMENTED
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

    Expr Mutate_(const Reduce* op, const Expr& e) {
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

      Array<Expr> new_result;
      for (const auto& res : op->combiner->result) {
        Expr new_res = make_zero(res.type());
        for (size_t i = 0; i < op->combiner->lhs.size(); ++i) {
          Expr res_di = Derivative(res, op->combiner->lhs[i]);
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

      Array<Expr> new_identity;
      for (const auto& id : op->combiner->identity_element)
        new_identity.push_back(Mutate(id));
      for (const auto& id : op->combiner->identity_element)
        new_identity.push_back(id);

      Array<IterVar> new_axis;
      std::unordered_map<const Variable*, Expr> vmap;
      for (IterVar iv : op->axis) {
        IterVar new_v =
          IterVarNode::make(iv->dom, iv->var.copy_with_suffix(".jac.red"),
              iv->iter_type, iv->thread_tag);
        new_axis.push_back(new_v);
        vmap[iv->var.operator->()] = new_v;
      }

      Array<Expr> op_source_with_newaxis;
      for (const auto& src : op->source)
        op_source_with_newaxis.push_back(Substitute(src, vmap));

      Array<Expr> new_source;
      for (const auto& src : op_source_with_newaxis)
        new_source.push_back(Mutate(src));
      for (const auto& src : op_source_with_newaxis)
        new_source.push_back(src);

      CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
      return Reduce::make(new_combiner, new_source, new_axis, op->condition, op->value_index);
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
    std::cout << "Jacobian of " << output << " = " << op->body << std::endl;
    std::cout << "wrt " << input << std::endl;

    Array<IterVar> new_axis;
    std::unordered_map<const Variable*, Expr> vmap;
    for (IterVar iv : op->axis) {
      IterVar new_v =
        IterVarNode::make(iv->dom, iv->var.copy_with_suffix(".jac.out"),
            iv->iter_type, iv->thread_tag);
      new_axis.push_back(new_v);
      vmap[iv->var.operator->()] = new_v;
    }

    Array<Expr> input_itervars;
    size_t i = 0;
    for (Expr ext : input->shape) {
      IterVar new_v =
        IterVarNode::make(Range(0, ext), Var("jacobian_i" + std::to_string(i)),
            IterVarType::kDataPar);
      new_axis.push_back(new_v);
      input_itervars.push_back(new_v);
      ++i;
    }

    Expr new_body =
      Jacobian(Substitute(op->body[output->value_index], vmap), input, input_itervars);

    std::cout << "resulting body = " << new_body << "\n" << std::endl;

    int value_index = 0;
    Array<Expr> new_bodies;

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

    Array<Expr> new_shape = output->shape;
    for (const auto& e : input->shape)
      new_shape.push_back(e);

    return TensorNode::make(new_shape, output->dtype, new_op, value_index);
  }
  else
    NOT_IMPLEMENTED;
}

inline tvm::Tensor generalized_matmul(const tvm::Tensor& A,
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

    for (auto& subtensor : subtensors.subtensors) {
      Tensor part;
      if (subtensor->op.as<PlaceholderOpNode>()) {
        if (subtensor == input)
          part = generalized_matmul(head, Jacobian(output, subtensor), output->shape.size());
        else
          continue;
      }
      else {
        Tensor new_head =
          generalized_matmul(head, Jacobian(output, subtensor), output->shape.size());
        part = JacobianRecursive(subtensor, input, new_head);
      }

      if (res.operator->())
        res = topi::add(res, part);
      else
        res = part;
    }

    if (res.operator->())
      return res;
    else {
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
