/*!
 *  Copyright (c) 2018 by Contributors
 * \file autodiff.cc
 * \brief Automatic differentiation of IR Expr
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>


namespace tvm {
namespace ir {

#define NOT_IMPLEMENTED { throw dmlc::Error("Derivative of this op is not implemented"); }

class JacobianMutator : public IRMutator {
  public:
    explicit JacobianMutator(Tensor input, Array<Expr> indices)
      : input_(input), indices_(indices), zero_(0.0) {}

    explicit JacobianMutator(VarExpr input)
      : input_var_(input), zero_(0.0) {}

    Expr Mutate_(const Variable* op, const Expr& e) {
      if (input_var_.operator->() && input_var_.get() == op)
        return FloatImm::make(op->type, 1.0);
      else
        return Cast::make(op->type, zero_);
    }

    Expr Mutate_(const Load* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const Let* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const Call* op, const Expr& e) {
      if (op->call_type == Call::CallType::Halide) {
        if (input_.operator->() && op->func.same_as(input_->op)) {
          Expr condition = UIntImm::make(Bool(), 1);
          for (size_t i = 0; i < input_.ndim(); ++i) {
            condition = And::make(condition, EQ::make(indices_[i], op->args[i]));
          }
          return Cast::make(op->type, condition);
        }
        else
          return Cast::make(op->type, this->zero_);
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
        Expr new_res = FloatImm::make(res.type(), 0.0);
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

      Array<Expr> new_source;
      for (const auto& src : op->source)
        new_source.push_back(Mutate(src));
      for (const auto& src : op->source)
        new_source.push_back(src);

      CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
      return Reduce::make(new_combiner, new_source, op->axis, op->condition, op->value_index);
    }

    Expr Mutate_(const Cast* op, const Expr& e) {
      if (op->type.is_float())
        return Cast::make(op->type, Mutate(op->value));
      else
        return Cast::make(op->type, zero_);
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
    Expr zero_;
};

Expr Jacobian(Expr expr, Tensor input, Array<Expr> indices) {
  return JacobianMutator(input, indices).Mutate(expr);
}

Expr Derivative(Expr expr, VarExpr var) {
  return JacobianMutator(var).Mutate(expr);
}

}  // namespace ir
}  // namespace tvm
