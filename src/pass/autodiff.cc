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

#define NOT_IMPLEMENTED { throw "Not implemented"; }

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

    Expr Mutate_(const Min* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const Max* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const EQ* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const NE* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const LT* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const LE* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const GT* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const GE* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const And* op, const Expr& e) NOT_IMPLEMENTED
    Expr Mutate_(const Or* op, const Expr& e) NOT_IMPLEMENTED

    Expr Mutate_(const Reduce* op, const Expr& e) {
        NOT_IMPLEMENTED
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
