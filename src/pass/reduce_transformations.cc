/*!
 *  Copyright (c) 2018 by Contributors
 * \file reduce_transformations.cc
 * \brief Transformations of reduce expression.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include "../op/op_util.h"
#include <tvm/api_registry.h>
#include "arithmetic/ModulusRemainder.h"

namespace tvm {
namespace ir {

using HalideIR::Internal::gcd;
using HalideIR::Internal::lcm;

// TODO: Maybe move somewhere, a similar thing is used in combine_context_call
struct ExprLess {
    bool operator()(const Expr& l, const Expr& r) const {
      return Compare(l, r) < 0;
    }
};

struct ExprEq {
    bool operator()(const Expr& l, const Expr& r) const {
      return Compare(l, r) == 0;
    }
};

// TODO: Move somewhere
template <class container>
Expr All(const container& c) {
  Expr res;
  for (const auto& e : c)
    if (res.get())
      res = res && e;
    else
      res = e;
  if (res.get())
    return res;
  else
    return make_const(Bool(1), true);
}

// TODO: Move somewhere
template <class container>
Expr Minimum(const container& c, Type t) {
  Expr res;
  for (const auto& e : c)
    if (res.get())
      res = min(res, e);
    else
      res = e;
  if (res.get())
    return res;
  else
    return t.min();
}

// TODO: Move somewhere
template <class container>
Expr Maximum(const container& c, Type t) {
  Expr res;
  for (const auto& e : c)
    if (res.get())
      res = max(res, e);
    else
      res = e;
  if (res.get())
    return res;
  else
    return t.max();
}

// TODO: Same thing is done in Simplify, merge the code
Expr RemoveEmptyReduction(const Expr& e) {
  const Reduce* r = e.as<Reduce>();
  if (r && r->axis.empty()) {
    return Select::make(r->condition,
                        r->source[r->value_index],
                        r->combiner->identity_element[r->value_index]);
  }
  return e;
}

Expr SimplifyCombiner(const Expr& expr, bool prune_unused_components) {
  const Reduce* op = expr.as<Reduce>();

  // First simplify the results
  Array<Expr> simplified_result;
  for (const auto& res : op->combiner->result)
    simplified_result.push_back(Simplify(res));

  // Which components to keep
  std::vector<int> used(op->combiner->result.size(), false);

  if (prune_unused_components) {
    // This function recursively marks the used components starting from
    // the index idx
    std::function<void(int)> mark_used;
    mark_used = [&used, &simplified_result, op, &mark_used](size_t idx) {
      // if the idx-th component was mark as used before, do nothing
      if (used[idx]) return;
      used[idx] = true;

      // check if the idx-th result expr uses some lhs or rhs variables
      // and recursively mark the corresponding components
      for (size_t i = 0; i < simplified_result.size(); ++i)
        if (!used[i]) {
          if (ExprUseVar(simplified_result[idx], op->combiner->lhs[i]) ||
              ExprUseVar(simplified_result[idx], op->combiner->rhs[i]))
            mark_used(i);
        }
    };

    // mark all used components starting from the value_index
    mark_used(op->value_index);
  }
  else {
    // if pruning was not requested, keep all components
    used.assign(used.size(), true);
  }

  int new_value_index = op->value_index;
  Array<Expr> new_result;
  Array<Expr> new_identity;
  Array<Var> new_lhs;
  Array<Var> new_rhs;
  Array<Expr> new_source;

  // new stuff is old stuff which is used
  for (size_t i = 0; i < used.size(); ++i) {
    if (used[i]) {
      // We simplify the result and identity, but not the source
      new_result.push_back(simplified_result[i]);
      new_identity.push_back(Simplify(op->combiner->identity_element[i]));
      new_lhs.push_back(op->combiner->lhs[i]);
      new_rhs.push_back(op->combiner->rhs[i]);
      new_source.push_back(op->source[i]);
    }
    else if (static_cast<int>(i) < op->value_index)
      // value_index should also be adjusted
      new_value_index--;
  }

  CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
  return Reduce::make(new_combiner, new_source, op->axis, op->condition, new_value_index);
}

// clone iter vars and return both the new vars and the substitution
std::pair<Array<IterVar>, std::unordered_map<const Variable*, Expr>>
CloneIterVars(const Array<IterVar>& vars) {
  Array<IterVar> new_vars;
  std::unordered_map<const Variable*, Expr> vmap;
  for (IterVar iv : vars) {
    IterVar new_v =
      IterVarNode::make(iv->dom, iv->var.copy_with_suffix(".copy"),
          iv->iter_type, iv->thread_tag);
    new_vars.push_back(new_v);
    vmap[iv->var.get()] = new_v;
  }
  return std::make_pair(std::move(new_vars), std::move(vmap));
}

// clone reduction by cloning the axis variables
// TODO: when nested reductions are allowed, replace this with a mutator that does it recursively
Expr CloneReduction(const Expr& expr) {
  if (const Reduce* red = expr.as<Reduce>()) {
    Array<IterVar> new_axis;
    std::unordered_map<const Variable*, Expr> vmap;
    std::tie(new_axis, vmap) = CloneIterVars(red->axis);

    Array<Expr> src_with_newaxis;
    for (const auto& src : red->source)
      src_with_newaxis.push_back(Substitute(src, vmap));

    return Reduce::make(red->combiner, src_with_newaxis,
        new_axis, Substitute(red->condition, vmap), red->value_index);
  }
  else
    return expr;
}

// return true if this combiner is just a sum
bool IsSumCombiner(const CommReducer& combiner) {
  if (combiner->identity_element.size() != 1)
    return false;

  auto type = combiner->identity_element[0].type();
  Var src("src", type);
  auto cond = make_const(Bool(1), true);
  return Equal(Reduce::make(combiner, {src}, {}, cond, 0), tvm::sum(src, {}));
}

// Return true if zero may be factored out of a reduction with this combiner,
// i.e. `(a, 0) combine (b, 0) = (c, 0)` for any a, b, some c, and 0 being at the
// value_index position. All combiners generated by autodiff are such.
bool CanFactorZeroFromCombiner(const CommReducer& combiner, int value_index) {
  if (!is_const_scalar(combiner->identity_element[value_index], 0))
    return false;

  Expr zero = make_zero(combiner->result[value_index].type());
  Expr in = Substitute(combiner->result[value_index],
                       {{combiner->lhs[value_index], zero},
                        {combiner->rhs[value_index], zero}});
  in = Simplify(in);

  return Equal(zero, in);
}

class FuseTensorsMutator : public IRMutator {
  public:
    explicit FuseTensorsMutator(Array<Tensor> inlineable) {
      for (const Tensor& tensor : inlineable)
        inlineable_.emplace(tensor->op.operator->(), tensor->value_index);
    }

    Expr Mutate_(const Variable* op, const Expr& e) { return e; }
    Expr Mutate_(const Load* op, const Expr& e) { return e; }
    Expr Mutate_(const Let* op, const Expr& e) { return e; }

    Expr Mutate_(const Call* op, const Expr& e) {
      if (op->call_type == Call::CallType::Halide && op->value_index == 0) {
        const ComputeOpNode* op_comp = op->func.as<ComputeOpNode>();
        if (op_comp && op_comp->num_outputs() == 1) {
          if (inlineable_.empty() || inlineable_.count(std::make_pair(op_comp, op->value_index))) {
            if (const Reduce* op_red = op_comp->body[0].as<Reduce>()) {
              if (!IsSumCombiner(op_red->combiner))
                return e;
            }

            Array<Var> tensor_axes;
            for (const auto& var : op_comp->axis)
              tensor_axes.push_back(var->var);

            Expr new_e =
              Inline(Evaluate::make(e), op->func, tensor_axes,
                     op_comp->body[op->value_index]).as<ir::Evaluate>()->value;

            new_e = CloneReduction(new_e);

            return Mutate(new_e);
          }
        }
      }

      return e;
    }

    // TODO: This is possible by introducing an additional itervar but I'm not sure if it's useful
    // Also if the two reductions may be aligned then the loops can be fused
    Expr Mutate_(const Add* op, const Expr& e)  { return e; }
    Expr Mutate_(const Sub* op, const Expr& e)  { return e; }

    Expr Mutate_(const Mul* op, const Expr& e) {
      Expr ma = Mutate(op->a);
      Expr mb = Mutate(op->b);
      if (ma.same_as(op->a) && mb.same_as(op->b))
        return e;

      return MakeMul(ma, mb);
    }

    Expr MakeMul(const Expr& a, const Expr& b) {
      const Reduce* a_red = a.as<Reduce>();
      if (a_red && IsSumCombiner(a_red->combiner))
        return MakeSum(MakeMul(a_red->source[0], b), a_red);

      const Reduce* b_red = b.as<Reduce>();
      if (b_red && IsSumCombiner(b_red->combiner))
        return MakeSum(MakeMul(a, b_red->source[0]), b_red);

      return Mul::make(a, b);
    }

    // TODO: This is questionable, at least for ints, maybe remove this case
    Expr Mutate_(const Div* op, const Expr& e) {
      Expr a = Mutate(op->a);

      const Reduce* a_red = a.as<Reduce>();
      if (a_red && IsSumCombiner(a_red->combiner))
        return MakeSum(Div::make(a_red->source[0], op->b), a_red);

      return e;
    }

    Expr Mutate_(const Mod* op, const Expr& e) { return e; }
    Expr Mutate_(const Min* op, const Expr& e) { return e; }
    Expr Mutate_(const Max* op, const Expr& e) { return e; }

    Expr Mutate_(const Reduce* op, const Expr& e) {
      if (IsSumCombiner(op->combiner))
        return MakeSum(Mutate(op->source[0]), op);
      else
        return e;
    }

    Expr Mutate_(const Cast* op, const Expr& e) {
      if (op->type == op->value.type())
        return Mutate(op->value);
      else
        return e; // TODO: In some cases this may be safe
    }

    // TODO: Either rewrite as c*a + (1-c)*b or do something equivalent
    Expr Mutate_(const Select* op, const Expr& e) {
      return e;
      //return MakeSelect(op->condition, Mutate(op->true_value), Mutate(op->false_value));
    }

    Expr MakeSelect(const Expr& cond, const Expr& a, const Expr& b) {
      // TODO
    }

    Expr MakeSum(const Expr& source, const Reduce* red) {
      const Reduce* src_red = source.as<Reduce>();

      if (src_red && IsSumCombiner(src_red->combiner)) {
        // TODO: Check types
        Array<IterVar> axes(red->axis);
        for (const IterVar& v : src_red->axis)
          axes.push_back(v);

        return Reduce::make(red->combiner, {src_red->source}, axes,
            And::make(red->condition, src_red->condition), 0);
      }
      else
        return Reduce::make(red->combiner, {source},
            red->axis, red->condition, red->value_index);
    }

  private:
    std::set<std::pair<const OperationNode*, int>> inlineable_;
};

Tensor FuseTensors(const Tensor& outer, const Array<Tensor>& to_inline) {
  if (const ComputeOpNode* outer_op = outer->op.as<ComputeOpNode>()) {
    FuseTensorsMutator mutator(to_inline);
    Array<Expr> fused_body;
    for (const Expr& e : outer_op->body)
      fused_body.push_back(mutator.Mutate(e));
    return ComputeOpNode::make(outer_op->name, outer_op->tag, outer_op->attrs,
        outer_op->axis, fused_body).output(outer->value_index);
  }
  else
    CHECK(false) << "Not implemented";
}


Expr InlineThisCall(const Expr& expr) {
  if (const Call* op = expr.as<Call>()) {
    if (op->call_type == Call::CallType::Halide) {
      if (const ComputeOpNode* op_comp = op->func.as<ComputeOpNode>()) {
        Array<Var> tensor_axes;
        for (const auto& var : op_comp->axis)
          tensor_axes.push_back(var->var);

        Expr new_expr = Inline(Evaluate::make(expr), op->func, tensor_axes,
                               op_comp->body[op->value_index]).as<ir::Evaluate>()->value;
        // If it is a reduction, clone it
        return CloneReduction(new_expr);
      }
    }
  }

  return expr;
}

Tensor InlineTailCall(const Tensor& tensor) {
  return op::TransformBody(tensor, InlineThisCall);
}


class InlineNonReductionsMutator : public IRMutator {
  public:
    InlineNonReductionsMutator(const Array<Tensor>& inlineable) {
      for (const Tensor& tensor : inlineable)
        inlineable_.emplace(tensor->op.operator->(), tensor->value_index);
    }

    Expr Mutate_(const Call* op, const Expr& e) {
      if (op->call_type == Call::CallType::Halide) {
        const ComputeOpNode* op_comp = op->func.as<ComputeOpNode>();
        if (inlineable_.empty() || inlineable_.count(std::make_pair(op_comp, op->value_index))) {
          if (op_comp && !op_comp->body[0].as<Reduce>()) {
            Array<Var> tensor_axes;
            for (const auto& var : op_comp->axis)
              tensor_axes.push_back(var->var);

            Expr new_e =
              Inline(Evaluate::make(e), op->func, tensor_axes,
                     op_comp->body[op->value_index]).as<ir::Evaluate>()->value;

            return Mutate(new_e);
          }
        }
      }

      return e;
    }

  private:
    std::set<std::pair<const OperationNode*, int>> inlineable_;
};

Tensor InlineNonReductions(const Tensor& tensor, const Array<Tensor>& inlineable) {
  auto transformation =
    [inlineable](const Expr& e) { return InlineNonReductionsMutator(inlineable).Mutate(e); };
  return op::TransformBody(tensor, transformation);
}


class NonzeronessCondition {
  public:
    static std::pair<Expr, Expr> Nonzeroness(const Expr& e) {
      const static FType& f = vtable();
      return f(e, e);
    }

    using FType = IRFunctor<std::pair<Expr, Expr> (const NodeRef&, const Expr&)>;
    static FType& vtable() {
      static FType inst;
      return inst;
    }

    static Expr PairToExpr(const std::pair<Expr, Expr>& p) {
      return Select::make(p.first, p.second, make_zero(p.second.type()));
    }

    static std::pair<Expr, Expr> DefaultFunc(const NodeRef&, const Expr& e) {
      return std::make_pair(UIntImm::make(Bool(), 1), e);
    };

    template <class TNode>
    static std::pair<Expr, Expr> Const(const TNode* op, const Expr& e) {
      if (op->value == 0)
        return std::make_pair(UIntImm::make(Bool(), 0), e);
      else
        return std::make_pair(UIntImm::make(Bool(), 1), e);
    };

    template <class TNode>
    static std::pair<Expr, Expr> BinOpAddLike(const TNode* op, const Expr& e) {
      auto pair_a = Nonzeroness(op->a);
      auto pair_b = Nonzeroness(op->b);

      if (Equal(pair_a.first, pair_b.first)) {
        if (pair_a.second.same_as(op->a) && pair_b.second.same_as(op->b))
          return std::make_pair(pair_a.first, e);
        else
          return std::make_pair(pair_a.first, TNode::make(pair_a.second, pair_b.second));
      }
      else {
        Expr new_cond = CanonicalSimplify(Simplify(Or::make(pair_a.first, pair_b.first)));
        Expr new_a = Equal(pair_a.first, new_cond) ? pair_a.second : PairToExpr(pair_a);
        Expr new_b = Equal(pair_b.first, new_cond) ? pair_b.second : PairToExpr(pair_b);
        Expr new_expr = TNode::make(new_a, new_b);
        return std::make_pair(new_cond, new_expr);
      }
    }

    template <class TNode>
    static std::pair<Expr, Expr> BinOpMulLike(const TNode* op, const Expr& e) {
      auto pair_a = Nonzeroness(op->a);
      auto pair_b = Nonzeroness(op->b);

      Expr new_cond = CanonicalSimplify(Simplify(pair_a.first && pair_b.first));

      if (pair_a.second.same_as(op->a) && pair_b.second.same_as(op->b))
        return std::make_pair(new_cond, e);
      else
        return std::make_pair(new_cond, TNode::make(pair_a.second, pair_b.second));
    }

    template <class TNode>
    static std::pair<Expr, Expr> BinOpDivLike(const TNode* op, const Expr& e) {
      auto pair_a = Nonzeroness(op->a);

      if (pair_a.second.same_as(op->a))
        return std::make_pair(pair_a.first, e);
      else
        return std::make_pair(pair_a.first, TNode::make(pair_a.second, op->b));
    }
};

TVM_STATIC_IR_FUNCTOR(NonzeronessCondition, vtable)
.set_dispatch<Variable>(NonzeronessCondition::DefaultFunc)
.set_dispatch<Call>(NonzeronessCondition::DefaultFunc)
.set_dispatch<IntImm>(NonzeronessCondition::Const<IntImm>)
.set_dispatch<UIntImm>(NonzeronessCondition::Const<UIntImm>)
.set_dispatch<FloatImm>(NonzeronessCondition::Const<FloatImm>)
.set_dispatch<StringImm>(NonzeronessCondition::DefaultFunc)
.set_dispatch<Add>(NonzeronessCondition::BinOpAddLike<Add>)
.set_dispatch<Sub>(NonzeronessCondition::BinOpAddLike<Sub>)
.set_dispatch<Mul>(NonzeronessCondition::BinOpMulLike<Mul>)
.set_dispatch<Div>(NonzeronessCondition::BinOpDivLike<Div>)
.set_dispatch<Mod>(NonzeronessCondition::BinOpDivLike<Mod>)
.set_dispatch<Min>(NonzeronessCondition::BinOpAddLike<Min>)
.set_dispatch<Max>(NonzeronessCondition::BinOpAddLike<Max>)
.set_dispatch<Cast>([](const Cast* op, const Expr& e) {
  if (op->value.type().is_bool())
    return std::make_pair(op->value, make_const(e.type(), 1));
  else {
    auto pair_a = NonzeronessCondition::Nonzeroness(op->value);

    if (pair_a.second.same_as(op->value))
      return std::make_pair(pair_a.first, e);
    else
      return std::make_pair(pair_a.first, Cast::make(op->type, pair_a.second));
  }
})
.set_dispatch<Select>([](const Select* op, const Expr& e) {
  auto pair_a = NonzeronessCondition::Nonzeroness(op->true_value);
  auto pair_b = NonzeronessCondition::Nonzeroness(op->false_value);

  if (is_const_scalar(pair_b.second, 0)) {
    Expr new_cond = CanonicalSimplify(Simplify(pair_a.first && op->condition));
    return std::make_pair(new_cond, pair_a.second);
  }

  if (is_const_scalar(pair_a.second, 0)) {
    Expr new_cond = CanonicalSimplify(Simplify(pair_b.first && !op->condition));
    return std::make_pair(new_cond, pair_b.second);
  }

  Expr new_cond =
    CanonicalSimplify(Simplify(Or::make(op->condition && pair_a.first,
                                        !op->condition &&  pair_b.first)));
  if (pair_a.second.same_as(op->true_value) && pair_b.second.same_as(op->false_value))
    return std::make_pair(new_cond, e);
  else
    return std::make_pair(new_cond, Select::make(op->condition, pair_a.second, pair_b.second));
});

Expr LiftNonzeronessCondition(const Expr& expr) {
  return NonzeronessCondition::PairToExpr(NonzeronessCondition::Nonzeroness(expr));
}


class NormalizeComparisonsMutator : public IRMutator {
  public:
    virtual Expr Mutate_(const EQ* op, const Expr& e) { return Make<EQ>(op->a, op->b); }
    virtual Expr Mutate_(const NE* op, const Expr& e) { return Make<NE>(op->a, op->b); }
    virtual Expr Mutate_(const LT* op, const Expr& e) { return Make<LT>(op->a, op->b); }
    virtual Expr Mutate_(const LE* op, const Expr& e) { return Make<LE>(op->a, op->b); }
    virtual Expr Mutate_(const GT* op, const Expr& e) { return Make<LT>(op->b, op->a); }
    virtual Expr Mutate_(const GE* op, const Expr& e) { return Make<LE>(op->b, op->a); }

  private:
    template <class TNode>
    Expr Make(const Expr& a, const Expr& b) {
      // rewrite LT to LE for ints
      if (std::is_same<TNode, LT>::value && (a.type().is_int() || a.type().is_uint()))
        return LE::make(CanonicalSimplify(Simplify(a - b + 1)), make_zero(a.type()));
      return TNode::make(CanonicalSimplify(Simplify(a - b)), make_zero(a.type()));
    }
};

// Rewrite every comparison into the form a == 0, a != 0, a <= 0, and sometimes for floats a < 0
Expr NormalizeComparisons(const Expr& expr) {
  return NormalizeComparisonsMutator().Mutate(expr);
}


// TODO: This is easier to express with a bunch of ifs, not a functor with dispatch
class FactorOutAtomicFormulas {
  public:
    static std::pair<std::vector<Expr>, Expr> Factor(const Expr& e) {
      const static FType& f = vtable();
      return f(e, e);
    }

    static Expr PairToExpr(const std::pair<std::vector<Expr>, Expr>& p) {
      Expr res = p.second;
      for (const Expr& e : p.first)
        res = And::make(e, res);
      return res;
    }

    using FType = IRFunctor<std::pair<std::vector<Expr>, Expr> (const NodeRef&, const Expr&)>;
    static FType& vtable() {
      static FType inst;
      return inst;
    }

    static std::pair<std::vector<Expr>, Expr> Atomic(const NodeRef&, const Expr& e) {
      return std::make_pair<std::vector<Expr>, Expr>({e}, make_const(e.type(), 1));
    }
};

TVM_STATIC_IR_FUNCTOR(FactorOutAtomicFormulas, vtable)
.set_dispatch<Variable>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<Call>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<IntImm>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<UIntImm>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<EQ>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<NE>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<LE>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<LT>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<GE>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<GT>(FactorOutAtomicFormulas::Atomic)
.set_dispatch<And>([](const And* op, const Expr& e) {
  auto pair_a = FactorOutAtomicFormulas::Factor(op->a);
  auto pair_b = FactorOutAtomicFormulas::Factor(op->b);

  std::vector<Expr> res;
  res.reserve(pair_a.first.size() + pair_b.first.size());
  std::set_union(pair_a.first.begin(), pair_a.first.end(),
                 pair_b.first.begin(), pair_b.first.end(),
                 std::back_inserter(res),
                 ExprLess());

  return std::make_pair(res, pair_a.second && pair_b.second);
})
.set_dispatch<Or>([](const Or* op, const Expr& e) {
  auto pair_a = FactorOutAtomicFormulas::Factor(op->a);
  auto pair_b = FactorOutAtomicFormulas::Factor(op->b);

  std::vector<Expr> res;
  res.reserve(std::min(pair_a.first.size(), pair_b.first.size()));
  std::set_intersection(pair_a.first.begin(), pair_a.first.end(),
                        pair_b.first.begin(), pair_b.first.end(),
                        std::back_inserter(res),
                        ExprLess());

  std::vector<Expr> new_cond_a;
  new_cond_a.reserve(pair_a.first.size() - res.size());
  std::set_difference(pair_a.first.begin(), pair_a.first.end(),
                      res.begin(), res.end(),
                      std::back_inserter(new_cond_a),
                      ExprLess());

  std::vector<Expr> new_cond_b;
  new_cond_b.reserve(pair_b.first.size() - res.size());
  std::set_difference(pair_b.first.begin(), pair_b.first.end(),
                      res.begin(), res.end(),
                      std::back_inserter(new_cond_b),
                      ExprLess());

  pair_a.first = std::move(new_cond_a);
  pair_b.first = std::move(new_cond_b);

  return std::make_pair(res, Or::make(FactorOutAtomicFormulas::PairToExpr(pair_a),
                                      FactorOutAtomicFormulas::PairToExpr(pair_b)));
});

struct VarBounds {
  Expr coef;
  Array<Expr> lower;
  Array<Expr> equal;
  Array<Expr> upper;

  Array<Expr> get_var_upper_bounds() const {
    Array<Expr> res;
    for (Expr e : equal)
      res.push_back(e/coef);
    for (Expr e : upper)
      res.push_back(e/coef);
    return res;
  }

  Array<Expr> get_var_lower_bounds() const {
    Array<Expr> res;
    for (Expr e : equal)
      res.push_back(e/coef);
    for (Expr e : lower)
      res.push_back(e/coef);
    return res;
  }

  VarBounds substitute(const std::unordered_map<const Variable*, Expr>& subst) const {
    auto apply_fun = [&subst](const Expr& e) { return Substitute(e, subst); };
    return {Substitute(coef, subst),
            UpdateArray(lower, apply_fun),
            UpdateArray(equal, apply_fun),
            UpdateArray(upper, apply_fun)};
  }
};

struct SolveSystemOfInequalitiesResult {
  Array<Var> variables;
  std::unordered_map<const Variable*, VarBounds> bounds;
  Array<Expr> other_conditions;

  Array<Expr> as_conditions() const {
    Array<Expr> res;
    for (const Var& v : variables) {
      auto it = bounds.find(v.get());
      CHECK(it != bounds.end());
      const VarBounds& bnds = it->second;
      Expr lhs = bnds.coef * v;
      for (const Expr& rhs : bnds.equal)
        res.push_back(EQ::make(lhs, rhs));
      for (const Expr& rhs : bnds.lower)
        res.push_back(GE::make(lhs, rhs));
      for (const Expr& rhs : bnds.upper)
        res.push_back(LE::make(lhs, rhs));
    }
    for (const Expr& e : other_conditions)
      res.push_back(e);
    return res;
  }
};

// Rewrite the system of inequalities using Fourier-Motzkin elimination
SolveSystemOfInequalitiesResult SolveSystemOfInequalities(const Array<Expr>& inequalities,
                                                          const Array<Var>& variables) {
  SolveSystemOfInequalitiesResult res;
  res.variables = variables;

  std::vector<Expr> current;
  std::vector<Expr> new_current;
  std::vector<std::pair<int64_t, Expr>> coef_pos;
  std::vector<std::pair<int64_t, Expr>> coef_neg;

  std::cout << "\nSolveSystemOfInequalities\n";
  std::cout << "  ineqs: " << inequalities << "\n  vars: " << variables << "\n";

  // formulas we don't what to do with
  std::vector<Expr> rest;

  for (const Expr& ineq : inequalities)
    current.push_back(NormalizeComparisons(ineq));

  for (const Var& v : variables) {
    CHECK(!res.bounds.count(v.get())) <<
      "Variable " << v << " appears several times in the `variables` which might be a bug";

    new_current.clear();
    coef_pos.clear();
    coef_neg.clear();

    //std::cout << "\n";
    //std::cout << "  var " << v << "\n";
    //std::cout << "  current " << Array<Expr>(current) << "\n";

    for (const Expr& ineq : current) {
      if (const LE* le = ineq.as<LE>()) {
        Array<Expr> coef = arith::DetectLinearEquation(le->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0)
            new_current.push_back(ineq);
          else if (coef0 > 0)
            coef_pos.push_back(std::make_pair(coef0, coef[1]));
          else if (coef0 < 0)
            coef_neg.push_back(std::make_pair(coef0, coef[1]));
          continue;
        }
      }
      else if (const EQ* eq = ineq.as<EQ>()) {
        Array<Expr> coef = arith::DetectLinearEquation(eq->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0)
            new_current.push_back(ineq);
          else if (coef0 > 0) {
            coef_pos.push_back(std::make_pair(coef0, coef[1]));
            coef_neg.push_back(std::make_pair(-coef0, -coef[1]));
          }
          else if (coef0 < 0) {
            coef_pos.push_back(std::make_pair(-coef0, -coef[1]));
            coef_neg.push_back(std::make_pair(coef0, coef[1]));
          }
          continue;
        }
      }

      // if nothing worked, put it in rest
      rest.push_back(ineq);
    }

    // Combine each positive inequality with each negative one
    for (const auto& pos : coef_pos)
      for (const auto& neg : coef_neg) {
        auto first_gcd = gcd(pos.first, -neg.first);
        Expr c_pos = make_const(v.type(), neg.first/first_gcd);
        Expr c_neg = make_const(v.type(), pos.first/first_gcd);
        new_current.push_back(LE::make(c_neg*neg.second - c_pos*pos.second,
                                       make_zero(pos.second.type())));
      }

    // Find the common denominator in a sense
    // We will generate equations of the form coef_lcm*v <= bound
    int64_t coef_lcm = 1;
    for (const auto& pos : coef_pos)
      coef_lcm = lcm(coef_lcm, pos.first);
    for (const auto& neg : coef_neg)
      coef_lcm = lcm(coef_lcm, -neg.first);

    std::vector<Expr> upper_bounds;
    std::vector<Expr> lower_bounds;
    upper_bounds.reserve(coef_pos.size());
    lower_bounds.reserve(coef_neg.size());

    for (const auto& pos : coef_pos) {
      Expr bound = make_const(v.type(), -coef_lcm/pos.first)*pos.second;
      upper_bounds.push_back(CanonicalSimplify(Simplify(bound)));
    }
    for (const auto& neg : coef_neg) {
      Expr bound = make_const(v.type(), -coef_lcm/neg.first)*neg.second;
      lower_bounds.push_back(CanonicalSimplify(Simplify(bound)));
    }

    for (std::vector<Expr>* bounds : {&upper_bounds, &lower_bounds}) {
      std::sort(bounds->begin(), bounds->end(), ExprLess());
      bounds->erase(std::unique(bounds->begin(), bounds->end(), ExprEq()), bounds->end());
    }

    std::vector<Expr> equal;
    equal.reserve(std::min(upper_bounds.size(), lower_bounds.size()));
    std::set_intersection(upper_bounds.begin(), upper_bounds.end(),
                          lower_bounds.begin(), lower_bounds.end(),
                          std::back_inserter(equal), ExprLess());

    std::vector<Expr> new_upper;
    new_upper.reserve(upper_bounds.size() - equal.size());
    std::set_difference(upper_bounds.begin(), upper_bounds.end(),
                        equal.begin(), equal.end(),
                        std::back_inserter(new_upper), ExprLess());

    std::vector<Expr> new_lower;
    new_lower.reserve(lower_bounds.size() - equal.size());
    std::set_difference(lower_bounds.begin(), lower_bounds.end(),
                        equal.begin(), equal.end(),
                        std::back_inserter(new_lower), ExprLess());

    auto& bnds = res.bounds[v.get()];
    bnds.coef = make_const(v.type(), coef_lcm);
    bnds.equal = equal;
    bnds.lower = new_lower;
    bnds.upper = new_upper;

    std::swap(current, new_current);
  }

  for(const Expr& e : current) {
    Expr e_simp = Simplify(e);
    if (is_const_int(e_simp, 0)) {
      // contradiction detected
      res.other_conditions = {make_const(Bool(1), 0)};
      return res;
    }
    else if (is_const_int(e_simp, 1))
      continue;
    else
      res.other_conditions.push_back(e_simp);
  }

  for(const Expr& e : rest)
    res.other_conditions.push_back(e);

  std::cout << "  res: " << res.as_conditions() << "\n" << std::endl;

  return res;
}

TVM_REGISTER_API("arith.SolveSystemOfInequalities")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = SolveSystemOfInequalities(args[0], args[1]).as_conditions();
  });

struct DomainSimplificationResult {
  Array<IterVar> axis;
  Array<Expr> conditions;
  std::unordered_map<const Variable*, Expr> old_to_new;
  std::unordered_map<const Variable*, Expr> new_to_old;
};

DomainSimplificationResult SimplifyDomain(const Expr& cond,
                                          const Array<IterVar>& axis,
                                          const Array<IterVar>& outer_axis) {
  Expr rest_of_cond;
  std::vector<Expr> atomic_formulas;
  std::tie(atomic_formulas, rest_of_cond) = FactorOutAtomicFormulas::Factor(cond);

  Array<Var> vars;
  for (const IterVar& v : axis)
    vars.push_back(v->var);
  for (const IterVar& v : outer_axis)
    vars.push_back(v->var);

  auto solved_system = SolveSystemOfInequalities(atomic_formulas, vars);

  DomainSimplificationResult res;
  std::unordered_map<const Variable*, IntSet> new_var_intsets;

  for (const IterVar& v : outer_axis)
    new_var_intsets[v->var.get()] = IntSet::range(v->dom);

  for (auto it = axis.rbegin(); it != axis.rend(); ++it) {
    const IterVar& iv = *it;
    auto& bnd = solved_system.bounds[iv->var.get()];
    bnd = bnd.substitute(res.old_to_new);
    if (is_one(bnd.coef) && !bnd.equal.empty()) {
      res.old_to_new[iv->var.get()] = bnd.equal[0];

      std::cout << "\nvar " << iv << " replaced with " << bnd.equal[0] << "\n";
    }
    else {
      Array<Expr> lowers = bnd.get_var_lower_bounds();
      Array<Expr> uppers = bnd.get_var_upper_bounds();

      Expr best_lower, best_diff, best_diff_upper;

      for (const Expr& low : lowers) {
        for (const Expr& upp : uppers) {
          Expr diff = Simplify(upp - low);
          Expr diff_upper = EvalSet(diff, new_var_intsets).max();

          if (!best_lower.get() || is_const_int(Simplify(diff_upper < best_diff_upper), 1)) {
            best_lower = low;
            best_diff = diff;
            best_diff_upper = diff_upper;
          }
        }
      }

      std::cout << "\nvar " << iv << " has best lower " << best_lower << "     and diff    " << best_diff << "\n";

      Var new_iv = iv->var.copy_with_suffix(".shifted");
      res.old_to_new[iv->var.get()] = new_iv + best_lower;
      res.new_to_old[new_iv.get()] = iv->var - best_lower;

      std::cout << "var " << iv << " replaced with " << res.old_to_new[iv->var.get()] << "\n";

      new_var_intsets[new_iv.get()] = IntSet::interval(make_zero(new_iv.type()), best_diff_upper);

      std::cout << "its ubound " << best_diff_upper;

      auto range = Range(make_zero(new_iv.type()), best_diff_upper + 1);
      res.axis.push_back(IterVarNode::make(range, new_iv, iv->iter_type, iv->thread_tag));

      std::cout << "new range " << range << "\n";
    }
  }

  for (const Expr& old_cond : solved_system.as_conditions())
    res.conditions.push_back(Substitute(old_cond, res.old_to_new));

  return res;
}

// Use the condition of a reduction op to simplify its domain (axis)
Expr SimplifyReductionDomain(const Expr& expr, const Array<IterVar>& outer_axis) {
  if (const Reduce* red = expr.as<Reduce>()) {
    auto res = SimplifyDomain(red->condition, red->axis, outer_axis);

    Array<Expr> new_source;
    for (const Expr& src : red->source)
      new_source.push_back(Substitute(src, res.old_to_new));

    std::cout << "\nred before simplify dom\n" << expr << "\n";
    std::cout << "\nred after simplify dom\n" << Reduce::make(red->combiner, new_source, res.axis, All(res.conditions), red->value_index) << "\n\n";

    return RemoveEmptyReduction(Reduce::make(red->combiner, new_source, res.axis,
                                             All(res.conditions), red->value_index));
  }
  else
    return expr;
}

// Extract from cond an implication of cond not containing vars
std::pair<Expr, Expr> ImplicationNotContainingVars(
                          const Expr& cond, const std::unordered_set<const Variable*>& vars) {
  // TODO: assert cond is bool
  // TODO: not
  if (const And* op = cond.as<And>()) {
    auto pair_a = ImplicationNotContainingVars(op->a, vars);
    auto pair_b = ImplicationNotContainingVars(op->b, vars);
    return std::make_pair(pair_a.first && pair_b.first,
                          pair_a.second && pair_b.second);
  }
  else if (const Or* op = cond.as<Or>()) {
    auto pair_a = ImplicationNotContainingVars(op->a, vars);
    auto pair_b = ImplicationNotContainingVars(op->b, vars);
    return std::make_pair(Or::make(pair_a.first, pair_b.first), cond);
  }
  else if (!ExprUseVar(cond, vars)) {
    return std::make_pair(cond, make_const(Bool(1), true));
  }
  else
    return std::make_pair(make_const(Bool(1), true), cond);
}


// TODO: Move somewere
// Convert an array of itervars to an array of inequalities
Array<Expr> IterVarsToInequalities(const Array<IterVar>& itervars) {
  Array<Expr> res;
  for (const IterVar& v : itervars) {
    res.push_back(GE::make(v->var, v->dom->min));
    res.push_back(LT::make(v->var, v->dom->min + v->dom->extent));
  }
  return res;
}


class RemoveRedundantInequalitiesMutator : public IRMutator {
  public:
    RemoveRedundantInequalitiesMutator(Array<Expr> known) {
      for (const Expr& cond : known)
        known_.push_back(CanonicalSimplify(Simplify(cond)));
    }

    virtual Expr Mutate_(const Select* op, const Expr& e) {
      Expr new_cond = Simplify(Mutate(op->condition));
      if (is_one(new_cond))
        return Mutate(op->true_value);
      else if (is_zero(new_cond))
        return Mutate(op->false_value);
      else {
        Array<Expr> new_known = known_;
        for (const Expr& atomic : FactorOutAtomicFormulas::Factor(new_cond).first)
          new_known.push_back(atomic);
        RemoveRedundantInequalitiesMutator new_mutator(new_known);
        // Note that we mutate with the new mutator only the true value
        // TODO: Update known conditions for the false value as well
        return Select::make(new_cond, new_mutator.Mutate(op->true_value), Mutate(op->false_value));
      }
    }

    virtual Expr Mutate_(const Reduce* op, const Expr& e) {
      Array<Expr> known_with_axes = known_;
      for (const Expr& axis_cond : IterVarsToInequalities(op->axis))
          known_with_axes.push_back(axis_cond);
      RemoveRedundantInequalitiesMutator mutator_with_axes(known_with_axes);

      Expr new_cond = mutator_with_axes.Mutate(op->condition);

      Array<Expr> new_known = known_with_axes;
      for (const Expr& atomic : FactorOutAtomicFormulas::Factor(new_cond).first)
        new_known.push_back(atomic);
      RemoveRedundantInequalitiesMutator new_mutator(new_known);

      Array<Expr> new_source;
      for (const Expr& src : op->source)
        new_source.push_back(new_mutator.Mutate(src));

      return Reduce::make(op->combiner, new_source, op->axis, new_cond, op->value_index);
    }

    virtual Expr Mutate_(const EQ* op, const Expr& e) { return MutateAtomic_(e); }
    virtual Expr Mutate_(const NE* op, const Expr& e) { return MutateAtomic_(e); }
    virtual Expr Mutate_(const LT* op, const Expr& e) { return MutateAtomic_(e); }
    virtual Expr Mutate_(const LE* op, const Expr& e) { return MutateAtomic_(e); }
    virtual Expr Mutate_(const GT* op, const Expr& e) { return MutateAtomic_(e); }
    virtual Expr Mutate_(const GE* op, const Expr& e) { return MutateAtomic_(e); }

    // Use eager constant folding to get rid of ubiquitous (uint1)1
    virtual Expr Mutate_(const And* op, const Expr& e) {
      return Mutate(op->a) && Mutate(op->b);
    }

  private:
    Expr MutateAtomic_(const Expr& e) {
      Expr simplified = CanonicalSimplify(Simplify(e));
      for (const Expr& other : known_)
        if (Equal(simplified, other))
          return make_const(Bool(1), true);
      return simplified;
    }

    Array<Expr> known_;
};

// Propagate information from conditions and remove redundant inequalities
Expr RemoveRedundantInequalities(const Expr& expr, const Array<Expr>& known) {
  return RemoveRedundantInequalitiesMutator(known).Mutate(expr);
}


// TODO: Move somewhere and use instead of directly
Expr IfThenElseZero(const Expr& cond, const Expr& on_true) {
  return Select::make(cond, on_true, make_zero(on_true.type()));
}

// TODO: Move somewhere, it is quite general
std::pair<Expr, Expr> LiftConditionsThroughReduction(const Expr& cond,
                                                     const Array<IterVar>& red_axis,
                                                     const Array<IterVar>& outer_axis) {
  Expr rest;
  Array<Expr> atomics;
  // Factor out atomics so that we can consider this as a system of inequalities
  std::tie(atomics, rest) = FactorOutAtomicFormulas().Factor(cond);

  Array<Var> allvars;
  for (const IterVar& v : red_axis)
    allvars.push_back(v->var);
  for (const IterVar& v : outer_axis)
    allvars.push_back(v->var);

  // start from reduction vars, so that input vars don't depend on them
  atomics = SolveSystemOfInequalities(atomics, allvars).as_conditions();

  // Append the rest part
  Expr rewritten_cond = All(atomics) && rest;

  std::unordered_set<const Variable*> vset;
  for (const IterVar& v : red_axis)
    vset.insert(v->var.get());

  // The outer (first) condition does not contain reduction vars,
  // the inner (second) condition is everything else
  return ImplicationNotContainingVars(rewritten_cond, vset);
}

class SplitIntoTensorsSmartlyMutator : public IRMutator {
  public:
    explicit SplitIntoTensorsSmartlyMutator(Array<IterVar> axis, std::string name="auto")
      : axis_(std::move(axis)), name_(std::move(name)) {}

    Expr Mutate_(const Reduce* op, const Expr& e) {
      Array<IterVar> combined_axis = axis_;
      for (const IterVar& v : op->axis)
        combined_axis.push_back(v);

      SplitIntoTensorsSmartlyMutator new_mutator(combined_axis);

      Array<Expr> new_source;
      for (const Expr& src : op->source)
        new_source.push_back(new_mutator.Mutate(src));

      Expr new_reduce =
        Reduce::make(op->combiner, new_source, op->axis, op->condition, op->value_index);

      auto newaxis_vmap_pair = CloneIterVars(axis_);
      Array<IterVar> new_axis = newaxis_vmap_pair.first;
      new_reduce = Substitute(new_reduce, newaxis_vmap_pair.second);

      const Reduce* red = new_reduce.as<Reduce>();

      Array<Expr> new_body;
      for (size_t i = 0; i < op->source.size(); ++i)
        new_body.push_back(Reduce::make(red->combiner, red->source, red->axis, red->condition, i));

      Tensor tensor =
        ComputeOpNode::make(name_ + ".extracted_reduction", tag_, attrs_, new_axis, new_body)
          .output(op->value_index);

      Array<Expr> args;
      for (const IterVar& v : axis_)
        args.push_back(v->var);

      return Call::make(e.type(), tensor->op->name, args,
                        Call::CallType::Halide, tensor->op, tensor->value_index);
    }

  private:
    Array<IterVar> axis_;
    std::string name_;
    std::string tag_;
    Map<std::string, NodeRef> attrs_;
};

// Introduce tensors wherever needed (on reductions) or makes sense (memoization)
// TODO: Do this smartly, currently we just extract reductions
Expr SplitIntoTensorsSmartly(const Expr& expr, const Array<IterVar>& axis) {
  return SplitIntoTensorsSmartlyMutator(axis).Mutate(expr);
}

Expr OptimizeAndLiftNonzeronessConditionsImpl(const Expr& expr, const Array<IterVar>& axis) {
  Array<Expr> axis_conds = IterVarsToInequalities(axis);

  Expr result;

  if (const Reduce* red = expr.as<Reduce>()) {
    bool is_sum = IsSumCombiner(red->combiner);
    if (is_sum || CanFactorZeroFromCombiner(red->combiner, red->value_index)) {
      Expr new_red = expr;

      // Here we add axis conditions to the reduce conditions and simplify the reduction
      {
        Array<Expr> red_axis_conds = IterVarsToInequalities(red->axis);

        Expr cond = All(axis_conds) && All(red_axis_conds) && red->condition;
        Array<Expr> source = red->source;

        // If it is summation then we can lift nonzeroness conditions from the source
        // and add them to the reduction conditions
        if (is_sum) {
          Expr nz_cond, nz_source;
          std::tie(nz_cond, nz_source) =
            NonzeronessCondition::Nonzeroness(red->source[red->value_index]);
          cond = nz_cond && cond;
          source.Set(0, nz_source);
        }

        new_red = Reduce::make(red->combiner, source, red->axis, cond, red->value_index);
        new_red = SimplifyReductionDomain(new_red, axis);
        red = new_red.as<Reduce>();

        // If the reduction disappears completely then transform the result as a non-reduction
        if (!red)
          return OptimizeAndLiftNonzeronessConditionsImpl(new_red, axis);
      }

      Expr new_outer_cond, new_reduce_cond;
      Array<Expr> new_source = red->source;

      // Since the reduction domain might have changed, add information about reduction
      // axes once again.
      // TODO: This might be unnecessary, because the information may be preserved in the cond,
      //       but I'm not sure
      Array<Expr> red_axis_conds = IterVarsToInequalities(red->axis);

      // Partially lift conditions from the reduce condition
      std::tie(new_outer_cond, new_reduce_cond) =
        LiftConditionsThroughReduction(red->condition && All(red_axis_conds), red->axis, axis);

      // If it's not sum then we haven't yet lifted nonzeroness cond from the source
      if (!is_sum) {
        Expr outer_nz_cond, nz_cond, nz_source;
        std::tie(nz_cond, nz_source) =
          NonzeronessCondition::Nonzeroness(red->source[red->value_index]);
        // Append conditions from the reduction (including conditions on parameters)
        nz_cond = red->condition && nz_cond;
        std::tie(outer_nz_cond, nz_cond) =
          LiftConditionsThroughReduction(nz_cond, red->axis, axis);
        new_outer_cond = new_outer_cond && outer_nz_cond;
        new_source.Set(red->value_index, IfThenElseZero(nz_cond, nz_source));
      }

      Expr new_reduce = Reduce::make(red->combiner, new_source, red->axis,
                                     red->condition, red->value_index);
      result = IfThenElseZero(new_outer_cond, new_reduce);
    }
    else
      return SimplifyReductionDomain(expr, axis);
  }
  else {
    Expr cond, new_expr;
    std::tie(cond, new_expr) = NonzeronessCondition::Nonzeroness(expr);
    result = IfThenElseZero(cond, new_expr);
  }

  result = RemoveRedundantInequalities(result, axis_conds);

  return SplitIntoTensorsSmartly(result, axis);
}

Tensor OptimizeAndLiftNonzeronessConditions(const Tensor& tensor) {
  return op::TransformBody(tensor, OptimizeAndLiftNonzeronessConditionsImpl);
}

TVM_REGISTER_API("arith.OptimizeAndLiftNonzeronessConditions")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = OptimizeAndLiftNonzeronessConditions(args[0]);
  });

}  // namespace ir
}  // namespace tvm
