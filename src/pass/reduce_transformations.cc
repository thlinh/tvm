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

namespace tvm {
namespace ir {

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

// clone reduction by cloning the axis variables
Expr CloneReduction(const Expr& expr) {
  if (const Reduce* red = expr.as<Reduce>()) {
    Array<IterVar> new_axis;
    std::unordered_map<const Variable*, Expr> vmap;
    for (IterVar iv : red->axis) {
      IterVar new_v =
        IterVarNode::make(iv->dom, iv->var.copy_with_suffix(".copy"),
            iv->iter_type, iv->thread_tag);
      new_axis.push_back(new_v);
      vmap[iv->var.operator->()] = new_v;
    }

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

      Expr new_cond = CanonicalSimplify(Simplify(And::make(pair_a.first, pair_b.first)));

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

  if (is_zero(pair_b.second)) {
    Expr new_cond = CanonicalSimplify(Simplify(And::make(pair_a.first, op->condition)));
    return std::make_pair(new_cond, pair_a.second);
  }

  if (is_zero(pair_a.second)) {
    Expr new_cond = CanonicalSimplify(Simplify(And::make(pair_b.first, Not::make(op->condition))));
    return std::make_pair(new_cond, pair_b.second);
  }

  Expr new_cond =
    CanonicalSimplify(Simplify(Or::make(And::make(op->condition, pair_a.first),
                                        And::make(Not::make(op->condition), pair_b.first))));
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
    virtual Expr Mutate_(const NE* op, const Expr& e) { return Make<NE>(op->a, op->b); };
    virtual Expr Mutate_(const LT* op, const Expr& e) { return Make<LT>(op->a, op->b); };
    virtual Expr Mutate_(const LE* op, const Expr& e) { return Make<LE>(op->a, op->b); };
    virtual Expr Mutate_(const GT* op, const Expr& e) { return Make<LT>(op->b, op->a); };
    virtual Expr Mutate_(const GE* op, const Expr& e) { return Make<LE>(op->b, op->a); };

  private:
    template <class TNode>
    Expr Make(const Expr& a, const Expr& b) {
      return TNode::make(CanonicalSimplify(Simplify(Sub::make(a, b))), make_zero(a.type()));
    }
};

// Rewrite every comparison into the form a == 0, a != 0, a <= 0, a < 0
Expr NormalizeComparisons(const Expr& expr) {
  return NormalizeComparisonsMutator().Mutate(expr);
}



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
                 [](const Expr& l, const Expr& r) { return Compare(l, r) < 0; });

  return std::make_pair(res, And::make(pair_a.second, pair_b.second));
})
.set_dispatch<Or>([](const Or* op, const Expr& e) {
  auto pair_a = FactorOutAtomicFormulas::Factor(op->a);
  auto pair_b = FactorOutAtomicFormulas::Factor(op->b);

  std::vector<Expr> res;
  res.reserve(std::min(pair_a.first.size(), pair_b.first.size()));
  std::set_intersection(pair_a.first.begin(), pair_a.first.end(),
                        pair_b.first.begin(), pair_b.first.end(),
                        std::back_inserter(res),
                        [](const Expr& l, const Expr& r) { return Compare(l, r) < 0; });

  std::vector<Expr> new_cond_a;
  new_cond_a.reserve(pair_a.first.size() - res.size());
  std::set_difference(pair_a.first.begin(), pair_a.first.end(),
                      res.begin(), res.end(),
                      std::back_inserter(new_cond_a),
                      [](const Expr& l, const Expr& r) { return Compare(l, r) < 0; });

  std::vector<Expr> new_cond_b;
  new_cond_b.reserve(pair_b.first.size() - res.size());
  std::set_difference(pair_b.first.begin(), pair_b.first.end(),
                      res.begin(), res.end(),
                      std::back_inserter(new_cond_b),
                      [](const Expr& l, const Expr& r) { return Compare(l, r) < 0; });

  pair_a.first = std::move(new_cond_a);
  pair_b.first = std::move(new_cond_b);

  return std::make_pair(res, Or::make(FactorOutAtomicFormulas::PairToExpr(pair_a),
                                      FactorOutAtomicFormulas::PairToExpr(pair_b)));
});


// returns a vector of coefficients [a, b, c] if expr is a*x + b*y + c
dmlc::optional<std::vector<Expr>> AsLinear(const Expr& expr, const std::vector<Expr>& vars) {
  for (size_t i = 0; i < vars.size(); ++i)
    if (expr.same_as(vars[i])) {
      std::vector<Expr> res(vars.size() + 1);
      
    }

  if (const Add* add = expr.as<Add>()) {
    auto coef_a = AsLinear(add->a, vars);
    auto coef_b = AsLinear(add->b, vars);
    if (coef_a.has_value() && coef_b.has_value()) {
      CHECK_EQ((*coef_a).size(), (*coef_b).size());
      for (size_t i = 0; i < coef_a.value().size(); ++i)
        (*coef_a)[i] = Add::make((*coef_a)[i], (*coef_b)[i]);
      return coef_a;
    }
    else
      return dmlc::optional<std::vector<Expr>>();
  }
  else if (const Mul* mul = expr.as<Mul>()) {
    
  }
}


Expr SimplifySumDomain(const Expr& expr) {
  const Reduce* red = expr.as<Reduce>();
  if (red && IsSumCombiner(red->combiner)) {
    Expr cond, source;
    std::tie(cond, source) = NonzeronessCondition::Nonzeroness(red->source[0]);

    cond = NormalizeComparisons(Simplify(And::make(red->condition, cond)));
    std::vector<Expr> atomic_formulas;
    std::tie(atomic_formulas, cond) = FactorOutAtomicFormulas::Factor(cond);

    std::unordered_set<IterVar> vars_left(red->axis.begin(), red->axis.end());

    for (Expr& formula : atomic_formulas) {
      if (const EQ* eq = formula.as<EQ>()) {
        for (const IterVar& var : vars_left)
      }
    }
  }
  else
    return expr;
}

}  // namespace ir
}  // namespace tvm
