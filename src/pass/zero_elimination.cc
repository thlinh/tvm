/*!
 *  Copyright (c) 2018 by Contributors
 * \file zero_elimination.cc
 * \brief Transform tensors in such a way as to eliminate summation over zeros.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include "./zero_elimination.h"
#include "../op/op_util.h"
#include <tvm/api_registry.h>
#include <tvm/ir_functor_ext.h>
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
template <class K, class V>
Map<K, V> Merge(Map<K, V> original, Map<K, V> update) {
  for (const auto& p : update)
    original.Set(p.first, p.second);
  return std::move(original);
}

// TODO: Move somewhere
template <class T>
Array<T> Concat(Array<T> a, const Array<T>& b) {
  for (const auto& x : b)
    a.push_back(x);
  return std::move(a);
}

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


bool CanProve(Expr e, const Map<Var, Range>& vranges = Map<Var, Range>()) {
  // the Simplifier is quite strange, e.g. it won't simplify `(ax1 - ax3) - ax1 <= 0`,
  // although `(ax1 - ax3) - ax1 >= 0` is simplified perfectly (the difference is >=)
  // As a workaround, we first canonical simplify, then simplify
  return is_one(Simplify(CanonicalSimplify(e, vranges), vranges));
}

// Simplify Really Really well, please
Expr SuperSimplify(Expr e, const Map<Var, Range>& vranges = Map<Var, Range>()) {
  return CanonicalSimplify(Simplify(CanonicalSimplify(e, vranges), vranges), vranges);
}


// TODO: Move somewhere
// Collect all tensors used by the given tensor
class IRCollectSubtensors : public IRVisitor {
  public:
    void Visit_(const Call* op) {
      if (op->call_type == Call::CallType::Halide)
        if (op->func->derived_from<OperationNode>()) {
          subtensors.insert(Downcast<Operation>(op->func).output(op->value_index));
        }
      for (auto& e : op->args)
        Visit(e);
    }

    std::unordered_set<Tensor> subtensors;
};

std::unordered_set<Tensor> Subtensors(const Expr& expr) {
    IRCollectSubtensors subtensors;
    subtensors.Visit(expr);
    return std::move(subtensors.subtensors);
}

std::string PrintTensorName(const Tensor& tensor) {
  if (!tensor.get())
    return "NULL_TENSOR";

  std::ostringstream oss;
  oss << tensor->op->name << "{" << tensor->op.get() << "}" << "[" << tensor->value_index << "]";
  return oss.str();
}

std::string PrintIterVars(const Array<IterVar>& itervars) {
  std::ostringstream oss;
  oss << "(";
  bool first = true;
  for (const IterVar& iv : itervars) {
    if (!first) oss << ", ";
    first = false;
    oss << iv->var << " : " << "[" << iv->dom->min
        << ", " << (iv->dom->min + iv->dom->extent - 1) << "]";
  }
  oss << ")";
  return oss.str();
}

std::string PrintTensorRecursively(const Tensor& tensor) {
  if (!tensor.get())
    return "NULL_TENSOR\n";

  std::vector<Tensor> unprocessed({tensor});
  std::unordered_set<Tensor> processed;
  std::ostringstream oss;

  while (!unprocessed.empty()) {
    Tensor cur = unprocessed.back();
    unprocessed.pop_back();
    processed.insert(cur);

    oss << "tensor " << PrintTensorName(cur) << " : " << cur->dtype << " " << cur->shape << "\n";
    if (const ComputeOpNode* comp = cur->op.as<ComputeOpNode>()) {
      oss << "axes " << PrintIterVars(comp->axis) << "\n";
      Expr body = comp->body[cur->value_index];

      for (const Tensor& t : Subtensors(body))
        if (processed.count(t) == 0)
          unprocessed.push_back(t);

      if (const Reduce* red = body.as<Reduce>()) {
        oss << "Reduction\n";
        oss << "    identity " << red->combiner->identity_element << "\n";
        oss << "    lhs " << red->combiner->lhs << "  rhs " << red->combiner->rhs << "\n";
        oss << "    combiner " << red->combiner->result << "\n";
        oss << "    axes " << PrintIterVars(red->axis) << "\n";
        oss << "    condition " << red->condition << "\n";
        for (size_t i = 0; i < red->source.size(); ++i)
          oss << "    source[" << i << "] = " << red->source[i] << "\n";
      } else
        oss << "    " << body << "\n";
    }
    else
      oss << "    " << cur->op << "\n";
    oss << "\n";
  }

  return oss.str();
}

TVM_REGISTER_API("PrintTensorRecursively")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = PrintTensorRecursively(args[0]);
  });


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
    simplified_result.push_back(SuperSimplify(res));

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
      new_identity.push_back(SuperSimplify(op->combiner->identity_element[i]));
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
      IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""),
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
  in = SuperSimplify(in);

  return Equal(zero, in);
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


// TODO: Move somewere
// Convert an array of itervars to a map from vars to ranges
Map<Var, Range> IterVarsToMap(const Array<IterVar>& itervars) {
  Map<Var, Range> res;
  for (const IterVar& v : itervars)
    res.Set(v->var, v->dom);
  return res;
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

TVM_REGISTER_API("ir_pass.InlineNonReductions")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = InlineNonReductions(args[0], args.size() > 1 ? args[1] : Array<Tensor>());
  });


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
        Expr new_cond = SuperSimplify(Or::make(pair_a.first, pair_b.first));
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

      Expr new_cond = SuperSimplify(pair_a.first && pair_b.first);

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
    Expr new_cond = SuperSimplify(pair_a.first && op->condition);
    return std::make_pair(new_cond, pair_a.second);
  }

  if (is_const_scalar(pair_a.second, 0)) {
    Expr new_cond = SuperSimplify(pair_b.first && !op->condition);
    return std::make_pair(new_cond, pair_b.second);
  }

  Expr new_cond =
    SuperSimplify(Or::make(op->condition && pair_a.first,
                           !op->condition &&  pair_b.first));
  if (pair_a.second.same_as(op->true_value) && pair_b.second.same_as(op->false_value))
    return std::make_pair(new_cond, e);
  else
    return std::make_pair(new_cond, Select::make(op->condition, pair_a.second, pair_b.second));
});

Expr LiftNonzeronessCondition(const Expr& expr) {
  return NonzeronessCondition::PairToExpr(NonzeronessCondition::Nonzeroness(expr));
}

TVM_REGISTER_API("ir_pass.LiftNonzeronessCondition")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = LiftNonzeronessCondition(args[0]);
  });


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
        return LE::make(SuperSimplify(a - b + 1), make_zero(a.type()));
      return TNode::make(SuperSimplify(a - b), make_zero(a.type()));
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


// Compute variable ranges from variable definitions and ranges
Map<Var, Range> EvalVarRanges(const Map<Var, Expr>& var_values, Map<Var, Range> var_ranges) {
  std::unordered_map<const Variable*, IntSet> var_intsets;
  for (const auto& p : var_ranges)
    var_intsets[p.first.get()] = IntSet::range(p.second);

  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& p : var_values) {
      IntSet intset = EvalSet(p.second, var_intsets);
      Range range = intset.cover_range(Range());
      if (range.get()) {
        var_intsets[p.first.get()] = IntSet::range(range);
        if (var_ranges.count(p.first)) {
          Range old_range = var_ranges[p.first];
          if (!old_range.same_as(range))
            if (!(Equal(range->min, old_range->min) && Equal(range->extent, old_range->extent)))
              changed = true;
        }
        else {
          changed = true;
        }
        var_ranges.Set(p.first, range);
      }
    }
  }

  return var_ranges;
}


struct EliminateDivModResult {
  Expr expr;
  Map<Var, Expr> substitution;
  Array<Var> new_variables;
  Array<Expr> conditions;
};

class EliminateDivModMutator : public IRMutator {
  public:
    Map<Var, Expr> substitution;
    Array<Var> new_variables;
    Array<Expr> conditions;

    EliminateDivModMutator() {}

    virtual Expr Mutate_(const Div* op, const Expr& e) {
      const IntImm* imm = op->b.as<IntImm>();
      if (imm && imm->value > 0) {
        auto it = expr_to_vars_.find(std::make_pair(op->a.get(), imm->value));
        if (it != expr_to_vars_.end())
          return it->second.first;

        Expr mutated_a = Mutate(op->a);
        return AddNewVarPair(op->a, mutated_a, imm->value).first;
      }

      return Div::make(Mutate(op->a), Mutate(op->b));
    }

    virtual Expr Mutate_(const Mod* op, const Expr& e) {
      const IntImm* imm = op->b.as<IntImm>();
      if (imm && imm->value > 0) {
        auto it = expr_to_vars_.find(std::make_pair(op->a.get(), imm->value));
        if (it != expr_to_vars_.end())
          return it->second.second;

        Expr mutated_a = Mutate(op->a);
        return AddNewVarPair(op->a, mutated_a, imm->value).second;
      }

      return Mod::make(Mutate(op->a), Mutate(op->b));
    }

  private:
    std::pair<Var, Var> AddNewVarPair(const Expr& e, const Expr& mut, int64_t val) {
      Expr val_e = make_const(e.type(), val);
      idx_ += 1;
      auto div = Var("div" + std::to_string(idx_), e.type());
      auto mod = Var("mod" + std::to_string(idx_), e.type());
      substitution.Set(div, mut / val_e);
      substitution.Set(mod, mut % val_e);
      new_variables.push_back(div);
      new_variables.push_back(mod);
      conditions.push_back(mut == div*val_e + mod);
      conditions.push_back(mod <= val_e - 1);
      // TODO: controversial, depends on % semantics
      conditions.push_back(mod >= 0);//1 - val_e);
      //conditions.push_back(mod == mut % val_e);
      //conditions.push_back(div == mut / val_e);
      auto p = std::make_pair(div, mod);
      expr_to_vars_[std::make_pair(e.get(), val)] = p;
      return p;
    }

    int idx_{0};
    std::map<std::pair<const HalideIR::Internal::IRNode*, int64_t>, std::pair<Var, Var>>
      expr_to_vars_;
};

// replace every subexpr of the form e/const and e % const with a new variable
EliminateDivModResult EliminateDivMod(const Expr& expr) {
  EliminateDivModResult res;
  EliminateDivModMutator mutator;
  res.expr = mutator.Mutate(expr);
  res.conditions = std::move(mutator.conditions);
  res.new_variables = std::move(mutator.new_variables);
  res.substitution = std::move(mutator.substitution);
  return res;
}

// run EliminateDivMod from the condition of a reduction
Expr EliminateDivModFromReductionCondition(const Expr& expr,
                                           Map<Var, Range> ranges = Map<Var, Range>()) {
  // TODO: Maybe also replace subexprs with variables inside the source
  if (const Reduce* red = expr.as<Reduce>()) {
    auto elim_res = EliminateDivMod(red->condition);

    for (const IterVar& iv : red->axis)
      ranges.Set(iv->var, iv->dom);

    ranges = EvalVarRanges(elim_res.substitution, ranges);

    Array<IterVar> new_axis = red->axis;
    for (const Var& v : elim_res.new_variables) {
      auto range = ranges[v];
      new_axis.push_back(IterVarNode::make(range, v, IterVarType::kCommReduce));
    }

    Expr new_cond = elim_res.expr && All(elim_res.conditions);

    return Reduce::make(red->combiner, red->source, new_axis, new_cond, red->value_index);
  }
  else
    return expr;
}


struct VarBounds {
  Expr coef;
  Array<Expr> lower;
  Array<Expr> equal;
  Array<Expr> upper;

  Array<Expr> get_var_upper_bounds() const {
    Array<Expr> res;
    for (Expr e : equal)
      res.push_back(SuperSimplify(e/coef));
    for (Expr e : upper)
      res.push_back(SuperSimplify(e/coef));
    return res;
  }

  Array<Expr> get_var_lower_bounds() const {
    Array<Expr> res;
    for (Expr e : equal)
      res.push_back(SuperSimplify(e/coef));
    for (Expr e : lower)
      res.push_back(SuperSimplify(e/coef));
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
// Note that variable ranges help a lot, so this parameter is even non-optional
// TODO: Probably add all vranges as additional inequalities to make the resulting set of
// inequalities self-sufficient
SolveSystemOfInequalitiesResult SolveSystemOfInequalities(const Array<Expr>& inequalities,
                                                          const Array<Var>& variables,
                                                          const Map<Var, Range>& vranges) {
  SolveSystemOfInequalitiesResult res;
  res.variables = variables;

  std::set<Expr, ExprLess> current;
  std::set<Expr, ExprLess> new_current;
  std::vector<std::pair<int64_t, Expr>> coef_pos;
  std::vector<std::pair<int64_t, Expr>> coef_neg;

  // std::cout << "\nSolveSystemOfInequalities\n";
  // std::cout << "  ineqs: " << inequalities << "\n  vars: " << variables << "\n";
  // std::cout << "  ranges: " << vranges << "\n";

  // formulas we don't know what to do with
  std::vector<Expr> rest;

  // A helper that adds an inequality to new_current if it's not obviously redundant
  auto add_to_new_current = [&new_current, &vranges] (const Expr& new_ineq) {
    if (CanProve(new_ineq, vranges))
        return;
    if (const LE* new_le = new_ineq.as<LE>()) {
        // A heuristic: check if the new inequality is a consequence of one
        // of its future neighbors (in this case don't add it) or if a future neighbor is
        // a consequence of the new ineq (in which case remove the neighbor)
        auto it_neighbor = new_current.lower_bound(new_ineq);
        if (it_neighbor != new_current.begin()) {
          const LE* le = std::prev(it_neighbor)->as<LE>();
          if (le && CanProve(new_le->a - le->a <= 0, vranges))
            return;
          else if(le && CanProve(le->a - new_le->a <= 0, vranges))
            new_current.erase(std::prev(it_neighbor));
          // else if(le)
          //   std::cout << "Incomparable " << le->a << "  " << new_le->a << std::endl;
        }
        // Check the other neighbor
        if (it_neighbor != new_current.end()) {
          const LE* le = it_neighbor->as<LE>();
          if (le && CanProve(new_le->a - le->a <= 0, vranges))
            return;
          else if(le && CanProve(le->a - new_le->a <= 0, vranges))
            it_neighbor = new_current.erase(it_neighbor);
          // else if(le)
          //   std::cout << "Incomparable " << le->a << "  " << new_le->a << std::endl;
        }

        new_current.insert(it_neighbor, new_ineq);
    }
    else
      new_current.insert(new_ineq);
  };

  for (const Expr& ineq : inequalities)
    add_to_new_current(NormalizeComparisons(SuperSimplify(ineq)));

  std::swap(current, new_current);

  for (const Var& v : variables) {
    CHECK(!res.bounds.count(v.get())) <<
      "Variable " << v << " appears several times in the `variables` which might be a bug";

    new_current.clear();
    coef_pos.clear();
    coef_neg.clear();

    // Add bounds from vranges
    if (vranges.count(v)) {
      const Range& range = vranges[v];
      Expr range_lbound = SuperSimplify(range->min);
      Expr range_ubound = SuperSimplify(range->min + range->extent - 1);
      coef_neg.push_back(std::make_pair(1, range_lbound));
      coef_pos.push_back(std::make_pair(1, -range_ubound));
    }

    // std::cout << "\nVariable " << v << std::endl;
    // for (auto p : current) std::cout << "current " << p << std::endl;
    // std::cout << std::endl;

    for (const Expr& ineq : current) {
      if (const LE* le = ineq.as<LE>()) {
        Array<Expr> coef = arith::DetectLinearEquation(le->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0)
            add_to_new_current(ineq);
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
            add_to_new_current(ineq);
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
        Expr new_lhs = c_neg*neg.second - c_pos*pos.second;
        Expr new_ineq = LE::make(new_lhs, make_zero(pos.second.type()));
        new_ineq = NormalizeComparisons(SuperSimplify(new_ineq));
        add_to_new_current(new_ineq);
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
      bound = SuperSimplify(bound);
      // Don't add if any of the existing bounds is better
      if (std::any_of(upper_bounds.begin(), upper_bounds.end(),
                      [&bound, &vranges](const Expr& o) { return CanProve(o - bound <= 0,
                                                                          vranges); }))
        continue;
      // Erase all worse bounds
      upper_bounds.erase(
        std::remove_if(upper_bounds.begin(), upper_bounds.end(),
                       [&bound, &vranges](const Expr& o) { return CanProve(o - bound >= 0,
                                                                           vranges); }),
        upper_bounds.end());
      // Add
      upper_bounds.push_back(bound);
    }
    for (const auto& neg : coef_neg) {
      Expr bound = make_const(v.type(), -coef_lcm/neg.first)*neg.second;
      bound = SuperSimplify(bound);
      // Don't add if any of the existing bounds is better
      if (std::any_of(lower_bounds.begin(), lower_bounds.end(),
                      [&bound, &vranges](const Expr& o) { return CanProve(o - bound >= 0,
                                                                          vranges); }))
        continue;
      // Erase all worse bounds
      lower_bounds.erase(
        std::remove_if(lower_bounds.begin(), lower_bounds.end(),
                       [&bound, &vranges](const Expr& o) { return CanProve(o - bound <= 0,
                                                                           vranges); }),
        lower_bounds.end());
      // Add
      lower_bounds.push_back(bound);
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
    Expr e_simp = SuperSimplify(e);
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

  // std::cout << "  res: " << res.as_conditions() << "\n" << std::endl;

  return res;
}

TVM_REGISTER_API("arith.SolveSystemOfInequalities")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = SolveSystemOfInequalities(args[0], args[1], args[2]).as_conditions();
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
  Expr rest_of_cond =
    cond && All(IterVarsToInequalities(axis)) && All(IterVarsToInequalities(outer_axis));
  std::vector<Expr> atomic_formulas;
  std::tie(atomic_formulas, rest_of_cond) = FactorOutAtomicFormulas::Factor(rest_of_cond);

  Array<Var> vars;
  for (const IterVar& v : axis)
    vars.push_back(v->var);
  for (const IterVar& v : outer_axis)
    vars.push_back(v->var);

  auto vranges = Merge(IterVarsToMap(axis), IterVarsToMap(outer_axis));
  auto solved_system = SolveSystemOfInequalities(atomic_formulas, vars, vranges);

  DomainSimplificationResult res;
  std::unordered_map<const Variable*, IntSet> new_var_intsets;

  for (const IterVar& v : outer_axis)
    new_var_intsets[v->var.get()] = IntSet::range(v->dom);

  for (auto it = axis.rbegin(); it != axis.rend(); ++it) {
    const IterVar& iv = *it;
    auto& bnd = solved_system.bounds[iv->var.get()];
    // Note that we replace old vars with new ones
    bnd = bnd.substitute(res.old_to_new);
    // std::cout << "\nvar " << bnd.coef << " * " << iv << "\n";
    // std::cout << "equal " << bnd.equal << "\n";
    if (is_one(bnd.coef) && !bnd.equal.empty()) {
      res.old_to_new[iv->var.get()] = bnd.equal[0];

      // std::cout << "\nvar " << iv << " replaced with " << bnd.equal[0] << "\n";
    }
    else {
      Array<Expr> lowers = Concat(bnd.equal, bnd.lower);
      Array<Expr> uppers = Concat(bnd.equal, bnd.upper);

      // std::cout << "lowers " << lowers << "\n";
      // std::cout << "uppers " << uppers << "\n";

      Expr best_lower = iv->dom->min * bnd.coef;
      Expr best_diff = (iv->dom->extent - 1) * bnd.coef;
      Expr best_diff_upper = best_diff;

      for (const Expr& low : lowers) {
        for (const Expr& upp : uppers) {
          Expr diff = SuperSimplify(upp - low);
          Expr diff_upper = EvalSet(diff, new_var_intsets).max();

          // std::cout << "checking diff " << diff << " its upper " << diff_upper << std::endl;

          if (diff_upper.same_as(HalideIR::Internal::Interval::pos_inf))
            continue;

          if (can_prove(diff_upper - best_diff_upper < 0)) {
            best_lower = low;
            best_diff = diff;
            best_diff_upper = diff_upper;
          }
        }
      }

      // std::cout << "\nvar " << iv << " has best lower " << best_lower << "     and diff    " << best_diff << "\n";

      if (is_const_int(best_diff, 0)) {
        // In this case coef*iv = best_lower
        // Don't create an itervar, just replace it everywhere with its min
        res.old_to_new[iv->var.get()] = SuperSimplify(best_lower / bnd.coef);
        // To assure correctness, we have to add a condition that best_lower can be divided by coef
        res.conditions.push_back(SuperSimplify(best_lower % bnd.coef == 0));
        // std::cout << "var " << iv << " conditionally replaced with " << res.old_to_new[iv->var.get()] << "\n";
      }
      else {
        std::string suffix = Equal(best_lower, iv->dom->min * bnd.coef) ? "" : ".shifted";
        Var new_iv = iv->var.copy_with_suffix(suffix);
        // We use rounding-up division here
        Expr shift = SuperSimplify(Select::make(best_lower <= 0,
                                                best_lower / bnd.coef,
                                                (best_lower + bnd.coef - 1)/bnd.coef));
        Expr diff = SuperSimplify(best_diff_upper / bnd.coef);

        if (is_const_int(diff, 0)) {
          // Don't create an itervar, just replace it everywhere with its min
          res.old_to_new[iv->var.get()] = shift;
          // std::cout << "var " << iv << " replaced with " << res.old_to_new[iv->var.get()] << "\n";
        }
        else {
          res.old_to_new[iv->var.get()] = new_iv + shift;
          // Note that we are substituting old with new, so best_lower contains new var,
          // that is we have to substitute new with old in best_lower here
          res.new_to_old[new_iv.get()] = SuperSimplify(iv->var - Substitute(shift, res.new_to_old));

          // std::cout << "var " << iv << " replaced with " << res.old_to_new[iv->var.get()] << "\n";
          // std::cout << "back " << new_iv << " -> " << res.new_to_old[new_iv.get()] << "\n";

          new_var_intsets[new_iv.get()] = IntSet::interval(make_zero(new_iv.type()), diff);

          // std::cout << "its ubound " << diff << "\n";

          auto range = Range(make_zero(new_iv.type()), diff + 1);
          res.axis.push_back(IterVarNode::make(range, new_iv, iv->iter_type, iv->thread_tag));

          // std::cout << "new range " << range << "\n";
        }
      }
    }
  }

  for (const Expr& old_cond : solved_system.as_conditions())
    res.conditions.push_back(SuperSimplify(Substitute(old_cond, res.old_to_new)));

  return res;
}

// Use the condition of a reduction op to simplify its domain (axis)
Expr SimplifyReductionDomain(const Expr& expr, const Array<IterVar>& outer_axis) {
  if (const Reduce* red = expr.as<Reduce>()) {
    Map<Var, Range> vranges = Merge(IterVarsToMap(outer_axis), IterVarsToMap(red->axis));
    Expr expr_with_divmod_eliminated = EliminateDivModFromReductionCondition(expr, vranges);
    // std::cout << "\nred before simplify dom\n" << expr << "\n";
    // std::cout << "\nred after eliminate div mod\n" << expr_with_divmod_eliminated << "\n";
    red = expr_with_divmod_eliminated.as<Reduce>();
    auto res = SimplifyDomain(red->condition, red->axis, outer_axis);

    Array<Expr> new_source;
    for (const Expr& src : red->source)
      new_source.push_back(Substitute(src, res.old_to_new));

    // std::cout << "\nred before simplify dom\n" << expr << "\n";
    // std::cout << "\nred after eliminate div mod\n" << expr_with_divmod_eliminated << "\n";
    // std::cout << "\nred after simplify dom\n" << Reduce::make(red->combiner, new_source, res.axis, All(res.conditions), red->value_index) << "\n\n";

    return RemoveEmptyReduction(Reduce::make(red->combiner, new_source, res.axis,
                                             All(res.conditions), red->value_index));
  }
  else
    return expr;
}

// Extract the given expr under the given condition as a separate tensor if the volume of the
// extracted tensor will be less than the volume of the outer_axis
Expr ExtractAsTensorMaybe(const Expr& e, const Expr& cond, const Array<IterVar>& outer_axis) {
  auto res = SimplifyDomain(cond, outer_axis, {});

  Expr new_expr = SuperSimplify(Substitute(e, res.old_to_new), IterVarsToMap(outer_axis));

  // Keep only those variables of the new axis which are used in the new_expr
  {
    Array<IterVar> used_res_axis;
    for (const IterVar& iv : res.axis)
      if (ExprUseVar(new_expr, iv->var))
        used_res_axis.push_back(iv);

    res.axis = std::move(used_res_axis);
  }

  // If the expression does not use vars then it is probably better to keep it inlined
  if (res.axis.empty())
    return new_expr;

  // Compute volumes before and after
  Expr old_volume = make_const(Int(64), 1);
  for (const IterVar& iv : outer_axis)
    old_volume = old_volume * iv->dom->extent;

  Expr new_volume = make_const(Int(64), 1);
  for (const IterVar& iv : res.axis)
    new_volume = new_volume * iv->dom->extent;

  // if we can prove that the old volume is not greater than the new volume then
  // prefer the old expression.
  if (can_prove(old_volume <= new_volume))
    return e;

  Tensor tensor = op::TensorFromExpr(new_expr, res.axis);

  Array<Expr> args;
  for (const IterVar& iv : res.axis)
    args.push_back(res.new_to_old[iv->var.get()]);

  return Call::make(e.type(), tensor->op->name, args,
                    Call::CallType::Halide, tensor->op, tensor->value_index);
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


class RemoveRedundantInequalitiesMutator : public IRMutator {
  public:
    RemoveRedundantInequalitiesMutator(Array<Expr> known) {
      for (const Expr& cond : known)
        known_.push_back(SuperSimplify(cond));
    }

    virtual Expr Mutate_(const Select* op, const Expr& e) {
      Expr new_cond = SuperSimplify(Mutate(op->condition));
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
      Expr simplified = SuperSimplify(e);
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

  auto vranges = Merge(IterVarsToMap(red_axis), IterVarsToMap(outer_axis));
  // start from reduction vars, so that input vars don't depend on them
  atomics = SolveSystemOfInequalities(atomics, allvars, vranges).as_conditions();

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
    explicit SplitIntoTensorsSmartlyMutator(Array<IterVar> axis, std::string name="extracted")
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
      new_reduce = SuperSimplify(Substitute(new_reduce, newaxis_vmap_pair.second),
                                 IterVarsToMap(new_axis));

      Tensor tensor = op::TensorFromExpr(new_reduce, new_axis, name_, tag_, attrs_);

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

  // std::cout << "\n========== OptimizeAndLiftNonzeronessConditionsImpl ========\n";
  // std::cout << expr << "\n" << std::endl;

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
      new_reduce = ExtractAsTensorMaybe(new_reduce, new_outer_cond, axis);
      result = IfThenElseZero(new_outer_cond, new_reduce);
    }
    else
      return SimplifyReductionDomain(expr, axis);
  }
  else {
    Expr cond, new_expr;
    std::tie(cond, new_expr) = NonzeronessCondition::Nonzeroness(expr);
    new_expr = ExtractAsTensorMaybe(new_expr, cond, axis);
    result = IfThenElseZero(cond, new_expr);
  }

  result = RemoveRedundantInequalities(result, axis_conds);

  // Sometimes ExtractAsTensorMaybe doesn't perform extraction, so there may be some non-top
  // reductions left, take care of them
  return SuperSimplify(SplitIntoTensorsSmartly(result, axis), IterVarsToMap(axis));
}

Tensor OptimizeAndLiftNonzeronessConditions(const Tensor& tensor) {
  return op::TransformBody(tensor, OptimizeAndLiftNonzeronessConditionsImpl);
}

TVM_REGISTER_API("ir_pass.OptimizeAndLiftNonzeronessConditions")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = OptimizeAndLiftNonzeronessConditions(args[0]);
  });

}  // namespace ir
}  // namespace tvm
