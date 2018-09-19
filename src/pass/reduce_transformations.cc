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

            std::cout << "Inlining:\n" << e << "\nbody: " << op_comp->body[0] << "\nnew_e: " << new_e << "\ne_type: " << e.type() << "\nbody_type: " << op_comp->body[0].type() << "\nnewe_type: " << new_e.type() << std::endl;

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
      std::cout << "Mul:\n" << e << "\na: " << op->a << "\nb: " << op->b << "\nma: " << ma << "\nmb: " << mb << std::endl;

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

}  // namespace ir
}  // namespace tvm
