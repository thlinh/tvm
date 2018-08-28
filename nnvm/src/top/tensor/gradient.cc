/*!
 *  Copyright (c) 2018 by Contributors
 * \file gradient.cc
 * \brief Automatic gradients for any operation
 */
#include <topi/nn.h>
#include <topi/broadcast.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include <tvm/operation.h>
#include <tvm/ir_pass.h>
#include <tvm/ir.h>

namespace nnvm {
namespace top {


/*!
 * \brief A map from string to string but with an operator>> defined.
 */
class StrDict {
 public:
  std::unordered_map<std::string, std::string> dict;
  /*!
   * \brief Save StrDict to JSON.
   * \param writer JSONWriter
   */
  inline void Save(dmlc::JSONWriter* writer) const {
    writer->Write(dict);
  }
  /*!
   * \brief Load StrDict from JSON.
   * \param reader JSONReader
   */
  inline void Load(dmlc::JSONReader* reader) {
    reader->Read(&dict);
  }
  /*!
   * \brief allow output string of tuple to ostream
   * \param os the output stream
   * \param t the tuple
   * \return the ostream
   */
  friend std::ostream &operator<<(std::ostream &os, const StrDict &sd) {
    dmlc::JSONWriter(&os).Write(sd);
    return os;
  }
  /*!
   * \brief read tuple from the istream
   * \param is the input stream
   * \param t The tuple
   * \return the istream
   */
  friend std::istream &operator>>(std::istream &is, StrDict &sd) {
    dmlc::JSONReader(&is).Read(&sd);
    return is;
  }
};



using namespace nnvm::compiler;

struct GradientParam : public dmlc::Parameter<GradientParam> {
  std::string original_op;
  std::string original_name;
  StrDict original_attrs;

  DMLC_DECLARE_PARAMETER(GradientParam) {
    DMLC_DECLARE_FIELD(original_op);
    DMLC_DECLARE_FIELD(original_name);
    DMLC_DECLARE_FIELD(original_attrs);
  }

  NodeAttrs original() const {
    NodeAttrs res;
    res.op = Op::Get(original_op);
    for (const auto& key_val_pair : original_attrs.dict)
      res.dict[key_val_pair.first] = key_val_pair.second;
    res.name = original_name;
    if (res.op->attr_parser) {
      res.op->attr_parser(&res);
    }
    return res;
  }
};

DMLC_REGISTER_PARAMETER(GradientParam);

inline bool GradientShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  static auto& finfershape = nnvm::Op::GetAttr<FInferShape>("FInferShape");

  NodeAttrs o_attrs = nnvm::get<GradientParam>(attrs.parsed).original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

  uint32_t o_num_outputs = o_attrs.op->num_outputs;
  if (o_attrs.op->get_num_outputs)
    o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

  CHECK_EQ(in_attrs->size(), o_num_inputs + o_num_outputs);
  CHECK_EQ(out_attrs->size(), o_num_inputs);

  for (size_t i = 0; i < o_num_inputs; ++i) {
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, i, (*in_attrs)[i]);
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_attrs, i, (*out_attrs)[i]);
  }

  if (finfershape.count(o_attrs.op)) {
    std::vector<TShape> o_out_attrs(in_attrs->begin() + o_num_inputs, in_attrs->end());
    finfershape[o_attrs.op](o_attrs, out_attrs, &o_out_attrs);
    for (size_t i = 0; i < o_num_outputs; ++i) {
      NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_attrs, o_num_inputs + i, o_out_attrs[i]);
    }
  }

  return true;
}

inline bool GradientType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  static auto& finfertype = nnvm::Op::GetAttr<FInferType>("InferType");

  NodeAttrs o_attrs = nnvm::get<GradientParam>(attrs.parsed).original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

  uint32_t o_num_outputs = o_attrs.op->num_outputs;
  if (o_attrs.op->get_num_outputs)
    o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

  CHECK_EQ(in_attrs->size(), o_num_inputs + o_num_outputs);
  CHECK_EQ(out_attrs->size(), o_num_inputs);

  for (size_t i = 0; i < o_num_inputs; ++i) {
    NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, i, (*in_attrs)[i]);
    NNVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, i, (*out_attrs)[i]);
  }

  if (finfertype.count(o_attrs.op)) {
    std::vector<int> o_out_attrs(in_attrs->begin() + o_num_inputs, in_attrs->end());
    finfertype[o_attrs.op](o_attrs, out_attrs, &o_out_attrs);
    for (size_t i = 0; i < o_num_outputs; ++i) {
      NNVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, o_num_inputs + i, o_out_attrs[i]);
    }
  }

  return true;
}

Array<Tensor> GradientCompute(const NodeAttrs& attrs,
                              const Array<Tensor>& inputs,
                              const Array<Tensor>& out_info) {
  static auto& ftvmcompute = nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");

  NodeAttrs o_attrs = nnvm::get<GradientParam>(attrs.parsed).original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

  uint32_t o_num_outputs = o_attrs.op->num_outputs;
  if (o_attrs.op->get_num_outputs)
    o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

  Array<Tensor> o_inputs(inputs.begin(), inputs.begin() + o_num_inputs);
  Array<Tensor> head_grads(inputs.begin() + o_num_inputs, inputs.end());

  Array<Tensor> input_placeholders;
  std::unordered_map<Tensor, Tensor> placeholders_to_inputs;
  for (const Tensor& input : o_inputs) {
    Tensor place =
      tvm::PlaceholderOpNode::make(input->op->name, input->shape, input->dtype).output(0);
    input_placeholders.push_back(place);
    placeholders_to_inputs[place] = input;
  }

  Array<Tensor> forward = ftvmcompute[o_attrs.op](o_attrs, input_placeholders, head_grads);

  Array<Tensor> results;

  for (const Tensor& place : input_placeholders) {
    Tensor res;
    auto head_grads_iter = head_grads.begin();
    for (const Tensor& out : forward) {
      Tensor jac = tvm::ir::Jacobian(out, place);

      Array<tvm::Expr> res_shape(jac->shape.begin() + o_num_outputs, jac->shape.end());
      Array<tvm::Expr> iter_vars_expr;
      Array<tvm::IterVar> iter_vars;
      for (size_t i = 0; i < o_num_inputs; ++i) {
        auto ivar = tvm::reduce_axis(tvm::Range(0, jac->shape[o_num_outputs + i]), "k");
        iter_vars.push_back(ivar);
        iter_vars_expr.push_back(ivar);
      }
      auto func =
        [head_grads_iter, &iter_vars, &iter_vars_expr, &jac](const Array<tvm::Var>& input_indices) {
          Array<tvm::Expr> jac_indices(iter_vars_expr);
          for (auto& v : input_indices)
            jac_indices.push_back(v);
          return tvm::sum((*head_grads_iter)(iter_vars_expr)*jac(jac_indices), iter_vars);
        };
      Tensor part = tvm::compute(res_shape, func, "gradient", topi::kMatMul);

      if (res.operator->()) {
        res = topi::add(res, part);
      }
      else
        res = part;

      ++head_grads_iter;
    }
    results.push_back(res);
  }

  return results;
}

NNVM_REGISTER_OP(gradient)
.describe(R"doc(Gradients for any specified operation.
)doc" NNVM_ADD_FILELINE)
//.set_support_level(1)
.set_num_inputs(([](const NodeAttrs& attrs) {
    NodeAttrs o_attrs = nnvm::get<GradientParam>(attrs.parsed).original();

    uint32_t o_num_inputs = o_attrs.op->num_inputs;
    if (o_attrs.op->get_num_inputs)
      o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

    uint32_t o_num_outputs = o_attrs.op->num_outputs;
    if (o_attrs.op->get_num_outputs)
      o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

    return o_num_inputs + o_num_outputs;
  }))
.set_num_outputs(([](const NodeAttrs& attrs) {
    NodeAttrs o_attrs = nnvm::get<GradientParam>(attrs.parsed).original();
    if (o_attrs.op->get_num_inputs)
      return o_attrs.op->get_num_inputs(o_attrs);
    else
      return o_attrs.op->num_inputs;
  }))
.set_attr_parser(ParamParser<GradientParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<GradientParam>)
.add_arguments(GradientParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", GradientShape)
.set_attr<FInferType>("FInferType", GradientType)
//.set_attr<FCorrectLayout>("FCorrectLayout", DotCorrectLayout)
.set_attr<FTVMCompute>("FTVMCompute", GradientCompute)
.set_attr<FTVMSchedule>("FTVMSchedule",
  [](const NodeAttrs& attrs, const Array<Tensor>& outs, const std::string& target) {
    Array<tvm::Operation> out_ops;
    for (auto t : outs)
      out_ops.push_back(t->op);
    return create_schedule(out_ops);
  });

}  // namespace top
}  // namespace nnvm
