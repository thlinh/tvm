/*!
 *  Copyright (c) 2018 by Contributors
 * \file jacobian.cc
 * \brief Automatic jacobians for any operation
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

struct JacobianParam : public dmlc::Parameter<JacobianParam> {
  std::string original_op;
  std::string original_name;
  StrDict original_attrs;
  int output;

  DMLC_DECLARE_PARAMETER(JacobianParam) {
    DMLC_DECLARE_FIELD(original_op);
    DMLC_DECLARE_FIELD(original_name);
    DMLC_DECLARE_FIELD(original_attrs);
    DMLC_DECLARE_FIELD(output);
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

DMLC_REGISTER_PARAMETER(JacobianParam);

inline bool JacobianShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  static auto& finfershape = nnvm::Op::GetAttr<FInferShape>("FInferShape");

  auto parsed = nnvm::get<JacobianParam>(attrs.parsed);
  NodeAttrs o_attrs = parsed.original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

  uint32_t o_num_outputs = o_attrs.op->num_outputs;
  if (o_attrs.op->get_num_outputs)
    o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

  CHECK_EQ(in_attrs->size(), o_num_inputs);
  CHECK_EQ(out_attrs->size(), o_num_inputs);

  std::vector<TShape> o_out_attrs(o_num_outputs);
  finfershape[o_attrs.op](o_attrs, in_attrs, &o_out_attrs);

  for (size_t i = 0; i < o_num_inputs; ++i) {
    TShape ith_out_shape(o_out_attrs[parsed.output].ndim() + (*in_attrs)[i].ndim());

    size_t j = 0;
    for (auto k : o_out_attrs[parsed.output]) {
      ith_out_shape[j] = k;
      ++j;
    }
    for (auto k : (*in_attrs)[i]) {
      ith_out_shape[j] = k;
      ++j;
    }

    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, i, ith_out_shape);
  }

  return true;
}

inline bool JacobianType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  static auto& finfertype = nnvm::Op::GetAttr<FInferType>("FInferType");

  auto parsed = nnvm::get<JacobianParam>(attrs.parsed);
  NodeAttrs o_attrs = parsed.original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

  uint32_t o_num_outputs = o_attrs.op->num_outputs;
  if (o_attrs.op->get_num_outputs)
    o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

  CHECK_EQ(in_attrs->size(), o_num_inputs);
  CHECK_EQ(out_attrs->size(), o_num_inputs);

  std::vector<int> o_out_attrs(o_num_outputs);
  finfertype[o_attrs.op](o_attrs, in_attrs, &o_out_attrs);

  for (size_t i = 0; i < o_num_inputs; ++i)
    NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, i, o_out_attrs[parsed.output]);

  return true;
}

Array<Tensor> JacobianCompute(const NodeAttrs& attrs,
                              const Array<Tensor>& inputs,
                              const Array<Tensor>& out_info) {
  static auto& ftvmcompute = nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");

  auto parsed = nnvm::get<JacobianParam>(attrs.parsed);
  NodeAttrs o_attrs = parsed.original();

  Array<Tensor> input_placeholders;
  std::unordered_map<Tensor, Tensor> placeholders_to_inputs;
  for (const Tensor& input : inputs) {
    Tensor place =
      tvm::PlaceholderOpNode::make(input->op->name, input->shape, input->dtype).output(0);
    input_placeholders.push_back(place);
    placeholders_to_inputs[place] = input;
  }

  Array<Tensor> output_placeholders;
  auto inputs_iter = inputs.begin();
  for (const Tensor& out : out_info) {
    uint32_t out_size = out->shape.size() - (*inputs_iter)->shape.size();
    Array<tvm::Expr> out_shape(out->shape.begin(), out->shape.begin() + out_size);
    Tensor place =
      tvm::PlaceholderOpNode::make(out->op->name, out_shape, out->dtype).output(0);
    output_placeholders.push_back(place);
    ++inputs_iter;
  }

  Tensor forward =
    ftvmcompute[o_attrs.op](o_attrs, input_placeholders, output_placeholders)[parsed.output];

  Array<Tensor> results;

  for (const Tensor& place : input_placeholders) {
    Tensor jac = tvm::ir::Jacobian(forward, place);
    jac = tvm::TensorNode::make(jac->shape, jac->dtype,
            jac->op->ReplaceInputs(jac->op, placeholders_to_inputs), jac->value_index);
    results.push_back(jac);
  }

  return results;
}

NNVM_REGISTER_OP(jacobian)
.describe(R"doc(Jacobians for any specified operation.
)doc" NNVM_ADD_FILELINE)
//.set_support_level(1)
.set_num_inputs(([](const NodeAttrs& attrs) {
    NodeAttrs o_attrs = nnvm::get<JacobianParam>(attrs.parsed).original();
    if (o_attrs.op->get_num_inputs)
      return o_attrs.op->get_num_inputs(o_attrs);
    else
      return o_attrs.op->num_inputs;
  }))
.set_num_outputs(([](const NodeAttrs& attrs) {
    NodeAttrs o_attrs = nnvm::get<JacobianParam>(attrs.parsed).original();
    if (o_attrs.op->get_num_inputs)
      return o_attrs.op->get_num_inputs(o_attrs);
    else
      return o_attrs.op->num_inputs;
  }))
.set_attr_parser(ParamParser<JacobianParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<JacobianParam>)
.add_arguments(JacobianParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", JacobianShape)
.set_attr<FInferType>("FInferType", JacobianType)
//.set_attr<FCorrectLayout>("FCorrectLayout",
  //[](const NodeAttrs& attrs,
     //std::vector<Layout> *in_layouts,
     //const std::vector<Layout> *last_in_layouts,
     //std::vector<Layout> *out_layouts) {
    //return true;
  //})
.set_attr<FTVMCompute>("FTVMCompute", JacobianCompute)
.set_attr<FTVMSchedule>("FTVMSchedule",
  [](const NodeAttrs& attrs, const Array<Tensor>& outs, const std::string& target) {
    Array<tvm::Operation> out_ops;
    for (auto t : outs)
      out_ops.push_back(t->op);
    return create_schedule(out_ops);
  });

}  // namespace top
}  // namespace nnvm
