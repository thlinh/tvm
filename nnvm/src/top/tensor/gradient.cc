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
#include <tvm/api_registry.h>

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
  StrDict original_attrs;
  int input_index;

  DMLC_DECLARE_PARAMETER(GradientParam) {
    DMLC_DECLARE_FIELD(original_op);
    DMLC_DECLARE_FIELD(original_attrs);
    DMLC_DECLARE_FIELD(input_index);
  }

  NodeAttrs original() const {
    NodeAttrs res;
    res.op = Op::Get(original_op);
    for (const auto& key_val_pair : original_attrs.dict)
      res.dict[key_val_pair.first] = key_val_pair.second;
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
  auto parsed = nnvm::get<GradientParam>(attrs.parsed);
  NodeAttrs o_attrs = parsed.original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

  uint32_t o_num_outputs = o_attrs.op->num_outputs;
  if (o_attrs.op->get_num_outputs)
    o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

  CHECK_EQ(in_attrs->size(), o_num_inputs + o_num_outputs);
  CHECK_EQ(out_attrs->size(), 1);

  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, (*in_attrs)[parsed.input_index]);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_attrs, parsed.input_index, (*out_attrs)[0]);

  return true;
}

inline bool GradientType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  auto parsed = nnvm::get<GradientParam>(attrs.parsed);
  NodeAttrs o_attrs = parsed.original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

  uint32_t o_num_outputs = o_attrs.op->num_outputs;
  if (o_attrs.op->get_num_outputs)
    o_num_outputs = o_attrs.op->get_num_outputs(o_attrs);

  CHECK_EQ(in_attrs->size(), o_num_inputs + o_num_outputs);
  CHECK_EQ(out_attrs->size(), 1);

  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, (*in_attrs)[parsed.input_index]);
  NNVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, parsed.input_index, (*out_attrs)[0]);

  return true;
}

Array<Tensor> GradientCompute(const NodeAttrs& attrs,
                              const Array<Tensor>& inputs,
                              const Array<Tensor>& out_info) {
  static auto& ftvmcompute = nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");

  auto parsed = nnvm::get<GradientParam>(attrs.parsed);
  NodeAttrs o_attrs = parsed.original();

  uint32_t o_num_inputs = o_attrs.op->num_inputs;
  if (o_attrs.op->get_num_inputs)
    o_num_inputs = o_attrs.op->get_num_inputs(o_attrs);

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

  const Tensor& place = input_placeholders[parsed.input_index];
  Tensor res;
  auto head_grads_iter = head_grads.begin();

  for (const Tensor& out : forward) {
    Tensor part = tvm::ir::JacobianRecursive(out, {place}, *head_grads_iter)[0];
    part = tvm::TensorNode::make(part->shape, part->dtype,
            part->op->ReplaceInputs(part->op, placeholders_to_inputs), part->value_index);

    if (res.operator->()) {
      res = topi::add(res, part);
    }
    else
      res = part;

    ++head_grads_iter;
  }

  return {res};
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
.set_num_outputs(1)
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

std::vector<NodeEntry> AutomaticFGradient(const NodePtr& n, const std::vector<NodeEntry>& ograds) {
  std::vector<NodeEntry> grad_inputs(n->inputs);
  grad_inputs.insert(grad_inputs.end(), ograds.begin(), ograds.end());

  std::vector<NodeEntry> result;

  for (uint32_t i = 0; i < n->num_inputs(); ++i) {
    std::unordered_map<std::string, std::string> grad_attrs;
    std::ostringstream orig_attrs;
    orig_attrs << StrDict{n->attrs.dict};
    grad_attrs["original_op"] = n->op()->name;
    grad_attrs["original_attrs"] = orig_attrs.str();
    grad_attrs["input_index"] = std::to_string(i);
    result.push_back(nnvm::MakeNode("gradient", n->attrs.name + "_grad", grad_inputs, grad_attrs));
  }

  return result;
}

void MakeDifferentiable(const std::string& op_name, int plevel = 100) {
  Op& op = dmlc::Registry<Op>::Get()->__REGISTER_OR_GET__(op_name);
  if (op.num_inputs == kVarg)
    std::cerr << op_name <<
      " accepts variable number of arguments, automatic gen of gradients is not supported\n";
  else
    op.set_attr<FGradient>("FGradient", AutomaticFGradient, plevel);
}

void MakeDifferentiableAll(int plevel = 100) {
  for (auto op : dmlc::Registry<Op>::List()) {
    MakeDifferentiable(op->name, plevel);
  }
}

TVM_REGISTER_API("MakeDifferentiable")
  .set_body([](tvm::TVMArgs args,  tvm::TVMRetValue *ret) {
      MakeDifferentiable(args[0]);
    });
TVM_REGISTER_API("MakeDifferentiableAll")
  .set_body([](tvm::TVMArgs args,  tvm::TVMRetValue *ret) {
      MakeDifferentiableAll();
    });

}  // namespace top
}  // namespace nnvm
