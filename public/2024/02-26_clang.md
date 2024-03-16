```py
init_rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2), 'emb_dropout': jax.random.PRNGKey(3)}
key = jax.random.PRNGKey(5)
features = jax.random.normal(key, (4, 256, 1024))
merger = PatchMerger(dim=1024, num_tokens_out=8)
mei

```
## Test
```py
cpp_file_path = '/home/leondgarse/workspace/ModelLink/mindie_ref/mindie_llm/atb_models/models/llama/layer/flash_attention_layer.cpp'
from clang import cindex
from llm.transform import utils
from llm.transform import transform_quant
from llm.transform.transform_quant_cpp_layer_function import TransformQuantCppLayerFunction

def print_spelling(param, info="", level="debug"):
    param = param.get_children() if hasattr(param, "get_children") else param
    message = info + "[" + ", ".join([ii.spelling for ii in param]) + "]"
    print(message)

cursor, contents = transform_quant.parse_file_as_cursor(cpp_file_path)
children = list(next(list(cursor.get_children())[-1].get_children()).get_children())
print_spelling(children, info="Children parts from cpp: ", level="info")

indent = 4
in_tensor_added = transform_quant.add_scale_bias_in_enum(contents, children[3], indent)[-1]

cur_cursor = children[-1]
aa = TransformQuantCppLayerFunction(contents, children[-1], in_tensor_added, indent=indent)
```
```py
cpp_file = "/home/leondgarse/workspace/ModelLink/mindie_ref/mindie_llm/atb_models/models/baichuan2/7b/layer/flash_attention_rope_layer.cpp"
h_file = "/home/leondgarse/workspace/ModelLink/mindie_ref/mindie_llm/atb_models/models/baichuan2/7b/layer/flash_attention_rope_layer.h"

```
## Parse
```py
input_code = """
struct Person
{
    int age;
    const char* name;
};

int main()
{
    Person person = { 1, "John" };
    person.age = 2;
    int cc = 2;
    return 0;
}
"""

from clang import cindex

tu = cindex.Index.create(excludeDecls=True).parse('main.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', input_code)])
translation_unit = index.parse('example.cpp', args=['-std=c++11'], unsaved_files=[('example.cpp', source_code)])

aa, bb = list(tu.cursor.get_children())
ff = list(list(bb.get_children())[0].get_children())[1]
print([ii.spelling for ii in ff.get_tokens()])

gg = list(ff.get_tokens())[-1]
print(input_code[:gg.extent.start.offset] + "4" + input_code[gg.extent.end.offset:])
```
## Functions
```py
Please learn the style of this example thansforming c++ code using python clang interface.
contents = """
namespace test_model {
enum FlashAttentionRopeLayerTensorId : int {
    IN_HIDDENSTATES = 0,

    IN_NORMWEIGHT,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,    
    OUT_LAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
};
}
"""

USING_SCALE_BIAS_ITEMS = ["IN_QKV", "IN_SELFOUT", "IN_MLP"]
INTERMIDATE_PREFIX = "INTERMIDATE_"

def find_first_enum_recursion(cursor):
    for cur in cursor.get_children():
        if cur.kind == cindex.CursorKind.ENUM_DECL:
            return cur
        ret = find_first_enum_recursion(cur)
        if ret is not None:
            return ret


def add_scale_bias_in_enum(contents, enum_cursor, indent=4):
    added_items, insert_position, is_intermodate_found = [], enum_cursor.extent.end.offset - 1, False
    for enum_item in enum_cursor.get_children():
        enum_item_spelling = enum_item.spelling
        if any([ii in enum_item_spelling for ii in USING_SCALE_BIAS_ITEMS]):
            # print(enum_item_spelling)
            added_items.append(enum_item_spelling + "_DEQSCALE,")
            added_items.append(enum_item_spelling + "_BIAS,")  # [TODO] check if bias already exists
        if not is_intermodate_found and enum_item_spelling.startswith(INTERMIDATE_PREFIX):
            insert_position = contents[:enum_item.extent.start.offset].rfind('\n') + 1
            is_intermodate_found = True

    indent_prefix = "\n" + " " * indent
    insert_contents = indent_prefix + "// Quant weights" + indent_prefix + indent_prefix.join(added_items) + "\n"
    return insert_contents, insert_position, insert_position
    # return added_items

from clang import cindex

# contents = open('flash_attention_rope_layer.cpp').read()
tu = cindex.Index.create(excludeDecls=True).parse('temp.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', contents)])
attention_enum = find_first_enum_recursion(tu.cursor)
# print([ii.spelling for ii in list(attention_enum.get_children())])
insert_contents, insert_start, insert_end = add_scale_bias_in_enum(contents, attention_enum)
print(contents[:insert_start] + insert_contents + contents[insert_end:])

Now try using this coding style and clang==14.0 interface, give me python code transforming:
    atb::Status FlashAttentionRopeLayer(){
        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.commDownParam.backend = param.backend;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        mlpParam.transposeB = true;
        mlpParam.isBias = false;
        mlpParam.isPack = false;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    }
To:
    atb::Status FlashAttentionRopeLayer(){
        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.isBias = true;
        mlpParam.isPack = false;
        mlpParam.isQuant = true;
        mlpParam.transposeB = true;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.commDownParam.backend = param.backend;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        // add quant op
        mlpParam.quantDownParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantDownParam.isQuantOp = true;
        mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        mlpParam.quantDownParam.inputScale = param.down_projInputScale;
        mlpParam.quantDownParam.inputOffset = param.down_projInputOffset;

        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    }

Yes, now try using this coding style and python clang interface, write a python code transforming:
    atb::Status FlashAttentionRopeLayer()
    {
        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.commDownParam.backend = param.backend;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        mlpParam.transposeB = true;
        mlpParam.isBias = false;
        mlpParam.isPack = false;
        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    }
To:
    atb::Status FlashAttentionRopeLayer()
    {
        atb_speed::common::MlpGateParamV2 mlpParam;
        mlpParam.commDownParam.rank = param.rank;
        mlpParam.commDownParam.rankSize = param.rankSize;
        mlpParam.commDownParam.backend = param.backend;
        mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
        mlpParam.transposeB = true;
        mlpParam.isBias = true;
        mlpParam.isPack = false;
        mlpParam.isQuant = true;
        // add quant op
        mlpParam.quantDownParam.quantType = atb::infer::QUANT_INT8;
        mlpParam.quantDownParam.isQuantOp = true;
        mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
        mlpParam.quantDownParam.inputScale = param.down_projInputScale;
        mlpParam.quantDownParam.inputOffset = param.down_projInputOffset;

        atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    }

Please notice those parameters like `mlpParam.isBias` or `mlpParam.transposeB` may exist or not, and its value may different from the target added, so all of them should be checked. If those parameters not exists in source code, should also add them. Also please use `get_tokens` for parsing function's contents

```
```py
from clang import cindex

def find_function_cursor(cursor, function_name):
    for cur in cursor.get_children():
        if cur.kind == cindex.CursorKind.FUNCTION_DECL and cur.spelling == function_name:
            return cur
        result = find_function_cursor(cur, function_name)
        if result:
            return result

def add_parameter_if_not_exists(contents, function_cursor, param_name, default_value):
    tokens = list(function_cursor.get_tokens())
    for token in tokens:
        if token.spelling == param_name:
            return contents

    indent = " " * 4
    insert_contents = "\n" + indent + f"{param_name} = {default_value};"
    return contents[:-1] + insert_contents + contents[-1]

def transform_code(contents, function_name):
    tu = cindex.Index.create().parse('temp.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', contents)])
    function_cursor = find_function_cursor(tu.cursor, function_name)

    if function_cursor:
        params_to_add = {
            "isBias": "true",
            "isPack": "false",
            "isQuant": "true",
            "quantDownParam.quantType": "atb::infer::QUANT_INT8",
            "quantDownParam.isQuantOp": "true",
            "quantDownParam.elewiseType": "atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT",
            "quantDownParam.inputScale": "param.down_projInputScale",
            "quantDownParam.inputOffset": "param.down_projInputOffset"
        }

        for param_name, default_value in params_to_add.items():
            contents = add_parameter_if_not_exists(contents, function_cursor, param_name, default_value)

    return contents

source_code = """
atb::Status FlashAttentionRopeLayer(){
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.commDownParam.backend = param.backend;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = true;
    mlpParam.isBias = false;
    mlpParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
}
"""

transformed_code = transform_code(source_code, "FlashAttentionRopeLayer")
print(transformed_code)
```
## MLP
```cpp
Transforming this code using python clang==14.0 interface like get_children and get_tokens.
Source:
struct Person
{
    int age;
    const char* name;
};

int main()
{
    Person person = { 1, "John" };
    person.age = 2;
    int cc = 2;
    return 0;
}

Target:
struct Person
{
    int age;
    const char* name;
};

int main()
{
    Person person = { 1, "John" };
    person.age = 4;
    int cc = 2;
    return 0;
}
```
## Practices
```py
from clang import cindex
from clang.cindex import Index, Config, CursorKind, TypeKind

file_path = "test.cpp"
index = Index.create()
tu = index.parse(file_path)

AST_root_node = tu.cursor
print(AST_root_node)

node_list = []
def preorder_travers_AST(cursor, depth=0):
    for cur in cursor.get_children():
        print(depth, cur.spelling)
        print("  ", [token.spelling for token in cur.get_tokens()])
        preorder_travers_AST(cur, depth=depth+1)
preorder_travers_AST(AST_root_node)

cursor_content = ""
for token in AST_root_node.get_tokens():
    print(token.spelling)
```

Give me a runable example transform c++ code using python clang interface. Like transformering:
#include <aaa.hpp>
int main() {
  int aa = 11;
  int bb = 11;
}
to:
#include <aaa_qq.hpp>
int main() {
  int aa = 22;
  int bb = 11;
}
```py
import clang.cindex

input_code = """
#include <aaa.hpp>
int main() {
  int aa = 11;
  int bb = 11;
}
"""

index = clang.cindex.Index.create()
translation_unit = index.parse('temp.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', input_code)])
for node in translation_unit.cursor.walk_preorder():
    if node.kind == clang.cindex.CursorKind.INCLUSION_DIRECTIVE and 'aaa.hpp' in node.displayname:
        new_text = '#include <aaa_qq.hpp>'
        node.extent.replace(new_text)
    elif node.kind == clang.cindex.CursorKind.INTEGER_LITERAL and node.displayname == '11':
        node.extent.replace('22')

def transform_cpp_code(input_code):
    # Parse the input C++ code
    index = clang.cindex.Index.create()
    translation_unit = index.parse('temp.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', input_code)])

    # Traverse the AST to find and modify nodes
    for node in translation_unit.cursor.walk_preorder():
        if node.kind == clang.cindex.CursorKind.INCLUSION_DIRECTIVE and 'aaa.hpp' in node.displayname:
            # Modify the inclusion directive
            new_text = '#include <aaa_qq.hpp>'
            node.extent.replace(new_text)

        elif node.kind == clang.cindex.CursorKind.INTEGER_LITERAL and node.displayname == '11':
            # Modify the integer literal
            node.extent.replace('22')

    # Generate the transformed code
    transformed_code = translation_unit.cursor.translation_unit.spelling

    return transformed_code

if __name__ == "__main__":
    # Example C++ code

"""
