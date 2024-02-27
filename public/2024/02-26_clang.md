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
aa, bb = list(tu.cursor.get_children())
ff = list(list(bb.get_children())[0].get_children())[1]
print([ii.spelling for ii in ff.get_tokens()])

gg = list(ff.get_tokens())[-1]
print(input_code[:gg.extent.start.offset] + "4" + input_code[gg.extent.end.offset:])
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
