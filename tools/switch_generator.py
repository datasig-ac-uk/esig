import os


class SwitchGenerator:
    _path = "src"
    _default_spec = {
    #letters_range : max_depth
        (2, 2) : 16,
        (3, 3) : 10,
        (4, 4) : 8,
        (5, 6) : 6,
        (7, 9) : 5,
        #(10, 16) : 4,
        #(17, 40) : 3,
        #(41, 256) : 2
    }
    indent_str = "    "
    endln = "\n"
    template_fn_macro = "TemplatedFn"

    @property
    def default_spec(self):
        return dict(
            (i, d) for (mina, maxa), d in self._default_spec.items()
            for i in range(mina, maxa+1)
        )


    def __init__(self, spec=None, types=None, path=None):
        self.spec = spec or self.default_spec
        self.types = types or ["DPReal", "SPReal"]
        self.indentation = 0
        self._file = None
        self.path = path or self._path


    def _write_file(self):
        self.enter_switch("width")
        for k, v in self.spec.items():
            self.write_case(k)
            self.write_depth_switch(k, v)
        self.write_width_default()
        self.exit_switch()

    def write_depth_switch(self, w, max_depth):
        self.enter_switch("depth")
        for d in range(2, max_depth+1):
            self.write_case(d)
            self.write_templatefn(w, d, None)
            #self.write_type_switch(w, d)
        self.write_depth_default(w, max_depth)
        self.exit_switch()
        self.write_break()

    def write_type_switch(self, w, d):
        self.enter_switch("coeff")
        for typ in self.types:
            self.write_case(typ)
            self.write_templatefn(w, d, typ)
        self.write_type_default()
        self.exit_switch()

    def write_file(self):
        path = os.path.join(self.path, "switch.h")
        with open(path, "wt", encoding="UTF-8") as f:
            self._file = f
            self._write_file()
        self._file = None
        # self.write_config_bounds_header()

    def enter_switch(self, var):
        self.writeln("switch ({}) {{".format(var))
        self.indentation += 1

    def exit_switch(self):
        self.indentation -= 1
        self.writeln("}")

    def write_case(self, n):
        self.writeln("case {} :".format(n))

    def write_templatefn(self, w, d, dtype):
        text = "return {}({}, {});".format(
            self.template_fn_macro, w, d,# dtype
        )
        self.writeln(text)
        self.write_break()

    def write_break(self):
        self.writeln("break;" + self.endln)

    def write_depth_default(self, w, d):
        self.writeln("default :")
        text = "Legitimate depth of 2<->{} for records with width {} exceeds limit"
        self.writeln(
            "throw std::runtime_error ( \"{}\" );".format(
                text.format(d, w)
            )
        )

    def write_type_default(self):
        self.writeln("default :")
        text = "This type is not supported."
        self.writeln("throw std::runtime_error ( \"{}\" );".format(text))

    def write_width_default(self):
        self.writeln("default :")
        text = "Legitimate width 2 <-> 256 exceeded"
        self.writeln(
            "throw std::runtime_error ( \"{}\" );".format(text)
        )

    def writeln(self, val):
        assert self._file
        self._file.write(
            self.indent_str*self.indentation + val + self.endln
        )


    def write_config_bounds_header(self):
        path = os.path.join(self.path, "config_bounds.h")
        with open(path, "wt", encoding="UTF-8") as f:
            self._file = f
            self._write_config_bounds_header()
            self.write_checker_switch_function()
            self.write_get_bounds_switch_function()
        self._file = None

    def _write_config_bounds_header(self):
        self.start_internal_namespace()
        self.write_struct()
        for depth in self.spec:
            self.write_struct(depth)
        self.end_internal_namespace()

    def start_internal_namespace(self):
        self.writeln("namespace {")
        self.indentation += 1

    def end_internal_namespace(self):
        self.indentation -= 1
        self.writeln("}")
        self.writeln(self.endln)

    def write_struct(self, width=None):
        self.write_struct_head(width)
        self.write_struct_values(width)
        self.write_struct_end()


    def write_struct_head(self, width):
        template_str = "" if width else "DEG W"
        template_args = "<{width}>".format(width=width) if width else ""
        self.writeln("template <{template_str}>".format(template_str=template_str))
        self.writeln("struct config_check{template_args}".format(template_args=template_args))
        self.writeln("{")
        self.indentation += 1

    def write_struct_values(self, width):
        if not width:
            self.writeln("")
            return None
        min_depth = 2
        max_depth = self.spec[width]
        self.writeln("static const DEG min_depth = {min_depth};".format(min_depth=min_depth))
        self.writeln("static const DEG max_depth = {max_depth};".format(max_depth=max_depth))

    def write_struct_end(self):
        self.indentation -= 1
        self.writeln("};")
        self.writeln(self.endln)

    def write_checker_switch_function(self):
        self.writeln("inline bool check_depth_config(DEG width, DEG depth)")
        self.writeln("{")
        self.indentation += 1

        self.writeln("DEG min_depth, max_depth;")
        self.enter_switch("width")
        for width in self.spec:
            typename = "config_check<{width}>".format(width=width)
            self.write_case(width)
            self.writeln("min_depth = {typename}::min_depth;".format(typename=typename))
            self.writeln("max_depth = {typename}::max_depth;".format(typename=typename))
            self.write_break()
        self.writeln("default:")
        self.writeln("return false;")
        self.exit_switch()

        self.writeln("return (depth <= max_depth && depth >= min_depth);")
        self.indentation -= 1
        self.writeln("}")
        self.writeln(self.endln)

    def write_get_bounds_switch_function(self):
        self.writeln("inline std::pair<DEG, DEG> get_depth_bounds_config(DEG width, DEG depth)")
        self.writeln("{")
        self.indentation += 1

        self.writeln("DEG min_depth, max_depth;")
        self.enter_switch("width")
        for width in self.spec:
            typename = "config_check<{width}>".format(width=width)
            self.write_case(width)
            self.writeln("min_depth = {typename}::min_depth;".format(typename=typename))
            self.writeln("max_depth = {typename}::max_depth;".format(typename=typename))
            self.write_break()
        self.writeln("default:")
        self.writeln("return std::make_pair(0, 0);")
        self.exit_switch()

        self.writeln("return std::make_pair(min_depth, max_depth);")
        self.indentation -= 1
        self.writeln("}")
        self.writeln(self.endln)


if __name__ == "__main__":
    g = SwitchGenerator(path="test.txt")

    g.write_file()