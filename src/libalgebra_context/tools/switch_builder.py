import argparse
import collections
import functools
import json
import sys

from pathlib import Path

SWITCH_INL_NAME = "switch.inl"
ENABLED_INL_NAME = "switch_enabled.inl"
CMAKE_LIST_SEP = ";"


def print_message(message):
    print(message, file=sys.stderr)


def print_file(filename):
    print(filename, end=CMAKE_LIST_SEP)


class Writer:
    indent_string = ' '*4

    def __init__(self, file, default_context=None):
        self.file = file
        self._indentation = 0
        self.ctx = collections.ChainMap(default_context or {})

    def indentation(self):
        return self.indent_string * self._indentation

    def increase_indent(self):
        self._indentation += 1

    def decrease_indent(self):
        if self._indentation > 0:
            self._indentation -= 1

    def push_context(self, ctx):
        self.ctx = self.ctx.new_child(ctx)

    def pop_context(self):
        self.ctx = self.ctx.parents

    def _write(self, format_string):
        self.file.write(self.indentation() + format_string.format(**self.ctx))

    def __lshift__(self, arg):
        if isinstance(arg, (list, tuple)):
            for s in arg:
                self._write(str(s) + "\n")
        else:
            self._write(str(arg))
        return self


class SwitchCase:

    def __init__(self, case_val, format_func):
        self.case_val = case_val
        self.format_func = format_func

    def __call__(self, writer):
        return self.format_func(writer)


class SwitchStatement:

    def __init__(self, writer, switch_var):
        self.switch_var = switch_var
        self.writer = writer
        self.ctx = {switch_var: None}

    def __enter__(self):
        self.writer << "switch (%s) {{\n" % self.switch_var
        self.writer.increase_indent()
        self.writer.push_context(self.ctx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_val
        self.writer.pop_context()
        self.writer.decrease_indent()
        self.writer << "}}\n"

    def __lshift__(self, arg):
        if isinstance(arg, SwitchCase):
            self.ctx[self.switch_var] = arg.case_val
            self.writer << "case {}:\n".format(arg.case_val)
            self.writer.increase_indent()
            arg(self.writer)
            # self.writer << "\n"
            self.writer.decrease_indent()
            return self
        return NotImplemented

    def write_default(self, default):
        if default is not None:
            self.writer << "default:\n"
            self.writer.increase_indent()
            default(self.writer)
            self.writer.decrease_indent()


class RegisterFunction:

    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.name = "register_la_context_{width}_{depth}".format(width=width, depth=depth)

    def __call__(self, dir, namespaces, fq_mapname):
        fq_name = "::".join([*namespaces, self.name])
        path = dir / "{}.cpp".format(self.name)
        with open(path, "wt") as fp:
            fp.writelines([
                "#include \"register_la_contexts.h\"\n",
                "void {name}({mapname}& map)\n".format(name=fq_name, mapname=fq_mapname),
                "{\n",
                "    using context_t = esig::algebra::libalgebra_context<{width}, "
                "{depth}>;\n".format(width=self.width, depth=self.depth),
                "    map[{{{width}, {depth}}}] = std::shared_ptr<esig::algebra::context>(new "
                "context_t());\n".format(width=self.width, depth=self.depth),
                "}\n"
            ])
        print_file(path)


class MainRegisterFunction:
    main_fname = "register_la_contexts"
    seen_macro = "ESIG_LIBALGEBRA_CONTEXT_REGISTRY_H_"
    namespaces = ("esig", "algebra", "dtl")

    def __init__(self, dir):
        self.dir = dir
        self.header_path = header_path = dir / "{}.h".format(self.main_fname)
        self.file_path = file_path = dir / "{}.cpp".format(self.main_fname)
        self.file_writer = Writer(open(file_path, "wt"))
        self.header_writer = Writer(open(header_path, "wt"))
        self.mapname = "context_map"
        self.fq_mapname = "::".join([*self.namespaces, self.mapname])
        print_file(file_path)
        print_file(header_path)

    def _start_header(self):
        self.header_writer << [
            "#ifndef {}".format(self.seen_macro),
            "#define {}".format(self.seen_macro),
            "#include <esig/libalgebra_context/libalgebra_context.h>\n",
        ] + ["namespace {} {{{{".format(name) for name in self.namespaces]

    def _end_header(self):
        self.header_writer << "void {}({}& map);\n".format(self.main_fname, self.mapname)
        self.header_writer << ["}}}} // namespace {}".format(name) for name in reversed(self.namespaces)]
        self.header_writer << "#endif // {}\n".format(self.seen_macro)

    def __enter__(self):
        self.dir.mkdir(exist_ok=True, parents=True)
        self._start_header()

        fq_name = "::".join([*self.namespaces, self.main_fname])
        try:
            self.file_writer << "#include \"{}.h\"\n".format(self.main_fname)
            self.file_writer << "void {}({}& map)\n{{{{\n".format(fq_name, self.fq_mapname)
        except BaseException:
            self.file_writer.file.close()
            self.header_writer.file.close()
            raise

        self.file_writer.increase_indent()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.file_writer.file.close()
            self.header_writer.file.close()
            raise exc_val

        self._end_header()
        self.file_writer.decrease_indent()
        self.file_writer << "}}\n"
        self.file_writer.file.close()
        self.header_writer.file.close()

    def __lshift__(self, other):
        if isinstance(other, RegisterFunction):
            fq_name = "::".join([*self.namespaces, other.name])
            self.header_writer << "void {}({}& map);\n".format(other.name, self.mapname)
            self.file_writer << "{}(map);\n".format(fq_name)
            other(self.dir, self.namespaces, self.fq_mapname)
            return self
        return NotImplemented


def write_template_fn(writer):
    writer << "return TemplatedFn({width}, {depth});\n"


def wd_oob_exception(writer):
    writer << "throw std::runtime_error(\"invalid width-depth combination {width}-{depth}\");\n"


def w_oob_exception(writer):
    writer << "throw std::runtime_error(\"invalid width {width}\");\n"


def write_depth_switch(inner, cases, writer, default=None):
    with SwitchStatement(writer, "depth") as switch:
        for case in cases:
            switch << SwitchCase(case, inner)
        switch.write_default(default)


def write_width_switch(inner, cases, writer, width_default=None, depth_default=None):
    with SwitchStatement(writer, "width") as switch:
        for wcase, dcases in cases.items():
            func = functools.partial(write_depth_switch, inner, dcases, default=depth_default)
            switch << SwitchCase(wcase, func)
        switch.write_default(width_default)


def write_get_switch(cases, out_dir):
    path = out_dir / SWITCH_INL_NAME
    out_dir.mkdir(exist_ok=True, parents=True)
    with path.open("wt") as f:
        writer = Writer(f)
        write_width_switch(write_template_fn, cases, writer,
                           width_default=w_oob_exception,
                           depth_default=wd_oob_exception)


def write_return_false(writer):
    writer << "return false;"


def write_enabled_switch(cases, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    path = out_dir / ENABLED_INL_NAME
    with path.open("wt") as f:
        writer = Writer(f)
        write_width_switch(lambda w: w << "return true;", cases, writer,
                           width_default=write_return_false,
                           depth_default=write_return_false)


class ExternHeader:
    guard_name = "ESIG_LA_CONFIG_EXTERN_H_SEEN_"
    main_header_name = "esig_la_default_context_header.h"
    extern_cpp_name = "esig_extern_{width}_{depth}.cpp"

    def __init__(self, cases, classname, namespaces=None):
        self.cases = cases
        self.classname = classname
        self.namespaces = namespaces or []

    def _write_guard_head(self, file):
        file.write("#ifndef {guard}\n#define {guard}\n\n".format(guard=self.guard_name))

    def _write_guard_tail(self, file):
        file.write("\n\n#endif // {}\n".format(self.guard_name))

    def _open_namespaces(self, file):
        for ns in self.namespaces:
            file.write("namespace {} {{\n".format(ns))

    def _close_namespaces(self, file):
        for ns in self.namespaces:
            file.write("}} // namespace {}\n".format(ns))

    def _write_extern_decl(self, file, width, depth):
        file.write("extern template class {name}<{width},{depth}>;\n".format(
            name=self.classname, width=width, depth=depth
        ))

    def _write_extern_cpp(self, width, depth, dir):
        path = dir / self.extern_cpp_name.format(width=width, depth=depth)
        print_file(path)
        fq_name = "::".join([*self.namespaces, self.classname])
        with path.open("wt") as fp:
            # fp.write("#include \"{}\"\n\n".format(self.main_header_name))
            # fp.write("template class {name}<{width}, {depth}>;\n".format(
            #     name=fq_name, width=width, depth=depth
            # ))
            fp.write("#include <esig/libalgebra_context/libalgebra_context.h>\n")
            fp.write("ESIG_DECLARE_LA_CONTEXT({width}, {depth})\n".format(width=width, depth=depth))

    def write(self, dir):
        dir.mkdir(exist_ok=True, parents=True)
        main_path = dir / self.main_header_name

        # with main_path.open("wt") as fp:
        #     self._write_guard_head(fp)
        #     fp.write("#include <esig/libalgebra_context/libalgebra_context.h>\n\n")
        #     self._open_namespaces(fp)
        #     for wcase, dcases in self.cases.items():
        #         for dcase in dcases:
        #             self._write_extern_decl(fp, wcase, dcase)
        #     self._close_namespaces(fp)
        #     self._write_guard_tail(fp)


        for wcase, dcases in self.cases.items():
            for dcase in dcases:
                self._write_extern_cpp(wcase, dcase, dir)


def load_config(path):
    with path.open("rt") as f:
        return json.load(f)


def get_register_functions(config):
    return [RegisterFunction(width, depth)
            for width, depths in config.items()
            for depth in depths]


def write_register_funcs(dir, funcs):
    with MainRegisterFunction(dir) as mfunc:
        for fn in funcs:
            mfunc << fn


def main():
    default_path = Path(__file__).parent.parent / "config" / "default.json"
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=Path, default=Path.cwd(),
                        help="path to directory where files should be created")
    parser.add_argument("-c", "--config", type=Path, default=default_path,
                        help="path to configuration (json) file")
    # parser.add_argument("--names-only", action="store_true")
    # parser.add_argument("--no-switch", action="store_true")
    # parser.add_argument("--no-enable", action="store_true")
    # parser.add_argument("--no-extern", action="store_true")

    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        parser.exit(1, "no such file {}\n".format(config_path))

    config = load_config(config_path)

    output_path = args.output.resolve()
    if output_path.is_file():
        parser.exit(1, "{} is a file\n".format(output_path))

    # if args.names_only:
    #     print_file(output_path / ExternHeader.main_header_name)
    #     for wcase, dcases in config.items():
    #         for dcase in dcases:
    #             print_file(output_path / ExternHeader.extern_cpp_name.format(
    #                 width=wcase, depth=dcase
    #             ))
    #     parser.exit(0)
    #
    # if not args.no_switch:
    #     write_get_switch(config, output_path)
    #
    # if not args.no_enable:
    #     write_enabled_switch(config, output_path)
    #
    # if not args.no_extern:
    output_path.mkdir(exist_ok=True, parents=True)
    write_register_funcs(output_path, get_register_functions(config))
        # extern_header = ExternHeader(config, "libalgebra_context", ["esig", "algebra"])
        # extern_header.write(output_path)

    parser.exit(0)


if __name__ == "__main__":
    main()
