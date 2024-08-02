"""GenEM Physics-Informed Generative Cryo-Electron Microscopy"""


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    import genem

    parser.add_argument(
        "--version", action="version", version="GenEM " + genem.__version__
    )

    import genem.commands.train
    import genem.commands.test
    import genem.commands.gen_data
    import genem.commands.esti_ice
    import genem.commands.video
    import genem.commands.gallery
    import genem.commands.analysis_fspace
    

    modules = [
        genem.commands.train,
        genem.commands.test,
        genem.commands.gen_data,
        genem.commands.esti_ice,
        genem.commands.video,
        genem.commands.gallery,
        genem.commands.analysis_fspace,
    ]

    subparsers = parser.add_subparsers(title="Choose a command")
    subparsers.required = True

    def get_str_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]

    for module in modules:
        this_parser = subparsers.add_parser(
            get_str_name(module), description=module.__doc__
        )
        module.add_args(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
