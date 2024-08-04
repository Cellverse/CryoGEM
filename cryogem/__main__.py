"""CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy"""


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    import cryogem

    parser.add_argument(
        "--version", action="version", version="CryoGEM " + cryogem.__version__
    )

    import cryogem.commands.train
    import cryogem.commands.test
    import cryogem.commands.gen_data
    import cryogem.commands.esti_ice
    import cryogem.commands.video
    import cryogem.commands.gallery
    import cryogem.commands.analysis_fspace
    

    modules = [
        cryogem.commands.train,
        cryogem.commands.test,
        cryogem.commands.gen_data,
        cryogem.commands.esti_ice,
        cryogem.commands.video,
        cryogem.commands.gallery,
        cryogem.commands.analysis_fspace,
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
