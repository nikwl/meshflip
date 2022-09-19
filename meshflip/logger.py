import logging


LOG = logging.getLogger("meshflip")


def add_logger_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    if args.debug:
        LOG.setLevel(logging.DEBUG)
    elif args.quiet:
        LOG.setLevel(logging.WARNING)
    else:
        LOG.setLevel(logging.INFO)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        LOG.addHandler(file_logger_handler)
